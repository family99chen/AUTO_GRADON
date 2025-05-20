import os
import torch
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time
from tqdm import tqdm, trange
import logging
from datetime import datetime
from contextlib import contextmanager
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

# Set tokenizers parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 导入自定义模块
from configuration.promptmaker import PromptMakerConfiguration
from create_config import create_config, load_and_generate_nodes
from evaluator import TestEvaluator
from grpo import GRPOTrainer

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing
# 设置Python标准多进程为spawn
multiprocessing.set_start_method('spawn', force=True)


class RAGOptimizer:
    """使用GRPO优化RAG系统的多组件配置"""
    
    # 所有可能的组件列表
    ALL_COMPONENTS = [
        "vectordb", 
        "query_expansion",
        "retrieval",
        "passage_augmenter",
        "passage_reranker",
        "passage_filter",
        "passage_compressor",
        "prompt_maker",
        "generator"
    ]
    
    @contextmanager
    def timer(self, name):
        start = time.time()
        yield
        end = time.time()
        self.logger.info(f"{name} 耗时: {end - start:.2f} 秒")

    def __init__(self, 
                 qa_data_path: str, 
                 corpus_data_path: str, 
                 project_dir: str,
                 config_path: str = "/home/xwh/AutoRAG/ragas/configuration/config.yaml",
                 trial_name: str = "rag_optimization",
                 num_gpus: int = 1,
                 target_components: Optional[List[str]] = None,
                 fixed_components: Optional[Dict[str, str]] = None,
                 use_cache: bool = True):
        """
        初始化优化器
        
        Args:
            qa_data_path: 问答数据路径
            corpus_data_path: 语料库数据路径
            project_dir: 项目目录
            config_path: 配置文件路径
            trial_name: 试验名称
            num_gpus: 可用GPU数量
            target_components: 需要优化的组件列表，如果为None则优化除generator外的所有组件
            fixed_components: 固定组件的配置，格式为 {"组件名": "方法名"}
            use_cache: 是否使用评估缓存
        """
        # 设置GPU数量
        self.num_gpus = num_gpus
        self.logger_initialized = False
        
        # 初始化评估器
        self.evaluator = TestEvaluator(qa_data_path, corpus_data_path, project_dir)
        self.evaluator.init_trial(trial_name)
        
        # 保存路径 
        self.config_path = config_path
        self.project_dir = project_dir
        self.trial_dir = os.path.join(project_dir, trial_name)
        os.makedirs(os.path.join(self.trial_dir, "configs"), exist_ok=True)
        
        # 设置日志
        self._setup_logger(trial_name)
        
        # 加载基础配置
        self.base_config = self._load_base_config()
        
        # 设置目标组件和固定组件
        self.fixed_components = fixed_components or {}
        self._setup_target_components(target_components)
        
        # 初始化节点的配置
        self._initialize_node_configs()
        
        # 设置缓存
        self.use_cache = use_cache
        if self.use_cache:
            self._initialize_cache()
        
    def _setup_logger(self, trial_name):
        """设置日志器"""
        log_dir = os.path.join(self.trial_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'grpo_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"日志文件保存在: {log_file}")
        self.logger_initialized = True
        
    def _load_base_config(self) -> Dict:
        """加载基础配置"""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)
            
    def _setup_target_components(self, target_components: Optional[List[str]]):
        """设置目标优化组件"""
        if target_components is None:
            # 默认优化除generator外的所有组件
            self.components = [c for c in self.ALL_COMPONENTS if c != "generator" and c in self.base_config]
        else:
            # 验证用户提供的组件是否有效
            valid_components = []
            for component in target_components:
                if component in self.ALL_COMPONENTS and component in self.base_config:
                    valid_components.append(component)
                else:
                    if self.logger_initialized:
                        self.logger.warning(f"组件 '{component}' 无效或不在配置文件中，将被忽略")
            
            self.components = valid_components
            
        # 从组件列表中移除固定组件
        self.components = [c for c in self.components if c not in self.fixed_components]
        
        if self.logger_initialized:
            self.logger.info(f"将优化以下 {len(self.components)} 个组件: {self.components}")
            if self.fixed_components:
                self.logger.info(f"以下组件将使用固定配置: {self.fixed_components}")
    
    def _initialize_node_configs(self):
        """初始化所有节点的配置选项"""
        # 初始化节点配置
        self.nodes_config = {}
        self.method_counts = {}  # 存储每个节点的方法数量
        
        for idx, component in enumerate(self.components):
            node_key = f"node{idx+1}"
            
            # 获取组件的方法列表
            methods = self.base_config[component].get("method", [])
            
            # 确保是列表
            if not isinstance(methods, list):
                methods = [methods]
                
            # 记录方法数量
            self.method_counts[node_key] = len(methods)
            
            # 创建节点配置映射
            self.nodes_config[node_key] = {
                str(i): method for i, method in enumerate(methods)
            }
        
        self.logger.info(f"初始化了{len(self.components)}个节点的配置:")
        for node, methods in self.nodes_config.items():
            self.logger.info(f"  {node}: {methods}")
    
    def setup_grpo_trainer(self, d_model: int = 256, nhead: int = 8, num_layers: int = 3):
        """
        设置GRPO训练器
        
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
        """
        # 确定每个节点的操作数量
        operations = [self.method_counts[f"node{i+1}"] for i in range(len(self.components))]
        
        # 取最大操作数量作为通用操作数
        max_operation = max(operations)
        
        self.logger.info(f"设置GRPO训练器，共{len(self.components)}个节点，操作维度: {operations}")
        self.logger.info(f"使用最大操作数 {max_operation} 作为统一操作数")
        
        self.trainer = GRPOTrainer(
            num_process=len(self.components),  # 优化指定数量的节点
            d_model=d_model,
            nhead=nhead, 
            num_layers=num_layers,
            operation=max_operation,  # 使用最大操作数
            kl_coeff=0.01,
            clip_eps=0.2
        )
        
        # 存储每个节点的实际操作数以便后续处理
        self.node_operations = operations
        
        self.logger.info(f"GRPO训练器初始化完成，策略网络结构: d_model={d_model}, nhead={nhead}, layers={num_layers}")
        return self.trainer
    
    def generate_config(self, actions: np.ndarray) -> str:
        """
        根据动作生成配置文件
        
        Args:
            actions: 动作数组，长度为组件数量
            
        Returns:
            配置文件路径
        """
        # 创建一个唯一的配置文件名
        config_id = "_".join([str(a) for a in actions])
        config_file = os.path.join(self.trial_dir, "configs", f"config_{config_id}.yaml")
        
        # 获取每个节点选择的方法
        selected_methods = {}
        for i, component in enumerate(self.components):
            node_key = f"node{i+1}"
            action_idx = str(actions[i])
            if action_idx in self.nodes_config[node_key]:
                selected_methods[component] = self.nodes_config[node_key][action_idx]
        
        # 添加固定组件配置
        for component, method in self.fixed_components.items():
            selected_methods[component] = method
            
        self.logger.info(f"选择的方法: {selected_methods}")
        
        # 准备生成配置所需的node_lines
        node_lines_list = []
        vectordb_line = None
        retriever_cfg = None
        
        # 首先处理vectordb，因为它需要作为create_config的第一个参数
        try:
            vectordb_lines, _ = load_and_generate_nodes(
                self.config_path, 
                "vectordb", 
                size=1, 
                exhaustive=False
            )
            vectordb_line = vectordb_lines[0]  # 使用第一个vectordb配置
        except Exception as e:
            self.logger.error(f"生成vectordb配置失败: {str(e)}")
            return None  # 如果vectordb配置失败，无法继续
        
        # 为每个组件生成配置
        for component in self.ALL_COMPONENTS:
            if component == "vectordb":  # 已经处理过
                continue
            
            if component not in self.base_config:
                continue
            
            # 获取该组件选择的方法
            method = selected_methods.get(component)
            
            try:
                # 生成多个配置以增加找到匹配方法的机会
                lines, cfg = load_and_generate_nodes(
                    self.config_path,
                    component,
                    size=10,  # 生成更多配置以增加匹配成功率
                    exhaustive=True  # 使用穷举模式
                )
                
                # 如果是retrieval组件，保存配置对象以便后续提取bm25_tokenizer
                if component == "retrieval":
                    retriever_cfg = cfg
                
                # 确保lines是列表
                if not isinstance(lines, list):
                    lines = [lines]
                
                # 如果指定了方法，尝试找到匹配的配置行
                if method:
                    matched_line = None
                    
                    for line in lines:
                        # 尝试将line转为字符串以进行匹配检查
                        line_str = str(line)
                        
                        # 根据不同组件的配置格式，检查方法名是否在配置中
                        if f"module_type: {method}" in line_str or \
                           f"\"{method}\"" in line_str or \
                           f"'{method}'" in line_str:
                            matched_line = line
                            self.logger.info(f"为组件 {component} 找到匹配方法 {method} 的配置")
                            break
                    
                    # 如果找到匹配，使用它；否则使用第一个配置
                    if matched_line:
                        node_lines_list.append(matched_line)
                    else:
                        node_lines_list.append(lines[0])
                        self.logger.warning(f"无法为组件 {component} 找到匹配方法 {method} 的配置，使用默认配置")
                else:
                    # 如果没有指定方法，使用第一个配置
                    node_lines_list.append(lines[0])
                
            except Exception as e:
                self.logger.error(f"为组件 {component} 生成配置失败: {str(e)}")
                # 尝试获取一个基本配置
                try:
                    basic_lines, _ = load_and_generate_nodes(
                        self.config_path,
                        component,
                        size=1,
                        exhaustive=False
                    )
                    if isinstance(basic_lines, list) and basic_lines:
                        node_lines_list.append(basic_lines[0])
                    elif basic_lines:
                        node_lines_list.append(basic_lines)
                except Exception as inner_e:
                    self.logger.error(f"无法为组件 {component} 获取基本配置: {str(inner_e)}")
        
        # 准备额外参数
        extra_params = {'strategies': {'metrics': ['meteor', 'rouge', 'bert_score']}}
        
        # 处理bm25_tokenizer特殊情况
        if retriever_cfg and "bm25" in selected_methods.get("retrieval", ""):
            if hasattr(retriever_cfg, 'cs') and '[bm25]bm25_tokenizer' in retriever_cfg.cs:
                extra_params['bm25_tokenizer_list'] = retriever_cfg.cs.get('[bm25]bm25_tokenizer').choices
            else:
                extra_params['bm25_tokenizer_list'] = ['porter_stemmer', 'space']
        
        # 创建配置文件
        try:
            # 记录将要使用的配置
            self.logger.info(f"生成配置文件，使用了 {len(node_lines_list)} 个组件配置")
            
            # 使用create_config函数生成配置文件
            # 注意vectordb_line需要作为第一个参数
            create_config(vectordb_line, *node_lines_list, extra_params=extra_params, save_path=config_file)
            
            # 验证生成的配置文件是否包含期望的方法
            self._verify_config_file(config_file, selected_methods)
            
        except Exception as e:
            self.logger.error(f"创建配置文件失败: {str(e)}")
            raise
        
        return config_file
    
    def _verify_config_file(self, config_file, selected_methods):
        """验证生成的配置文件是否包含预期的方法"""
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                
            missing_methods = []
            for component, method in selected_methods.items():
                if method not in content:
                    missing_methods.append(f"{component}:{method}")
                
            if missing_methods:
                self.logger.warning(f"配置文件可能未正确包含以下组件的方法: {', '.join(missing_methods)}")
            else:
                self.logger.info("配置文件验证通过：所有选定的方法都在配置中")
                
        except Exception as e:
            self.logger.error(f"验证配置文件时出错: {str(e)}")
    
    def _evaluate_batch(self, actions_batch, gpu_id: int) -> List[float]:
        """
        在特定GPU上评估一批配置
        
        Args:
            actions_batch: 一批动作
            gpu_id: GPU ID
            
        Returns:
            奖励值列表
        """
        # 设置使用特定GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        rewards = []
        
        for i, actions in enumerate(actions_batch):
            start_time = time.time()
            
            try:
                # 生成配置文件
                config_file = self.generate_config(actions)
                
                # 初始化runner
                self.evaluator.init_runner_from_yaml(config_file)
                
                # 评估配置
                yaml_name = f"config_{'_'.join([str(a) for a in actions])}.yaml"
                summary_df = self.evaluator.run_with_qa_eval(yaml_name=yaml_name)
                
                # 计算原始奖励
                rouge_score = summary_df["rouge"].values[0]
                meteor_score = summary_df["meteor"].values[0]
                bert_score = summary_df.get("bert_score", pd.Series([0])).values[0]
                
                raw_reward = 0.4 * rouge_score + 0.4 * meteor_score + 0.2 * bert_score
                
                # 应用非线性变换来放大小差异
                # 使用指数变换强化小差异
                transformed_reward = np.exp(20 * raw_reward)
                
                rewards.append(transformed_reward)
                
                self.logger.info(f"[GPU {gpu_id}] 配置 {i+1} 评估完成: ROUGE={rouge_score:.4f}, METEOR={meteor_score:.4f}, BERT={bert_score:.4f}")
                self.logger.info(f"[GPU {gpu_id}] 奖励转换: 原始={raw_reward:.4f}, 变换后={transformed_reward:.4f}")
                
            except Exception as e:
                self.logger.error(f"[GPU {gpu_id}] 评估配置 {i+1} 时出错: {str(e)}")
                rewards.append(0.0)
            
            eval_time = time.time() - start_time
            self.logger.info(f"[GPU {gpu_id}] 配置 {i+1} 评估耗时: {eval_time:.2f} 秒")
        
        return rewards
    
    def _initialize_cache(self):
        """初始化评估结果缓存"""
        # 缓存文件路径
        self.cache_file = os.path.join(self.trial_dir, "evaluation_cache.json")
        
        # 尝试加载现有缓存
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.evaluation_cache = json.load(f)
                self.logger.info(f"加载评估缓存，包含 {len(self.evaluation_cache)} 条记录")
            except Exception as e:
                self.logger.warning(f"加载缓存失败: {str(e)}，将创建新缓存")
                self.evaluation_cache = {}
        else:
            self.evaluation_cache = {}
        
        # 缓存命中统计
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 在_initialize_cache方法中添加
        self.use_raw_rewards_in_cache = False  # 标记缓存中存储的是变换后的奖励

        # 如果是旧缓存，可能需要添加迁移逻辑
        if os.path.exists(self.cache_file) and not self.use_raw_rewards_in_cache:
            try:
                # 迁移旧缓存
                with open(self.cache_file, 'r') as f:
                    old_cache = json.load(f)
                
                # 转换奖励
                new_cache = {}
                for key, value in old_cache.items():
                    new_cache[key] = np.exp(20 * float(value))
                
                self.evaluation_cache = new_cache
                self.logger.info(f"已将{len(old_cache)}条缓存记录从原始奖励转换为非线性奖励")
            except Exception as e:
                self.logger.error(f"缓存迁移失败: {str(e)}")

    def _save_cache(self):
        """保存评估缓存到文件"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.evaluation_cache, f)
            self.logger.info(f"评估缓存已保存，共 {len(self.evaluation_cache)} 条记录")
        except Exception as e:
            self.logger.error(f"保存缓存失败: {str(e)}")

    def evaluate_configs(self, actions_batch) -> List[float]:
        """
        评估多个配置，支持多GPU并行，使用缓存优化
        
        Args:
            actions_batch: 动作批次
            
        Returns:
            奖励值列表
        """
        total_start = time.time()
        
        # 将动作转换为可哈希的格式用于缓存查找
        actions_tuples = [tuple(map(int, action)) for action in actions_batch]
        
        # 检查哪些配置需要评估，哪些可以从缓存中获取
        cache_keys = []
        need_evaluate_indices = []
        need_evaluate_actions = []
        
        # 打印每个配置对应的策略
        for i, action_tuple in enumerate(actions_tuples):
            # 获取当前动作对应的策略
            selected_methods = {}
            for j, component_idx in enumerate(action_tuple):
                if j < len(self.components):
                    component = self.components[j]
                    node_key = f"node{j+1}"
                    action_idx = str(component_idx)
                    if action_idx in self.nodes_config[node_key]:
                        selected_methods[component] = self.nodes_config[node_key][action_idx]
            
            # 添加固定组件配置
            for component, method in self.fixed_components.items():
                selected_methods[component] = method
            
            # 生成缓存键
            cache_key = str(action_tuple)
            cache_keys.append(cache_key)
            
            # 检查是否命中缓存
            if cache_key in self.evaluation_cache:
                self.cache_hits += 1
                # 打印命中缓存的策略和奖励值
                reward = self.evaluation_cache[cache_key]
                self.logger.info(f"配置 {i+1} 命中缓存: 策略={selected_methods}, 奖励={reward:.4f}")
            else:
                self.cache_misses += 1
                need_evaluate_indices.append(i)
                need_evaluate_actions.append(actions_batch[i])
        
        self.logger.info(f"开始评估 {len(actions_batch)} 个配置，其中 {len(need_evaluate_actions)} 个需要评估，{len(actions_batch) - len(need_evaluate_actions)} 个使用缓存")
        
        # 如果有需要评估的配置
        if need_evaluate_actions:
            need_evaluate_actions = np.array(need_evaluate_actions)
            
            if self.num_gpus <= 1:
                # 单GPU评估
                new_rewards = self._evaluate_batch(need_evaluate_actions, 0)
            else:
                # 多GPU并行评估
                new_rewards = []
                
                # 将样本分成多个批次
                batches = np.array_split(need_evaluate_actions, min(self.num_gpus, len(need_evaluate_actions)))
                
                # 多进程并行处理
                with ProcessPoolExecutor(max_workers=self.num_gpus) as executor:
                    futures = []
                    for i, batch in enumerate(batches):
                        # 提交任务到进程池
                        futures.append(executor.submit(self._evaluate_batch, batch, i % self.num_gpus))
                    
                    # 收集结果
                    for future in as_completed(futures):
                        try:
                            batch_rewards = future.result()
                            new_rewards.extend(batch_rewards)
                        except Exception as e:
                            self.logger.error(f"处理评估结果时出错: {str(e)}")
                            # 对于失败的批次，添加相应数量的0奖励
                            new_rewards.extend([0.0] * len(batches[len(new_rewards) % len(batches)]))
            
            # 将新评估的结果添加到缓存
            for i, idx in enumerate(need_evaluate_indices):
                self.evaluation_cache[cache_keys[idx]] = new_rewards[i]
            
            # 定期保存缓存
            if self.cache_misses % 10 == 0:
                self._save_cache()
        
        # 组装所有结果（从缓存中获取和新评估的）
        rewards = []
        for cache_key in cache_keys:
            rewards.append(self.evaluation_cache[cache_key])
        
        total_time = time.time() - total_start
        self.logger.info(f"批次评估完成，总耗时: {total_time:.2f} 秒，平均每个配置耗时: {total_time/len(actions_batch):.2f} 秒")
        self.logger.info(f"缓存统计 - 命中: {self.cache_hits}, 未命中: {self.cache_misses}, 命中率: {self.cache_hits/(self.cache_hits+self.cache_misses):.2%}")
        
        # 汇总评估结果，按照奖励值从高到低排序并显示策略
        self._summarize_batch_results(actions_batch, rewards)
        
        return rewards

    def _summarize_batch_results(self, actions_batch, rewards):
        """汇总批次评估结果，显示每个策略的奖励值"""
        # 创建策略-奖励对
        strategy_reward_pairs = []
        
        for i, actions in enumerate(actions_batch):
            # 获取策略
            selected_methods = {}
            for j, action in enumerate(actions):
                if j < len(self.components):
                    component = self.components[j]
                    node_key = f"node{j+1}"
                    action_idx = str(int(action))
                    if action_idx in self.nodes_config[node_key]:
                        selected_methods[component] = self.nodes_config[node_key][action_idx]
            
            # 添加固定组件配置
            for component, method in self.fixed_components.items():
                selected_methods[component] = method
            
            strategy_reward_pairs.append((selected_methods, rewards[i]))
        
        # 按奖励值从高到低排序
        strategy_reward_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 输出排序后的结果
        self.logger.info(f"本批次评估结果汇总 (共{len(strategy_reward_pairs)}个配置):")
        for i, (strategy, reward) in enumerate(strategy_reward_pairs[:5]):  # 只显示前5个
            self.logger.info(f"  排名 {i+1}: 奖励={reward:.4f}, 策略={strategy}")
        
        # 如果有更多配置，简单提示
        if len(strategy_reward_pairs) > 5:
            self.logger.info(f"  ... 以及 {len(strategy_reward_pairs)-5} 个更多配置")
    
    def run_optimization(self, num_epochs: int = 20, batch_size: int = 20):
        """
        运行优化流程
        
        Args:
            num_epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            tuple: (最佳配置, 最佳奖励)
        """
        optimization_start = time.time()
        self.logger.info(f"开始GRPO优化，{len(self.components)}个组件，{num_epochs}轮训练，批量大小{batch_size}")
        
        with self.timer("设置GRPO训练器"):
            self.setup_grpo_trainer()
        
        history = {
            "epoch": [],
            "avg_reward": [],
            "max_reward": [],
            "best_actions": [],
            "best_config": [],
            "epoch_time": []
        }
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            self.logger.info(f"\n{'='*20} Epoch {epoch+1}/{num_epochs} {'='*20}")
            
            # 自适应探索率：从大到小逐渐衰减
            epsilon = max(0.7 * (1.0 - epoch / num_epochs), 0.05)
            self.logger.info(f"当前探索率 (epsilon): {epsilon:.3f}")
            
            with self.timer(f"Epoch {epoch+1} 生成动作样本"):
                actions = self.trainer.generate_actions(batch_size, epsilon)
                actions_np = actions.numpy()
                
                # 裁剪动作到有效范围
                for sample_idx in range(actions_np.shape[0]):
                    for node_idx in range(actions_np.shape[1]):
                        # 确保动作不超过该节点的实际操作数
                        actions_np[sample_idx, node_idx] = min(
                            actions_np[sample_idx, node_idx],
                            self.node_operations[node_idx] - 1
                        )
                
                self.logger.info(f"生成的动作形状: {actions_np.shape}")
            
            with self.timer(f"Epoch {epoch+1} 评估配置"):
                rewards = self.evaluate_configs(actions_np)
                if rewards:
                    self.logger.info(f"本批次奖励值统计: 最小={min(rewards):.4f}, 最大={max(rewards):.4f}, 平均={np.mean(rewards):.4f}")
                else:
                    self.logger.warning("本批次没有有效的奖励值")
                    continue
            
            with self.timer(f"Epoch {epoch+1} 更新策略"):
                metrics = self.trainer.update_policy(actions_np, rewards)
            
            epoch_time = time.time() - epoch_start
            history["epoch"].append(epoch)
            history["avg_reward"].append(metrics["avg_reward"])
            history["max_reward"].append(metrics["max_reward"])
            
            # 修复这里，检查是列表还是numpy数组，适当处理
            if isinstance(metrics["best_actions"], np.ndarray):
                history["best_actions"].append(metrics["best_actions"].tolist())
            else:
                # 如果已经是列表，直接添加
                history["best_actions"].append(metrics["best_actions"])
            
            # 创建最佳配置的描述
            best_config = {}
            for i, component in enumerate(self.components):
                action_idx = str(metrics["best_actions"][i])
                node_key = f"node{i+1}"
                if action_idx in self.nodes_config[node_key]:
                    best_config[component] = self.nodes_config[node_key][action_idx]
            
            # 添加固定组件配置
            for component, method in self.fixed_components.items():
                best_config[component] = method
            
            history["best_config"].append(best_config)
            history["epoch_time"].append(epoch_time)
            
            self.logger.info(f"Epoch {epoch+1} 统计:")
            self.logger.info(f"  - 平均奖励: {metrics['avg_reward']:.4f}")
            self.logger.info(f"  - 最大奖励: {metrics['max_reward']:.4f}")
            self.logger.info(f"  - 当前最佳动作: {metrics['best_actions']}")
            self.logger.info(f"  - 当前最佳配置: {best_config}")
            self.logger.info(f"  - 本轮耗时: {epoch_time:.2f} 秒")
            
            # 打印当前策略分布
            current_policy = self.trainer.generate_actions(1, 0).tolist()
            self.logger.info(f"  - 当前策略分布: {current_policy}")
            
            # 保存当前最佳模型
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                checkpoint_path = os.path.join(self.trial_dir, f"checkpoint_epoch_{epoch}.pt")
                state_dict = {
                    'policy_state': self.trainer.policy.state_dict(),
                    'old_policy_state': self.trainer.old_policy.state_dict(),
                    'best_actions': self.trainer.best_actions,
                    'best_reward': self.trainer.best_reward
                }
                torch.save(state_dict, checkpoint_path)
                self.logger.info(f"保存检查点到: {checkpoint_path}")
        
        total_time = time.time() - optimization_start
        self.logger.info(f"\n优化完成!")
        self.logger.info(f"总耗时: {total_time:.2f} 秒")
        self.logger.info(f"平均每轮耗时: {total_time/num_epochs:.2f} 秒")
        
        # 获取最终的最佳配置
        best_actions = self.trainer.best_actions
        best_reward = self.trainer.best_reward
        
        best_config = {}
        for i, component in enumerate(self.components):
            action_idx = str(best_actions[i])
            node_key = f"node{i+1}"
            if action_idx in self.nodes_config[node_key]:
                best_config[component] = self.nodes_config[node_key][action_idx]
        
        # 添加固定组件配置
        for component, method in self.fixed_components.items():
            best_config[component] = method
        
        self.logger.info(f"最佳动作: {best_actions}")
        self.logger.info(f"最佳配置: {best_config}")
        self.logger.info(f"最佳奖励: {best_reward:.4f}")
        
        # 保存优化历史
        history_df = pd.DataFrame(history)
        history_path = os.path.join(self.trial_dir, "optimization_history.csv")
        history_df.to_csv(history_path, index=False)
        self.logger.info(f"优化历史已保存至: {history_path}")
        
        # 保存最佳配置文件
        best_config_file = self.generate_config(best_actions)
        best_config_path = os.path.join(self.trial_dir, "best_config.yaml")
        import shutil
        shutil.copy2(best_config_file, best_config_path)
        self.logger.info(f"最佳配置文件已保存至: {best_config_path}")
        
        # 保存最终评估缓存
        if self.use_cache:
            self._save_cache()
            self.logger.info(f"评估缓存统计 - 总记录数: {len(self.evaluation_cache)}, 命中率: {self.cache_hits/(self.cache_hits+self.cache_misses):.2%}")
        
        return best_config, best_reward


# 使用示例
if __name__ == "__main__":
    # 配置路径
    qa_data_path = "../data/5dataset_100/qa100.parquet"
    corpus_data_path = "../data/5dataset_100/corpus_relate.parquet"
    project_dir = "../experiments/4-100-0502"
    
    # 示例1: 优化所有组件
    # optimizer = RAGOptimizer(
    #     qa_data_path=qa_data_path,
    #     corpus_data_path=corpus_data_path,
    #     project_dir=project_dir,
    #     trial_name="full_rag_opt",
    #     num_gpus=4 
    # )
    
    # 示例2: 只优化特定组件
    # optimizer = RAGOptimizer(
    #     qa_data_path=qa_data_path,
    #     corpus_data_path=corpus_data_path,
    #     project_dir=project_dir,
    #     trial_name="retrieval_prompt_opt",
    #     num_gpus=4,
    #     target_components=["retrieval", "prompt_maker"]  # 只优化这两个组件
    # )
    
    # 示例3: 固定某些组件的配置，优化其他组件
    optimizer = RAGOptimizer(
        qa_data_path=qa_data_path,
        corpus_data_path=corpus_data_path,
        project_dir=project_dir,
        trial_name="strategy_opt",
        num_gpus=4,
        target_components=["retrieval", "query_expansion", "passage_reranker", "passage_filter", "passage_compressor"],
        fixed_components={
            # 如果有些组件你想固定为特定方法，在这里指定
            # 例如: "vectordb": "chroma"
        },
        use_cache=True  # 启用评估缓存
    )
    
    # 运行优化，使用较小的batch size
    best_config, best_reward = optimizer.run_optimization(num_epochs=100, batch_size=4)