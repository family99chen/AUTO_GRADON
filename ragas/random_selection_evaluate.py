import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import time
import logging
from datetime import datetime
from contextlib import contextmanager
import json
import random

# 导入自定义模块
from create_config import create_config, load_and_generate_nodes
from evaluator import TestEvaluator


class RandomRAGOptimizer:
    """使用随机选择方法优化RAG系统的多组件配置，用于与GRPO对比"""
    
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
                 config_path: str = "/home/cz/AUTO_GRADON/ragas/configuration/config210.yaml",
                 trial_name: str = "random_rag_optimization",
                 target_components: Optional[List[str]] = None,
                 fixed_components: Optional[Dict[str, str]] = None,
                 use_cache: bool = True,
                 random_seed: int = 42):
        """
        初始化随机优化器
        
        Args:
            qa_data_path: 问答数据路径
            corpus_data_path: 语料库数据路径
            project_dir: 项目目录
            config_path: 配置文件路径
            trial_name: 试验名称
            target_components: 需要优化的组件列表，如果为None则优化除generator外的所有组件
            fixed_components: 固定组件的配置，格式为 {"组件名": "方法名"}
            use_cache: 是否使用评估缓存
            random_seed: 随机种子
        """
        self.logger_initialized = False
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 保存数据路径
        self.qa_data_path = qa_data_path
        self.corpus_data_path = corpus_data_path
        
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
        
        # 记录已尝试的配置，确保不重复
        self.tried_configs: Set[Tuple] = set()
        
    def _setup_logger(self, trial_name):
        """设置日志器"""
        log_dir = os.path.join(self.trial_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'random_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
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
            
        # 计算总的配置空间大小
        total_combinations = 1
        for node_key in self.method_counts:
            total_combinations *= self.method_counts[node_key]
        self.logger.info(f"总配置空间大小: {total_combinations}")
    
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

    def _save_cache(self):
        """保存评估缓存到文件"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.evaluation_cache, f)
            self.logger.info(f"评估缓存已保存，共 {len(self.evaluation_cache)} 条记录")
        except Exception as e:
            self.logger.error(f"保存缓存失败: {str(e)}")

    def generate_random_config(self, max_attempts: int = 1000) -> Optional[List[int]]:
        """
        生成一个随机配置，确保不重复
        
        Args:
            max_attempts: 最大尝试次数，避免无限循环
            
        Returns:
            随机配置的动作列表，如果无法生成不重复配置则返回None
        """
        for attempt in range(max_attempts):
            # 为每个组件随机选择一个方法
            actions = []
            for i, component in enumerate(self.components):
                node_key = f"node{i+1}"
                method_count = self.method_counts[node_key]
                action = random.randint(0, method_count - 1)
                actions.append(action)
            
            # 检查是否重复
            config_tuple = tuple(actions)
            if config_tuple not in self.tried_configs:
                self.tried_configs.add(config_tuple)
                return actions
        
        self.logger.warning(f"经过{max_attempts}次尝试，无法生成新的不重复配置")
        return None

    def generate_config(self, actions: List[int]) -> str:
        """
        根据动作生成配置文件
        
        Args:
            actions: 动作列表，长度为组件数量
            
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
            
        except Exception as e:
            self.logger.error(f"创建配置文件失败: {str(e)}")
            raise
        
        return config_file

    def evaluate_config(self, actions: List[int]) -> float:
        """
        评估单个配置
        
        Args:
            actions: 动作列表
            
        Returns:
            奖励值
        """
        # 将动作转换为可哈希的格式用于缓存查找
        cache_key = str(tuple(actions))
        
        # 检查缓存
        if self.use_cache and cache_key in self.evaluation_cache:
            self.cache_hits += 1
            reward = self.evaluation_cache[cache_key]
            self.logger.info(f"配置命中缓存: 动作={actions}, 奖励={reward:.4f}")
            return reward
        
        self.cache_misses += 1
        
        try:
            # 生成配置文件
            config_file = self.generate_config(actions)
            
            # 初始化runner
            self.evaluator.init_runner_from_yaml(config_file)
            
            # 评估配置
            yaml_name = f"config_{'_'.join([str(a) for a in actions])}.yaml"
            summary_df = self.evaluator.run_with_qa_eval(yaml_name=yaml_name)
            
            # 计算奖励（与GRPO相同的奖励函数）
            rouge_score = summary_df["rouge"].values[0]
            meteor_score = summary_df["meteor"].values[0]
            bert_score = summary_df.get("bert_score", pd.Series([0])).values[0]
            
            # 使用与GRPO相同的奖励函数
            reward = 0.4 * rouge_score + 0.4 * meteor_score + 0.2 * bert_score
            
            # 缓存结果
            if self.use_cache:
                self.evaluation_cache[cache_key] = reward
            
            self.logger.info(f"配置评估完成: ROUGE={rouge_score:.4f}, METEOR={meteor_score:.4f}, BERT={bert_score:.4f}, 奖励={reward:.4f}")
            
            return reward
            
        except Exception as e:
            self.logger.error(f"评估配置时出错: {str(e)}")
            return 0.0

    def run_optimization(self, n_trials: int = 25) -> Tuple[Dict, float]:
        """
        运行随机优化
        
        Args:
            n_trials: 试验次数
            
        Returns:
            tuple: (最佳配置, 最佳奖励)
        """
        optimization_start = time.time()
        self.logger.info(f"开始随机优化，{len(self.components)}个组件，{n_trials}次试验")
        
        # 存储所有试验结果
        all_results = []
        best_reward = -float('inf')
        best_actions = None
        best_methods = None
        
        for trial_idx in range(n_trials):
            trial_start = time.time()
            self.logger.info(f"\n{'='*20} Trial {trial_idx+1}/{n_trials} {'='*20}")
            
            # 生成随机配置
            with self.timer(f"Trial {trial_idx+1} 生成随机配置"):
                actions = self.generate_random_config()
                
            if actions is None:
                self.logger.error(f"无法生成第{trial_idx+1}个不重复配置，跳过")
                continue
            
            # 获取配置描述
            selected_methods = {}
            for i, component in enumerate(self.components):
                node_key = f"node{i+1}"
                action_idx = str(actions[i])
                if action_idx in self.nodes_config[node_key]:
                    selected_methods[component] = self.nodes_config[node_key][action_idx]
            
            # 添加固定组件配置
            for component, method in self.fixed_components.items():
                selected_methods[component] = method
            
            self.logger.info(f"Trial {trial_idx+1} 配置: {selected_methods}")
            
            # 评估配置
            with self.timer(f"Trial {trial_idx+1} 评估配置"):
                reward = self.evaluate_config(actions)
            
            # 记录结果
            trial_time = time.time() - trial_start
            result = {
                "trial": trial_idx + 1,
                "actions": actions,
                "selected_methods": selected_methods,
                "reward": reward,
                "trial_time": trial_time
            }
            all_results.append(result)
            
            # 更新最佳结果
            if reward > best_reward:
                best_reward = reward
                best_actions = actions
                best_methods = selected_methods
                self.logger.info(f"🎉 发现新的最佳配置！奖励: {best_reward:.4f}")
            
            self.logger.info(f"Trial {trial_idx+1} 完成: 奖励={reward:.4f}, 耗时={trial_time:.2f}秒")
            self.logger.info(f"当前最佳奖励: {best_reward:.4f}")
        
        total_time = time.time() - optimization_start
        self.logger.info(f"\n随机优化完成!")
        self.logger.info(f"总耗时: {total_time:.2f} 秒")
        self.logger.info(f"平均每次试验耗时: {total_time/len(all_results):.2f} 秒")
        self.logger.info(f"最佳动作: {best_actions}")
        self.logger.info(f"最佳配置: {best_methods}")
        self.logger.info(f"最佳奖励: {best_reward:.4f}")
        
        # 保存优化历史
        self._save_optimization_history(all_results)
        
        # 保存最佳配置文件
        if best_actions:
            best_config_file = self.generate_config(best_actions)
            best_config_path = os.path.join(self.trial_dir, "best_config.yaml")
            import shutil
            shutil.copy2(best_config_file, best_config_path)
            self.logger.info(f"最佳配置文件已保存至: {best_config_path}")
        
        # 保存最终评估缓存
        if self.use_cache:
            self._save_cache()
            total_evaluations = self.cache_hits + self.cache_misses
            if total_evaluations > 0:
                self.logger.info(f"评估缓存统计 - 总记录数: {len(self.evaluation_cache)}, 命中率: {self.cache_hits/total_evaluations:.2%}")
        
        return best_methods, best_reward

    def _save_optimization_history(self, all_results: List[Dict]):
        """保存优化历史"""
        # 保存详细历史
        history_df = pd.DataFrame(all_results)
        history_path = os.path.join(self.trial_dir, "optimization_history.csv")
        history_df.to_csv(history_path, index=False)
        self.logger.info(f"优化历史已保存至: {history_path}")
        
        # 保存JSON格式的详细信息
        detailed_history_path = os.path.join(self.trial_dir, "detailed_history.json")
        with open(detailed_history_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        self.logger.info(f"详细历史已保存至: {detailed_history_path}")
        
        # 统计信息
        rewards = [r["reward"] for r in all_results]
        self.logger.info(f"\n=== 随机优化统计 ===")
        self.logger.info(f"试验次数: {len(all_results)}")
        self.logger.info(f"最高奖励: {max(rewards):.4f}")
        self.logger.info(f"最低奖励: {min(rewards):.4f}")
        self.logger.info(f"平均奖励: {np.mean(rewards):.4f}")
        self.logger.info(f"奖励标准差: {np.std(rewards):.4f}")
        
        # 保存统计信息
        stats = {
            "n_trials": len(all_results),
            "best_reward": max(rewards),
            "worst_reward": min(rewards),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "unique_configs_tried": len(self.tried_configs),
            "total_config_space": np.prod(list(self.method_counts.values()))
        }
        
        stats_path = os.path.join(self.trial_dir, "optimization_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        self.logger.info(f"统计信息已保存至: {stats_path}")


# 使用示例
if __name__ == "__main__":
    # 配置路径
    qa_data_path = "../data/5dataset_100/qa100.parquet"
    corpus_data_path = "../data/5dataset_100/corpus.parquet"
    project_dir = "../experiments/210-random_comparison"
    
    # 示例1: 优化所有组件
    # optimizer = RandomRAGOptimizer(
    #     qa_data_path=qa_data_path,
    #     corpus_data_path=corpus_data_path,
    #     project_dir=project_dir,
    #     trial_name="full_random_opt"
    # )
    
    # 示例2: 只优化特定组件
    # optimizer = RandomRAGOptimizer(
    #     qa_data_path=qa_data_path,
    #     corpus_data_path=corpus_data_path,
    #     project_dir=project_dir,
    #     trial_name="retrieval_prompt_random_opt",
    #     target_components=["retrieval", "prompt_maker"]  # 只优化这两个组件
    # )
    
    # 示例3: 固定某些组件的配置，优化其他组件（与GRPO对比实验相同设置）
    optimizer = RandomRAGOptimizer(
        qa_data_path=qa_data_path,
        corpus_data_path=corpus_data_path,
        project_dir=project_dir,
        trial_name="random_strategy_opt",
        target_components=["retrieval", "query_expansion", "passage_reranker", "passage_filter", "passage_compressor"],
        fixed_components={
            # 如果有些组件你想固定为特定方法，在这里指定
            # 例如: "vectordb": "chroma"
        },
        use_cache=True,  # 启用评估缓存
        random_seed=42   # 设置随机种子以确保可重现性
    )
    
    # 运行优化，25次不重复的随机试验
    best_config, best_reward = optimizer.run_optimization(n_trials=25)
    
    print(f"\n🎯 随机优化最终结果:")
    print(f"最佳配置: {best_config}")
    print(f"最佳奖励: {best_reward:.4f}")
