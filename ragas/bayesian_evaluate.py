import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time
import logging
from datetime import datetime
from contextlib import contextmanager
import json

# 贝叶斯优化相关库
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# 导入自定义模块
from create_config import create_config, load_and_generate_nodes
from evaluator import TestEvaluator


class BayesianRAGOptimizer:
    """使用TPE贝叶斯优化方法优化RAG系统的多组件配置"""
    
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
                 config_path: str = "/home/cz/AUTO_GRADON/ragas/configuration/config206.yaml",
                 trial_name: str = "bayesian_rag_optimization",
                 target_components: Optional[List[str]] = None,
                 fixed_components: Optional[Dict[str, str]] = None,
                 use_cache: bool = True):
        """
        初始化贝叶斯优化器
        
        Args:
            qa_data_path: 问答数据路径
            corpus_data_path: 语料库数据路径
            project_dir: 项目目录
            config_path: 配置文件路径
            trial_name: 试验名称
            target_components: 需要优化的组件列表，如果为None则优化除generator外的所有组件
            fixed_components: 固定组件的配置，格式为 {"组件名": "方法名"}
            use_cache: 是否使用评估缓存
        """
        self.logger_initialized = False
        
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
        
        # 初始化Optuna study
        self._setup_optuna_study()
        
        # 🚀 新增：收敛检测相关变量
        self.convergence_history = []  # 记录最佳值历史
        self.early_stop_triggered = False  # 是否触发早停
        
    def _setup_logger(self, trial_name):
        """设置日志器"""
        log_dir = os.path.join(self.trial_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'bayesian_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
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
    
    def _setup_optuna_study(self):
        """设置Optuna研究"""
        # 创建TPE采样器
        sampler = TPESampler(
            n_startup_trials=10,  # 前10次试验使用随机采样
            n_ei_candidates=24,   # 期望改进候选数
            seed=442               # 随机种子
        )
        
        # 创建剪枝器（可选）
        pruner = MedianPruner(
            n_startup_trials=5,   # 前5次试验不剪枝
            n_warmup_steps=10     # 预热步数
        )
        
        # 创建study
        study_name = f"bayesian_rag_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        storage_path = os.path.join(self.trial_dir, "optuna_study.db")
        
        self.study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{storage_path}",
            direction="maximize",  # 最大化奖励
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
        
        self.logger.info(f"Optuna study 已创建: {study_name}")
        self.logger.info(f"数据库路径: {storage_path}")
    
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

    def _check_convergence(self, patience: int = 15, min_improvement: float = 0.001) -> bool:
        """
        检查是否收敛
        
        Args:
            patience: 容忍多少轮没有显著改进
            min_improvement: 最小改进阈值
            
        Returns:
            是否应该早停
        """
        if len(self.convergence_history) < patience + 5:  # 至少需要一些试验才能判断收敛
            return False
        
        # 获取最近的最佳值
        recent_best = max(self.convergence_history[-patience:])
        historical_best = max(self.convergence_history[:-patience]) if len(self.convergence_history) > patience else 0
        
        # 计算改进程度
        if historical_best == 0:
            improvement = float('inf')
        else:
            improvement = (recent_best - historical_best) / abs(historical_best)
        
        # 检查是否收敛
        converged = improvement < min_improvement
        
        if converged:
            self.logger.info(f"🛑 检测到收敛！最近{patience}轮的最佳改进仅为 {improvement:.4f} (< {min_improvement})")
            self.logger.info(f"   历史最佳: {historical_best:.4f}, 最近最佳: {recent_best:.4f}")
        
        return converged

    def _check_plateau(self, plateau_trials: int = 20, plateau_threshold: float = 0.0005) -> bool:
        """
        检查是否进入平台期（连续多轮没有明显改进）
        
        Args:
            plateau_trials: 平台期试验数
            plateau_threshold: 平台期阈值
            
        Returns:
            是否进入平台期
        """
        if len(self.convergence_history) < plateau_trials:
            return False
        
        recent_values = self.convergence_history[-plateau_trials:]
        max_val = max(recent_values)
        min_val = min(recent_values)
        
        # 计算最近试验的变化幅度
        variation = (max_val - min_val) / max_val if max_val != 0 else 0
        
        plateau = variation < plateau_threshold
        
        if plateau:
            self.logger.info(f"📊 检测到平台期！最近{plateau_trials}轮的变化幅度仅为 {variation:.4f} (< {plateau_threshold})")
        
        return plateau

    def _should_early_stop(self, 
                          patience: int = 15, 
                          min_improvement: float = 0.001,
                          plateau_trials: int = 20,
                          plateau_threshold: float = 0.0005,
                          min_trials: int = 25) -> bool:
        """
        综合判断是否应该早停
        
        Args:
            patience: 收敛检测的容忍轮数
            min_improvement: 最小改进阈值
            plateau_trials: 平台期检测的试验数
            plateau_threshold: 平台期阈值
            min_trials: 最少试验数（避免过早停止）
            
        Returns:
            是否应该早停
        """
        # 至少要完成最少试验数
        if len(self.convergence_history) < min_trials:
            return False
        
        # 检查收敛
        converged = self._check_convergence(patience, min_improvement)
        
        # 检查平台期
        plateau = self._check_plateau(plateau_trials, plateau_threshold)
        
        # 任一条件满足就早停
        should_stop = converged or plateau
        
        if should_stop:
            self.logger.info(f"🎯 满足早停条件：收敛={converged}, 平台期={plateau}")
        
        return should_stop

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
            
            # 计算奖励
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

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna目标函数
        
        Args:
            trial: Optuna试验对象
            
        Returns:
            目标值（奖励）
        """
        # 为每个组件选择一个方法
        actions = []
        for i, component in enumerate(self.components):
            node_key = f"node{i+1}"
            method_count = self.method_counts[node_key]
            
            # 使用trial.suggest_int选择方法索引
            action = trial.suggest_int(f"{component}_method", 0, method_count - 1)
            actions.append(action)
        
        # 记录当前试验的配置
        selected_methods = {}
        for i, component in enumerate(self.components):
            node_key = f"node{i+1}"
            action_idx = str(actions[i])
            if action_idx in self.nodes_config[node_key]:
                selected_methods[component] = self.nodes_config[node_key][action_idx]
        
        # 添加固定组件配置
        for component, method in self.fixed_components.items():
            selected_methods[component] = method
        
        self.logger.info(f"Trial {trial.number}: 测试配置 {selected_methods}")
        
        # 评估配置
        reward = self.evaluate_config(actions)
        
        # 🚀 新增：更新收敛历史
        self.convergence_history.append(reward)
        
        # 记录试验结果
        trial.set_user_attr("actions", actions)
        trial.set_user_attr("selected_methods", selected_methods)
        trial.set_user_attr("reward", reward)
        
        # 🚀 新增：检查是否应该早停
        if self._should_early_stop():
            self.early_stop_triggered = True
            self.logger.info(f"🛑 触发早停机制，在第 {trial.number + 1} 轮停止优化")
            # 通过抛出异常来停止优化
            trial.study.stop()
        
        return reward

    def run_optimization(self, 
                        n_trials: int = 400, 
                        timeout: Optional[int] = None,
                        # 🚀 新增早停参数
                        enable_early_stopping: bool = True,
                        patience: int = 15,
                        min_improvement: float = 0.001,
                        plateau_trials: int = 20,
                        plateau_threshold: float = 0.0005,
                        min_trials: int = 25) -> Tuple[Dict, float]:
        """
        运行贝叶斯优化
        
        Args:
            n_trials: 最大试验次数
            timeout: 超时时间（秒）
            enable_early_stopping: 是否启用早停
            patience: 收敛检测的容忍轮数
            min_improvement: 最小改进阈值
            plateau_trials: 平台期检测的试验数
            plateau_threshold: 平台期阈值
            min_trials: 最少试验数
            
        Returns:
            tuple: (最佳配置, 最佳奖励)
        """
        optimization_start = time.time()
        
        if enable_early_stopping:
            self.logger.info(f"开始贝叶斯优化（启用早停），{len(self.components)}个组件，最多{n_trials}次试验")
            self.logger.info(f"早停参数：patience={patience}, min_improvement={min_improvement}, plateau_trials={plateau_trials}")
        else:
            self.logger.info(f"开始贝叶斯优化（禁用早停），{len(self.components)}个组件，{n_trials}次试验")
        
        # 🚀 重置早停状态
        self.convergence_history = []
        self.early_stop_triggered = False
        
        # 运行优化
        with self.timer("贝叶斯优化"):
            try:
                self.study.optimize(
                    self.objective,
                    n_trials=n_trials,
                    timeout=timeout,
                    show_progress_bar=True
                )
            except optuna.exceptions.OptunaError as e:
                if "Study has been stopped" in str(e):
                    self.logger.info("✅ 优化因早停而正常结束")
                else:
                    raise e
        
        # 获取最佳结果
        best_trial = self.study.best_trial
        best_reward = best_trial.value
        best_actions = best_trial.user_attrs["actions"]
        best_methods = best_trial.user_attrs["selected_methods"]
        
        total_time = time.time() - optimization_start
        actual_trials = len(self.study.trials)
        
        self.logger.info(f"\n优化完成!")
        self.logger.info(f"实际试验次数: {actual_trials}/{n_trials}")
        if self.early_stop_triggered:
            self.logger.info(f"🛑 因早停机制提前结束，节省了 {n_trials - actual_trials} 次试验")
        self.logger.info(f"总耗时: {total_time:.2f} 秒")
        self.logger.info(f"平均每次试验耗时: {total_time/actual_trials:.2f} 秒")
        self.logger.info(f"最佳动作: {best_actions}")
        self.logger.info(f"最佳配置: {best_methods}")
        self.logger.info(f"最佳奖励: {best_reward:.4f}")
        
        # 保存优化历史
        self._save_optimization_history()
        
        # 保存最佳配置文件
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

    def _save_optimization_history(self):
        """保存优化历史"""
        # 保存试验历史
        trials_df = self.study.trials_dataframe()
        history_path = os.path.join(self.trial_dir, "optimization_history.csv")
        trials_df.to_csv(history_path, index=False)
        self.logger.info(f"优化历史已保存至: {history_path}")
        
        # 保存详细的试验信息
        detailed_history = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                detailed_history.append({
                    "trial_number": trial.number,
                    "value": trial.value,
                    "actions": trial.user_attrs.get("actions", []),
                    "selected_methods": trial.user_attrs.get("selected_methods", {}),
                    "params": trial.params,
                    "datetime_start": trial.datetime_start,
                    "datetime_complete": trial.datetime_complete,
                    "duration": (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else None
                })
        
        detailed_history_path = os.path.join(self.trial_dir, "detailed_history.json")
        with open(detailed_history_path, 'w') as f:
            json.dump(detailed_history, f, indent=2, default=str)
        self.logger.info(f"详细历史已保存至: {detailed_history_path}")
        
        # 🚀 新增：保存收敛历史
        convergence_path = os.path.join(self.trial_dir, "convergence_history.json")
        convergence_data = {
            "convergence_history": self.convergence_history,
            "early_stop_triggered": self.early_stop_triggered,
            "total_trials": len(self.study.trials),
            "best_value": self.study.best_value if self.study.best_trial else None
        }
        with open(convergence_path, 'w') as f:
            json.dump(convergence_data, f, indent=2)
        self.logger.info(f"收敛历史已保存至: {convergence_path}")

    def get_optimization_insights(self):
        """获取优化洞察"""
        if len(self.study.trials) == 0:
            self.logger.warning("没有完成的试验，无法提供洞察")
            return
        
        self.logger.info("\n=== 优化洞察 ===")
        
        # 最佳试验信息
        best_trial = self.study.best_trial
        self.logger.info(f"最佳试验编号: {best_trial.number}")
        self.logger.info(f"最佳奖励值: {best_trial.value:.4f}")
        
        # 🚀 新增：收敛分析
        if len(self.convergence_history) > 10:
            self.logger.info(f"\n收敛分析:")
            self.logger.info(f"  总试验数: {len(self.convergence_history)}")
            self.logger.info(f"  早停触发: {'是' if self.early_stop_triggered else '否'}")
            
            # 找到最佳值出现的位置
            best_idx = self.convergence_history.index(max(self.convergence_history))
            self.logger.info(f"  最佳值出现在第 {best_idx + 1} 轮")
            
            # 计算改进趋势
            if len(self.convergence_history) >= 20:
                early_avg = np.mean(self.convergence_history[:10])
                late_avg = np.mean(self.convergence_history[-10:])
                improvement = (late_avg - early_avg) / early_avg * 100 if early_avg != 0 else 0
                self.logger.info(f"  前10轮平均: {early_avg:.4f}, 后10轮平均: {late_avg:.4f}")
                self.logger.info(f"  整体改进: {improvement:.2f}%")
        
        # 参数重要性分析
        try:
            importance = optuna.importance.get_param_importances(self.study)
            self.logger.info("\n参数重要性排序:")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {param}: {imp:.4f}")
        except Exception as e:
            self.logger.warning(f"无法计算参数重要性: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 配置路径
    #qa_data_path = "../data/5dataset_100/qa100.parquet"
    #corpus_data_path = "../data/5dataset_100/corpus.parquet"
    qa_data_path = "../data/5dataset_100/sampled20_qa.parquet"
    corpus_data_path = "../data/5dataset_100/sampled20_corpus.parquet"
    project_dir = "../experiments/206-qa20_bayesian_comparison"
    
    # 示例3: 固定某些组件的配置，优化其他组件（与GRPO对比实验相同设置）
    optimizer = BayesianRAGOptimizer(
        qa_data_path=qa_data_path,
        corpus_data_path=corpus_data_path,
        project_dir=project_dir,
        trial_name="bayesian_strategy_opt_early_stop",
        target_components=["retrieval", "query_expansion", "passage_reranker"],
        fixed_components={
            # 如果有些组件你想固定为特定方法，在这里指定
            # 例如: "vectordb": "chroma"
        },
        use_cache=True  # 启用评估缓存
    )
    
    # 🚀 运行优化，启用早停机制
    best_config, best_reward = optimizer.run_optimization(
        n_trials=400,  # 最大试验次数
        enable_early_stopping=True,  # 启用早停
        patience=15,  # 15轮没有显著改进就停止
        min_improvement=0.001,  # 最小改进阈值 0.1%
        plateau_trials=20,  # 20轮变化很小也停止
        plateau_threshold=0.0005,  # 平台期阈值 0.05%
        min_trials=25  # 至少试验25次
    )
    
    # 获取优化洞察
    optimizer.get_optimization_insights()
