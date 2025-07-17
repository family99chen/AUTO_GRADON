import torch
import torch.nn.functional as F
import pandas as pd
import os
import yaml
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from datetime import datetime

from AUTO_GRADON.ragas.grpo_policy_based_on_env import DirectPolicyNetwork, GRPOTrainer

class GRPOModelEvaluator:
    """GRPO训练模型的加载和评估器"""
    
    def __init__(self, checkpoint_path: str, config_path: str = None, logger: Optional[logging.Logger] = None):
        """
        初始化GRPO模型评估器
        
        Args:
            checkpoint_path: .pt检查点文件路径
            config_path: config.yaml配置文件路径
            logger: 日志记录器
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path or "configuration/config.yaml"
        self.logger = logger or self._setup_logger()
        
        # 模型相关
        self.policy_network = None
        self.best_actions = None
        self.best_reward = None
        self.model_config = None
        
        # 从配置文件加载组件配置
        self._load_config_and_initialize_nodes()
        
        # 添加训练器实例，用于调用generate_actions方法
        self.trainer = None
        
        self.logger.info(f"GRPO模型评估器初始化完成")
        self.logger.info(f"检查点路径: {checkpoint_path}")
        if config_path:
            self.logger.info(f"配置文件路径: {config_path}")
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("GRPOEvaluator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config_and_initialize_nodes(self):
        """从config.yaml加载配置并初始化节点配置"""
        try:
            self.logger.info(f"正在加载配置文件: {self.config_path}")
            
            # 加载配置文件
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.original_config = yaml.safe_load(f)
                self.base_config = self.original_config
            
            # 🔧 临时修复：手动设置与训练时一致的组件
            self.components = [
                "retrieval", 
                "query_expansion", 
                "passage_reranker", 
                "passage_filter", 
            ]
            
            self.logger.info(f"🔧 手动设置组件列表（与训练时一致）: {self.components}")
            
            # 重新初始化节点配置
            self.nodes_config = {}
            for idx, component in enumerate(self.components):
                node_key = f"node{idx+1}"
                methods = self.base_config[component].get("method", [])
                if not isinstance(methods, list):
                    methods = [methods]
                self.nodes_config[node_key] = {
                    str(i): method for i, method in enumerate(methods)
                }
            
            self.logger.info(f"重新初始化了{len(self.components)}个节点的配置:")
            for node, methods in self.nodes_config.items():
                self.logger.info(f"  {node}: {methods}")
            
        except Exception as e:
            self.logger.error(f"❌ 配置加载失败: {str(e)}")
            raise
    
    def _use_default_config(self):
        """使用默认配置作为后备"""
        self.logger.warning("使用默认节点配置")
        
        self.components = ["retrieval", "query_expansion", "passage_reranker", 
                          "passage_filter", "passage_compressor"]
        
        # 默认节点配置
        self.nodes_config = {
            "node1": {"0": "bm25", "1": "vectordb", "2": "hybrid"},
            "node2": {"0": "pass_query_expansion", "1": "QueryDecompose", "2": "HyDE", "3": "multi_query_expansion"},
            "node3": {"0": "pass_passage_reranker", "1": "upr", "2": "tart"},
            "node4": {"0": "pass_passage_filter", "1": "percentile_cutoff", "2": "threshold_cutoff"},
            "node5": {"0": "pass_passage_compressor", "1": "tree_summarize", "2": "refine"}
        }
        
        self.method_counts = {node: len(methods) for node, methods in self.nodes_config.items()}

    def load_model(self) -> bool:
        """加载训练好的GRPO模型"""
        try:
            # 加载检查点
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # 🔍 详细检查 policy_state
            policy_state = checkpoint['policy_state']
            print(f"🔍 Policy state 包含的层:")
            for key, tensor in policy_state.items():
                print(f"  {key}: {tensor.shape}")
            
            # 提取模型信息
            self.best_actions = checkpoint.get('best_actions')
            self.best_reward = checkpoint.get('best_reward')
            
            # 获取或推断模型配置
            if 'model_config' in checkpoint:
                self.model_config = checkpoint['model_config']
                print(f"🔍 使用保存的配置: {self.model_config}")
            else:
                self.model_config = self._infer_model_config(policy_state)
                print(f"🔍 推断的配置: {self.model_config}")
            
            # 🔧 创建完整的训练器实例，而不只是策略网络
            self.trainer = GRPOTrainer(
                num_process=self.model_config['num_process'],
                d_model=self.model_config['d_model'],
                nhead=self.model_config['nhead'],
                num_layers=self.model_config['num_layers'],
                operation=self.model_config['operation_dim']
            )
            
            # 加载策略网络权重
            self.trainer.policy.load_state_dict(policy_state)
            self.trainer.policy.eval()
            
            # 同步旧策略网络
            self.trainer.old_policy.load_state_dict(policy_state)
            self.trainer.old_policy.eval()
            
            # 恢复训练状态
            self.trainer.best_actions = self.best_actions
            self.trainer.best_reward = self.best_reward
            
            # 保持向后兼容
            self.policy_network = self.trainer.policy
            
            # 🔍 验证加载后的输出
            with torch.no_grad():
                output = self.trainer.policy()
                strategy = output.argmax(dim=-1).squeeze(0).tolist()
                
                print(f"🔍 加载后网络输出:")
                print(f"  输出形状: {output.shape}")
                print(f"  输出策略: {strategy}")
                print(f"  输出值范围: {output.min():.4f} ~ {output.max():.4f}")
                
                # 🔍 详细分析每个位置的输出
                output_squeezed = output.squeeze(0)  # [num_process, operation_dim]
                print(f"🔍 每个位置的详细输出:")
                for i in range(output_squeezed.shape[0]):
                    pos_output = output_squeezed[i]  # [operation_dim]
                    pos_probs = torch.softmax(pos_output, dim=0)
                    print(f"  位置{i}: logits={pos_output.tolist()}")
                    print(f"         probs={pos_probs.tolist()}")
                    print(f"         argmax={pos_output.argmax().item()}")
                
                # 检查是否所有位置的第0个动作都是最大值
                first_action_dominant = all(output_squeezed[i, 0] == output_squeezed[i].max() 
                                          for i in range(output_squeezed.shape[0]))
                if first_action_dominant:
                    print("🚨 发现问题：所有位置的第0个动作都是最大值！")
                    print("   这可能是因为：")
                    print("   1. 网络权重初始化问题")
                    print("   2. 训练过程中网络陷入了局部最优")
                    print("   3. 训练数据中第0个动作被过度强化")
            
            self.logger.info("✅ GRPO模型加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _infer_model_config(self, policy_state: Dict) -> Dict:
        """从策略网络状态推断模型配置"""
        # 从权重形状推断参数
        process_embedding_weight = policy_state['process_embedding.weight']
        output_layer_weight = policy_state['output_layer.weight']
        
        d_model = process_embedding_weight.shape[0]
        operation_dim = output_layer_weight.shape[0]
        
        # 从transformer层推断其他参数
        transformer_keys = [k for k in policy_state.keys() if 'transformer_encoder' in k]
        num_layers = len(set(k.split('.')[2] for k in transformer_keys if len(k.split('.')) > 2))
        
        # 从attention权重推断nhead (这里使用默认值，因为难以从权重推断)
        nhead = 8  # 默认值
        num_process = len(self.components)  # 使用实际组件数量
        
        return {
            'num_process': num_process,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'operation_dim': operation_dim
        }
    
    def generate_strategy_samples(self, num_samples: int = 10, epsilon: float = 0.1) -> List[List[int]]:
        """使用训练器的generate_actions方法生成策略样本"""
        if self.trainer is None:
            raise ValueError("训练器尚未加载，请先调用load_model()")
        
        # 🔍 添加详细调试信息
        print(f"🔍 调试 generate_actions:")
        print(f"  trainer.process_num: {self.trainer.process_num}")
        print(f"  model_config num_process: {self.model_config['num_process']}")
        
        with torch.no_grad():
            # 检查网络实际输出形状
            logits = self.trainer.policy()
            print(f"  实际网络输出形状: {logits.shape}")
            
            probs = F.softmax(logits, dim=-1)
            best_actions = probs.argmax(dim=-1).squeeze(0)
            print(f"  best_actions形状: {best_actions.shape}")
            print(f"  best_actions值: {best_actions.tolist()}")
            
            # 检查随机动作生成
            random_actions = torch.randint(0, 4, (num_samples, self.trainer.process_num))
            print(f"  random_actions形状: {random_actions.shape}")
            
            # 检查是否会出现维度不匹配
            try:
                expanded_best = best_actions.expand_as(random_actions)
                print(f"  expand成功，形状: {expanded_best.shape}")
            except Exception as e:
                print(f"  ❌ expand失败: {e}")
                return []
        
        # 🔧 调用原始方法
        try:
            actions_tensor = self.trainer.generate_actions(num_samples, epsilon)
            actions_list = actions_tensor.tolist()
            
            print(f"🎲 成功生成 {len(actions_list)} 个样本:")
            for i, actions in enumerate(actions_list[:3]):
                print(f"  样本{i}: {actions}")
            
            return actions_list
            
        except Exception as e:
            print(f"❌ generate_actions失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def predict_best_strategy(self, deterministic: bool = True) -> Tuple[List[int], Dict[str, str]]:
        """
        使用训练好的策略网络预测最佳策略
        
        Args:
            deterministic: 是否使用确定性预测(argmax)，否则使用采样
            
        Returns:
            Tuple[List[int], Dict[str, str]]: (动作序列, 配置字典)
        """
        if self.trainer is None:
            raise ValueError("训练器尚未加载，请先调用load_model()")
        
        if deterministic:
            # 🔧 使用确定性预测：直接获取网络输出的argmax
            with torch.no_grad():
                logits = self.trainer.policy()  # [1, num_process, operation_dim]
                actions = logits.argmax(dim=-1).squeeze(0).tolist()  # [num_process]
        else:
            # 🔧 使用epsilon=0的generate_actions来获取随机采样
            actions_tensor = self.trainer.generate_actions(1, epsilon=0.0)  # [1, num_process]
            actions = actions_tensor[0].tolist()  # [num_process]
        
        # 转换为配置字典
        config_dict = self._actions_to_config(actions)
        
        self.logger.info(f"🎯 预测的最佳策略:")
        self.logger.info(f"   动作序列: {actions}")
        self.logger.info(f"   配置详情: {config_dict}")
        
        return actions, config_dict
    
    def get_trained_best_strategy(self) -> Tuple[List[int], Dict[str, str]]:
        """
        获取训练过程中发现的最佳策略
        
        Returns:
            Tuple[List[int], Dict[str, str]]: (最佳动作序列, 配置字典)
        """
        if self.best_actions is None:
            raise ValueError("模型尚未加载，请先调用load_model()")
        
        config_dict = self._actions_to_config(self.best_actions)
        
        self.logger.info(f"🏆 训练发现的最佳策略:")
        self.logger.info(f"   动作序列: {self.best_actions}")
        self.logger.info(f"   最佳奖励: {self.best_reward:.4f}")
        self.logger.info(f"   配置详情: {config_dict}")
        
        return self.best_actions, config_dict
    
    def _actions_to_config(self, actions: List[int]) -> Dict[str, str]:
        """将动作序列转换为配置字典"""
        config_dict = {}
        
        for i, action in enumerate(actions):
            if i < len(self.components):
                component = self.components[i]
                node_key = f"node{i+1}"
                action_str = str(action)
                
                if node_key in self.nodes_config and action_str in self.nodes_config[node_key]:
                    config_dict[component] = self.nodes_config[node_key][action_str]
                else:
                    self.logger.warning(f"未找到组件 {component} 的动作 {action} 对应的配置")
                    config_dict[component] = "unknown"
        
        return config_dict
    
    def _actions_to_config_dict(self, actions: List[int]) -> Dict[str, str]:
        """
        将动作序列转换为配置字典
        
        Args:
            actions: 动作序列
            
        Returns:
            Dict[str, str]: 组件到方法的映射
        """
        config_dict = {}
        for i, action in enumerate(actions):
            component = self.components[i]
            node_key = f"node{i+1}"
            action_key = str(action)
            
            if node_key in self.nodes_config and action_key in self.nodes_config[node_key]:
                config_dict[component] = self.nodes_config[node_key][action_key]
            else:
                self.logger.warning(f"未找到组件 {component} 的动作 {action}，使用默认值")
                config_dict[component] = "pass"
        
        return config_dict
    
    def _generate_full_config(self, config_dict: Dict[str, str]) -> Dict:
        """
        基于选定的方法生成完整的配置文件
        
        Args:
            config_dict: 组件到方法的映射
            
        Returns:
            Dict: 完整的配置字典
        """
        # 深拷贝原始配置
        import copy
        full_config = copy.deepcopy(self.original_config)
        
        # 更新每个组件的方法
        for component, method in config_dict.items():
            if component == "retrieval":
                # retrieval 组件映射到 vectordb 配置
                if "vectordb" in full_config:
                    # 保持原有配置，只更新 method
                    if method == "hybrid_rrf":
                        full_config["vectordb"]["method"] = ["chroma"]  # 使用原配置的方法
                        # 添加 hybrid 相关配置
                        if "hybrid" not in full_config:
                            full_config["hybrid"] = {
                                "method": ["rrf"],
                                "rrf": {"rrf_k": 60},
                                "cc": {"alpha": 0.5}
                            }
                    elif method == "bm25":
                        full_config["vectordb"]["method"] = ["chroma"]
                        # 可以添加 BM25 特定配置
                    elif method == "vectordb":
                        full_config["vectordb"]["method"] = ["chroma"]
                    elif method == "hybrid_cc":
                        full_config["vectordb"]["method"] = ["chroma"]
                        if "hybrid" not in full_config:
                            full_config["hybrid"] = {
                                "method": ["cc"],
                                "rrf": {"rrf_k": 60},
                                "cc": {"alpha": 0.5}
                            }
            
            elif component in full_config:
                # 直接更新方法
                if isinstance(full_config[component].get("method"), list):
                    full_config[component]["method"] = [method]
                else:
                    full_config[component]["method"] = method
        
        return full_config

    def generate_autorag_config(self, actions: List[int], filename: str = None) -> str:
        """
        根据动作序列生成AutoRAG配置文件
        
        Args:
            actions: 动作序列
            filename: 输出文件名，如果为None则自动生成带时间戳的文件名
            
        Returns:
            str: 生成的配置文件路径
        """
        if filename is None:
            # 自动生成带时间戳的文件名，避免覆盖
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluated_strategy_config_{timestamp}.yaml"
        
        # 检查是否会覆盖训练器生成的文件
        output_dir = os.path.dirname(self.checkpoint_path)
        config_path = os.path.join(output_dir, filename)
        
        # 检查是否存在训练器生成的关键文件
        training_files = ["best_config.yaml", "best_strategy_config.yaml"]
        if any(fname in filename for fname in training_files):
            # 如果可能冲突，添加前缀
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_{timestamp}_{filename}"
            config_path = os.path.join(output_dir, filename)
            self.logger.warning(f"检测到可能的文件名冲突，重命名为: {filename}")
        
        # 转换动作为配置
        config_dict = self._actions_to_config_dict(actions)
        
        # 生成完整的配置文件
        full_config = self._generate_full_config(config_dict)
        
        # 保存配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(full_config, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"✅ AutoRAG配置文件已生成: {config_path}")
        return config_path
    
    def compare_with_exploration(self, num_samples: int = 20, epsilon: float = 0.2) -> Dict:
        """
        比较确定性策略和探索性策略
        
        Args:
            num_samples: 探索样本数量
            epsilon: 探索概率
            
        Returns:
            Dict: 比较结果
        """
        if self.trainer is None:
            raise ValueError("训练器尚未加载，请先调用load_model()")
        
        # 获取确定性策略
        deterministic_actions, deterministic_config = self.predict_best_strategy(deterministic=True)
        
        # 获取训练最佳策略
        best_actions, best_config = self.get_trained_best_strategy()
        
        # 生成探索样本
        exploration_samples = self.generate_strategy_samples(num_samples, epsilon)
        
        # 分析探索样本的多样性
        unique_samples = list(set(tuple(sample) for sample in exploration_samples))
        diversity_ratio = len(unique_samples) / len(exploration_samples)
        
        # 检查探索样本中是否包含训练最佳策略
        best_in_exploration = best_actions in exploration_samples
        deterministic_in_exploration = deterministic_actions in exploration_samples
        
        comparison = {
            "deterministic_strategy": {
                "actions": deterministic_actions,
                "config": deterministic_config
            },
            "training_best": {
                "actions": best_actions,
                "config": best_config,
                "reward": self.best_reward
            },
            "exploration_analysis": {
                "total_samples": num_samples,
                "unique_samples": len(unique_samples),
                "diversity_ratio": diversity_ratio,
                "epsilon": epsilon,
                "best_in_exploration": best_in_exploration,
                "deterministic_in_exploration": deterministic_in_exploration
            },
            "strategy_consistency": {
                "deterministic_vs_best": deterministic_actions == best_actions,
                "action_differences": sum(1 for a, b in zip(deterministic_actions, best_actions) if a != b)
            }
        }
        
        self.logger.info(f"🔍 策略探索分析:")
        self.logger.info(f"   确定性策略: {deterministic_actions}")
        self.logger.info(f"   训练最佳策略: {best_actions}")
        self.logger.info(f"   策略一致性: {deterministic_actions == best_actions}")
        self.logger.info(f"   探索样本多样性: {diversity_ratio:.2f} ({len(unique_samples)}/{num_samples})")
        self.logger.info(f"   最佳策略在探索中: {best_in_exploration}")
        
        return comparison

    def analyze_strategy_distribution(self, num_samples: int = 100) -> Dict:
        """
        分析策略分布和网络行为
        
        Args:
            num_samples: 分析样本数量
            
        Returns:
            Dict: 分析结果
        """
        if self.trainer is None:
            raise ValueError("训练器尚未加载，请先调用load_model()")
        
        # 生成大量样本进行分析
        samples_low_epsilon = self.generate_strategy_samples(num_samples, epsilon=0.1)
        samples_high_epsilon = self.generate_strategy_samples(num_samples, epsilon=0.5)
        
        # 统计每个位置的动作分布
        position_stats = {}
        for pos in range(self.model_config['num_process']):
            position_stats[f"position_{pos}"] = {
                "low_epsilon": {},
                "high_epsilon": {}
            }
            
            # 统计低epsilon样本
            for sample in samples_low_epsilon:
                action = sample[pos]
                position_stats[f"position_{pos}"]["low_epsilon"][action] = \
                    position_stats[f"position_{pos}"]["low_epsilon"].get(action, 0) + 1
            
            # 统计高epsilon样本
            for sample in samples_high_epsilon:
                action = sample[pos]
                position_stats[f"position_{pos}"]["high_epsilon"][action] = \
                    position_stats[f"position_{pos}"]["high_epsilon"].get(action, 0) + 1
        
        # 分析网络输出分布
        with torch.no_grad():
            logits = self.trainer.policy()
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # [num_process, operation_dim]
            
            network_distribution = {}
            for pos in range(self.model_config['num_process']):
                network_distribution[f"position_{pos}"] = {
                    "probabilities": probs[pos].tolist(),
                    "preferred_action": probs[pos].argmax().item(),
                    "confidence": probs[pos].max().item()
                }
        
        analysis = {
            "sample_statistics": position_stats,
            "network_distribution": network_distribution,
            "exploration_comparison": {
                "low_epsilon_unique": len(set(tuple(s) for s in samples_low_epsilon)),
                "high_epsilon_unique": len(set(tuple(s) for s in samples_high_epsilon)),
                "total_samples": num_samples
            }
        }
        
        self.logger.info(f"📊 策略分布分析:")
        for pos in range(self.model_config['num_process']):
            component = self.components[pos] if pos < len(self.components) else f"component_{pos}"
            preferred = network_distribution[f"position_{pos}"]["preferred_action"]
            confidence = network_distribution[f"position_{pos}"]["confidence"]
            self.logger.info(f"   {component}: 偏好动作{preferred} (置信度: {confidence:.3f})")
        
        return analysis

    def print_config_summary(self):
        """打印配置摘要信息"""
        self.logger.info("📋 当前配置摘要:")
        self.logger.info(f"   配置文件: {self.config_path}")
        self.logger.info(f"   目标组件: {self.components}")
        self.logger.info(f"   节点配置:")
        
        for node, methods in self.nodes_config.items():
            self.logger.info(f"     {node}: {list(methods.values())}")

    def get_method_name(self, component: str, method_index: int) -> str:
        """获取组件方法的名称"""
        try:
            if component in self.nodes_config:
                methods = list(self.nodes_config[component].values())
                if 0 <= method_index < len(methods):
                    return methods[method_index]
                else:
                    return f"Unknown_{method_index}"
            else:
                return f"Unknown_Component_{method_index}"
        except Exception as e:
            self.logger.warning(f"获取方法名称失败: {e}")
            return f"Method_{method_index}"

    def print_model_analysis(self):
        """打印模型分析和策略解释"""
        self.logger.info("🧠 GRPO策略网络分析:")
        self.logger.info("="*50)
        
        # 网络结构信息
        total_params = sum(p.numel() for p in self.policy_network.parameters())
        self.logger.info(f"📊 网络参数总数: {total_params:,}")
        self.logger.info(f"🔧 网络结构: {len(self.components)}个组件 → Transformer → 动作分布")
        
        # 策略分析
        with torch.no_grad():
            logits = self.policy_network()
            probs = F.softmax(logits, dim=-1)
            
            self.logger.info(f"\n🎯 学习到的策略分布:")
            for i, component in enumerate(self.components):
                component_probs = probs[0, i, :].tolist()
                max_prob_idx = torch.argmax(probs[0, i, :]).item()
                
                self.logger.info(f"  {component}:")
                for j, prob in enumerate(component_probs):
                    marker = "🏆" if j == max_prob_idx else "  "
                    method_name = self.get_method_name(component, j)
                    self.logger.info(f"    {marker} 方法{j}({method_name}): {prob:.3f}")

    def explain_strategy_generation(self):
        """解释策略生成机制"""
        self.logger.info("\n🔍 策略生成机制解释:")
        self.logger.info("="*50)
        
        self.logger.info("1️⃣ 输入机制:")
        self.logger.info("   • 使用固定随机状态 torch.rand(1, 10, 10)")
        self.logger.info("   • 固定种子确保每次输入相同")
        self.logger.info("   • ❌ 不依赖实时环境信息")
        
        self.logger.info("\n2️⃣ 网络处理:")
        self.logger.info("   • 嵌入层: 10维 → 128维特征")
        self.logger.info("   • Transformer: 学习组件间依赖关系")
        self.logger.info("   • 输出层: 128维 → 4维动作分布")
        
        self.logger.info("\n3️⃣ 策略输出:")
        self.logger.info("   • 每个组件的方法选择概率分布")
        self.logger.info("   • 确定性模式: 选择概率最高的方法")
        self.logger.info("   • 随机模式: 按概率分布采样")
        
        self.logger.info("\n4️⃣ 训练学习:")
        self.logger.info("   • 网络通过GRPO算法学习")
        self.logger.info("   • 高奖励动作 → 增加选择概率")
        self.logger.info("   • 低奖励动作 → 减少选择概率")
        self.logger.info("   • 🧠 最优策略被编码到网络权重中")
        
        self.logger.info("\n5️⃣ 推理特点:")
        self.logger.info("   • ✅ 无需实时输入，直接输出最优策略")
        self.logger.info("   • ✅ 推理速度极快（单次前向传播）")
        self.logger.info("   • ❌ 无法适应新的数据集或任务")
        self.logger.info("   • 🧠 专门针对训练数据集优化的策略")

    def evaluate_on_data(self, data_path: str, config_dict: Dict[str, str], 
                        output_path: str = None, use_best_strategy: bool = False) -> Dict:
        """
        使用指定配置在数据集上进行评估
        
        Args:
            data_path: 数据集路径
            config_dict: 配置字典 {组件: 方法}
            output_path: 结果输出路径
            use_best_strategy: 是否使用训练发现的最佳策略（忽略config_dict）
            
        Returns:
            Dict: 评估结果
        """
        try:
            # 🔧 如果使用最佳策略，则覆盖config_dict
            if use_best_strategy:
                if self.best_actions is None:
                    raise ValueError("没有可用的最佳策略，请先加载模型")
                config_dict = self._actions_to_config(self.best_actions)
                self.logger.info(f"🏆 使用训练发现的最佳策略: {self.best_actions}")
            
            # 生成完整配置
            full_config = self._generate_full_config(config_dict)
            
            # 创建临时配置文件
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(full_config, f, default_flow_style=False, allow_unicode=True)
                temp_config_path = f.name
            
            self.logger.info(f"🔧 使用配置进行评估:")
            for component, method in config_dict.items():
                self.logger.info(f"   {component}: {method}")
            
            # 执行评估
            from autorag.evaluator import Evaluator
            
            evaluator = Evaluator(
                qa_data_path=data_path,
                config_path=temp_config_path,
                project_dir=self.project_dir
            )
            
            # 运行评估
            result = evaluator.start_trial()
            
            # 清理临时文件
            import os
            os.unlink(temp_config_path)
            
            # 提取关键指标
            if result and 'summary' in result:
                summary = result['summary']
                evaluation_result = {
                    'config': config_dict,
                    'metrics': summary,
                    'overall_score': self._calculate_overall_score(summary),
                    'success': True,
                    'used_best_strategy': use_best_strategy
                }
            else:
                evaluation_result = {
                    'config': config_dict,
                    'metrics': {},
                    'overall_score': 0.0,
                    'success': False,
                    'error': 'No summary in result',
                    'used_best_strategy': use_best_strategy
                }
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(evaluation_result, f, default_flow_style=False, allow_unicode=True)
                self.logger.info(f"📄 评估结果已保存: {output_path}")
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"❌ 评估失败: {str(e)}")
            return {
                'config': config_dict,
                'metrics': {},
                'overall_score': 0.0,
                'success': False,
                'error': str(e),
                'used_best_strategy': use_best_strategy
            }
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """
        计算总体评分
        
        Args:
            metrics: 评估指标字典
            
        Returns:
            float: 总体评分
        """
        # 定义权重
        weights = {
            'retrieval_f1': 0.3,
            'retrieval_recall': 0.2,
            'generation_bleu': 0.2,
            'generation_rouge': 0.15,
            'generation_meteor': 0.15
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                total_score += metrics[metric] * weight
                total_weight += weight
        
        # 如果没有找到任何指标，返回0
        if total_weight == 0:
            return 0.0
        
        # 归一化到0-1范围
        return total_score / total_weight
    
    def batch_evaluate_strategies(self, data_path: str, strategies: List[Tuple[List[int], Dict[str, str]]], 
                                output_dir: str = None) -> List[Dict]:
        """
        批量评估多个策略
        
        Args:
            data_path: 数据集路径
            strategies: 策略列表 [(动作序列, 配置字典), ...]
            output_dir: 输出目录
            
        Returns:
            List[Dict]: 评估结果列表
        """
        results = []
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for i, (actions, config_dict) in enumerate(strategies):
            self.logger.info(f"🔄 评估策略 {i+1}/{len(strategies)}: {actions}")
            
            # 设置输出路径
            if output_dir:
                output_path = os.path.join(output_dir, f"strategy_{i+1}_result.yaml")
            else:
                output_path = None
            
            # 执行评估
            result = self.evaluate_on_data(data_path, config_dict, output_path)
            result['strategy_index'] = i + 1
            result['actions'] = actions
            
            results.append(result)
            
            self.logger.info(f"✅ 策略 {i+1} 评估完成，得分: {result['overall_score']:.4f}")
        
        # 按得分排序
        results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # 保存汇总结果
        if output_dir:
            summary_path = os.path.join(output_dir, "evaluation_summary.yaml")
            summary = {
                'total_strategies': len(strategies),
                'best_strategy': results[0] if results else None,
                'all_results': results
            }
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"📋 评估汇总已保存: {summary_path}")
        
        return results


def main():
    """主函数示例"""
    # 配置路径
    checkpoint_path = "/home/cz/AUTO_GRADON/experiments/214-re-qe-par8-paf-pac-pau-pm/strategy_opt/checkpoint_epoch_95.pt"
    config_path = "/home/cz/AUTO_GRADON/ragas/configuration/config.yaml"
    qa_data_path = "/home/cz/AUTO_GRADON/data/5dataset_100/qa100.parquet"
    corpus_data_path = "/home/cz/AUTO_GRADON/data/5dataset_100/corpus.parquet"
    
    # 创建评估器（传入配置文件路径）
    evaluator = GRPOModelEvaluator(checkpoint_path, config_path)
    
    # 🔍 新增：详细的模型分析
    print("\n" + "="*60)
    print("🧠 GRPO策略网络深度分析")
    print("="*60)
    
    # 加载模型
    if not evaluator.load_model():
        print("❌ 模型加载失败，退出")
        return
    
    # 打印模型分析
    evaluator.print_model_analysis()
    
    # 解释策略生成机制
    evaluator.explain_strategy_generation()
    
    # 获取并分析训练最佳策略
    print("\n" + "="*50)
    print("🏆 训练阶段学到的最佳策略")
    print("="*50)
    best_actions, best_config = evaluator.get_trained_best_strategy()
    
    # 获取当前网络预测
    print("\n" + "="*50)
    print("🎯 当前网络直接输出的策略")
    print("="*50)
    current_actions, current_config = evaluator.predict_best_strategy()
    
    # 策略一致性验证
    print("\n" + "="*50)
    print("🔍 策略一致性验证")
    print("="*50)
    if best_actions == current_actions:
        print("✅ 网络输出与训练最佳策略完全一致")
        print("🎯 说明网络成功学习并记住了最优策略")
    else:
        print("⚠️ 网络输出与训练最佳策略不一致")
        print("🤔 可能原因：网络还在继续优化或存在随机性")
    
    # 生成配置文件（使用时间戳避免冲突）
    print("\n" + "="*50)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = evaluator.generate_autorag_config(
        best_actions, 
        f"evaluated_best_strategy_{timestamp}.yaml"  # 明确的评估文件名
    )
    
    # 🔧 修复：移除 output_prefix 参数，直接指定完整的输出路径
    output_path = f"/home/cz/AUTO_GRADON/experiments/28-re-qe-par-paf/strategy_opt/best_strategy_evaluation_{timestamp}.yaml"
    
    eval_result = evaluator.evaluate_on_data(
        data_path=qa_data_path,
        config_dict=best_config,
        output_path=output_path,  # 🔧 直接使用完整路径
        use_best_strategy=True
    )
    
    # 🔧 新增：使用训练器的generate_actions进行策略探索
    print("\n" + "="*50)
    print("🎲 策略探索分析")
    print("="*50)
    
    # 生成策略样本
    strategy_samples = evaluator.generate_strategy_samples(num_samples=10, epsilon=0.2)
    
    # 比较确定性策略和探索策略
    exploration_comparison = evaluator.compare_with_exploration(num_samples=20, epsilon=0.3)
    
    # 分析策略分布
    distribution_analysis = evaluator.analyze_strategy_distribution(num_samples=50)
    
    # 🔧 新增：检查探索是否能找到更好的策略
    print("\n" + "="*50)
    print("🔍 探索策略分析")
    print("="*50)
    
    if exploration_comparison["exploration_analysis"]["best_in_exploration"]:
        print("✅ 探索过程中发现了训练最佳策略")
        print("🎯 说明generate_actions方法工作正常")
    else:
        print("⚠️ 探索过程中未发现训练最佳策略")
        print("🤔 可能需要增加探索样本数量或epsilon值")
    
    print(f"📊 探索多样性: {exploration_comparison['exploration_analysis']['diversity_ratio']:.2f}")
    print(f"🎲 独特样本: {exploration_comparison['exploration_analysis']['unique_samples']}/{exploration_comparison['exploration_analysis']['total_samples']}")
    
    print(f"\n🎉 GRPO模型评估完成!")
    print(f"   最佳策略配置文件: {config_file}")
    print(f"   评估结果: {eval_result}")


if __name__ == "__main__":
    main()
