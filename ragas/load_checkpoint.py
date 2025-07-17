import os
import torch
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# 导入必要的模块
from AUTO_GRADON.ragas.grpo_policy_based_on_env import GRPOTrainer
from configuration.promptmaker import PromptMakerConfiguration
from create_config import create_config, load_and_generate_nodes
from evaluator import TestEvaluator


class CheckpointLoader:
    """加载和分析GRPO训练的checkpoint文件"""
    
    def __init__(self, config_path: str = "/home/cz/AUTO_GRADON/ragas/configuration/config210.yaml"):
        self.config_path = config_path
        self.base_config = self._load_base_config()
        
    def _load_base_config(self) -> Dict:
        """加载基础配置"""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)
    
    def load_checkpoint(self, checkpoint_path: str, 
                       target_components: Optional[List[str]] = None,
                       fixed_components: Optional[Dict[str, str]] = None) -> Dict:
        """
        加载checkpoint文件
        
        Args:
            checkpoint_path: pt文件路径
            target_components: 优化的组件列表（需要与训练时一致）
            fixed_components: 固定的组件配置
            
        Returns:
            包含模型状态和元信息的字典
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
        
        print(f"加载checkpoint: {checkpoint_path}")
        
        # 加载checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print("✅ Checkpoint加载成功")
        except Exception as e:
            raise RuntimeError(f"加载checkpoint失败: {str(e)}")
        
        # 检查checkpoint内容
        expected_keys = ['policy_state', 'old_policy_state', 'best_actions', 'best_reward']
        missing_keys = [key for key in expected_keys if key not in checkpoint]
        if missing_keys:
            print(f"⚠️  警告: checkpoint中缺少以下键: {missing_keys}")
        
        print("📋 Checkpoint信息:")
        print(f"  - 最佳奖励: {checkpoint.get('best_reward', 'N/A')}")
        print(f"  - 最佳动作: {checkpoint.get('best_actions', 'N/A')}")
        
        return checkpoint
    
    def create_trainer_from_checkpoint(self, checkpoint: Dict, 
                                     target_components: Optional[List[str]] = None,
                                     fixed_components: Optional[Dict[str, str]] = None,
                                     d_model: int = 256, nhead: int = 8, num_layers: int = 3) -> GRPOTrainer:
        """
        从checkpoint创建训练器
        
        Args:
            checkpoint: 加载的checkpoint数据
            target_components: 优化的组件列表
            fixed_components: 固定的组件配置
            其他参数: 网络架构参数
            
        Returns:
            配置好的GRPOTrainer实例
        """
        # 设置组件（与训练时保持一致）
        ALL_COMPONENTS = [
            "vectordb", "query_expansion", "retrieval", "passage_augmenter",
            "passage_reranker", "passage_filter", "passage_compressor",
            "prompt_maker", "generator"
        ]
        
        fixed_components = fixed_components or {}
        
        if target_components is None:
            components = [c for c in ALL_COMPONENTS if c != "generator" and c in self.base_config]
        else:
            components = [c for c in target_components if c in ALL_COMPONENTS and c in self.base_config]
        
        components = [c for c in components if c not in fixed_components]
        
        print(f"🔧 组件配置:")
        print(f"  - 优化组件: {components}")
        print(f"  - 固定组件: {fixed_components}")
        
        # 初始化节点配置
        nodes_config = {}
        method_counts = {}
        
        for idx, component in enumerate(components):
            node_key = f"node{idx+1}"
            methods = self.base_config[component].get("method", [])
            if not isinstance(methods, list):
                methods = [methods]
            
            method_counts[node_key] = len(methods)
            nodes_config[node_key] = {str(i): method for i, method in enumerate(methods)}
        
        # 确定每个节点的操作数量
        operations = [method_counts[f"node{i+1}"] for i in range(len(components))]
        max_operation = max(operations) if operations else 4
        
        print(f"  - 节点操作数: {operations}")
        print(f"  - 最大操作数: {max_operation}")
        
        # 创建训练器
        trainer = GRPOTrainer(
            num_process=len(components),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            operation=max_operation,
            kl_coeff=0.01,
            clip_eps=0.2
        )
        
        # 加载模型状态
        if 'policy_state' in checkpoint:
            trainer.policy.load_state_dict(checkpoint['policy_state'])
            print("✅ Policy状态加载成功")
        
        if 'old_policy_state' in checkpoint:
            trainer.old_policy.load_state_dict(checkpoint['old_policy_state'])
            print("✅ Old Policy状态加载成功")
        
        if 'best_actions' in checkpoint:
            trainer.best_actions = checkpoint['best_actions']
        
        if 'best_reward' in checkpoint:
            trainer.best_reward = checkpoint['best_reward']
        
        # 存储配置信息以供后续使用
        trainer.components = components
        trainer.nodes_config = nodes_config
        trainer.node_operations = operations
        trainer.fixed_components = fixed_components
        
        return trainer
    
    def get_strategy_from_trainer(self, trainer: GRPOTrainer, epsilon: float = 0.0) -> Dict:
        """
        从训练器获取当前策略
        
        Args:
            trainer: 训练好的GRPOTrainer
            epsilon: 探索率（0表示确定性策略）
            
        Returns:
            策略信息字典
        """
        print(f"🎯 生成策略 (epsilon={epsilon}):")
        
        # 生成动作
        actions = trainer.generate_actions(1, epsilon)
        actions_np = actions.numpy()[0]  # 取第一个样本
        
        # 将动作转换为具体的方法配置
        strategy = {}
        for i, component in enumerate(trainer.components):
            if i < len(actions_np):
                action_idx = str(int(actions_np[i]))
                node_key = f"node{i+1}"
                if action_idx in trainer.nodes_config[node_key]:
                    strategy[component] = trainer.nodes_config[node_key][action_idx]
                else:
                    strategy[component] = "unknown"
        
        # 添加固定组件
        strategy.update(trainer.fixed_components)
        
        # 获取网络的原始输出（logits和概率）
        with torch.no_grad():
            logits = trainer.policy()
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        result = {
            'actions': actions_np.tolist(),
            'strategy': strategy,
            'best_actions': trainer.best_actions,
            'best_reward': trainer.best_reward,
            'logits': logits[0].numpy().tolist(),  # [num_process, operation_dim]
            'probabilities': probs[0].numpy().tolist()
        }
        
        print(f"  - 动作: {result['actions']}")
        print(f"  - 策略: {result['strategy']}")
        print(f"  - 最佳历史动作: {result['best_actions']}")
        print(f"  - 最佳历史奖励: {result['best_reward']}")
        
        return result
    
    def analyze_checkpoint_directory(self, trial_dir: str) -> List[Dict]:
        """
        分析整个trial目录中的所有checkpoint
        
        Args:
            trial_dir: trial目录路径
            
        Returns:
            所有checkpoint的分析结果列表
        """
        print(f"🔍 分析目录: {trial_dir}")
        
        checkpoint_files = []
        for file in os.listdir(trial_dir):
            if file.startswith("checkpoint_epoch_") and file.endswith(".pt"):
                checkpoint_files.append(os.path.join(trial_dir, file))
        
        checkpoint_files.sort()  # 按文件名排序
        
        if not checkpoint_files:
            print("❌ 未找到checkpoint文件")
            return []
        
        print(f"📁 找到 {len(checkpoint_files)} 个checkpoint文件")
        
        results = []
        for checkpoint_path in checkpoint_files:
            try:
                # 提取epoch信息
                filename = os.path.basename(checkpoint_path)
                epoch = int(filename.replace("checkpoint_epoch_", "").replace(".pt", ""))
                
                # 加载checkpoint
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                result = {
                    'epoch': epoch,
                    'checkpoint_path': checkpoint_path,
                    'best_reward': checkpoint.get('best_reward', None),
                    'best_actions': checkpoint.get('best_actions', None)
                }
                
                results.append(result)
                print(f"  Epoch {epoch}: 奖励={result['best_reward']:.4f}, 动作={result['best_actions']}")
                
            except Exception as e:
                print(f"  ❌ 加载 {checkpoint_path} 失败: {str(e)}")
        
        return results


def main():
    """示例用法"""
    # 创建加载器
    loader = CheckpointLoader()
    
    # 方式1: 加载单个checkpoint
    print("=" * 60)
    print("方式1: 加载单个checkpoint")
    print("=" * 60)
    
    checkpoint_path = "../experiments/4-100-0504/strategy_opt/checkpoint_epoch_99.pt"
    
    try:
        # 加载checkpoint
        checkpoint = loader.load_checkpoint(checkpoint_path)
        
        # 创建训练器（需要与训练时的配置保持一致）
        trainer = loader.create_trainer_from_checkpoint(
            checkpoint,
            target_components=["retrieval", "query_expansion", "passage_reranker", 
                             "passage_filter", "passage_compressor"],
            fixed_components={},  # 根据训练时的配置调整
            d_model=256, nhead=8, num_layers=3
        )
        
        # 获取策略
        strategy_info = loader.get_strategy_from_trainer(trainer, epsilon=0.0)
        
        print("\n📊 完整策略信息:")
        for key, value in strategy_info.items():
            if key not in ['logits', 'probabilities']:  # 跳过太长的数组
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
    
    # 方式2: 分析整个目录
    print("\n" + "=" * 60)
    print("方式2: 分析整个trial目录")
    print("=" * 60)
    
    trial_dir = "../experiments/4-100-0504/strategy_opt"
    
    try:
        analysis_results = loader.analyze_checkpoint_directory(trial_dir)
        
        if analysis_results:
            print(f"\n📈 训练进度分析:")
            print(f"{'Epoch':<10} {'奖励':<15} {'最佳动作'}")
            print("-" * 50)
            for result in analysis_results[-10:]:  # 显示最后10个
                actions_str = str(result['best_actions'])[:30] + "..." if len(str(result['best_actions'])) > 30 else str(result['best_actions'])
                print(f"{result['epoch']:<10} {result['best_reward']:<15.4f} {actions_str}")
    
    except Exception as e:
        print(f"❌ 错误: {str(e)}")


if __name__ == "__main__":
    main()