import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from pathlib import Path
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import yaml
import json

# 导入GRPO相关类
from AUTO_GRADON.ragas.grpo_policy_based_on_env import DirectPolicyNetwork, GRPOTrainer


class GRPOPolicyAnalyzer:
    """GRPO策略分析器 - 用于读取和分析保存的.pt文件中的策略矩阵"""
    
    def __init__(self, 
                 checkpoint_dir: str,
                 config_path: str = "/home/cz/AUTO_GRADON/ragas/configuration/config214new.yaml",
                 components: Optional[List[str]] = None):
        """
        初始化策略分析器
        
        Args:
            checkpoint_dir: 检查点文件目录
            config_path: 配置文件路径
            components: 组件列表，如果为None则使用默认
        """
        self.checkpoint_dir = checkpoint_dir
        self.config_path = config_path
        
        # 默认组件列表（与RAGOptimizer中保持一致）
        self.default_components = [
            "retrieval", "query_expansion", "passage_reranker", 
            "passage_filter", "passage_compressor", "passage_augmenter", "prompt_maker"
        ]
        
        self.components = components or self.default_components
        
        # 加载配置以获取节点配置
        self._load_config()
        self._initialize_node_configs()
        
        print(f"初始化策略分析器:")
        print(f"  - 检查点目录: {checkpoint_dir}")
        print(f"  - 组件数量: {len(self.components)}")
        print(f"  - 组件列表: {self.components}")
    
    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, "r") as f:
                self.base_config = yaml.safe_load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            self.base_config = {}
    
    def _initialize_node_configs(self):
        """初始化节点配置映射"""
        self.nodes_config = {}
        self.method_counts = {}
        
        for idx, component in enumerate(self.components):
            node_key = f"node{idx+1}"
            
            # 获取组件的方法列表
            if component in self.base_config:
                methods = self.base_config[component].get("method", [])
                if not isinstance(methods, list):
                    methods = [methods]
            else:
                # 如果配置中没有该组件，使用默认方法
                methods = [f"method_{i}" for i in range(4)]
            
            self.method_counts[node_key] = len(methods)
            self.nodes_config[node_key] = {
                str(i): method for i, method in enumerate(methods)
            }
        
        print(f"节点配置初始化完成:")
        for node, config in self.nodes_config.items():
            print(f"  {node}: {config}")
    
    def find_checkpoint_files(self) -> List[str]:
        """查找所有检查点文件"""
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt")
        checkpoint_files = glob.glob(pattern)
        
        # 按epoch编号排序
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        print(f"找到 {len(checkpoint_files)} 个检查点文件:")
        for file in checkpoint_files:
            print(f"  - {os.path.basename(file)}")
        
        return checkpoint_files
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        加载检查点文件
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            检查点数据字典
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            print(f"成功加载检查点: {os.path.basename(checkpoint_path)}")
            print(f"  - 最佳动作: {checkpoint.get('best_actions', 'N/A')}")
            print(f"  - 最佳奖励: {checkpoint.get('best_reward', 'N/A')}")
            
            return checkpoint
        except Exception as e:
            print(f"加载检查点失败 {checkpoint_path}: {e}")
            return None
    
    def create_policy_network(self, checkpoint: Dict[str, Any]) -> DirectPolicyNetwork:
        """
        根据检查点数据创建策略网络
        
        Args:
            checkpoint: 检查点数据
            
        Returns:
            策略网络实例
        """
        # 从检查点推断网络参数
        policy_state = checkpoint['policy_state']
        
        # 从state_dict推断网络结构参数
        # 检查process_embedding的输入维度
        process_embedding_weight = policy_state['process_embedding.weight']
        d_model = process_embedding_weight.shape[0]
        
        # 检查transformer的参数来推断nhead和num_layers
        transformer_keys = [k for k in policy_state.keys() if 'transformer_encoder' in k]
        num_layers = len([k for k in transformer_keys if 'layers' in k and 'self_attn.in_proj_weight' in k])
        
        # 从自注意力层推断注意力头数
        if f'transformer_encoder.layers.0.self_attn.in_proj_weight' in policy_state:
            attn_weight = policy_state['transformer_encoder.layers.0.self_attn.in_proj_weight']
            # in_proj_weight的形状是 [3*d_model, d_model]，其中3表示Q,K,V
            nhead = d_model // (attn_weight.shape[0] // (3 * d_model))  # 简单推断
            nhead = 8  # 如果推断失败，使用默认值
        else:
            nhead = 8
            
        # 检查输出层来推断操作维度
        output_weight = policy_state['output_layer.weight']
        operation_dim = output_weight.shape[0]
        
        print(f"推断的网络参数:")
        print(f"  - 进程数: {len(self.components)}")
        print(f"  - d_model: {d_model}")
        print(f"  - nhead: {nhead}")
        print(f"  - num_layers: {num_layers}")
        print(f"  - operation_dim: {operation_dim}")
        
        # 创建网络
        policy_network = DirectPolicyNetwork(
            num_process=len(self.components),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            operation_dim=operation_dim
        )
        
        # 加载参数
        policy_network.load_state_dict(policy_state)
        policy_network.eval()
        
        return policy_network
    
    def get_policy_matrix(self, policy_network: DirectPolicyNetwork, 
                         input_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取策略矩阵
        
        Args:
            policy_network: 策略网络
            input_state: 输入状态，如果为None则使用默认状态
            
        Returns:
            (logits, probabilities) - 原始logits和概率分布
        """
        with torch.no_grad():
            # 获取网络输出
            logits = policy_network(input_state)  # [1, num_process, operation_dim]
            
            # 计算概率分布
            probabilities = F.softmax(logits, dim=-1)
            
            return logits.squeeze(0), probabilities.squeeze(0)  # [num_process, operation_dim]
    
    def analyze_single_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        分析单个检查点文件
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            分析结果字典
        """
        # 加载检查点
        checkpoint = self.load_checkpoint(checkpoint_path)
        if checkpoint is None:
            return None
        
        # 创建策略网络
        policy_network = self.create_policy_network(checkpoint)
        
        # 获取策略矩阵
        logits, probabilities = self.get_policy_matrix(policy_network)
        
        # 获取最优动作
        best_actions = probabilities.argmax(dim=-1).numpy()
        
        # 计算策略熵（多样性指标）
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1)
        
        # 获取最高概率
        max_probs = probabilities.max(dim=-1)[0]
        
        # 提取epoch编号
        epoch = int(os.path.basename(checkpoint_path).split('_')[-1].split('.')[0])
        
        return {
            'epoch': epoch,
            'checkpoint_path': checkpoint_path,
            'logits': logits.numpy(),
            'probabilities': probabilities.numpy(),
            'best_actions': best_actions,
            'checkpoint_best_actions': checkpoint.get('best_actions', None),
            'best_reward': checkpoint.get('best_reward', None),
            'entropy': entropy.numpy(),
            'max_probs': max_probs.numpy(),
            'policy_network': policy_network
        }
    
    def analyze_all_checkpoints(self) -> List[Dict[str, Any]]:
        """分析所有检查点文件"""
        checkpoint_files = self.find_checkpoint_files()
        
        results = []
        for checkpoint_path in checkpoint_files:
            print(f"\n分析检查点: {os.path.basename(checkpoint_path)}")
            result = self.analyze_single_checkpoint(checkpoint_path)
            if result:
                results.append(result)
        
        return results
    
    def plot_policy_evolution(self, results: List[Dict[str, Any]], save_dir: Optional[str] = None):
        """
        绘制策略演化图
        
        Args:
            results: 分析结果列表
            save_dir: 保存目录
        """
        if not results:
            print("没有分析结果可以绘制")
            return
        
        # 创建保存目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 1. 策略概率演化热力图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 提取数据
        epochs = [r['epoch'] for r in results]
        entropies = [r['entropy'] for r in results]
        max_probs = [r['max_probs'] for r in results]
        rewards = [r['best_reward'] for r in results if r['best_reward'] is not None]
        
        # 策略熵演化
        axes[0, 0].plot(epochs, np.mean(entropies, axis=1), 'b-', label='平均策略熵')
        axes[0, 0].fill_between(epochs, 
                               np.mean(entropies, axis=1) - np.std(entropies, axis=1),
                               np.mean(entropies, axis=1) + np.std(entropies, axis=1), 
                               alpha=0.3)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('策略熵')
        axes[0, 0].set_title('策略多样性演化')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 最大概率演化
        axes[0, 1].plot(epochs, np.mean(max_probs, axis=1), 'r-', label='平均最大概率')
        axes[0, 1].fill_between(epochs,
                               np.mean(max_probs, axis=1) - np.std(max_probs, axis=1),
                               np.mean(max_probs, axis=1) + np.std(max_probs, axis=1),
                               alpha=0.3)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('最大概率')
        axes[0, 1].set_title('策略确定性演化')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 奖励演化
        if rewards:
            reward_epochs = [r['epoch'] for r in results if r['best_reward'] is not None]
            axes[1, 0].plot(reward_epochs, rewards, 'g-o', label='最佳奖励')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('奖励值')
            axes[1, 0].set_title('奖励演化')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 策略动作演化
        best_actions_matrix = np.array([r['best_actions'] for r in results])
        im = axes[1, 1].imshow(best_actions_matrix.T, aspect='auto', cmap='viridis')
        axes[1, 1].set_xlabel('Epoch索引')
        axes[1, 1].set_ylabel('组件索引')
        axes[1, 1].set_title('最优动作演化')
        
        # 添加组件标签
        axes[1, 1].set_yticks(range(len(self.components)))
        axes[1, 1].set_yticklabels(self.components)
        
        plt.colorbar(im, ax=axes[1, 1], label='动作索引')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'policy_evolution.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_policy_heatmaps(self, results: List[Dict[str, Any]], save_dir: Optional[str] = None):
        """
        绘制策略概率热力图
        
        Args:
            results: 分析结果列表
            save_dir: 保存目录
        """
        if not results:
            print("没有分析结果可以绘制")
            return
        
        # 创建保存目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 选择几个关键的检查点进行可视化
        key_results = [results[0], results[len(results)//2], results[-1]] if len(results) >= 3 else results
        
        fig, axes = plt.subplots(1, len(key_results), figsize=(5*len(key_results), 6))
        if len(key_results) == 1:
            axes = [axes]
        
        for i, result in enumerate(key_results):
            probs = result['probabilities']
            epoch = result['epoch']
            
            # 创建热力图
            sns.heatmap(probs, 
                       annot=True, 
                       fmt='.3f', 
                       cmap='YlOrRd',
                       ax=axes[i],
                       cbar_kws={'label': '概率'})
            
            axes[i].set_title(f'Epoch {epoch} 策略概率矩阵')
            axes[i].set_xlabel('动作索引')
            axes[i].set_ylabel('组件索引')
            
            # 添加组件标签
            axes[i].set_yticklabels(self.components, rotation=0)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'policy_heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_strategy_report(self, results: List[Dict[str, Any]], save_dir: Optional[str] = None) -> pd.DataFrame:
        """
        生成策略分析报告
        
        Args:
            results: 分析结果列表
            save_dir: 保存目录
            
        Returns:
            策略分析DataFrame
        """
        if not results:
            print("没有分析结果可以生成报告")
            return None
        
        # 创建报告数据
        report_data = []
        
        for result in results:
            epoch = result['epoch']
            probs = result['probabilities']
            best_actions = result['best_actions']
            entropy = result['entropy']
            max_probs = result['max_probs']
            
            # 为每个组件生成策略信息
            for i, component in enumerate(self.components):
                component_probs = probs[i]
                best_action_idx = best_actions[i]
                
                # 获取方法名
                node_key = f"node{i+1}"
                method_name = self.nodes_config[node_key].get(str(best_action_idx), f"action_{best_action_idx}")
                
                report_data.append({
                    'epoch': epoch,
                    'component': component,
                    'component_idx': i,
                    'best_action_idx': best_action_idx,
                    'best_method': method_name,
                    'best_action_prob': component_probs[best_action_idx],
                    'entropy': entropy[i],
                    'max_prob': max_probs[i],
                    'all_probs': component_probs.tolist()
                })
        
        # 创建DataFrame
        df = pd.DataFrame(report_data)
        
        # 保存报告
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            report_path = os.path.join(save_dir, 'strategy_analysis_report.csv')
            df.to_csv(report_path, index=False)
            print(f"策略分析报告已保存至: {report_path}")
        
        return df
    
    def compare_strategies(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        比较不同epoch的策略差异
        
        Args:
            results: 分析结果列表
            
        Returns:
            策略比较结果
        """
        if len(results) < 2:
            print("需要至少2个检查点才能进行比较")
            return None
        
        # 计算策略变化
        strategy_changes = []
        
        for i in range(1, len(results)):
            prev_probs = results[i-1]['probabilities']
            curr_probs = results[i]['probabilities']
            
            # 计算KL散度
            kl_div = np.sum(curr_probs * np.log((curr_probs + 1e-8) / (prev_probs + 1e-8)), axis=1)
            
            # 计算动作变化
            prev_actions = results[i-1]['best_actions']
            curr_actions = results[i]['best_actions']
            action_changes = np.sum(prev_actions != curr_actions)
            
            strategy_changes.append({
                'from_epoch': results[i-1]['epoch'],
                'to_epoch': results[i]['epoch'],
                'kl_divergence': kl_div,
                'mean_kl_div': np.mean(kl_div),
                'action_changes': action_changes,
                'changed_components': [self.components[j] for j in range(len(prev_actions)) 
                                     if prev_actions[j] != curr_actions[j]]
            })
        
        return {
            'strategy_changes': strategy_changes,
            'total_epochs': len(results),
            'final_strategy': {
                'best_actions': results[-1]['best_actions'],
                'best_methods': [self.nodes_config[f"node{i+1}"].get(str(action), f"action_{action}") 
                               for i, action in enumerate(results[-1]['best_actions'])],
                'final_reward': results[-1]['best_reward']
            }
        }


def main():
    """主函数 - 展示如何使用GRPOPolicyAnalyzer"""
    
    # 配置参数
    checkpoint_dir = "../experiments/204-re-qe_new/strategy_opt"  # 修改为你的检查点目录
    config_path = "/home/cz/AUTO_GRADON/ragas/configuration/config204new.yaml"
    
    # 创建分析器
    analyzer = GRPOPolicyAnalyzer(
        checkpoint_dir=checkpoint_dir,
        config_path=config_path
    )
    
    # 分析所有检查点
    print("\n开始分析所有检查点...")
    results = analyzer.analyze_all_checkpoints()
    
    if not results:
        print("没有找到有效的检查点文件")
        return
    
    print(f"\n成功分析了 {len(results)} 个检查点")
    
    # 创建输出目录
    output_dir = os.path.join(checkpoint_dir, "policy_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制策略演化图
    print("\n绘制策略演化图...")
    analyzer.plot_policy_evolution(results, output_dir)
    
    # 绘制策略热力图
    print("\n绘制策略热力图...")
    analyzer.plot_policy_heatmaps(results, output_dir)
    
    # 生成策略报告
    print("\n生成策略分析报告...")
    report_df = analyzer.generate_strategy_report(results, output_dir)
    
    # 比较策略变化
    print("\n分析策略变化...")
    comparison = analyzer.compare_strategies(results)
    
    # 打印最终策略
    if comparison:
        final_strategy = comparison['final_strategy']
        print(f"\n最终策略:")
        print(f"  - 最佳奖励: {final_strategy['final_reward']}")
        print(f"  - 最佳动作: {final_strategy['best_actions']}")
        print(f"  - 最佳方法组合:")
        for component, method in zip(analyzer.components, final_strategy['best_methods']):
            print(f"    {component}: {method}")
    
    print(f"\n分析完成！结果已保存至: {output_dir}")


def analyze_cache_precision_issue():
    """分析缓存和检查点奖励值精度差异的工具函数"""
    
    print("🔍 缓存与检查点精度差异分析")
    print("="*50)
    
    # 1. 加载evaluation_cache.json
    cache_file = "../experiments/204-re-qe_new/strategy_opt/evaluation_cache.json"
    
    try:
        with open(cache_file, 'r') as f:
            evaluation_cache = json.load(f)
        
        print(f"✅ 成功加载缓存文件，共 {len(evaluation_cache)} 条记录")
        
        # 2. 找到最高奖励的配置
        max_reward_item = max(evaluation_cache.items(), key=lambda x: x[1])
        cache_best_action = max_reward_item[0]
        cache_best_reward = max_reward_item[1]
        
        print(f"\n📊 缓存中的最佳结果:")
        print(f"   动作: {cache_best_action}")
        print(f"   奖励: {cache_best_reward:.15f} (缓存)")
        
        # 3. 加载最新检查点
        checkpoint_dir = "../experiments/214-re-qe-par8-paf-pac-pau-pm/strategy_opt"
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
        
        if checkpoint_files:
            # 获取最新的检查点
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            checkpoint_best_actions = checkpoint.get('best_actions', None)
            checkpoint_best_reward = checkpoint.get('best_reward', None)
            
            print(f"\n📊 检查点中的最佳结果:")
            print(f"   动作: {checkpoint_best_actions}")
            print(f"   奖励: {checkpoint_best_reward:.15f} (检查点)")
            
            # 4. 计算精度差异
            if cache_best_action == str(tuple(checkpoint_best_actions)):
                reward_diff = abs(cache_best_reward - checkpoint_best_reward)
                print(f"\n🎯 精度差异分析:")
                print(f"   绝对差异: {reward_diff:.2e}")
                print(f"   相对差异: {reward_diff/checkpoint_best_reward:.2e}")
                
                # 5. 分析精度问题的来源
                print(f"\n🔧 精度问题分析:")
                print(f"   1. float32 vs float64 精度差异")
                print(f"   2. PyTorch tensor.item() 转换精度损失")
                print(f"   3. 非线性变换 exp(20*x) 的数值误差累积")
                print(f"   4. 多次数据类型转换的累积误差")
                
                # 6. 验证精度范围是否可接受
                if reward_diff < 1e-5:
                    print(f"\n✅ 结论: 精度差异在可接受范围内 (<1e-5)")
                    print(f"   这是由于浮点数精度和数据类型转换造成的正常现象")
                else:
                    print(f"\n⚠️  警告: 精度差异较大，可能存在其他问题")
            else:
                print(f"\n❌ 动作不匹配，可能存在逻辑错误:")
                print(f"   缓存动作: {cache_best_action}")
                print(f"   检查点动作: {tuple(checkpoint_best_actions)}")
                
        else:
            print("❌ 未找到检查点文件")
            
    except Exception as e:
        print(f"❌ 分析过程出错: {str(e)}")
    
    print("\n" + "="*50)


def fix_precision_issue_demo():
    """演示如何修复精度问题的示例函数"""
    
    print("\n💡 精度问题修复建议:")
    print("="*40)
    
    print("1. 统一数据类型:")
    print("   - 在缓存和检查点保存时使用相同的数据类型")
    print("   - 建议使用 float64 以提高精度")
    
    print("\n2. 修改GRPO trainer的update_policy方法:")
    print("   ```python")
    print("   # 原来的代码:")
    print("   max_reward = rewards[max_reward_idx].item()  # 可能损失精度")
    print("   ")
    print("   # 修复后的代码:")
    print("   max_reward = float(rewards[max_reward_idx].double())  # 保持精度")
    print("   ```")
    
    print("\n3. 修改缓存存储逻辑:")
    print("   ```python")
    print("   # 确保缓存和检查点使用相同的奖励值来源")
    print("   self.evaluation_cache[cache_keys[idx]] = float(new_rewards[i])")
    print("   ```")
    
    print("\n4. 添加精度容忍度比较:")
    print("   ```python")
    print("   def rewards_equal(r1, r2, tolerance=1e-6):")
    print("       return abs(r1 - r2) < tolerance")
    print("   ```")


if __name__ == "__main__":
    # 运行主分析
    main()
    
    # 运行精度问题分析
    print("\n" + "="*60)
    analyze_cache_precision_issue()
    
    # 显示修复建议
    fix_precision_issue_demo()
