import torch
import torch.nn.functional as F
import numpy as np
from AUTO_GRADON.ragas.grpo_policy_based_on_env import DirectPolicyNetwork

class SimplePolicyReader:
    """简单的策略读取器 - 只读取单个.pt文件的策略概率矩阵"""
    
    def __init__(self, checkpoint_path: str):
        """
        初始化策略读取器
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        self.checkpoint_path = checkpoint_path
        self.policy_network = None
        
    def load_checkpoint(self):
        """加载检查点文件"""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            print(f"✅ 成功加载检查点: {self.checkpoint_path}")
            return checkpoint
        except Exception as e:
            print(f"❌ 加载检查点失败: {e}")
            return None
    
    def create_policy_network(self, checkpoint):
        """根据检查点创建策略网络"""
        policy_state = checkpoint['policy_state']
        
        # 从state_dict推断网络参数
        process_embedding_weight = policy_state['process_embedding.weight']
        d_model = process_embedding_weight.shape[0]
        
        # 推断层数
        transformer_keys = [k for k in policy_state.keys() if 'transformer_encoder' in k]
        num_layers = len([k for k in transformer_keys if 'layers' in k and 'self_attn.in_proj_weight' in k])
        
        # 推断注意力头数 (简单设置为8)
        nhead = 8
        
        # 推断操作维度
        output_weight = policy_state['output_layer.weight']
        operation_dim = output_weight.shape[0]
        
        # 推断进程数 - 从输入embedding的维度推断
        input_dim = process_embedding_weight.shape[1]  # 输入维度
        num_process = 2  # 根据你的设置，应该是7个进程
        
        print(f"📊 网络参数:")
        print(f"   进程数: {num_process}")
        print(f"   d_model: {d_model}")
        print(f"   num_layers: {num_layers}")
        print(f"   operation_dim: {operation_dim}")
        
        # 创建策略网络
        policy_network = DirectPolicyNetwork(
            num_process=num_process,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            operation_dim=operation_dim
        )
        
        # 加载参数
        policy_network.load_state_dict(policy_state)
        policy_network.eval()
        
        return policy_network
    
    def get_policy_matrix(self, input_state=None, use_fixed_seed=True):
        """
        获取策略概率矩阵
        
        Args:
            input_state: 输入状态，如果为None则使用默认状态
            use_fixed_seed: 是否使用固定的全0张量确保结果一致性
            
        Returns:
            tuple: (logits, probabilities) - 原始logits和概率分布
        """
        if self.policy_network is None:
            print("❌ 策略网络未加载")
            return None, None
            
        with torch.no_grad():
            # 🔧 使用全0张量作为输入，与grpo.py中的DirectPolicyNetwork.forward()保持一致
            if input_state is None:
                # 获取网络的进程数
                num_process = self.policy_network.num_process
                input_state = torch.zeros(1, num_process, 10)
                print("🎯 使用固定的全0张量作为输入状态，确保与训练时完全一致")
            
            # 获取网络输出
            logits = self.policy_network(input_state)  # [1, num_process, operation_dim]
            
            # 计算概率分布
            probabilities = F.softmax(logits, dim=-1)
            
            # 去掉batch维度
            logits = logits.squeeze(0)  # [num_process, operation_dim]
            probabilities = probabilities.squeeze(0)  # [num_process, operation_dim]
            
            return logits.numpy(), probabilities.numpy()
    
    def read_policy(self, show_details=True):
        """
        读取策略矩阵的主函数
        
        Args:
            show_details: 是否显示详细信息
            
        Returns:
            tuple: (logits, probabilities) 或 None
        """
        # 1. 加载检查点
        checkpoint = self.load_checkpoint()
        if checkpoint is None:
            return None
        
        # 🔧 提取并显示保存的best_actions
        saved_best_actions = checkpoint.get('best_actions', None)
        saved_best_reward = checkpoint.get('best_reward', None)
        
        print(f"\n📋 检查点中保存的信息:")
        print(f"   保存的最佳动作: {saved_best_actions}")
        print(f"   保存的最佳奖励: {saved_best_reward}")
        if saved_best_actions:
            print(f"   动作类型: {type(saved_best_actions)}")
            print(f"   动作长度: {len(saved_best_actions)}")
        
        # 2. 创建策略网络
        self.policy_network = self.create_policy_network(checkpoint)
        
        # 3. 获取策略矩阵
        logits, probabilities = self.get_policy_matrix()
        
        if show_details and probabilities is not None:
            print(f"\n📋 实时生成的策略概率矩阵:")
            print(f"形状: {probabilities.shape}")
            print(f"概率矩阵:")
            print(probabilities)
            
            print(f"\n🎯 每个进程的最优动作:")
            predicted_best_actions = probabilities.argmax(axis=1)
            for i, action in enumerate(predicted_best_actions):
                max_prob = probabilities[i, action]
                print(f"   进程 {i}: 动作 {action} (概率: {max_prob:.4f})")
            
            print(f"\n📊 实时预测的最优动作组合: {predicted_best_actions.tolist()}")
            
            # 🔧 详细对比保存的动作和预测的动作
            self._compare_actions(saved_best_actions, predicted_best_actions.tolist(), probabilities)
        
        return logits, probabilities
    
    def _compare_actions(self, saved_actions, predicted_actions, probabilities):
        """
        详细对比保存的动作和预测的动作
        
        Args:
            saved_actions: 检查点中保存的最佳动作
            predicted_actions: 实时预测的最佳动作
            probabilities: 策略概率矩阵
        """
        print(f"\n🔄 动作对比分析:")
        print("="*50)
        
        if saved_actions is None:
            print("❌ 检查点中没有保存best_actions")
            return
        
        if len(saved_actions) != len(predicted_actions):
            print(f"❌ 动作长度不匹配:")
            print(f"   保存的动作长度: {len(saved_actions)}")
            print(f"   预测的动作长度: {len(predicted_actions)}")
            return
        
        # 逐个对比每个进程的动作
        matches = 0
        total = len(saved_actions)
        
        print(f"进程对比详情:")
        for i in range(total):
            saved_action = saved_actions[i]
            predicted_action = predicted_actions[i]
            match = saved_action == predicted_action
            
            if match:
                matches += 1
                status = "✅"
            else:
                status = "❌"
            
            # 显示该进程所有动作的概率
            process_probs = probabilities[i]
            saved_prob = process_probs[saved_action] if saved_action < len(process_probs) else 0.0
            predicted_prob = process_probs[predicted_action]
            
            print(f"   进程 {i}: {status}")
            print(f"      保存动作: {saved_action} (概率: {saved_prob:.6f})")
            print(f"      预测动作: {predicted_action} (概率: {predicted_prob:.6f})")
            
            if not match:
                # 如果不匹配，显示所有动作的概率排序
                sorted_indices = np.argsort(process_probs)[::-1]  # 从高到低排序
                print(f"      所有动作概率排序:")
                for rank, action_idx in enumerate(sorted_indices):
                    prob = process_probs[action_idx]
                    marker = "📌" if action_idx == saved_action else "🔸" if action_idx == predicted_action else "  "
                    print(f"         {marker} 动作{action_idx}: {prob:.6f}")
            print()
        
        print(f"📊 总体匹配情况:")
        print(f"   匹配数量: {matches}/{total}")
        print(f"   匹配率: {matches/total*100:.1f}%")
        
        if matches == total:
            print(f"🎉 完全匹配！保存的动作与实时预测完全一致")
        else:
            print(f"⚠️  存在 {total - matches} 个不匹配的动作")
            
            # 分析可能的原因
            print(f"\n🔍 不匹配原因分析:")
            print(f"   1. 训练时的状态输入与当前生成的可能不同")
            print(f"   2. 网络参数在保存后可能发生了变化")
            print(f"   3. 随机种子设置可能不一致")
            print(f"   4. 网络结构参数可能推断错误")
    
    def verify_consistency(self, num_tests=3):
        """
        验证多次调用是否产生一致的结果
        
        Args:
            num_tests: 测试次数
            
        Returns:
            bool: 是否所有结果都一致
        """
        print(f"\n🔍 验证策略矩阵生成的一致性 (测试{num_tests}次):")
        
        results = []
        for i in range(num_tests):
            _, probabilities = self.get_policy_matrix()
            if probabilities is not None:
                results.append(probabilities)
        
        if len(results) < 2:
            print("❌ 无法进行一致性验证")
            return False
        
        # 检查所有结果是否相同
        all_consistent = True
        for i in range(1, len(results)):
            if not np.allclose(results[0], results[i], atol=1e-10):
                print(f"❌ 第{i+1}次测试与第1次测试结果不一致")
                all_consistent = False
            else:
                print(f"✅ 第{i+1}次测试与第1次测试结果一致")
        
        if all_consistent:
            print("🎉 所有测试结果完全一致！")
        
        return all_consistent


def main():
    """使用示例"""
    # 检查点文件路径 (修改为你的实际路径)
    checkpoint_path = "../experiments/204-re-qe_new/strategy_opt/checkpoint_epoch_99.pt"
    
    # 创建策略读取器
    reader = SimplePolicyReader(checkpoint_path)
    
    # 读取策略矩阵
    logits, probabilities = reader.read_policy(show_details=True)
    
    if probabilities is not None:
        print(f"\n✅ 策略矩阵读取完成!")
        print(f"   logits形状: {logits.shape}")
        print(f"   probabilities形状: {probabilities.shape}")
        
        # 验证一致性
        reader.verify_consistency(num_tests=5)
        
        # 你可以在这里对概率矩阵进行进一步处理
        # 例如：
        # np.save('policy_matrix.npy', probabilities)
        # 或者进行其他分析
    else:
        print("❌ 策略矩阵读取失败")


if __name__ == "__main__":
    main() 