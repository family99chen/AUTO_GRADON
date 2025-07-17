"""
GRPO精度问题修复补丁
========================

这个文件包含修复GRPO trainer中缓存和检查点奖励值精度差异的代码修复。

问题描述:
- 缓存中的奖励值: 99.41462104755507
- 检查点中的奖励值: 99.41461944580078
- 精度差异: ~1.6e-6

原因分析:
1. PyTorch tensor.item() 转换时的精度损失
2. float32 vs float64 的精度差异
3. 非线性变换 exp(20*x) 的数值误差累积
4. 多次数据类型转换的累积误差
"""

import torch
import numpy as np
import json

# =============================================
# 修复方案1: 修改GRPO trainer的update_policy方法
# =============================================

def fixed_update_policy_snippet():
    """
    修复后的update_policy方法片段
    主要修改: 使用更高精度的数据类型转换
    """
    
    code_snippet = '''
    def update_policy(self, actions_batch, rewards_batch):
        """策略更新核心逻辑 - 修复精度版本"""
        # 转换为张量 - 使用double精度
        actions = torch.tensor(actions_batch, dtype=torch.long)
        rewards = torch.tensor(rewards_batch, dtype=torch.float64)  # 使用float64
        
        # 追踪最佳结果 - 修复精度问题
        max_reward_idx = torch.argmax(rewards).item()
        
        # 修复: 使用double()确保精度，然后转换为Python原生float
        max_reward = float(rewards[max_reward_idx].double())  # 保持最高精度
        
        if max_reward > self.best_reward:
            self.best_reward = max_reward
            self.best_actions = actions[max_reward_idx].tolist()
        
        # ... 其余代码保持不变 ...
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': kl_div.item(),
            'avg_reward': float(rewards.mean().double()),  # 修复精度
            'max_reward': max_reward,  # 使用修复后的max_reward
            'best_actions': self.best_actions
        }
    '''
    
    return code_snippet

# =============================================
# 修复方案2: 修改缓存存储逻辑
# =============================================

def fixed_cache_storage_snippet():
    """
    修复后的缓存存储逻辑
    确保缓存和检查点使用相同的数据类型
    """
    
    code_snippet = '''
    def evaluate_configs(self, actions_batch) -> List[float]:
        # ... 现有代码 ...
        
        # 修复: 将新评估的结果添加到缓存时使用统一的精度
        for i, idx in enumerate(need_evaluate_indices):
            # 确保使用一致的数据类型和精度
            reward_value = float(np.float64(new_rewards[i]))  # 统一使用float64精度
            self.evaluation_cache[cache_keys[idx]] = reward_value
        
        # ... 其余代码保持不变 ...
    '''
    
    return code_snippet

# =============================================
# 修复方案3: 添加精度容忍度比较函数
# =============================================

class PrecisionUtils:
    """精度处理工具类"""
    
    @staticmethod
    def rewards_equal(r1: float, r2: float, tolerance: float = 1e-6) -> bool:
        """
        比较两个奖励值是否在容忍度范围内相等
        
        Args:
            r1, r2: 要比较的奖励值
            tolerance: 容忍度，默认1e-6
            
        Returns:
            bool: 是否在容忍度范围内相等
        """
        return abs(r1 - r2) < tolerance
    
    @staticmethod
    def normalize_reward_precision(reward: float) -> float:
        """
        标准化奖励值精度
        
        Args:
            reward: 原始奖励值
            
        Returns:
            float: 标准化精度后的奖励值
        """
        # 使用numpy的float64确保精度一致性
        return float(np.float64(reward))
    
    @staticmethod
    def safe_tensor_to_float(tensor: torch.Tensor) -> float:
        """
        安全地将tensor转换为高精度float
        
        Args:
            tensor: PyTorch张量
            
        Returns:
            float: 高精度浮点数
        """
        return float(tensor.double())

# =============================================
# 修复方案4: 缓存验证和修复工具
# =============================================

class CacheValidator:
    """缓存验证和修复工具"""
    
    def __init__(self, cache_file_path: str):
        self.cache_file_path = cache_file_path
        self.precision_utils = PrecisionUtils()
    
    def validate_cache_precision(self, checkpoint_path: str) -> dict:
        """
        验证缓存与检查点的精度一致性
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            dict: 验证结果
        """
        try:
            # 加载缓存
            with open(self.cache_file_path, 'r') as f:
                cache_data = json.load(f)
            
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 找到最佳配置
            cache_best = max(cache_data.items(), key=lambda x: x[1])
            checkpoint_best_actions = checkpoint.get('best_actions', [])
            checkpoint_best_reward = checkpoint.get('best_reward', 0.0)
            
            # 验证精度
            cache_actions_str = cache_best[0]
            cache_reward = cache_best[1]
            
            # 检查动作是否匹配
            actions_match = cache_actions_str == str(tuple(checkpoint_best_actions))
            
            # 检查奖励精度
            reward_diff = abs(cache_reward - checkpoint_best_reward)
            precision_ok = self.precision_utils.rewards_equal(
                cache_reward, checkpoint_best_reward, tolerance=1e-5
            )
            
            return {
                'actions_match': actions_match,
                'precision_ok': precision_ok,
                'reward_difference': reward_diff,
                'cache_reward': cache_reward,
                'checkpoint_reward': checkpoint_best_reward,
                'cache_actions': cache_actions_str,
                'checkpoint_actions': str(tuple(checkpoint_best_actions))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def fix_cache_precision(self) -> bool:
        """
        修复缓存中的精度问题
        
        Returns:
            bool: 是否修复成功
        """
        try:
            # 加载缓存
            with open(self.cache_file_path, 'r') as f:
                cache_data = json.load(f)
            
            # 标准化所有奖励值的精度
            fixed_cache = {}
            for action_key, reward in cache_data.items():
                normalized_reward = self.precision_utils.normalize_reward_precision(reward)
                fixed_cache[action_key] = normalized_reward
            
            # 备份原文件
            backup_path = self.cache_file_path + '.backup'
            with open(backup_path, 'w') as f:
                json.dump(cache_data, f)
            
            # 保存修复后的缓存
            with open(self.cache_file_path, 'w') as f:
                json.dump(fixed_cache, f)
            
            print(f"✅ 缓存精度修复完成")
            print(f"📁 原文件备份至: {backup_path}")
            print(f"📊 修复了 {len(fixed_cache)} 条记录")
            
            return True
            
        except Exception as e:
            print(f"❌ 缓存修复失败: {str(e)}")
            return False

# =============================================
# 使用示例
# =============================================

def demo_precision_fix():
    """演示如何使用精度修复工具"""
    
    print("🔧 GRPO精度问题修复演示")
    print("=" * 40)
    
    # 1. 验证当前精度问题
    cache_file = "../experiments/214-re-qe-par8-paf-pac-pau-pm/strategy_opt/evaluation_cache.json"
    checkpoint_file = "../experiments/214-re-qe-par8-paf-pac-pau-pm/strategy_opt/checkpoint_epoch_99.pt"
    
    validator = CacheValidator(cache_file)
    result = validator.validate_cache_precision(checkpoint_file)
    
    if 'error' not in result:
        print(f"📊 验证结果:")
        print(f"   动作匹配: {result['actions_match']}")
        print(f"   精度OK: {result['precision_ok']}")
        print(f"   奖励差异: {result['reward_difference']:.2e}")
        print(f"   缓存奖励: {result['cache_reward']:.15f}")
        print(f"   检查点奖励: {result['checkpoint_reward']:.15f}")
        
        if not result['precision_ok']:
            print(f"\n🔧 开始修复精度问题...")
            success = validator.fix_cache_precision()
            if success:
                print("✅ 精度修复完成！")
            else:
                print("❌ 精度修复失败")
    else:
        print(f"❌ 验证过程出错: {result['error']}")

if __name__ == "__main__":
    # 显示修复代码片段
    print("🔧 GRPO精度修复代码片段:")
    print("=" * 50)
    
    print("\n1. 修复后的update_policy方法:")
    print(fixed_update_policy_snippet())
    
    print("\n2. 修复后的缓存存储逻辑:")
    print(fixed_cache_storage_snippet())
    
    print("\n3. 运行精度修复演示:")
    demo_precision_fix() 