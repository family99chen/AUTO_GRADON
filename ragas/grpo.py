import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class DirectPolicyNetwork(nn.Module):
    def __init__(self, num_process=10, d_model=128, nhead=8, num_layers=10, operation_dim=4):
        super().__init__()
        self.num_process = num_process

        # 初始嵌入层
        self.process_embedding = nn.Linear(10, d_model)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            activation='gelu',
            batch_first=True,
            dropout=0.1,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.output_layer = nn.Linear(d_model, operation_dim)

    def forward(self, state=None):
        if state is None:
            torch.manual_seed(42)  # 使用固定种子
            state = torch.rand(1, self.num_process, 10)
        # state = torch.rand(1, self.num_process, 10)
        # 生成初始嵌入 [batch_size, num_process, d_model]
        src = self.process_embedding(state)  # [1, self.num_process, 10] -> [1, self.num_process, 128]

        # 通过Transformer编码器
        encoded = self.transformer_encoder(src)  # [1, self.num_process, 128]
        # 生成最终输出
        logits = self.output_layer(encoded)  # [1, self.num_process, 4]
        return logits


class GRPOTrainer:
    def __init__(self, num_process=10, d_model=128, nhead=4, num_layers=3, operation=4, normalize_advantages=True, 
                kl_coeff=0.01, clip_eps=0.2, adv_lambda=0.95):
        # 初始化当前策略和旧策略网络
        self.policy = DirectPolicyNetwork(num_process, d_model, nhead, num_layers, operation)
        self.old_policy = DirectPolicyNetwork(num_process, d_model, nhead, num_layers, operation)
        self.old_policy.load_state_dict(self.policy.state_dict())  # 参数同步

        # 优化器和超参数
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff
        self.process_num = num_process
        self.operation_dim = operation
        
        # 高级优势估计选项
        self.normalize_advantages = normalize_advantages
        self.adv_lambda = adv_lambda  # GAE-lambda系数，用于平衡bias和variance
        
        # 追踪最佳结果
        self.best_reward = float('-inf')
        self.best_actions = None

    def generate_actions(self, num_samples, epsilon):
        """使用旧策略生成动作样本（带Epsilon-Greedy探索）"""
        with torch.no_grad():
            logits = self.policy()  # [1,10,4]
            probs = F.softmax(logits, dim=-1)

            # 获取最优动作 [1,10]
            best_actions = probs.argmax(dim=-1).squeeze(0)  # [10]

            # 生成随机动作矩阵 [num_samples, num_process]
            random_actions = torch.randint(0, 4, (num_samples, self.process_num))

            # 创建epsilon-greedy掩码 [num_samples, num_process]
            mask = torch.rand(num_samples, self.process_num) < epsilon

            # 组合选择结果
            actions = torch.where(
                mask,
                random_actions,
                best_actions.expand_as(random_actions)
            )

            return actions  # [num_samples, num_process]

    def _compute_advantages(self, rewards, normalize=True):
        """计算更高级的优势函数估计"""
        # 基础方式: 去均值，除以标准差
        advantages = rewards - rewards.mean()
        
        if normalize:
            # 使用rolling statistics减小方差
            std = rewards.std()
            if std > 0:
                advantages = advantages / (std + 1e-8)
            
            # 应用截断来降低极端值对训练的影响
            advantages = torch.clamp(advantages, min=-10.0, max=10.0)
        
        return advantages

    def _compute_kl_divergence(self, new_logits, old_logits):
        """计算更高效的KL散度"""
        # 正向KL：new distribution与old distribution的差异
        new_log_probs = F.log_softmax(new_logits, dim=-1)
        old_probs = F.softmax(old_logits.detach(), dim=-1) 
        
        # 计算KL散度: KL(new||old) = sum(new * log(new/old))
        kl = F.kl_div(
            new_log_probs,
            old_probs,
            reduction='none',
            log_target=False
        )
        
        # 计算每个进程的平均KL散度
        kl = kl.sum(dim=-1).mean()
        
        return kl

    def update_policy(self, actions_batch, rewards_batch):
        """策略更新核心逻辑"""
        # 转换为张量
        actions = torch.tensor(actions_batch, dtype=torch.long)  # if self.process_num=10, [batch, 10]
        rewards = torch.tensor(rewards_batch, dtype=torch.float32)  # [batch]
        
        # 追踪最佳结果
        max_reward_idx = torch.argmax(rewards).item()
        max_reward = rewards[max_reward_idx].item()
        if max_reward > self.best_reward:
            self.best_reward = max_reward
            self.best_actions = actions[max_reward_idx].tolist()

        # 获取新旧策略的logits
        with torch.no_grad():
            old_logits = self.old_policy()  # 使用旧策略网络
        new_logits = self.policy()  # 当前策略网络

        # 计算重要性采样比率
        ratios = []
        for i in range(self.process_num):
            # print(i, new_logits.shape, old_logits.shape)
            new_prob = F.softmax(new_logits[0, i, :], dim=-1)
            old_prob = F.softmax(old_logits[0, i, :], dim=-1)

            # 计算每个动作的概率比
            action_idx = actions[:, i]
            ratio = new_prob[action_idx] / (old_prob[action_idx] + 1e-8)
            ratios.append(ratio)

        ratios = torch.stack(ratios, dim=1)  # [batch, 10] <-- Epsilon-Greedy ?

        # 计算联合概率比和优势
        joint_ratios = ratios.prod(dim=1)  # 各流程选择的联合概率
        advantages = self._compute_advantages(rewards, self.normalize_advantages)

        # GRPO损失计算
        surr1 = joint_ratios * advantages
        surr2 = torch.clamp(joint_ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 改进的KL散度计算
        kl_div = self._compute_kl_divergence(new_logits, old_logits)

        # 总损失
        total_loss = policy_loss + self.kl_coeff * kl_div

        # 梯度更新
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        # 同步旧策略参数
        self.old_policy.load_state_dict(self.policy.state_dict())

        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': kl_div.item(),
            'avg_reward': rewards.mean().item(),
            'max_reward': max_reward,
            'best_actions': self.best_actions
        }

def generate_config_and_evaluate(config_set):
    re = {"node1": {"method_A":0.1, "method_B":0.1, "method_C":10.0, "None": 0.1},
          "node2": {"method_A":0.1, "method_B":10.0, "method_C":0.1, "None":0.1},
          "node3": {"method_A":10.0, "method_B":0.1, "method_C":0.1, "None":0.1},
          "node4": {"method_A":0.1, "method_B":0.1, "method_C":0.1, "None":10.0}}
    
    # 全新实现
    reward = 1.0
    nodes = list(re.keys())
    
    for i, node in enumerate(nodes):
        if i < len(config_set):
            method = config_set[i]
            if method in re[node]:
                reward += re[node][method]
    
    # 减小随机因素影响
    return reward/40 + 0.8 + np.random.rand()*0.05  # 固定随机范围

# 示例用法
if __name__ == "__main__":
    # 初始化训练器和黑盒评估函数
    num_process = 4
    trainer = GRPOTrainer(
        num_process=num_process,
        kl_coeff=0.01,    # 降低KL惩罚以加速收敛
        clip_eps=0.2      # 减小裁剪范围使得更新更频繁
    )


    # 假设的黑盒评估函数（需要用户自行实现）
    def black_box_evaluation(actions):
        """输入: [batch_size, 10]的numpy数组
        输出: [batch_size]的奖励值"""
        config = {"node1":{"0":"method_A", "1":"method_B", "2":"method_C", "3":"None"},
         "node2":{"0":"method_A", "1":"method_B", "2":"method_C", "3":"None"},
         "node3":{"0":"method_A", "1":"method_B", "2":"method_C", "3":"None"},
         "node4":{"0":"method_A", "1":"method_B", "2":"method_C", "3":"None"}}

        rewards = []
        batch = actions.shape[0]
        for single_sample in range(batch):
            single_action_set = []
            for it, (k, v) in enumerate(config.items()):
                single_action_set.append(v[str(actions[single_sample][it])])
            rewards.append(generate_config_and_evaluate(single_action_set))


        # 这里用随机值示例，实际应替换为真实评估逻辑
        return rewards # [batch_size]

        # 训练循环

    x = []
    y = []
    best_y = []
    
    # 训练参数
    num_epochs = 100
    batch_size = 64      # 增大批量大小提高估计稳定性
    
    print("开始训练GRPO...")
    for epoch in range(num_epochs):
        # 自适应探索率: 从大到小逐渐衰减
        epsilon = max(0.7 * (1.0 - epoch / num_epochs), 0.05)
        
        # 生成策略样本
        actions = trainer.generate_actions(batch_size, epsilon)
        actions_np = actions.numpy()  # 转换为numpy数组

        # 黑盒评估（假设支持批量评估）
        rewards = black_box_evaluation(actions_np)

        # 策略更新
        metrics = trainer.update_policy(actions_np, rewards)
        
        # 获取当前最优策略
        current_actions = trainer.generate_actions(1, 0).tolist()

        print(f"Optimized policy: {current_actions}")
        # 打印训练日志
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: "
                  f"Loss={metrics['total_loss']:.3f} | "
                  f"Avg Reward={metrics['avg_reward']:.3f} | "
                  f"Entropy={metrics['entropy']:.3f} | "
                  f"Epsilon={epsilon:.2f}")
            x.append(epoch/10.0)
            y.append(metrics['avg_reward'])
            best_y.append(metrics['max_reward'])
            
            print(f"Current best actions: {metrics['best_actions']} with reward {metrics['max_reward']:.4f}")
            print(f"Current policy outputs: {current_actions}")

    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='Average Reward')
    plt.plot(x, best_y, 'r--', label='Best Reward')
    plt.xlabel('Training Progress (Epoch/10)')
    plt.ylabel('Reward')
    plt.title('GRPO Training Performance')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 最终策略导出
    final_actions = trainer.generate_actions(1, 0).tolist()
    print(f"Optimized policy: {final_actions}")
    print(f"Best policy found: {trainer.best_actions} with reward {trainer.best_reward:.4f}")