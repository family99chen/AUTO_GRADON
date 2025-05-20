import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DirectPolicyNetwork(nn.Module):
    def __init__(self, num_process=10, d_model=128, nhead=4, num_layers=3, operation_dim=4):
        super().__init__()
        self.num_process = num_process

        # 初始嵌入层
        self.process_embedding = nn.Linear(num_process, d_model)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.output_layer = nn.Linear(d_model, operation_dim)

    def forward(self):
        # 生成初始嵌入 [batch_size, num_process, d_model]
        src = self.process_embedding(torch.ones(1, self.num_process))  # [1, 10] -> [1, 10, 128]

        # 通过Transformer编码器
        encoded = self.transformer_encoder(src)  # [1, 10, 128]

        # 生成最终输出
        logits = self.output_layer(encoded)  # [1, 10, 4]
        return logits


class GRPOTrainer:
    def __init__(self, num_process=10, d_model=128, nhead=4, num_layers=3, operation=4):
        # 初始化当前策略和旧策略网络
        self.policy = DirectPolicyNetwork(num_process, d_model, nhead, num_layers, operation)
        self.old_policy = DirectPolicyNetwork(num_process, d_model, nhead, num_layers, operation)
        self.old_policy.load_state_dict(self.policy.state_dict())  # 参数同步

        # 优化器和超参数
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.clip_eps = 0.2
        self.kl_coeff = 0.05
        self.process_num = num_process
        self.operation_dim = operation

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

    def update_policy(self, actions_batch, rewards_batch):
        """策略更新核心逻辑"""
        # 转换为张量
        actions = torch.tensor(actions_batch, dtype=torch.long)  # if self.process_num=10, [batch, 10]
        rewards = torch.tensor(rewards_batch, dtype=torch.float32)  # [batch]

        # 获取新旧策略的logits
        with torch.no_grad():
            old_logits = self.old_policy()  # 使用旧策略网络
        new_logits = self.policy()  # 当前策略网络

        # 计算重要性采样比率
        ratios = []
        for i in range(self.process_num):
            new_prob = F.softmax(new_logits[0, i, :], dim=-1)
            old_prob = F.softmax(old_logits[0, i, :], dim=-1)

            # 计算每个动作的概率比
            action_idx = actions[:, i]
            ratio = new_prob[action_idx] / (old_prob[action_idx] + 1e-8)
            ratios.append(ratio)

        ratios = torch.stack(ratios, dim=1)  # [batch, 10] <-- Epsilon-Greedy ?

        # 计算联合概率比和优势
        joint_ratios = ratios.prod(dim=1)  # 各流程选择的联合概率
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # PPO损失计算
        surr1 = joint_ratios * advantages
        surr2 = torch.clamp(joint_ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        kl_div = - F.kl_div(
            F.log_softmax(new_logits, dim=-1),
            F.softmax(old_logits.detach(), dim=-1),
            reduction='none'
        ).sum(dim=-1).mean()

        # 总损失
        total_loss = policy_loss + self.kl_coeff * kl_div

        # 同步旧策略参数
        self.old_policy.load_state_dict(self.policy.state_dict())

        # 梯度更新
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()



        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': kl_div.item(),
            'avg_reward': rewards.mean().item()
        }

def generate_config_and_evaluate(config_set):
    # generate config and evaluate them to choose the best one
    pass

# 示例用法
if __name__ == "__main__":
    # 初始化训练器和黑盒评估函数
    num_process = 10
    trainer = GRPOTrainer()


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
            for iter in range(num_process):
                for k, v in config.items():
                    single_action_set.append(v[str(actions[single_sample])])
                rewards.append(generate_config_and_evaluate(single_action_set))


        # 这里用随机值示例，实际应替换为真实评估逻辑
        return rewards # [batch_size]

        # 训练循环


    for epoch in range(1000):
        # 生成策略样本
        actions = trainer.generate_actions(10, 0.2)  # 生成1024个策略样本
        actions_np = actions.numpy()  # 转换为numpy数组

        # 黑盒评估（假设支持批量评估）
        rewards = black_box_evaluation(actions_np)

        # 策略更新
        metrics = trainer.update_policy(actions_np, rewards)

        # 打印训练日志
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: "
                  f"Loss={metrics['total_loss']:.3f} | "
                  f"Avg Reward={metrics['avg_reward']:.3f} | "
                  f"Entropy={metrics['entropy']:.3f}")

    # 最终策略导出
    final_actions = trainer.generate_actions(1)[0].tolist()
    print(f"Optimized policy: {final_actions}")