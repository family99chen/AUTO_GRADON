import os
import torch
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import time
from tqdm import tqdm, trange
import logging
from datetime import datetime
from contextlib import contextmanager

# Set tokenizers parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 导入自定义模块
from configuration.promptmaker import PromptMakerConfiguration
from create_config import create_config, load_and_generate_nodes
from evaluator import TestEvaluator
from grpo import GRPOTrainer

class PromptOptimizer:
    """使用GRPO优化RAG系统的Prompt配置"""
    
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
                 trial_name: str = "prompt_optimization"):
        """
        初始化优化器
        
        Args:
            qa_data_path: 问答数据路径
            corpus_data_path: 语料库数据路径
            project_dir: 项目目录
            config_path: 配置文件路径
            trial_name: 试验名称
        """
        # 初始化评估器
        self.evaluator = TestEvaluator(qa_data_path, corpus_data_path, project_dir)
        self.evaluator.init_trial(trial_name)
        
        # 保存路径 
        self.config_path = config_path
        self.project_dir = project_dir
        self.trial_dir = os.path.join(project_dir, trial_name)
        os.makedirs(os.path.join(self.trial_dir, "configs"), exist_ok=True)
        
        # 设置日志
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
        
        # 加载基础配置
        self.base_config = self._load_base_config()
        
        # 获取PromptMaker配置空间
        self.prompt_config = PromptMakerConfiguration.load_from_yaml(config_path, key="prompt_maker")
        
        # 从配置中提取prompt候选项
        self.prompt_candidates = self.base_config["prompt_maker"]["prompt"]
        print(f"初始prompt候选项: {self.prompt_candidates}")
        
    def _load_base_config(self) -> Dict:
        """加载基础配置"""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)
    
    def setup_grpo_trainer(self, 
                           num_prompts: int, 
                           d_model: int = 128, 
                           nhead: int = 4, 
                           num_layers: int = 3):
        """
        设置GRPO训练器
        
        Args:
            num_prompts: prompt候选数量
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
        """
        self.trainer = GRPOTrainer(
            num_process=1,  # 仅优化prompt
            d_model=d_model,
            nhead=nhead, 
            num_layers=num_layers,
            operation=num_prompts,  # 操作维度等于prompt候选数量
            kl_coeff=0.01,
            clip_eps=0.2
        )
        return self.trainer
    
    def generate_config(self, prompt_idx: int) -> str:
        """
        生成配置文件
        
        Args:
            prompt_idx: prompt索引
            
        Returns:
            配置文件路径
        """
        # 获取选择的prompt
        selected_prompt = self.prompt_candidates[prompt_idx]
        
        # 加载所有组件配置
        config_path = self.config_path
        keys = [
            "vectordb",
            "query_expansion",
            "retrieval",
            "passage_augmenter",
            "passage_reranker",
            "passage_filter",
            "passage_compressor",
            "generator",
        ]
        
        # 加载所有组件的node_lines
        node_lines_list = []
        retriever_cfg = None
        
        for key in keys:
            lines, cfg = load_and_generate_nodes(config_path, key, size=1, exhaustive=False)
            if key == "retrieval":
                retriever_cfg = cfg
            node_lines_list.append(lines[0])  # 取第一个配置
            
        # 创建自定义的prompt节点配置
        prompt_node_line = {
            "node_line_name": "prompt_maker_node_line",
            "nodes": [
                {
                    "node_type": "prompt_maker",
                    'strategy': {'metrics': ['meteor', 'rouge', 'bert_score']},
                    "modules": [
                        {
                            'module_type': 'fstring',
                            'prompt': selected_prompt
                        }
                    ]
                }
            ]
        }
        
        # 添加自定义的prompt配置
        all_node_lines = node_lines_list[:5] + [prompt_node_line] + node_lines_list[6:]
        
        # 设置额外参数
        extra_params = {
            'bm25_tokenizer_list': retriever_cfg.cs.get('[bm25]bm25_tokenizer').choices
                if '[bm25]bm25_tokenizer' in retriever_cfg.cs else ['porter_stemmer', 'space'],
            'strategies': {'metrics': ['meteor', 'rouge', 'bert_score']}
        }
        
        # 保存配置文件
        config_file = os.path.join(self.trial_dir, "configs", f"config_prompt_{prompt_idx}.yaml")
        create_config(node_lines_list[0], *all_node_lines[1:], extra_params=extra_params, save_path=config_file)
        
        return config_file
    
    def evaluate_configs(self, actions_batch) -> List[float]:
        """评估多个配置"""
        rewards = []
        total_start = time.time()
        
        for i, actions in enumerate(actions_batch):
            prompt_idx = actions[0]
            start_time = time.time()
            
            self.logger.info(f"开始评估配置 {i+1}/{len(actions_batch)} (prompt_idx: {prompt_idx})")
            
            try:
                with self.timer(f"生成配置文件 {prompt_idx}"):
                    config_file = self.generate_config(prompt_idx)
                
                with self.timer(f"评估配置 {prompt_idx}"):
                    self.evaluator.init_runner_from_yaml(config_file)
                    summary_df = self.evaluator.run_with_qa_eval(yaml_name=f"prompt_{prompt_idx}.yaml")
                    
                    rouge_score = summary_df["rouge"].values[0]
                    meteor_score = summary_df["meteor"].values[0]
                    bert_score = summary_df.get("bert_score", pd.Series([0])).values[0]
                    
                    reward = 0.4 * rouge_score + 0.4 * meteor_score + 0.2 * bert_score
                    rewards.append(reward)
                    
                    self.logger.info(f"配置 {prompt_idx} 评估完成:")
                    self.logger.info(f"  - ROUGE: {rouge_score:.4f}")
                    self.logger.info(f"  - METEOR: {meteor_score:.4f}")
                    self.logger.info(f"  - BERT: {bert_score:.4f}")
                    self.logger.info(f"  - 总奖励: {reward:.4f}")
                    
            except Exception as e:
                self.logger.error(f"评估配置 {prompt_idx} 时出错: {str(e)}")
                rewards.append(0.0)
            
            eval_time = time.time() - start_time
            self.logger.info(f"配置 {prompt_idx} 评估耗时: {eval_time:.2f} 秒")
        
        total_time = time.time() - total_start
        self.logger.info(f"批次评估完成，总耗时: {total_time:.2f} 秒，平均每个配置耗时: {total_time/len(actions_batch):.2f} 秒")
        
        return rewards
    
    def run_optimization(self, num_epochs: int = 20, batch_size: int = 4):
        """运行优化流程"""
        optimization_start = time.time()
        self.logger.info(f"开始GRPO优化，共{len(self.prompt_candidates)}个prompt候选项，{num_epochs}轮训练")
        
        with self.timer("设置GRPO训练器"):
            self.setup_grpo_trainer(num_prompts=len(self.prompt_candidates))
        
        history = {
            "epoch": [],
            "avg_reward": [],
            "max_reward": [],
            "best_prompt": [],
            "epoch_time": []
        }
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            self.logger.info(f"\n{'='*20} Epoch {epoch+1}/{num_epochs} {'='*20}")
            
            epsilon = max(0.7 * (1.0 - epoch / num_epochs), 0.05)
            self.logger.info(f"当前探索率 (epsilon): {epsilon:.3f}")
            
            with self.timer(f"Epoch {epoch+1} 生成动作样本"):
                actions = self.trainer.generate_actions(batch_size, epsilon)
                actions_np = actions.numpy()
            
            with self.timer(f"Epoch {epoch+1} 评估配置"):
                rewards = self.evaluate_configs(actions_np)
            
            with self.timer(f"Epoch {epoch+1} 更新策略"):
                metrics = self.trainer.update_policy(actions_np, rewards)
            
            epoch_time = time.time() - epoch_start
            history["epoch"].append(epoch)
            history["avg_reward"].append(metrics["avg_reward"])
            history["max_reward"].append(metrics["max_reward"])
            history["epoch_time"].append(epoch_time)
            
            best_prompt_idx = metrics["best_actions"][0]
            best_prompt = self.prompt_candidates[best_prompt_idx]
            history["best_prompt"].append(best_prompt)
            
            self.logger.info(f"Epoch {epoch+1} 统计:")
            self.logger.info(f"  - 平均奖励: {metrics['avg_reward']:.4f}")
            self.logger.info(f"  - 最大奖励: {metrics['max_reward']:.4f}")
            self.logger.info(f"  - 当前最佳prompt: {best_prompt}")
            self.logger.info(f"  - 本轮耗时: {epoch_time:.2f} 秒")
        
        total_time = time.time() - optimization_start
        self.logger.info(f"\n优化完成!")
        self.logger.info(f"总耗时: {total_time:.2f} 秒")
        self.logger.info(f"平均每轮耗时: {total_time/num_epochs:.2f} 秒")
        self.logger.info(f"最佳prompt: {self.prompt_candidates[self.trainer.best_actions[0]]}")
        self.logger.info(f"最佳奖励: {self.trainer.best_reward:.4f}")
        
        # 保存优化历史
        history_df = pd.DataFrame(history)
        history_path = os.path.join(self.trial_dir, "optimization_history.csv")
        history_df.to_csv(history_path, index=False)
        self.logger.info(f"优化历史已保存至: {history_path}")
        
        # 保存最佳配置文件
        best_prompt_idx = self.trainer.best_actions[0]
        best_config_file = self.generate_config(best_prompt_idx)
        best_config_path = os.path.join(self.trial_dir, "best_config.yaml")
        import shutil
        shutil.copy2(best_config_file, best_config_path)
        self.logger.info(f"最佳配置文件已保存至: {best_config_path}")
        
        return best_prompt, self.trainer.best_reward
    
    def extend_prompt_candidates(self, new_prompts: List[str]):
        """
        扩展prompt候选项
        
        Args:
            new_prompts: 新的prompt列表
        """
        self.prompt_candidates.extend(new_prompts)
        print(f"扩展后的prompt候选项: {self.prompt_candidates}")
        return self.prompt_candidates


# 使用示例
if __name__ == "__main__":
    # 配置路径
    qa_data_path = "../data/qa_small.parquet"
    corpus_data_path = "../data/corpus_small.parquet"
    project_dir = "../experiments"
    
    # 初始化优化器
    optimizer = PromptOptimizer(
        qa_data_path=qa_data_path,
        corpus_data_path=corpus_data_path,
        project_dir=project_dir,
        trial_name="prompt_opt_trial"
    )
    
    # 扩展prompt候选项
    optimizer.extend_prompt_candidates([
        "Based on the following information, answer the question: {query} \n Context: {retrieved_contents}",
        "Answer the question: {query} \n Using only the following context: {retrieved_contents}",
    ])
    
    # 运行优化
    best_prompt, best_reward = optimizer.run_optimization(num_epochs=20, batch_size=4)