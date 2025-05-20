import os
import sys
import torch.distributed as dist
import torch.multiprocessing as mp
import torch

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()

def run_training(rank, world_size):
    setup(rank, world_size)
    
    # 导入优化器
    from grpo_create_trainer_evaluate import PromptOptimizer
    
    # 配置路径
    qa_data_path = "../data/qa.parquet"
    corpus_data_path = "../data/corpus.parquet"
    project_dir = "../experiments"
    
    # 初始化优化器
    optimizer = PromptOptimizer(
        qa_data_path=qa_data_path,
        corpus_data_path=corpus_data_path,
        project_dir=project_dir,
        trial_name=f"prompt_opt_trial_rank_{rank}"
    )
    
    # 运行优化
    best_prompt, best_reward = optimizer.run_optimization(
        num_epochs=20,
        batch_size=4  # 基础batch size，会根据GPU数量自动调整
    )
    
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            run_training,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    else:
        print("需要至少2个GPU来运行分布式训练！")
        sys.exit(1)

if __name__ == "__main__":
    main()