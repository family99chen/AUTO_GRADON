from autorag import Evaluator, extract_best_config

# 初始化评估器
evaluator = Evaluator()

# 设置评估的数据集和配置
evaluator.init_trial(trial_name="my_trial", yaml_dir="path/to/config/dir")

# 运行评估
evaluator.run_single_pass(yaml_file="ollama_config_new.yaml", save_name="results.csv")

# 从评估结果中提取最佳配置
best_config = extract_best_config(
    yaml_path="ollama_config_new.yaml",
    summary_path="path/to/summary.csv"
)

# 保存最佳配置
best_config.save("best_config.yaml")

# 使用最佳配置运行
from autorag.deploy import Runner
runner = Runner.from_yaml("best_config.yaml")