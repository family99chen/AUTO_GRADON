from autorag.evaluator import Evaluator

evaluator = Evaluator(qa_data_path='/home/cz/AUTO_GRADON/data/5dataset_100/qa100.parquet', corpus_data_path='/home/cz/AUTO_GRADON/data/5dataset_100/corpus_relate.parquet')
# evaluator.start_trial('/home/xwh/AutoRAG/experiments/ollama_config_new.yaml')
evaluator.start_trial('/home/cz/AUTO_GRADON/experiments/4-100-0504/strategy_opt/best_config.yaml')