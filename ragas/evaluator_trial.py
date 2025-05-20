from autorag.evaluator import Evaluator

evaluator = Evaluator(qa_data_path='/home/xwh/AutoRAG/data/5dataset_100/qa100.parquet', corpus_data_path='/home/xwh/AutoRAG/data/5dataset_100/corpus.parquet')
# evaluator.start_trial('/home/xwh/AutoRAG/experiments/ollama_config_new.yaml')
evaluator.start_trial('/home/xwh/AutoRAG/experiments/4-100-0504/strategy_opt/best_config.yaml')