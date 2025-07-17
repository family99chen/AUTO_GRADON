import logging
import os
import pathlib
from copy import deepcopy
from typing import List, Dict, Optional

import pandas as pd

from autorag.nodes.retrieval.run import evaluate_retrieval_node
from autorag.schema.metricinput import MetricInput
from autorag.strategy import measure_speed, filter_by_threshold, select_best
from autorag.support import get_support_modules
from autorag.utils.util import make_combinations, explode

logger = logging.getLogger("AutoRAG")


def run_query_expansion_node(
	modules: List,
	module_params: List[Dict],
	previous_result: pd.DataFrame,
	node_line_dir: str,
	strategies: Dict,
) -> pd.DataFrame:
	"""
	Run evaluation and select the best module among query expansion node results.
	Initially, retrieval is run using expanded_queries, the result of the query_expansion module.
	The retrieval module is run as a combination of the retrieval_modules in strategies.
	If there are multiple retrieval_modules, run them all and choose the best result.
	If there are no retrieval_modules, run them with the default of bm25.
	In this way, the best result is selected for each module, and then the best result is selected.

	:param modules: Query expansion modules to run.
	:param module_params: Query expansion module parameters.
	:param previous_result: Previous result dataframe.
	    In this case, it would be qa data.
	:param node_line_dir: This node line's directory.
	:param strategies: Strategies for query expansion node.
	:return: The best result dataframe.
	"""
	if not os.path.exists(node_line_dir):
		os.makedirs(node_line_dir)
	node_dir = os.path.join(node_line_dir, "query_expansion")
	if not os.path.exists(node_dir):
		os.makedirs(node_dir)
	project_dir = pathlib.PurePath(node_line_dir).parent.parent

	# run query expansion
	results, execution_times = zip(
		*map(
			lambda task: measure_speed(
				task[0].run_evaluator,
				project_dir=project_dir,
				previous_result=previous_result,
				**task[1],
			),
			zip(modules, module_params),
		)
	)
	average_times = list(map(lambda x: x / len(results[0]), execution_times))

	# save results to folder
	pseudo_module_params = deepcopy(module_params)
	for i, module_param in enumerate(pseudo_module_params):
		if "prompt" in module_params:
			module_param["prompt"] = str(i)
	filepaths = list(
		map(lambda x: os.path.join(node_dir, f"{x}.parquet"), range(len(modules)))
	)
	list(
		map(lambda x: x[0].to_parquet(x[1], index=False), zip(results, filepaths))
	)  # execute save to parquet
	filenames = list(map(lambda x: os.path.basename(x), filepaths))

	# make summary file
	summary_df = pd.DataFrame(
		{
			"filename": filenames,
			"module_name": list(map(lambda module: module.__name__, modules)),
			"module_params": module_params,
			"execution_time": average_times,
		}
	)

	# Run evaluation when there are more than one module.
	if len(modules) > 1:
		# pop general keys from strategies (e.g. metrics, speed_threshold)
		general_key = ["metrics", "speed_threshold", "strategy"]
		general_strategy = dict(
			filter(lambda x: x[0] in general_key, strategies.items())
		)
		extra_strategy = dict(
			filter(lambda x: x[0] not in general_key, strategies.items())
		)

		# first, filter by threshold if it is enabled.
		if general_strategy.get("speed_threshold") is not None:
			results, filenames = filter_by_threshold(
				results, average_times, general_strategy["speed_threshold"], filenames
			)

		# check metrics in strategy
		if general_strategy.get("metrics") is None:
			raise ValueError(
				"You must at least one metrics for query expansion evaluation."
			)

		if extra_strategy.get("top_k") is None:
			extra_strategy["top_k"] = 10  # default value

		# get retrieval modules from strategy
		retrieval_callables, retrieval_params = make_retrieval_callable_params(
			extra_strategy
		)

		# get retrieval_gt
		retrieval_gt = pd.read_parquet(
			os.path.join(project_dir, "data", "qa.parquet"), engine="pyarrow"
		)["retrieval_gt"].tolist()

		# 添加详细的调试信息
		print(f"🔍 Debug - metric_inputs构造过程:")
		print(f"   project_dir: {project_dir}")
		print(f"   qa.parquet路径: {os.path.join(project_dir, 'data', 'qa.parquet')}")
		print(f"   retrieval_gt长度: {len(retrieval_gt)}")
		print(f"   previous_result shape: {previous_result.shape}")
		print(f"   previous_result['query']长度: {len(previous_result['query'].tolist())}")
		print(f"   previous_result['generation_gt']长度: {len(previous_result['generation_gt'].tolist())}")
		
		# 检查previous_result的列
		print(f"   previous_result列名: {previous_result.columns.tolist()}")
		
		# 检查retrieval_gt的前几个元素类型和内容
		print(f"   retrieval_gt前3个元素类型: {[type(x) for x in retrieval_gt[:3]]}")
		print(f"   retrieval_gt前3个元素内容: {retrieval_gt[:3]}")
		
		# 检查previous_result的前几行
		print(f"   previous_result前3行query: {previous_result['query'].head(3).tolist()}")
		print(f"   previous_result前3行generation_gt: {previous_result['generation_gt'].head(3).tolist()}")

		# make rows to metric_inputs
		query_list = previous_result["query"].tolist()
		generation_gt_list = previous_result["generation_gt"].tolist()
		
		print(f"🔍 Debug - 准备zip的三个列表长度:")
		print(f"   retrieval_gt: {len(retrieval_gt)}")
		print(f"   query_list: {len(query_list)}")
		print(f"   generation_gt_list: {len(generation_gt_list)}")
		
		# 计算zip后的预期长度
		expected_zip_length = min(len(retrieval_gt), len(query_list), len(generation_gt_list))
		print(f"   zip后预期长度: {expected_zip_length}")
		
		metric_inputs = [
			MetricInput(retrieval_gt=ret_gt, query=query, generation_gt=gen_gt)
			for ret_gt, query, gen_gt in zip(
				retrieval_gt,
				query_list,
				generation_gt_list,
			)
		]
		
		print(f"🔍 Debug - metric_inputs构造完成:")
		print(f"   实际生成的metric_inputs长度: {len(metric_inputs)}")
		
		# 检查前几个metric_inputs的内容
		for i, mi in enumerate(metric_inputs[:3]):
			print(f"   metric_inputs[{i}] - retrieval_gt类型: {type(mi.retrieval_gt)}, query: {mi.query[:50]}...")

		# run evaluation
		evaluation_results = list(
			map(
				lambda result: evaluate_one_query_expansion_node(
					retrieval_callables,
					retrieval_params,
					[
						setattr(metric_input, "queries", queries) or metric_input
						for metric_input, queries in zip(
							metric_inputs, result["queries"].to_list()
						)
					],
					general_strategy["metrics"],
					project_dir,
					previous_result,
					general_strategy.get("strategy", "mean"),
				),
				results,
			)
		)

		evaluation_df = pd.DataFrame(
			{
				"filename": filenames,
				**{
					f"query_expansion_{metric_name}": list(
						map(lambda x: x[metric_name].mean(), evaluation_results)
					)
					for metric_name in general_strategy["metrics"]
				},
			}
		)
		summary_df = pd.merge(
			on="filename", left=summary_df, right=evaluation_df, how="left"
		)

		best_result, best_filename = select_best(
			evaluation_results,
			general_strategy["metrics"],
			filenames,
			strategies.get("strategy", "mean"),
		)
		# change metric name columns to query_expansion_metric_name
		best_result = best_result.rename(
			columns={
				metric_name: f"query_expansion_{metric_name}"
				for metric_name in strategies["metrics"]
			}
		)
		best_result = best_result.drop(
			columns=["retrieved_contents", "retrieved_ids", "retrieve_scores"]
		)
	else:
		best_result, best_filename = results[0], filenames[0]
		best_result = pd.concat([previous_result, best_result], axis=1)

	# add 'is_best' column at summary file
	summary_df["is_best"] = summary_df["filename"] == best_filename

	# save files
	summary_df.to_csv(os.path.join(node_dir, "summary.csv"), index=False)
	best_result.to_parquet(
		os.path.join(node_dir, f"best_{os.path.splitext(best_filename)[0]}.parquet"),
		index=False,
	)

	return best_result


def evaluate_one_query_expansion_node(
	retrieval_funcs: List,
	retrieval_params: List[Dict],
	metric_inputs: List[MetricInput],
	metrics: List[str],
	project_dir,
	previous_result: pd.DataFrame,
	strategy_name: str,
) -> pd.DataFrame:
	previous_result["queries"] = [
		metric_input.queries for metric_input in metric_inputs
	]
	retrieval_results = list(
		map(
			lambda x: x[0].run_evaluator(
				project_dir=project_dir, previous_result=previous_result, **x[1]
			),
			zip(retrieval_funcs, retrieval_params),
		)
	)
	evaluation_results = list(
		map(
			lambda x: evaluate_retrieval_node(
				x,
				metric_inputs,
				metrics,
			),
			retrieval_results,
		)
	)
	best_result, _ = select_best(
		evaluation_results, metrics, strategy_name=strategy_name
	)
	best_result = pd.concat([previous_result, best_result], axis=1)
	return best_result


def make_retrieval_callable_params(strategy_dict: Dict):
	"""
	strategy_dict looks like this:

	.. Code:: json

	    {
	        "metrics": ["retrieval_f1", "retrieval_recall"],
	        "top_k": 50,
	        "retrieval_modules": [
	          {"module_type": "bm25"},
	          {"module_type": "vectordb", "embedding_model": ["openai", "huggingface"]}
	        ]
	      }

	"""
	node_dict = deepcopy(strategy_dict)
	retrieval_module_list: Optional[List[Dict]] = node_dict.pop(
		"retrieval_modules", None
	)
	if retrieval_module_list is None:
		retrieval_module_list = [
			{
				"module_type": "bm25",
			}
		]
	node_params = node_dict
	modules = list(
		map(
			lambda module_dict: get_support_modules(module_dict.pop("module_type")),
			retrieval_module_list,
		)
	)
	param_combinations = list(
		map(
			lambda module_dict: make_combinations({**module_dict, **node_params}),
			retrieval_module_list,
		)
	)
	return explode(modules, param_combinations)
