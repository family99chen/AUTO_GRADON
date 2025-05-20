from copy import deepcopy
from json import JSONDecoder
from typing import List, Callable, Dict, Optional, Any, Collection, Iterable
import os
import yaml
import re
import ast
import pandas as pd


def summary_df_to_yaml(summary_df: pd.DataFrame, config_dict: Dict) -> Dict:
    """
    Convert trial summary dataframe to config yaml file.

    :param summary_df: The trial summary dataframe of the evaluated trial.
    :param config_dict: The yaml configuration dict for the pipeline.
        You can load this to access trail_folder/config.yaml.
    :return: Dictionary of config yaml file.
        You can save this dictionary to yaml file.
    """

    # summary_df columns : 'node_line_name', 'node_type', 'best_module_filename',
    #                      'best_module_name', 'best_module_params', 'best_execution_time'
    node_line_names = extract_node_line_names(config_dict)
    node_strategies = extract_node_strategy(config_dict)
    strategy_df = pd.DataFrame(
        {
            "node_type": list(node_strategies.keys()),
            "strategy": list(node_strategies.values()),
        }
    )
    summary_df = summary_df.merge(strategy_df, on="node_type", how="left")
    summary_df["categorical_node_line_name"] = pd.Categorical(
        summary_df["node_line_name"], categories=node_line_names, ordered=True
    )
    summary_df = summary_df.sort_values(by="categorical_node_line_name")
    grouped = summary_df.groupby("categorical_node_line_name", observed=False)

    node_lines = [
        {
            "node_line_name": node_line_name,
            "nodes": [
                {
                    "node_type": row["node_type"],
                    "strategy": row["strategy"],
                    "modules": [
                        {
                            "module_type": row["best_module_name"],
                            **row["best_module_params"],
                        }
                    ],
                }
                for _, row in node_line.iterrows()
            ],
        }
        for node_line_name, node_line in grouped
    ]
    return {"node_lines": node_lines}


def extract_best_config(trial_path: str, output_path: Optional[str] = None) -> Dict:
    """
    Extract the optimal pipeline from the evaluated trial.

    :param trial_path: The path to the trial directory that you want to extract the pipeline from.
        Must already be evaluated.
    :param output_path: Output path that pipeline yaml file will be saved.
        Must be .yaml or .yml file.
        If None, it does not save YAML file and just returns dict values.
        Default is None.
    :return: The dictionary of the extracted pipeline.
    """
    summary_path = os.path.join(trial_path, "summary.csv")
    if not os.path.exists(summary_path):
        raise ValueError(f"summary.csv does not exist in {trial_path}.")
    trial_summary_df = load_summary_file(
        summary_path, dict_columns=["best_module_params"]
    )
    config_yaml_path = os.path.join(trial_path, "config.yaml")
    with open(config_yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    yaml_dict = summary_df_to_yaml(trial_summary_df, config_dict)
    yaml_dict["vectordb"] = extract_vectordb_config(trial_path)
    if output_path is not None:
        with open(output_path, "w") as f:
            yaml.safe_dump(yaml_dict, f)
    return yaml_dict


def extract_vectordb_config(trial_path: str) -> List[Dict]:
    # get vectordb.yaml file
    project_dir = pathlib.PurePath(os.path.realpath(trial_path)).parent
    vectordb_config_path = os.path.join(project_dir, "resources", "vectordb.yaml")
    if not os.path.exists(vectordb_config_path):
        raise ValueError(f"vectordb.yaml does not exist in {vectordb_config_path}.")
    with open(vectordb_config_path, "r") as f:
        vectordb_dict = yaml.safe_load(f)
    result = vectordb_dict.get("vectordb", [])
    if len(result) != 0:
        return result
    # return default setting of chroma
    return [
        {
            "name": "default",
            "db_type": "chroma",
            "client_type": "persistent",
            "embedding_model": "openai",
            "collection_name": "openai",
            "path": os.path.join(project_dir, "resources", "chroma"),
        }
    ]



def load_yaml_config(yaml_path: str, **kwargs) -> Dict:
	"""
	Load a YAML configuration file for AutoRAG.
	It contains safe loading, converting string to tuple, and insert environment variables.

	:param yaml_path: The path of the YAML configuration file.
	:return: The loaded configuration dictionary.
	"""
	if not os.path.exists(yaml_path):
		raise ValueError(f"YAML file {yaml_path} does not exist.")
	with open(yaml_path, "r", encoding="utf-8") as stream:
		try:
			yaml_dict = yaml.safe_load(stream, **kwargs)
		except yaml.YAMLError as exc:
			raise ValueError(f"YAML file {yaml_path} could not be loaded.") from exc

	yaml_dict = convert_string_to_tuple_in_dict(yaml_dict)
	yaml_dict = convert_env_in_dict(yaml_dict)
	return yaml_dict


def convert_string_to_tuple_in_dict(d):
	"""Recursively converts strings that start with '(' and end with ')' to tuples in a dictionary."""
	for key, value in d.items():
		# If the value is a dictionary, recurse
		if isinstance(value, dict):
			convert_string_to_tuple_in_dict(value)
		# If the value is a list, iterate through its elements
		elif isinstance(value, list):
			for i, item in enumerate(value):
				# If an item in the list is a dictionary, recurse
				if isinstance(item, dict):
					convert_string_to_tuple_in_dict(item)
				# If an item in the list is a string matching the criteria, convert it to a tuple
				elif (
					isinstance(item, str)
					and item.startswith("(")
					and item.endswith(")")
				):
					value[i] = ast.literal_eval(item)
		# If the value is a string matching the criteria, convert it to a tuple
		elif isinstance(value, str) and value.startswith("(") and value.endswith(")"):
			d[key] = ast.literal_eval(value)

	return d


def convert_env_in_dict(d: Dict):
	"""
	Recursively converts environment variable string in a dictionary to actual environment variable.

	:param d: The dictionary to convert.
	:return: The converted dictionary.
	"""
	env_pattern = re.compile(r".*?\${(.*?)}.*?")

	def convert_env(val: str):
		matches = env_pattern.findall(val)
		for match in matches:
			val = val.replace(f"${{{match}}}", os.environ.get(match, ""))
		return val

	for key, value in d.items():
		if isinstance(value, dict):
			convert_env_in_dict(value)
		elif isinstance(value, list):
			for i, item in enumerate(value):
				if isinstance(item, dict):
					convert_env_in_dict(item)
				elif isinstance(item, str):
					value[i] = convert_env(item)
		elif isinstance(value, str):
			d[key] = convert_env(value)
	return d