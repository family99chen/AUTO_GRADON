from abc import abstractmethod
from typing import List, Tuple, Union
from .util import load_yaml_config
import os
from pathlib import Path
from typing import Optional, Dict
from collections.abc import ItemsView, Iterable, Iterator
from typing import IO, TYPE_CHECKING, Any, Literal, Mapping, Sequence, overload
from ConfigSpace.util import generate_grid

import pandas as pd
import numpy as np
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, \
    NotEqualsCondition, Configuration, InCondition
from ConfigSpace.hyperparameters import Hyperparameter, CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant

from .zoo import *

def create_lines_with_nodes(node_line_name: str = "post_retrieve_node_line", *nodes) -> List[Dict]:

    node_lines = [{
        "node_line_name": node_line_name,
        "nodes": list(node_group)
    } for node_group in zip(*nodes)]
    return node_lines


def parse_hyperparameters_samples(hp_config: Mapping[str, Any], module_type: Optional[str] = None) -> Mapping[str, Any]:
    if module_type is None:
        try:
            module_type = hp_config['module_type']
        except KeyError:
            module_type = hp_config['db_type']

    prefix = f"[{module_type}]"
    parsed_dict = {}
    for key, value in hp_config.items():
        if key.startswith(prefix):
            # Remove the prefix '[module_type]'
            new_key = key[len(prefix):]
            parsed_dict[new_key] = value.item() if isinstance(value, np.generic) else value
        else:
            parsed_dict[key] = value.item() if isinstance(value, np.generic) else value

    return parsed_dict

def parse_hyperparameters_from_dict(
    items: Mapping[str, Any],
) -> Iterator[Hyperparameter]:
    for name, hp in items.items():
        # Anything that is a Hyperparameter already is good
        # Note that we discard the key name in this case in favour
        # of the name given in the dictionary
        if isinstance(hp, Hyperparameter):
            yield hp

        # Tuples are bounds, check if float or int
        elif isinstance(hp, tuple):
            if len(hp) != 2:
                raise ValueError(f"'{name}' must be (lower, upper) bound, got {hp}")

            lower, upper = hp
            if isinstance(lower, float):
                yield UniformFloatHyperparameter(name, lower, upper)
            else:
                yield UniformIntegerHyperparameter(name, lower, upper)

        # Lists are categoricals
        elif isinstance(hp, list):
            if len(hp) == 0:
                raise ValueError(f"Can't have empty list for categorical {name}")

            yield CategoricalHyperparameter(name, hp)
        else:
            # It's a constant
            yield Constant(name, hp)


class BaseConfiguration:

    def __init__(
        self,
        config: Dict,
        project_dir: Optional[str] = None,
        name: Optional[str] = "dummy_node",
    ):
        assert name in ["passage_augmenter", "query_expansion", "vectordb",
                        "retrieval", "prompt_maker", "passage_reranker", "passage_filter",
                        "passage_compressor", "generator", "dummy_node"], f"Unsupported node type: {name}"
        self.config = config
        self.project_dir = os.getcwd() if project_dir is None else project_dir
        self.name = name
        self.cs = self.build(config)

    def build(self, config: Dict) -> ConfigurationSpace:
        """
        Build the configuration space.

        :param config: The configuration dictionary.
        :return: The configuration space.
        """
        return self.build_from_yaml(config)

    def build_from_yaml(self, config: Dict) -> Configuration:
        assert "method" in config, "method is required in the configuration."
        used_methods = config.pop("method", [])
        method_weights = config.pop("method_weights", [1]*len(used_methods))

        cs = ConfigurationSpace(
            name=self.name,  # node_type
            space={
                "module_type": CategoricalHyperparameter("module_type", used_methods,
                                                           weights=method_weights),
            }
        )

        # Add method specific hyperparameters
        supported_methods = get_supported_methods(self.name)
        for method in supported_methods:
            method_params = config.pop(method, {})
            # add hyperparameter configuration space
            if method in used_methods:
                cs.add_configuration_space(
                    prefix="[{}]".format(method),
                    delimiter="",
                    configuration_space=ConfigurationSpace(method_params),
                    parent_hyperparameter={"parent": cs["module_type"], "value": method},
                )

        # Add general hyperparameters
        if len(config) > 0:
            params = list(parse_hyperparameters_from_dict(config))
            cs.add(params)

        return cs

    @classmethod
    def load_from_yaml(cls, path: str | Path | IO[str], key: Optional[str] = None, project_dir: Optional[str] = None, **kwargs: Any,) -> ConfigurationSpace:
        """Decode a serialized configuration space from a yaml file.
        """
        config = load_yaml_config(path, **kwargs)

        if key is not None:
            config = config[key]
            return cls(config, name=key)

        return cls(config)


    def load_static_params(self, module_type: str) -> Dict:
        """
        Get the default parameters for the module.
        These would not be used for hyperparameter optimization.
        """
        static_params = get_static_params(self.name)
        return static_params.get(module_type, {})


    def sampling(self, size: Optional[int] = 1, exhaustive = False) -> Union[Configuration, List[Configuration]]:
        """
        Sample the configuration.

        :param size: The number of configuration instances to generate.
        :param exhaustive: If True, generate all possible configurations.
        """
        hp_samples = None
        if exhaustive and (self.cs.estimate_size() != np.inf):
            try:
                hp_samples = generate_grid(self.cs)
            except ValueError:
                pass
        if hp_samples is None:
            hp_samples = self.cs.sample_configuration(size)
            if not isinstance(hp_samples, List):
                hp_samples = [hp_samples]
        return hp_samples


    @abstractmethod
    def create_node_lines(self,size: Optional[int] = 1,samples: Optional[List[Mapping[str, Any]]] = None,node_line_name: str = "retrieve_node_line",**kwargs) -> Dict:
        """
        Create node lines from the hyperparameters.

        :param size: The number of nodes to create.
        :param samples: The hyperparameters to use.
        :param node_line_name: The name of the node line.
        :param kwargs: Additional keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def create_nodes(self, size: Optional[int] = 1, samples: Optional[List[Mapping[str, Any]]] = None, **kwargs) -> Dict:
        """
        Create nodes from the hyperparameters.

        :param size: The number of nodes to create.
        :param samples: The hyperparameters to use.
        """
        raise NotImplementedError("Subclasses must implement this method")


#
#
# class PostRetrievalConfiguration(BaseConfiguration):
#
#     @abstractmethod
#     def post_build(self, cs: ConfigurationSpace) -> None:
#         """
#         Add generator hyperparameters to the configuration space.
#
#         :param cs: The configuration space to which hyperparameters will be added.
#         """
#         raise NotImplementedError("Subclasses must implement this method")
#
#     @abstractmethod
#     def create_nodes(self, size: Optional[int] = 1, samples: Optional[List[Mapping[str, Any]]] = None, **kwargs) -> Dict:
#         """
#         Create nodes from the hyperparameters.
#
#         :param size: The number of nodes to create.
#         :param samples: The hyperparameters to use.
#         """
#         raise NotImplementedError("Subclasses must implement this method")
#
#     def create_node_lines(self, size: Optional[int] = 1, samples: Optional[List[Mapping[str, Any]]] = None, **kwargs) -> List[Dict]:
#         nodes = self.create_nodes(size, samples, **kwargs)
#         node_lines = [{
#             "node_line_name": "post_retrieve_node_line",
#             "nodes": [node]
#         } for node in nodes]
#         return node_lines
