from .base import *
from .zoo import *
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal, Mapping, Sequence, overload

import ConfigSpace
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, InCondition, NotEqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant
from chromadb import (
	DEFAULT_TENANT,
	DEFAULT_DATABASE,
)


class RetrievalConfiguration(BaseConfiguration):

    def create_node_lines(self, size: Optional[int] = 1, samples: Optional[List[Mapping[str, Any]]] = None, **kwargs) -> Dict:
        """
            {
                'node_line_name': 'retrieve_node_line',
                'nodes': [
                    {
                        'strategy': {
                            'metrics': ['retrieval_f1', 'retrieval_recall', 'retrieval_precision']
                        },
                        'node_type': 'retrieval',
                        'modules': [
                            {'module_type': 'bm25', 'top_k': 5},
                            {'module_type': 'vectordb', 'vectordb': 'default_vectordb', 'top_k': 5}
                        ]
                    }
                ]
            }
        """
        if samples is None:
            samples = self.sampling(size, **kwargs)
        node_lines = []
        for hp_config in samples:
            hp_config_dict = parse_hyperparameters_samples(dict(hp_config))
            module_type = hp_config_dict["module_type"]
            static_params = self.load_static_params(module_type)
            hp_config_dict.update(static_params)
            if module_type.startswith("hybrid"):
                top_k = hp_config_dict.get("top_k", 5)
                hp_config_dict.update({
                    "target_module_params": [
                        {
                            "top_k": top_k,
                        },
                        {
                            "top_k": top_k,
                            "vectordb": "default_vectordb"
                        }
                    ],
                })
            node_line = {
                "node_line_name": "retrieve_node_line",
                "nodes": [{
                    "strategy": {
                        "metrics": ["retrieval_f1", "retrieval_recall", "retrieval_precision"]
                    },
                    "node_type": hp_config.config_space.name,
                    "modules": [hp_config_dict],
                }]
            }

            # if module_type.startswith("hybrid"):
                # node_line["nodes"][0]["modules"].extend([
                #     {'module_type': 'bm25', "top_k": top_k},
                #     {'module_type': 'vectordb', "vectordb": "default_vectordb", "top_k": top_k}
                # ])
            node_lines.append(node_line)

        return node_lines

