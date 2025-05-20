from .base import *
from .zoo import *

import ConfigSpace
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, NotEqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant


class PassageCompressorConfiguration(BaseConfiguration):

    def create_node_lines(self, size: Optional[int] = 1, samples: Optional[List[Mapping[str, Any]]] = None, **kwargs) -> Dict:
        if samples is None:
            samples = self.sampling(size, **kwargs)
        node_lines = []
        for hp_config in samples:
            hp_config_dict = parse_hyperparameters_samples(dict(hp_config))
            module_type = hp_config_dict["module_type"]
            static_params = self.load_static_params(module_type)
            hp_config_dict.update(static_params)

            node_line = {
                "node_line_name": "passage_compressor_node_line",
                "nodes": [{
                    "node_type": hp_config.config_space.name,
                    "modules": [hp_config_dict],
                }]
            }

            node_lines.append(node_line)

        return node_lines


