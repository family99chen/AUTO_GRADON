from .base import *
from .zoo import *

import ConfigSpace
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, NotEqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant




class PromptMakerConfiguration(BaseConfiguration):

    def create_node_lines(self, size: Optional[int] = 1, samples: Optional[List[Mapping[str, Any]]] = None, **kwargs) -> Dict:
        """
        As default, each node only contain one prompt_maker module,
        So the evaluation of Prompt Maker will be skipped in AutoRAG Search.
        In this way, we do not need to set the generator_modules in the Prompt Maker Params.
        {
            'node_type': 'prompt_maker',
            'strategy': {'metrics': ['meteor', 'rouge', 'bert_score']},
            'modules': [
                {
                    'module_type': 'fstring',
                    'prompt': 'Read the passages and answer the given question... '
                }
            ]
        },
        """
        if samples is None:
            samples = self.sampling(size, **kwargs)
        node_lines = []
        for hp_config in samples:
            hp_config_dict = parse_hyperparameters_samples(dict(hp_config))
            module_type = hp_config_dict["module_type"]
            static_params = self.load_static_params(module_type)
            hp_config_dict.update(static_params)

            node_lines.append({
                "node_line_name": "prompt_maker_node_line",
                "nodes": [
                    {
                        "node_type": hp_config.config_space.name,
                        'strategy': {'metrics': ['meteor', 'rouge', 'bert_score']},
                        "modules": [hp_config_dict],
                    }
                ]
            })

        return node_lines





