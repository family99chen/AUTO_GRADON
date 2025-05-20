from .base import *
from .zoo import *
import ConfigSpace
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, \
    NotEqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, \
    UniformFloatHyperparameter, Constant


class GeneratorConfiguration(BaseConfiguration):

    def build(self, config) -> Dict | None:
        cs = self.build_from_yaml(config)
        # add default conditions
        # TODO: Add conditions to check available models
        # cs.add([
        #     # USE cs["collection_name"] only if vectordb_name != "pinecone"
        #     NotEqualsCondition(cs["collection_name"], cs["vectordb_name"], "pinecone"),
        # ])

        return cs


    def create_node_lines(self, size: Optional[int] = 1, samples: Optional[List[Mapping[str, Any]]] = None, **kwargs) -> Dict:
        """
            {
                'node_type': 'generator',
                'strategy': {'metrics': ['meteor', 'rouge', 'bert_score']},
                'modules': [
                    {
                        'module_type': 'llama_index_llm',
                        'llm': 'ollama',
                        'model': 'llama3',
                        'temperature': 0.5,
                        'batch': 1
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

            node_lines.append({
                "node_line_name": "generator_node_line",
                "nodes": [
                    {
                        "node_type": hp_config.config_space.name,
                        'strategy': {'metrics': ['meteor', 'rouge', 'bert_score']},
                        "modules": [hp_config_dict],
                    }
                ]
            })

            # nodes.append({
            #     "node_type": hp_config.config_space.name,
            #     'strategy': {'metrics': ['meteor', 'rouge', 'bert_score']},
            #     "modules": [hp_config_dict],
            # })

        return node_lines


