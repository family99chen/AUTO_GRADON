from .base import *
from .zoo import *
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal, Mapping, Sequence, overload

import ConfigSpace
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause, InCondition, NotEqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant


class VectorDBConfiguration(BaseConfiguration):

    def build(self, config: Dict) -> Configuration:
        assert "method" in config, "method is required in the configuration."
        used_methods = config.pop("method", [])
        method_weights = config.pop("method_weights", [1]*len(used_methods))

        cs = ConfigurationSpace(
            name=self.name,
            space={
                "db_type": CategoricalHyperparameter("db_type", used_methods,
                                                           weights=method_weights),
            }
        )

        # Add method specific hyperparameters
        for method in get_supported_methods(self.name):
            method_params = config.pop(method, {})
            # add hyperparameter configuration space
            if method in used_methods:
                cs.add_configuration_space(
                    prefix="[{}]".format(method),
                    delimiter="",
                    configuration_space=ConfigurationSpace(method_params),
                    parent_hyperparameter={"parent": cs["db_type"], "value": method},
                )

        # Add general hyperparameters
        if len(config) > 0:
            params = list(parse_hyperparameters_from_dict(config))
            cs.add(params)

        # # Add general hyperparameters
        # embedding_model = CategoricalHyperparameter("embedding_model", choices=["openai", "bert", "gpt-3"])
        # embedding_batch = UniformIntegerHyperparameter("embedding_batch", lower=50, upper=200, default_value=100)
        # similarity_metric = CategoricalHyperparameter("similarity_metric", choices=["cosine", "euclidean", "ip"],
        #                                               default_value="cosine")
        # collection_name = CategoricalHyperparameter("collection_name",
        #                                             choices=["collection1", "collection2", "collection"])
        # ingest_batch = UniformIntegerHyperparameter("ingest_batch", lower=50, upper=200, default_value=100)
        # cs.add([embedding_model, embedding_batch, similarity_metric, collection_name, ingest_batch])

        # add conditions
        conditions = []
        if "pinecone" in cs["db_type"].choices:
            # USE cs["collection_name"] only if vectordb_name != "pinecone"
            conditions.append(NotEqualsCondition(cs["collection_name"], cs["db_type"], "pinecone"))
        if "couchbase" in cs["db_type"].choices:
            # cs["similarity_metric"] == "ip" if vectordb_name == "couchbase"
            conditions.append(ForbiddenAndConjunction(
                ForbiddenEqualsClause(cs["db_type"], "couchbase"),
                ForbiddenInClause(cs['similarity_metric'], ["cosine", "euclidean"])
            ))
        # USE cs["ingest_batch"]  if vectordb_name in ["couchbase", "milvus", "pinecone"]
        # if any(db in cs["db_type"].choices for db in ["couchbase", "milvus", "pinecone"]):
        #     conditions.append(InCondition(cs["ingest_batch"], cs["db_type"], ["couchbase", "milvus", "pinecone"]))
        cs.add(conditions)

        return cs

    def create_node_lines(self, size: Optional[int] = 1, samples: Optional[List[Mapping[str, Any]]] = None, **kwargs) -> Dict:
        """
        {
            'name': 'default_vectordb',
            'db_type': 'chroma',
            'client_type': 'persistent',
            'collection_name': 'huggingface_all_mpnet_base_v2',
            'embedding_model': 'huggingface_all_mpnet_base_v2',
        }
        """
        if samples is None:
            samples = self.sampling(size, **kwargs)
        node_lines = []
        for hp_config in samples:
            hp_config_dict = parse_hyperparameters_samples(dict(hp_config))
            module_type = hp_config_dict["db_type"]
            static_params = self.load_static_params(module_type)
            hp_config_dict.update(static_params)
            node_lines.append({
                "name": "default_vectordb",
                **hp_config_dict
            })
        return node_lines

