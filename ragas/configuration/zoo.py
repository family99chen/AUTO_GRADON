import time
from typing import Dict, List, Union
from enum import Enum
from dataclasses import dataclass
from chromadb import DEFAULT_TENANT, DEFAULT_DATABASE


class MethodType(Enum):
    QUERY_EXPANSION = "query_expansion"
    RETRIEVAL = "retrieval"
    PASSAGE_AUGMENTER = "passage_augmenter"
    PROMPT_MAKER = "prompt_maker"
    GENERATOR = "generator"
    VECTORDB = "vectordb"
    PASSAGE_RERANKER = "passage_reranker"
    PASSAGE_FILTER = "passage_filter"
    PASSAGE_COMPRESSOR = "passage_compressor"


class SupportedMethods:
    VECTOR_DBS = ["chroma", "couchbase", "milvus", "pinecone", "qdrant", "weaviate"]
    QUERY_EXPANSIONS = ["pass_query_expansion", "QueryDecompose", "HyDE", "multi_query_expansion"]
    RETRIEVERS = ["bm25", "vectordb", "hybrid_cc", "hybrid_rrf"]
    PROMPT_MAKERS = ['fstring', 'window_replacement', 'long_context_reorder']
    GENERATORS = ["llama_index_llm", "vllm", "openai_llm", "vllm_api",
                 "LlamaIndexLLM", "OpenAILLM", "Vllm", "VllmAPI"]
    PASSAGE_AUGMENTERS = ["pass_passage_augmenter", "prev_next_augmenter"]
    RERANKERS = [
        "pass_passage_filter", "upr", "tart", "monot5", "rankgpt",
        "colbert_reranker", "sentence_transformer_reranker",
        "flag_embedding_reranker", "flag_embedding_llm_reranker",
        "time_reranker", "openvino_reranker", "flashrank_reranker"
    ]
    PASSAGE_FILTERS = [
        "pass_passage_filter", "similarity_threshold_cutoff",
        "similarity_percentile_cutoff", "recency_filter",
        "threshold_cutoff", "percentile_cutoff"
    ]
    PASSAGE_COMPRESSORS = ["pass_compressor", "tree_summarize", "refine", "longllmlingua"]


@dataclass
class APIConfig:
    # # "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    # # "api_key": "sk-ab6eb49be7934c4f86678574618c646a", # OpenAI API key. You can also set this to env variable OPENAI_API_KEY.
    # BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    BASE_URL: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    API_KEY: str = "sk-e46fb251c74d4c64a4c2835333e994a3"
    TIME_OUT: float = 1200.0

class StaticParams:
    api_config = APIConfig()

    @classmethod
    def get_query_expansion_params(cls) -> Dict:
        return {
            method: {
                "api_base": cls.api_config.BASE_URL,
                "api_key": cls.api_config.API_KEY,
                "timeout": cls.api_config.TIME_OUT,
            } for method in SupportedMethods.QUERY_EXPANSIONS
        }

    @classmethod
    def get_retrieval_params(cls) -> Dict:
        return {
            "vectordb": {"vectordb": "default_vectordb"},
            "hybrid_cc": {"target_modules": "('bm25', 'vectordb')"},
            "hybrid_rrf": {"target_modules": "('bm25', 'vectordb')"},
        }

    @classmethod
    def get_reranker_params(cls) -> Dict:
        return {
            "rankgpt": {
                "api_base": cls.api_config.BASE_URL,
                "api_key": cls.api_config.API_KEY,
                "timeout": cls.api_config.TIME_OUT,
            },
            "upr": {
                "use_bf16": False,
                "prefix_prompt": "Passage: ",
                "suffix_prompt": "Please write a question based on this passage.",
            },
            "tart": {
                "instruction": "Find passage to answer given question",
            },
            "colbert_reranker": {
                "model_name": "colbert-ir/colbertv2.0",
            },
            "sentence_transformer_reranker": {
                "model_name": "cross-encoder/ms-marco-MiniLM-L-2-v2",
            },
            "flag_embedding_reranker": {
                "model_name": "BAAI/bge-reranker-large",
            },
            "flag_embedding_llm_reranker": {
                "use_fp16": False,
                "model_name": "BAAI/bge-reranker-v2-gemma",
            },
            "openvino_reranker": {
                "model_name": "BAAI/bge-reranker-large",
            },
            "flashrank_reranker": {
                "model_name": "ms-marco-MiniLM-L-12-v2",
            },
            "voyageai_reranker": {
                "api_key": "your_voyageai_api_key",
                "model": "rerank-2",
                "truncation": True
            },
            "mixedbreadai_reranker": {
                "api_key": "your_mixedbread_api_key"
            },
            "jina_reranker": {
                "api_key": "your_jina_api_key",
                "model": "jina-reranker-v1-base-en",
            },
            "cohere_reranker": {
                "api_key": "your_cohere_api_key",
                "batch": 64,
                "model": "rerank-multilingual-v2.0",
            },
        }

    @classmethod
    def get_passage_compressor_params(cls) -> Dict:
        return {
            method: {
                "api_base": cls.api_config.BASE_URL,
                "api_key": cls.api_config.API_KEY,
                "timeout": cls.api_config.TIME_OUT,
            } for method in ["tree_summarize", "refine"]
        }

    @classmethod
    def get_generator_params(cls) -> Dict:
        return {
            "LlamaIndexLLM": {
                "api_base": cls.api_config.BASE_URL,
                "api_key": cls.api_config.API_KEY,
                "timeout": cls.api_config.TIME_OUT,
            },
            "OpenAILLM": {
                "truncate": True,
                "base_url": cls.api_config.BASE_URL,
                "api_key": cls.api_config.API_KEY,
                "request_timeout": cls.api_config.TIME_OUT,
            },
            "Vllm": {},
            "VllmAPI": {
                "uri": "http://localhost:8012",
            }
        }

    @classmethod
    def get_vectordb_params(cls) -> Dict[str, Dict]:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        return {
            "chroma": {
                "client_type": "persistent",
                "host": "localhost",
                "port": 8000,
                "ssl": False,
                "headers": None,
                "api_key": None,
                "tenant": DEFAULT_TENANT,
                "database": DEFAULT_DATABASE,
                "path": f"../experiments/db_resources/db_20250501",
            },
            "couchbase": {
                "index_name": "my_vector_index",
                "host": "localhost",
                "port": 8091,
                "username": "Administrator",
                "password": "password",
                "bucket_name": "default",
                "scope_name": "_default",
                "connection_string": "couchbase://localhost",
                "text_key": "text",
                "embedding_key": "embedding",
                "scoped_index": True
            },
            "milvus": {
                "index_type": "IVF_FLAT",
                "uri": "http://localhost:19530",
                "db_name": "",
                "token": "",
                "user": "",
                "password": "",
                "timeout": 30.0,
            },
            "pinecone": {
                "index_name": "my_vector_index",
                "api_key": "your_api_key",
                "dimension": 1536,
                "cloud": "aws",
                "region": "us-east-1",
                "deletion_protection": "disabled",
                "namespace": "default",
            },
            "qdrant": {
                "client_type": "docker",
                "url": "http://localhost:6333",
                "host": "",
                "api_key": "",
                "dimension": 1536,
                "parallel": 1,
                "max_retries": 3,
            },
            "weaviate": {
                "client_type": "docker",
                "host": "localhost",
                "port": 8080,
                "grpc_port": 50051,
                "url": None,
                "api_key": None,
                "text_key": "content",
            }
        }

def get_supported_methods(method_type: Union[str, MethodType]) -> List[str]:
    if isinstance(method_type, str):
        method_type = MethodType(method_type.lower())

    method_map = {
        MethodType.QUERY_EXPANSION: SupportedMethods.QUERY_EXPANSIONS,
        MethodType.RETRIEVAL: SupportedMethods.RETRIEVERS,
        MethodType.PASSAGE_AUGMENTER: SupportedMethods.PASSAGE_AUGMENTERS,
        MethodType.PROMPT_MAKER: SupportedMethods.PROMPT_MAKERS,
        MethodType.GENERATOR: SupportedMethods.GENERATORS,
        MethodType.VECTORDB: SupportedMethods.VECTOR_DBS,
        MethodType.PASSAGE_RERANKER: SupportedMethods.RERANKERS,
        MethodType.PASSAGE_FILTER: SupportedMethods.PASSAGE_FILTERS,
        MethodType.PASSAGE_COMPRESSOR: SupportedMethods.PASSAGE_COMPRESSORS,
    }

    if method_type not in method_map:
        raise ValueError(f"Unknown method type: {method_type}")

    return method_map[method_type]

def get_static_params(method_type: Union[str, MethodType]) -> Dict:
    if isinstance(method_type, str):
        method_type = MethodType(method_type.lower())

    param_map = {
        MethodType.QUERY_EXPANSION: StaticParams.get_query_expansion_params(),
        MethodType.RETRIEVAL: StaticParams.get_retrieval_params(),
        MethodType.PASSAGE_AUGMENTER: {},
        MethodType.PROMPT_MAKER: {},
        MethodType.GENERATOR: StaticParams.get_generator_params(),
        MethodType.VECTORDB: StaticParams.get_vectordb_params(),
        MethodType.PASSAGE_RERANKER: StaticParams.get_reranker_params(),
        MethodType.PASSAGE_FILTER: {},
        MethodType.PASSAGE_COMPRESSOR: StaticParams.get_passage_compressor_params(),
    }

    if method_type not in param_map:
        raise ValueError(f"Unknown method type: {method_type}")

    return param_map[method_type]
