from configuration import *
from pathlib import Path
import yaml
import argparse
import os
from itertools import product
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', '-s', type=int, default=5, help="Number of sampled configurations")
    parser.add_argument('--config_dir', '-d', type=str, default="../experiments/candidate_config", help="Output dir for storing config files")
    parser.add_argument('--exhaustive', '-e', action='store_true', help="Use exhaustive search")
    parser.add_argument('--clean', '-c', action='store_true', help="Clean config dir before running")
    return parser.parse_args()

def create_config(vectordb_node_line, *node_lines, extra_params={}, save_path="./ollama_config.yaml"):
    data = {
        'vectordb': [vectordb_node_line],
        'node_lines': list(node_lines),
        **extra_params
    }
    with open(save_path, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=False, sort_keys=False)
    return data

def load_and_generate_nodes(config_path, key, size, exhaustive):
    config_cls = {
        "vectordb": VectorDBConfiguration,
        "query_expansion": QueryExpansionConfiguration,
        "passage_augmenter": PassageAugConfiguration,
        "retrieval": RetrievalConfiguration,
        "prompt_maker": PromptMakerConfiguration,
        "generator": GeneratorConfiguration,
        "passage_reranker": RerankerConfiguration,
        "passage_filter": PassageFilterConfiguration,
        "passage_compressor": PassageCompressorConfiguration
    }
    cfg = config_cls[key].load_from_yaml(config_path, key=key)
    return cfg.create_node_lines(size=size, exhaustive=exhaustive), cfg

def main():
    args = parse_args()
    config_dir = Path(args.config_dir)
    if args.clean:
        import shutil
        shutil.rmtree(config_dir, ignore_errors=True)
    os.makedirs(config_dir, exist_ok=True)

    config_path = "./configuration/config.yaml"
    keys = [
        "vectordb",
        "query_expansion",
        "retrieval",
        "passage_augmenter",
        "passage_reranker",
        "passage_filter",
        "passage_compressor",
        "prompt_maker",
        "generator",
    ]

    node_lines_list = []
    retriever_cfg = None

    for key in keys:
        lines, cfg = load_and_generate_nodes(config_path, key, args.size, args.exhaustive)
        if key == "retrieval":
            retriever_cfg = cfg
        node_lines_list.append(lines)

    vectordb_node_lines = node_lines_list[0]
    other_node_lines = node_lines_list[1:]

    extra_params = {
        'bm25_tokenizer_list': retriever_cfg.cs.get('[bm25]bm25_tokenizer').choices
            if '[bm25]bm25_tokenizer' in retriever_cfg.cs else ['porter_stemmer', 'space'],
        'strategies': {'metrics': ['meteor', 'rouge', 'bert_score']}
    }

    if args.exhaustive:
        all_combinations = list(product(*([vectordb_node_lines] + other_node_lines)))
        for i, combo in tqdm(enumerate(all_combinations), total=len(all_combinations), desc="Generating configs"):
            create_config(combo[0], *combo[1:], extra_params=extra_params, save_path=config_dir / f"ollama_config_{i}.yaml")
        print(f"[Exhaustive] Total configurations: {i + 1}")
    else:
        for i, combo in tqdm(enumerate(zip(*([vectordb_node_lines] + other_node_lines)))):
            create_config(combo[0], *combo[1:], extra_params=extra_params, save_path=config_dir / f"ollama_config_{i}.yaml")
        print(f"[Simple] Total configurations: {i + 1}")

if __name__ == '__main__':
    main()

# template = {
#     'vectordb': [
#         # Insert Here
#         # {
#         #     'name': 'chroma_mpnet',
#         #     'db_type': 'chroma',
#         #     'client_type': 'persistent',
#         #     'collection_name': 'huggingface_all_mpnet_base_v2',
#         #     'embedding_model': 'huggingface_all_mpnet_base_v2',
#         #     'path': '/home/lyb/RAG/experiments/chroma_mpnet'
#         # }
#     ],
#     'node_lines': [
#         # {
#         #     'node_line_name': 'retrieve_node_line',
#         #     'nodes': [
#         #         {
#         #             'node_type': 'retrieval',
#         #             'top_k': 3,
#         #             'modules': [{'module_type': 'bm25'}]
#         #         }
#         #     ]
#         # },
#         # {
#         #     'node_line_name': 'post_retrieve_node_line',
#         #     'nodes': [
#         #         {
#         #             'node_type': 'prompt_maker',
#         #             'strategy': {'metrics': ['meteor', 'rouge', 'bert_score']},
#         #             'modules': [
#         #                 {
#         #                     'module_type': 'fstring',
#         #                     'prompt': 'Read the passages and answer the given question. \n Question: {query} \n Passage: {retrieved_contents} \n Answer : '
#         #                 }
#         #             ]
#         #         },
#         #
#         #         {
#         #             'node_type': 'generator',
#         #             'strategy': {'metrics': ['meteor', 'rouge', 'bert_score']},
#         #             'modules': [
#         #                 {
#         #                     'module_type': 'llama_index_llm',
#         #                     'llm': 'ollama',
#         #                     'model': 'llama3',
#         #                     'temperature': 0.5,
#         #                     'batch': 1
#         #                 }
#         #             ]
#         #         }
#         #     ]
#         # }
#     ]
# }
