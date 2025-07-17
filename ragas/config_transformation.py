import yaml
from pathlib import Path
from typing import Dict, Any

def add_missing_strategies(config_path: str, output_path: str = None) -> str:
    """
    为配置文件添加缺失的 strategy 字段
    
    Args:
        config_path: 输入配置文件路径
        output_path: 输出配置文件路径，如果为None则自动生成
    
    Returns:
        输出文件路径
    """
    # 如果没有指定输出路径，自动生成
    if output_path is None:
        config_file = Path(config_path)
        output_path = str(config_file.parent / f"{config_file.stem}_fixed{config_file.suffix}")
    
    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 定义每种节点类型对应的策略
    node_strategies = {
        'query_expansion': {
            'metrics': ['retrieval_precision', 'retrieval_recall', 'retrieval_f1']
        },
        'retrieval': {
            'metrics': ['retrieval_f1', 'retrieval_recall', 'retrieval_precision']
        },
        'passage_augmenter': {
            'metrics': ['retrieval_f1', 'retrieval_recall', 'retrieval_precision']
        },
        'passage_reranker': {
            'metrics': ['retrieval_f1', 'retrieval_recall', 'retrieval_precision']
        },
        'passage_filter': {
            'metrics': ['retrieval_f1', 'retrieval_recall', 'retrieval_precision']
        },
        'passage_compressor': {
            'metrics': ['retrieval_token_recall', 'retrieval_token_precision', 'retrieval_token_f1']
        },
        'prompt_maker': {
            'metrics': ['meteor', 'rouge', 'bert_score']
        },
        'generator': {
            'metrics': ['meteor', 'rouge', 'bert_score']
        }
    }
    
    # 处理 node_lines
    if 'node_lines' in config:
        for node_line in config['node_lines']:
            if 'nodes' in node_line:
                for node in node_line['nodes']:
                    node_type = node.get('node_type')
                    
                    # 如果节点没有 strategy 字段，添加对应的策略
                    if node_type and 'strategy' not in node:
                        if node_type in node_strategies:
                            node['strategy'] = node_strategies[node_type].copy()
                            print(f"为 {node_type} 节点添加了 strategy 字段")
                        else:
                            print(f"警告: 未知的节点类型 {node_type}")
    
    # 确保有全局 strategies 字段
    if 'strategies' not in config:
        config['strategies'] = {
            'metrics': ['meteor', 'rouge', 'bert_score']
        }
        print("添加了全局 strategies 字段")
    
    # 确保有 bm25_tokenizer_list 字段
    if 'bm25_tokenizer_list' not in config:
        config['bm25_tokenizer_list'] = ['porter_stemmer', 'space']
        print("添加了 bm25_tokenizer_list 字段")
    
    # 写入修复后的配置文件
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"配置文件已修复并保存到: {output_path}")
    return output_path

def validate_config(config_path: str) -> bool:
    """
    验证配置文件是否包含所有必要字段
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        是否有效
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查必要的顶级字段
        required_top_fields = ['vectordb', 'node_lines']
        for field in required_top_fields:
            if field not in config:
                print(f"错误: 缺少顶级字段 '{field}'")
                return False
        
        # 检查 node_lines 中的 strategy 字段
        missing_strategies = []
        if 'node_lines' in config:
            for i, node_line in enumerate(config['node_lines']):
                if 'nodes' in node_line:
                    for j, node in enumerate(node_line['nodes']):
                        node_type = node.get('node_type')
                        if node_type and 'strategy' not in node:
                            missing_strategies.append(f"node_lines[{i}].nodes[{j}] ({node_type})")
        
        if missing_strategies:
            print(f"错误: 以下节点缺少 strategy 字段:")
            for missing in missing_strategies:
                print(f"  - {missing}")
            return False
        
        print("配置文件验证通过!")
        return True
        
    except Exception as e:
        print(f"验证配置文件时出错: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    
    # 支持命令行参数或默认路径
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # 默认修复 best_config.yaml
        config_path = "../experiments/28-bayesian_comparison/bayesian_strategy_opt_early_stop/best_config.yaml"
    
    print(f"开始修复配置文件: {config_path}")
    
    # 检查文件是否存在
    if not Path(config_path).exists():
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    # 先验证原文件
    print("\n=== 验证原配置文件 ===")
    is_valid_before = validate_config(config_path)
    
    # 修复配置文件
    print("\n=== 修复配置文件 ===")
    try:
        fixed_config_path = add_missing_strategies(config_path)
        
        # 验证修复后的文件
        print("\n=== 验证修复后的配置文件 ===")
        is_valid_after = validate_config(fixed_config_path)
        
        if is_valid_after:
            print(f"\n✅ 配置文件修复成功!")
            print(f"原文件: {config_path}")
            print(f"修复后文件: {fixed_config_path}")
        else:
            print(f"\n❌ 配置文件修复后仍有问题")
            
    except Exception as e:
        print(f"\n❌ 修复过程中出错: {str(e)}")
        sys.exit(1)
