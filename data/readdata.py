import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

def read_parquet_data(file_path: str) -> pd.DataFrame:
    """
    读取 parquet 文件
    
    Args:
        file_path: parquet 文件路径
        
    Returns:
        pd.DataFrame: 读取的数据
    """
    try:
        df = pd.read_parquet(file_path)
        print(f"成功读取文件: {file_path}")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

def debug_corpus_data(corpus_path: str, qa_path: str = None):
    """
    调试语料库数据，检查 doc_id 匹配问题
    
    Args:
        corpus_path: 语料库文件路径
        qa_path: QA数据文件路径（可选）
    """
    print("=" * 50)
    print("开始调试语料库数据")
    print("=" * 50)
    
    # 读取语料库数据
    corpus_df = read_parquet_data(corpus_path)
    if corpus_df is None:
        return
    
    print("\n语料库数据信息:")
    print(f"总行数: {len(corpus_df)}")
    print(f"列名: {list(corpus_df.columns)}")
    print("\n前5行数据:")
    print(corpus_df.head())
    
    # 检查 doc_id 列
    if 'doc_id' in corpus_df.columns:
        doc_ids = corpus_df['doc_id'].unique()
        print(f"\n唯一 doc_id 数量: {len(doc_ids)}")
        print(f"前10个 doc_id: {doc_ids[:10]}")
        
        # 检查是否有重复的 doc_id
        duplicates = corpus_df[corpus_df.duplicated(subset=['doc_id'], keep=False)]
        if not duplicates.empty:
            print(f"\n发现重复的 doc_id: {len(duplicates)}")
            print(duplicates[['doc_id']].head())
        else:
            print("\n没有重复的 doc_id")
            
        # 检查特定的 doc_id
        target_doc_id = "2WikiMultihopQA_C42_0"
        if target_doc_id in doc_ids:
            print(f"\n找到目标 doc_id: {target_doc_id}")
            target_row = corpus_df[corpus_df['doc_id'] == target_doc_id]
            print(target_row)
        else:
            print(f"\n未找到目标 doc_id: {target_doc_id}")
            # 查找相似的 doc_id
            similar_ids = [doc_id for doc_id in doc_ids if "2WikiMultihopQA" in str(doc_id)]
            print(f"包含 '2WikiMultihopQA' 的 doc_id 数量: {len(similar_ids)}")
            if similar_ids:
                print(f"前10个相似的 doc_id: {similar_ids[:10]}")
    else:
        print("\n警告: 语料库中没有 'doc_id' 列")
        print(f"可用列: {list(corpus_df.columns)}")
    
    # 如果提供了 QA 数据，也进行检查
    if qa_path:
        print("\n" + "=" * 30)
        print("检查 QA 数据")
        print("=" * 30)
        
        qa_df = read_parquet_data(qa_path)
        if qa_df is not None:
            print(f"\nQA 数据形状: {qa_df.shape}")
            print(f"QA 数据列名: {list(qa_df.columns)}")
            print("\nQA 数据前5行:")
            print(qa_df.head())
            
            # 检查 QA 数据中引用的 doc_id
            if 'retrieval_gt' in qa_df.columns:
                print("\n检查 retrieval_gt 中的 doc_id:")
                retrieval_gts = qa_df['retrieval_gt'].dropna()
                all_referenced_doc_ids = set()
                
                for i, gt in enumerate(retrieval_gts.head(10)):
                    try:
                        # 处理不同类型的数据
                        if isinstance(gt, np.ndarray):
                            # 如果是 numpy 数组，转换为列表
                            gt_list = gt.tolist()
                        elif isinstance(gt, str):
                            # 如果是字符串，尝试解析
                            if gt.startswith('['):
                                gt_list = eval(gt)
                            else:
                                gt_list = [gt]
                        elif isinstance(gt, list):
                            # 如果已经是列表
                            gt_list = gt
                        else:
                            # 其他类型，转换为字符串后处理
                            gt_str = str(gt)
                            if gt_str.startswith('['):
                                gt_list = eval(gt_str)
                            else:
                                gt_list = [gt_str]
                        
                        # 确保 gt_list 中的元素都是字符串
                        gt_list = [str(item) for item in gt_list]
                        
                        for doc_id in gt_list:
                            all_referenced_doc_ids.add(doc_id)
                        
                        print(f"第{i+1}行 retrieval_gt (类型: {type(gt).__name__}): {gt_list}")
                        
                    except Exception as e:
                        print(f"第{i+1}行解析失败 (类型: {type(gt).__name__}): {e}")
                        print(f"原始数据: {gt}")
                
                print(f"\nQA 数据中引用的唯一 doc_id 数量: {len(all_referenced_doc_ids)}")
                if all_referenced_doc_ids:
                    print(f"前10个引用的 doc_id: {list(all_referenced_doc_ids)[:10]}")
                
                # 检查缺失的 doc_id
                if 'doc_id' in corpus_df.columns:
                    corpus_doc_ids = set(corpus_df['doc_id'].unique())
                    missing_doc_ids = all_referenced_doc_ids - corpus_doc_ids
                    
                    if missing_doc_ids:
                        print(f"\n缺失的 doc_id 数量: {len(missing_doc_ids)}")
                        print(f"前10个缺失的 doc_id: {list(missing_doc_ids)[:10]}")
                    else:
                        print("\n所有引用的 doc_id 都在语料库中找到了")

def search_doc_id(corpus_path: str, search_term: str):
    """
    在语料库中搜索包含特定字符串的 doc_id
    
    Args:
        corpus_path: 语料库文件路径
        search_term: 搜索词
    """
    corpus_df = read_parquet_data(corpus_path)
    if corpus_df is None or 'doc_id' not in corpus_df.columns:
        return
    
    matching_ids = corpus_df[corpus_df['doc_id'].str.contains(search_term, na=False)]
    print(f"\n包含 '{search_term}' 的 doc_id:")
    print(f"找到 {len(matching_ids)} 个匹配项")
    
    if not matching_ids.empty:
        print(matching_ids[['doc_id']].head(20))

def fix_doc_id_format(corpus_path: str, output_path: str = None):
    """
    尝试修复 doc_id 格式问题
    
    Args:
        corpus_path: 输入语料库文件路径
        output_path: 输出文件路径（可选）
    """
    corpus_df = read_parquet_data(corpus_path)
    if corpus_df is None:
        return
    
    if 'doc_id' not in corpus_df.columns:
        print("语料库中没有 doc_id 列")
        return
    
    print("原始 doc_id 格式示例:")
    print(corpus_df['doc_id'].head(10).tolist())
    
    # 检查并修复可能的格式问题
    original_count = len(corpus_df)
    
    # 移除空值
    corpus_df = corpus_df.dropna(subset=['doc_id'])
    print(f"移除空 doc_id 后: {len(corpus_df)} 行 (原来 {original_count} 行)")
    
    # 转换为字符串并去除空格
    corpus_df['doc_id'] = corpus_df['doc_id'].astype(str).str.strip()
    
    # 移除重复项
    before_dedup = len(corpus_df)
    corpus_df = corpus_df.drop_duplicates(subset=['doc_id'])
    print(f"去重后: {len(corpus_df)} 行 (去重前 {before_dedup} 行)")
    
    print("修复后 doc_id 格式示例:")
    print(corpus_df['doc_id'].head(10).tolist())
    
    if output_path:
        corpus_df.to_parquet(output_path, index=False)
        print(f"修复后的数据已保存到: {output_path}")
    
    return corpus_df

def debug_chroma_collection(db_path: str, collection_name: str):
    """
    调试 Chroma 数据库中的 collection 内容
    
    Args:
        db_path: Chroma 数据库路径
        collection_name: collection 名称
    """
    try:
        import chromadb
        from chromadb.config import Settings
        
        print("=" * 50)
        print(f"开始调试 Chroma Collection: {collection_name}")
        print("=" * 50)
        
        # 连接到 Chroma 数据库
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 列出所有 collections
        collections = client.list_collections()
        print(f"数据库中的所有 collections:")
        for col in collections:
            print(f"  - {col.name} (元数据: {col.metadata})")
        
        # 检查目标 collection 是否存在
        collection_names = [col.name for col in collections]
        if collection_name not in collection_names:
            print(f"\n错误: Collection '{collection_name}' 不存在")
            print(f"可用的 collections: {collection_names}")
            return
        
        # 获取指定的 collection
        collection = client.get_collection(name=collection_name)
        
        # 获取 collection 信息
        count = collection.count()
        print(f"\nCollection '{collection_name}' 信息:")
        print(f"  - 文档数量: {count}")
        print(f"  - 元数据: {collection.metadata}")
        
        if count > 0:
            # 获取前10个文档
            results = collection.get(
                limit=10,
                include=['documents', 'metadatas', 'embeddings']
            )
            
            print(f"\n前10个文档:")
            for i, doc_id in enumerate(results['ids']):
                print(f"\n文档 {i+1}:")
                print(f"  ID: {doc_id}")
                
                # 安全地处理文档内容
                doc = None
                if results['documents'] is not None and i < len(results['documents']):
                    doc = results['documents'][i]
                
                if doc is not None:
                    print(f"  内容: {doc[:200]}..." if len(doc) > 200 else f"  内容: {doc}")
                else:
                    print(f"  内容: None")
                
                # 安全地处理元数据
                metadata = None
                if results['metadatas'] is not None and i < len(results['metadatas']):
                    metadata = results['metadatas'][i]
                print(f"  元数据: {metadata}")
                
                # 安全地处理向量
                embedding = None
                if results['embeddings'] is not None and i < len(results['embeddings']):
                    embedding = results['embeddings'][i]
                
                if embedding is not None:
                    print(f"  向量维度: {len(embedding)}")
                else:
                    print(f"  向量: None")
            
            # 搜索特定的 doc_id
            target_doc_id = "2WikiMultihopQA_C42_0"
            try:
                target_results = collection.get(
                    ids=[target_doc_id],
                    include=['documents', 'metadatas']
                )
                
                if target_results['ids']:
                    print(f"\n找到目标文档 '{target_doc_id}':")
                    doc_content = target_results['documents'][0] if target_results['documents'] else None
                    metadata_content = target_results['metadatas'][0] if target_results['metadatas'] else None
                    print(f"  内容: {doc_content}")
                    print(f"  元数据: {metadata_content}")
                else:
                    print(f"\n未找到目标文档 '{target_doc_id}'")
                    
                    # 搜索包含关键词的文档
                    all_results = collection.get(include=['metadatas'])
                    matching_ids = []
                    for doc_id in all_results['ids']:
                        if "2WikiMultihopQA" in str(doc_id):
                            matching_ids.append(doc_id)
                    
                    if matching_ids:
                        print(f"包含 '2WikiMultihopQA' 的文档 ID 数量: {len(matching_ids)}")
                        print(f"前10个匹配的文档 ID:")
                        for match_id in matching_ids[:10]:
                            print(f"  - {match_id}")
                    else:
                        print("没有找到包含 '2WikiMultihopQA' 的文档")
                    
            except Exception as e:
                print(f"搜索特定文档时出错: {e}")
        
        else:
            print("\nCollection 为空")
            
    except ImportError:
        print("错误: 需要安装 chromadb")
        print("请运行: pip install chromadb")
    except Exception as e:
        print(f"连接 Chroma 数据库时出错: {e}")
        import traceback
        traceback.print_exc()

def compare_corpus_and_chroma(corpus_path: str, db_path: str, collection_name: str):
    """
    比较语料库文件和 Chroma 数据库中的数据
    
    Args:
        corpus_path: 语料库文件路径
        db_path: Chroma 数据库路径
        collection_name: collection 名称
    """
    print("=" * 50)
    print("比较语料库文件和 Chroma 数据库")
    print("=" * 50)
    
    # 读取语料库文件
    corpus_df = read_parquet_data(corpus_path)
    if corpus_df is None:
        return
    
    corpus_doc_ids = set(corpus_df['doc_id'].unique()) if 'doc_id' in corpus_df.columns else set()
    print(f"语料库文件中的文档数量: {len(corpus_doc_ids)}")
    
    # 读取 Chroma 数据库
    try:
        import chromadb
        from chromadb.config import Settings
        
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection = client.get_collection(name=collection_name)
        chroma_results = collection.get(include=['metadatas'])
        chroma_doc_ids = set(chroma_results['ids'])
        
        print(f"Chroma 数据库中的文档数量: {len(chroma_doc_ids)}")
        
        # 比较差异
        only_in_corpus = corpus_doc_ids - chroma_doc_ids
        only_in_chroma = chroma_doc_ids - corpus_doc_ids
        common_docs = corpus_doc_ids & chroma_doc_ids
        
        print(f"\n比较结果:")
        print(f"  - 共同文档: {len(common_docs)}")
        print(f"  - 仅在语料库中: {len(only_in_corpus)}")
        print(f"  - 仅在 Chroma 中: {len(only_in_chroma)}")
        
        if only_in_corpus:
            print(f"\n仅在语料库中的前10个文档:")
            for doc_id in list(only_in_corpus)[:10]:
                print(f"  - {doc_id}")
        
        if only_in_chroma:
            print(f"\n仅在 Chroma 中的前10个文档:")
            for doc_id in list(only_in_chroma)[:10]:
                print(f"  - {doc_id}")
                
    except Exception as e:
        print(f"读取 Chroma 数据库时出错: {e}")

def check_doc_id_in_chroma(db_path: str, collection_name: str, doc_id: str):
    """
    检查 Chroma 数据库中是否存在特定的 doc_id
    
    Args:
        db_path: Chroma 数据库路径
        collection_name: collection 名称
        doc_id: 要检查的文档 ID
    """
    try:
        import chromadb
        from chromadb.config import Settings
        
        print("=" * 50)
        print(f"检查文档 ID: {doc_id}")
        print("=" * 50)
        
        # 连接到 Chroma 数据库
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 检查 collection 是否存在
        collections = client.list_collections()
        collection_names = [col.name for col in collections]
        
        if collection_name not in collection_names:
            print(f"错误: Collection '{collection_name}' 不存在")
            print(f"可用的 collections: {collection_names}")
            return False
        
        # 获取 collection
        collection = client.get_collection(name=collection_name)
        
        try:
            # 尝试获取特定的文档，只包含支持的字段
            results = collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas', 'embeddings']
            )
            
            if results['ids']:
                print(f"✅ 找到文档 '{doc_id}'")
                
                # 显示所有返回的字段
                print(f"\n返回结果的结构:")
                for key, value in results.items():
                    if value is not None:
                        if isinstance(value, list) and len(value) > 0:
                            print(f"  {key}: {type(value[0])} (长度: {len(value)})")
                            if key == 'documents':
                                print(f"    内容: {value[0]}")
                            elif key == 'metadatas':
                                print(f"    元数据: {value[0]}")
                            elif key == 'embeddings' and value[0] is not None:
                                print(f"    向量维度: {len(value[0])}")
                        else:
                            print(f"  {key}: {value}")
                    else:
                        print(f"  {key}: None")
                
                # 尝试不同的查询方式
                print(f"\n尝试其他查询方式:")
                
                # 只查询 documents
                doc_only = collection.get(ids=[doc_id], include=['documents'])
                print(f"仅查询 documents: {doc_only.get('documents', [None])[0] if doc_only.get('documents') else None}")
                
                # 只查询 metadatas
                meta_only = collection.get(ids=[doc_id], include=['metadatas'])
                print(f"仅查询 metadatas: {meta_only.get('metadatas', [None])[0] if meta_only.get('metadatas') else None}")
                
                # 查询所有字段
                all_fields = collection.get(ids=[doc_id])
                print(f"查询所有字段: {all_fields}")
                
                return True
            else:
                print(f"❌ 未找到文档 '{doc_id}'")
                
                # 尝试模糊搜索
                print(f"\n尝试模糊搜索包含 '{doc_id}' 的文档...")
                all_results = collection.get(include=['metadatas'])
                
                # 精确匹配
                exact_matches = [id for id in all_results['ids'] if id == doc_id]
                if exact_matches:
                    print(f"精确匹配: {exact_matches}")
                    return True
                
                # 部分匹配
                partial_matches = [id for id in all_results['ids'] if doc_id in str(id)]
                if partial_matches:
                    print(f"部分匹配 ({len(partial_matches)} 个):")
                    for match in partial_matches[:10]:
                        print(f"  - {match}")
                    return True
                
                # 反向匹配（检查 doc_id 是否包含数据库中的某个 ID）
                reverse_matches = [id for id in all_results['ids'] if str(id) in doc_id]
                if reverse_matches:
                    print(f"反向匹配 ({len(reverse_matches)} 个):")
                    for match in reverse_matches[:10]:
                        print(f"  - {match}")
                
                print("没有找到任何匹配的文档")
                return False
                
        except Exception as e:
            print(f"查询文档时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError:
        print("错误: 需要安装 chromadb")
        print("请运行: pip install chromadb")
        return False
    except Exception as e:
        print(f"连接 Chroma 数据库时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def diagnose_chroma_data_issue(db_path: str, collection_name: str):
    """
    诊断 Chroma 数据库的数据问题
    
    Args:
        db_path: Chroma 数据库路径
        collection_name: collection 名称
    """
    try:
        import chromadb
        from chromadb.config import Settings
        
        print("=" * 50)
        print("诊断 Chroma 数据库数据问题")
        print("=" * 50)
        
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection = client.get_collection(name=collection_name)
        
        # 获取前20个文档进行分析
        results = collection.get(
            limit=20,
            include=['documents', 'metadatas', 'embeddings']
        )
        
        print(f"分析前20个文档:")
        
        none_docs = 0
        none_metas = 0
        none_embeddings = 0
        valid_docs = 0
        
        for i, doc_id in enumerate(results['ids']):
            doc = results['documents'][i] if results['documents'] and i < len(results['documents']) else None
            meta = results['metadatas'][i] if results['metadatas'] and i < len(results['metadatas']) else None
            embedding = results['embeddings'][i] if results['embeddings'] and i < len(results['embeddings']) else None
            
            if doc is None:
                none_docs += 1
            else:
                valid_docs += 1
                
            if meta is None:
                none_metas += 1
                
            if embedding is None:
                none_embeddings += 1
            
            if i < 5:  # 显示前5个的详细信息
                print(f"\n文档 {i+1} (ID: {doc_id}):")
                print(f"  文档内容: {'None' if doc is None else f'{doc[:100]}...' if len(str(doc)) > 100 else doc}")
                print(f"  元数据: {meta}")
                print(f"  向量: {'None' if embedding is None else f'维度 {len(embedding)}'}")
        
        print(f"\n统计结果:")
        print(f"  总文档数: {len(results['ids'])}")
        print(f"  有效文档内容: {valid_docs}")
        print(f"  空文档内容: {none_docs}")
        print(f"  空元数据: {none_metas}")
        print(f"  空向量: {none_embeddings}")
        
        if none_docs > 0:
            print(f"\n⚠️  发现 {none_docs} 个文档内容为空，这可能是数据库构建时的问题")
            print("建议重新构建数据库或检查数据源")
            
    except Exception as e:
        print(f"诊断时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 使用示例
    corpus_path = "./5dataset_100/corpus_relate.parquet"
    qa_path = "./5dataset_100/qa100.parquet"
    db_path = "/home/cz/AUTO_GRADON/experiments/db_resources/db_20250523"
    collection_name = "test_collection_v3"
    
    # 调试数据
    debug_corpus_data(corpus_path, qa_path)
    
    # 调试 Chroma 数据库
    debug_chroma_collection(db_path, collection_name)
    
    # 比较语料库和数据库
    compare_corpus_and_chroma(corpus_path, db_path, collection_name)
    
    # 检查特定文档 ID
    target_doc_id = "2WikiMultihopQA_C36_0"
    check_doc_id_in_chroma(db_path, collection_name, target_doc_id)
