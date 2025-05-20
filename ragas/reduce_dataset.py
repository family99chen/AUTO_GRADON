# import pandas as pd
# import os

# # 数据目录路径
# data_dir = "/home/xwh/AutoRAG/data"  # 请确认这是正确的路径

# # 读取原始语料库数据
# corpus_path = os.path.join(data_dir, "corpus.parquet")
# corpus_data = pd.read_parquet(corpus_path)

# # 读取原始问答数据
# qa_path = os.path.join(data_dir, "qa.parquet")
# qa_data = pd.read_parquet(qa_path)

# print(f"原始语料库大小: {len(corpus_data)}条记录")
# print(f"原始问答数据大小: {len(qa_data)}条记录")

# # 创建原始数据备份
# corpus_backup_path = os.path.join(data_dir, "corpus_full.parquet")
# qa_backup_path = os.path.join(data_dir, "qa_full.parquet")

# # 保存原始数据备份
# corpus_data.to_parquet(corpus_backup_path)
# qa_data.to_parquet(qa_backup_path)
# print(f"原始数据已备份为: {corpus_backup_path} 和 {qa_backup_path}")

# # 只保留前100条语料库数据
# small_corpus = corpus_data.head(100)

# # 获取保留语料的ID
# corpus_ids = set(small_corpus['doc_id'])
# print(f"保留了{len(corpus_ids)}个语料库ID")

# # 根据retrieval_gt字段过滤问答数据
# filtered_qa = []
# for _, row in qa_data.iterrows():
#     # 检查问答对是否关联到我们保留的语料库ID
#     if 'retrieval_gt' in row:
#         retrieval_gt = row['retrieval_gt']
#         # 处理retrieval_gt可能是二维列表的情况
#         flat_refs = []
#         if isinstance(retrieval_gt, list):
#             if retrieval_gt and isinstance(retrieval_gt[0], list):
#                 # 如果是二维列表，扁平化它
#                 flat_refs = [doc_id for sublist in retrieval_gt for doc_id in sublist]
#             else:
#                 # 如果是一维列表，直接使用
#                 flat_refs = retrieval_gt
            
#             # 检查是否有引用在我们保留的语料中
#             if any(doc_id in corpus_ids for doc_id in flat_refs):
#                 filtered_qa.append(row)

# small_qa = pd.DataFrame(filtered_qa) if filtered_qa else pd.DataFrame()
# print(f"通过检索关联找到的问答数据: {len(small_qa)}条")

# # 如果过滤后的问答对太少，直接取前面的几条
# if len(small_qa) < 10:  # 假设我们希望至少有10个问答对
#     print("过滤后的问答数据太少，直接使用前20条问答数据")
#     small_qa = qa_data.head(20)
    
#     # 收集这些问答对引用的所有语料ID
#     referenced_ids = set()
#     for _, row in small_qa.iterrows():
#         if 'retrieval_gt' in row:
#             retrieval_gt = row['retrieval_gt']
#             if isinstance(retrieval_gt, list):
#                 if retrieval_gt and isinstance(retrieval_gt[0], list):
#                     # 如果是二维列表，扁平化它
#                     flat_refs = [doc_id for sublist in retrieval_gt for doc_id in sublist]
#                 else:
#                     # 如果是一维列表，直接使用
#                     flat_refs = retrieval_gt
#                 referenced_ids.update(flat_refs)
    
#     print(f"这些问答数据引用了{len(referenced_ids)}个语料ID")
    
#     # 确保语料库包含所有这些ID的文档
#     important_docs = corpus_data[corpus_data['doc_id'].isin(referenced_ids)]
#     print(f"找到了{len(important_docs)}条重要的语料库记录")
    
#     # 合并重要文档和前100条语料
#     small_corpus = pd.concat([small_corpus, important_docs]).drop_duplicates(subset=['doc_id'])
#     print(f"合并后的语料库大小: {len(small_corpus)}条记录")

# # 保存缩小后的数据集
# small_corpus_path = os.path.join(data_dir, "corpus_small.parquet")
# small_qa_path = os.path.join(data_dir, "qa_small.parquet")

# small_corpus.to_parquet(small_corpus_path)
# small_qa.to_parquet(small_qa_path)

# print(f"\n=== 数据集缩减结果 ===")
# print(f"原始语料库大小: {len(corpus_data)}条记录")
# print(f"缩小后语料库大小: {len(small_corpus)}条记录")
# print(f"原始问答数据大小: {len(qa_data)}条记录")
# print(f"缩小后问答数据大小: {len(small_qa)}条记录")
# print(f"缩小后的数据已保存为: {small_corpus_path} 和 {small_qa_path}")

import pandas as pd
import os
import numpy as np

# 数据目录路径
data_dir = "/home/xwh/AutoRAG/data"

# 读取原始语料库数据
corpus_path = os.path.join(data_dir, "corpus.parquet")
corpus_data = pd.read_parquet(corpus_path)

# 读取原始问答数据
qa_path = os.path.join(data_dir, "qa.parquet")
qa_data = pd.read_parquet(qa_path)

print(f"原始语料库大小: {len(corpus_data)}条记录")
print(f"原始问答数据大小: {len(qa_data)}条记录")

# 创建原始数据备份
corpus_backup_path = os.path.join(data_dir, "corpus_full.parquet")
qa_backup_path = os.path.join(data_dir, "qa_full.parquet")
corpus_data.to_parquet(corpus_backup_path)
qa_data.to_parquet(qa_backup_path)
print(f"原始数据已备份为: {corpus_backup_path} 和 {qa_backup_path}")

# 只保留前100条语料库数据
small_corpus = corpus_data.head(100)
corpus_ids = set(small_corpus['doc_id'])
print(f"保留了{len(corpus_ids)}个语料库ID")

# 专门处理这种嵌套NumPy数组的retrieval_gt
def extract_doc_ids(retrieval_gt):
    """从嵌套的NumPy数组中提取文档ID"""
    doc_ids = []
    
    # 如果是NumPy数组
    if isinstance(retrieval_gt, np.ndarray):
        # 遍历数组的每个元素
        for item in retrieval_gt:
            # 递归处理嵌套数组
            if isinstance(item, np.ndarray) or isinstance(item, list):
                doc_ids.extend(extract_doc_ids(item))
            else:
                doc_ids.append(item)
    # 如果是列表
    elif isinstance(retrieval_gt, list):
        for item in retrieval_gt:
            if isinstance(item, np.ndarray) or isinstance(item, list):
                doc_ids.extend(extract_doc_ids(item))
            else:
                doc_ids.append(item)
    # 如果是单个值
    else:
        doc_ids.append(retrieval_gt)
        
    return doc_ids

# 根据retrieval_gt字段过滤问答数据
filtered_qa = []
for idx, row in qa_data.iterrows():
    try:
        if 'retrieval_gt' in row:
            retrieval_gt = row['retrieval_gt']
            # 提取所有文档ID
            doc_ids = extract_doc_ids(retrieval_gt)
            # 检查是否有引用在我们保留的语料中
            if any(str(doc_id) in corpus_ids or doc_id in corpus_ids for doc_id in doc_ids):
                filtered_qa.append(row)
    except Exception as e:
        print(f"处理行 {idx} 时出错: {e}")

small_qa = pd.DataFrame(filtered_qa) if filtered_qa else pd.DataFrame()
print(f"通过检索关联找到的问答数据: {len(small_qa)}条")

# 如果过滤后的问答对太少，直接取前面的几条
if len(small_qa) < 10:
    print("过滤后的问答数据太少，直接使用前20条问答数据")
    small_qa = qa_data.head(20)
    
    # 收集这些问答对引用的所有语料ID
    referenced_ids = set()
    for _, row in small_qa.iterrows():
        if 'retrieval_gt' in row:
            try:
                retrieval_gt = row['retrieval_gt']
                doc_ids = extract_doc_ids(retrieval_gt)
                # 支持字符串和对象ID的比较
                referenced_ids.update(str(id) for id in doc_ids)
                referenced_ids.update(doc_ids)
            except Exception as e:
                print(f"收集引用ID时出错: {e}")
    
    print(f"这些问答数据引用了{len(referenced_ids)}个语料ID")
    
    # 确保语料库包含所有这些ID的文档
    # 使用字符串比较确保类型匹配
    corpus_data_with_str_id = corpus_data.copy()
    corpus_data_with_str_id['doc_id_str'] = corpus_data['doc_id'].astype(str)
    
    # 分别检查原始ID和字符串ID
    mask1 = corpus_data['doc_id'].isin([id for id in referenced_ids if not isinstance(id, str)])
    mask2 = corpus_data_with_str_id['doc_id_str'].isin([id for id in referenced_ids if isinstance(id, str)])
    
    important_docs = corpus_data[mask1 | mask2]
    print(f"找到了{len(important_docs)}条重要的语料库记录")
    
    # 合并重要文档和前100条语料
    small_corpus = pd.concat([small_corpus, important_docs]).drop_duplicates(subset=['doc_id'])
    print(f"合并后的语料库大小: {len(small_corpus)}条记录")

# 保存缩小后的数据集
small_corpus_path = os.path.join(data_dir, "corpus_small.parquet")
small_qa_path = os.path.join(data_dir, "qa_small.parquet")

small_corpus.to_parquet(small_corpus_path)
small_qa.to_parquet(small_qa_path)

print(f"\n=== 数据集缩减结果 ===")
print(f"原始语料库大小: {len(corpus_data)}条记录")
print(f"缩小后语料库大小: {len(small_corpus)}条记录")
print(f"原始问答数据大小: {len(qa_data)}条记录")
print(f"缩小后问答数据大小: {len(small_qa)}条记录")
print(f"缩小后的数据已保存为: {small_corpus_path} 和 {small_qa_path}")