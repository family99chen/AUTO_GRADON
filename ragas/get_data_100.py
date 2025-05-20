import pandas as pd
import numpy as np
import os

# 文件路径
qa_path = "/home/xwh/AutoRAG/data/5dataset_100/qa.parquet"
corpus_path = "/home/xwh/AutoRAG/data/5dataset_100/corpus.parquet"

# 加载数据
print("加载原始数据...")
qa_df = pd.read_parquet(qa_path)
corpus_df = pd.read_parquet(corpus_path)

print(f"原始QA数据: {len(qa_df)}条")
print(f"原始语料库数据: {len(corpus_df)}条")

# 获取前100条QA数据
qa_subset = qa_df.head(100)
print(f"已选择前100条QA数据")

# 收集所有需要的文档ID
all_doc_ids = set()

# 处理数据解析文档ID
for i, row in qa_subset.iterrows():
    retrieval_gt = row['retrieval_gt']
    
    if isinstance(retrieval_gt, np.ndarray):
        # 如果是numpy数组，需要进一步处理
        for item in retrieval_gt:
            if isinstance(item, np.ndarray):
                for doc_id in item:
                    if isinstance(doc_id, str):
                        all_doc_ids.add(doc_id)
            elif isinstance(item, str):
                all_doc_ids.add(item)
    elif isinstance(retrieval_gt, list):
        # 处理列表形式
        for item in retrieval_gt:
            if isinstance(item, np.ndarray):
                for doc_id in item:
                    if isinstance(doc_id, str):
                        all_doc_ids.add(doc_id)
            elif isinstance(item, str):
                all_doc_ids.add(item)

print(f"从QA数据中提取了{len(all_doc_ids)}个唯一文档ID")

# 检查所有文档ID是否存在于语料库中
found_ids = set(corpus_df[corpus_df['doc_id'].isin(all_doc_ids)]['doc_id'])
missing_ids = all_doc_ids - found_ids

print(f"在语料库中找到了{len(found_ids)}个文档ID")
if missing_ids:
    print(f"有{len(missing_ids)}个文档ID在语料库中找不到")
    
    # 尝试处理截断的ID问题
    print("检查是否存在截断ID问题...")
    replacement_map = {}
    for missing_id in missing_ids:
        # 寻找可能的扩展ID (例如 "2WikiMultihopQA_C83" -> "2WikiMultihopQA_C83_0")
        potential_ids = [id for id in corpus_df['doc_id'] if id.startswith(missing_id + "_")]
        if potential_ids:
            print(f"  ID '{missing_id}' 找到可能的完整ID: {potential_ids[:3]}")
            # 为了简单起见，使用第一个找到的扩展ID
            replacement_map[missing_id] = potential_ids[0]
            
            # 将这些完整ID添加到需要的文档集合中
            found_ids.add(potential_ids[0])

# 筛选出所有需要的语料库文档
corpus_subset = corpus_df[corpus_df['doc_id'].isin(found_ids)]

# 创建输出目录
output_dir = "/home/xwh/AutoRAG/data/5dataset_100"
os.makedirs(output_dir, exist_ok=True)

# 保存结果
qa_output = os.path.join(output_dir, "qa100.parquet")
corpus_output = os.path.join(output_dir, "corpus_relate.parquet")

qa_subset.to_parquet(qa_output)
corpus_subset.to_parquet(corpus_output)

print(f"\n已保存QA数据集前100条到: {qa_output}")
print(f"已保存相关语料库数据({len(corpus_subset)}条)到: {corpus_output}")
print("\n完成！")

# 额外提示
if len(corpus_subset) < len(found_ids):
    print("\n警告：保存的语料库数据少于找到的文档ID数量!")

if missing_ids:
    print("\n注意：如果在运行时仍然遇到文档ID找不到的问题，建议修改AutoRAG代码中的fetch_one_content函数，")
    print("让它在找不到精确ID时尝试查找前缀匹配的ID。这可能位于/home/xwh/AutoRAG/autorag/utils/util.py文件中。")