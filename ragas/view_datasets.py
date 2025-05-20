import pandas as pd
import os
import time
import yaml

# # 读取问答数据集
# qa_path = "../data/qa.parquet"  # 根据您的实际路径调整
# qa_data = pd.read_parquet(qa_path)

# # 读取语料库数据集
# corpus_path = "../data/corpus.parquet"  # 根据您的实际路径调整
# corpus_data = pd.read_parquet(corpus_path)

# # 显示问答数据集的前几行和列名
# print("=== 问答数据集(qa.parquet) ===")
# print(f"形状: {qa_data.shape}")
# print(f"列名: {list(qa_data.columns)}")
# print("前5行:")
# print(qa_data.head(5))

# # 显示语料库数据集的前几行和列名
# print("\n=== 语料库数据集(corpus.parquet) ===")
# print(f"形状: {corpus_data.shape}")
# print(f"列名: {list(corpus_data.columns)}")
# print("前5行:")
# print(corpus_data.head(5))

# # 可选: 显示数据类型的更多细节
# print("\n问答数据集的详细信息:")
# print(qa_data.info())

# print("\n语料库数据集的详细信息:")
# print(corpus_data.info())

# # 可选: 如果数据集比较大，随机抽样查看
# print("\n问答数据集的随机样本:")
# print(qa_data.sample(3))

# print("\n语料库数据集的随机样本:")
# print(corpus_data.sample(3))

import pandas as pd
import json

# 读取问答数据集
qa_path = "../data/5dataset_100/qa.parquet"
qa_data = pd.read_parquet(qa_path)

# 读取语料库数据集
corpus_path = "../data/5dataset_100/corpus.parquet"
corpus_data = pd.read_parquet(corpus_path)

# 辅助函数：格式化复杂对象用于展示
def format_value(value, max_length=200):
    if isinstance(value, list):
        if len(value) > 5:
            formatted = str(value[:5])[:-1] + ", ...]"
        else:
            formatted = str(value)
        if len(formatted) > max_length:
            return formatted[:max_length] + "..."
        return formatted
    elif isinstance(value, dict):
        formatted = json.dumps(value, ensure_ascii=False)
        if len(formatted) > max_length:
            return formatted[:max_length] + "..."
        return formatted
    else:
        formatted = str(value)
        if len(formatted) > max_length:
            return formatted[:max_length] + "..."
        return formatted

# 显示问答数据集的基本信息
print("=== 问答数据集(qa.parquet) ===")
print(f"形状: {qa_data.shape}")
print(f"列名: {list(qa_data.columns)}")

# 显示问答数据集的详细内容
print("\n详细的问答数据内容 (前3条):")
for i, row in qa_data.head(3).iterrows():
    print(f"\n记录 #{i+1}:")
    for col in qa_data.columns:
        print(f"  {col}: {format_value(row[col])}")

# 显示retrieval_gt和generation_gt的具体结构
if 'retrieval_gt' in qa_data.columns:
    sample = qa_data['retrieval_gt'].dropna().iloc[0] if not qa_data['retrieval_gt'].dropna().empty else None
    print(f"\nretrieval_gt示例结构: {type(sample)}")
    if isinstance(sample, list) and sample:
        print(f"  第一个元素类型: {type(sample[0])}")
        print(f"  第一个元素: {sample[0]}")

if 'generation_gt' in qa_data.columns:
    sample = qa_data['generation_gt'].dropna().iloc[0] if not qa_data['generation_gt'].dropna().empty else None
    print(f"\ngeneration_gt示例结构: {type(sample)}")
    if sample:
        print(f"  内容: {sample}")

# 显示语料库数据集的基本信息
print("\n\n=== 语料库数据集(corpus.parquet) ===")
print(f"形状: {corpus_data.shape}")
print(f"列名: {list(corpus_data.columns)}")

# 显示语料库数据集的详细内容
print("\n详细的语料库数据内容 (前3条):")
for i, row in corpus_data.head(3).iterrows():
    print(f"\n记录 #{i+1}:")
    for col in corpus_data.columns:
        print(f"  {col}: {format_value(row[col])}")

# 如果metadata列是字典类型，展示其结构
if 'metadata' in corpus_data.columns:
    sample = corpus_data['metadata'].dropna().iloc[0] if not corpus_data['metadata'].dropna().empty else None
    if isinstance(sample, dict):
        print(f"\nmetadata字段结构示例: {list(sample.keys())}")
        for key, value in sample.items():
            print(f"  {key}: {format_value(value, max_length=100)}")

# 随机抽样查看更多记录
print("\n\n随机抽样的问答数据 (3条):")
for i, row in qa_data.sample(3).iterrows():
    print(f"\n随机问答记录 #{i}:")
    for col in qa_data.columns:
        print(f"  {col}: {format_value(row[col])}")

print("\n\n随机抽样的语料库数据 (3条):")
for i, row in corpus_data.sample(3).iterrows():
    print(f"\n随机语料记录 #{i}:")
    for col in corpus_data.columns:
        print(f"  {col}: {format_value(row[col])}")

# 打印一个完整的示例（不截断），帮助理解数据结构
print("\n\n完整的问答记录示例:")
sample_qa = qa_data.iloc[0]
for col in qa_data.columns:
    print(f"{col}:")
    print(sample_qa[col])
    print("-" * 80)

print("\n\n完整的语料库记录示例:")
sample_corpus = corpus_data.iloc[0]
for col in corpus_data.columns:
    print(f"{col}:")
    print(sample_corpus[col])
    print("-" * 80)