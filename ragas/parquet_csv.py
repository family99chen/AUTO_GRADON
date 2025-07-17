import pandas as pd

# 文件路径
qa_path = "/home/cz/data/qa.parquet"
corpus_path = "/home/cz/data/corpus.parquet"

# 转换QA数据
qa_df = pd.read_parquet(qa_path)
qa_csv_path = "qa_data3.csv"
qa_df.to_csv(qa_csv_path, index=False, escapechar='\\', quoting=1)
print(f"已将QA数据转换为CSV并保存到: {qa_csv_path}")
print(f"QA数据共有{len(qa_df)}行, {qa_df.shape[1]}列")

# 转换语料库数据
corpus_df = pd.read_parquet(corpus_path)
corpus_csv_path = "corpus_data3.csv"
corpus_df.to_csv(corpus_csv_path, index=False, escapechar='\\', quoting=1, encoding='utf-8')
print(f"已将语料库数据转换为CSV并保存到: {corpus_csv_path}")
print(f"语料库数据共有{len(corpus_df)}行, {corpus_df.shape[1]}列")

# 查找特定的文档ID - 根据错误信息中提到的
doc_id = "2WikiMultihopQA_C83"
found = doc_id in corpus_df['doc_id'].values
print(f"文档ID '{doc_id}' 在语料库中{'存在' if found else '不存在'}")

# 如果找不到这个ID，搜索相似的ID
if not found:
    similar_ids = [id for id in corpus_df['doc_id'].values if id.startswith("2WikiMultihopQA_C")]
    print(f"找到{len(similar_ids)}个以'2WikiMultihopQA_C'开头的文档ID")
    print(f"示例: {similar_ids[:10]}")
    
    # 查找最接近的ID
    closest_ids = sorted([id for id in similar_ids if id.startswith("2WikiMultihopQA_C8")])
    if closest_ids:
        print(f"最接近的ID: {closest_ids[:10]}")