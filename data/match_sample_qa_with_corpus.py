#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA数据采样和语料库匹配脚本

从下载的HuggingFace RAG-Mini-BioASQ数据集中采样指定数量的QA，
并获取对应的语料库文档，保存为新的parquet文件。

支持新的数据格式：
- QA数据列名：qid, query, retrieval_gt, generation_gt
- 语料库数据列名：doc_id, contents, metadata
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json
import random
import re
from datetime import datetime

class QACorpusSampler:
    def __init__(self, input_dir: str = "./rag-mini-bioasq"):
        """
        初始化采样器
        
        Args:
            input_dir: 包含parquet文件的输入目录
        """
        self.input_dir = Path(input_dir)
        self.qa_file = self.input_dir / "qa.parquet"
        self.corpus_file = self.input_dir / "corpus.parquet"
        
        self.qa_df = None
        self.corpus_df = None
        self.corpus_id_map = {}  # passage_id -> corpus row mapping
        
        # 列名映射，会在load_data后初始化
        self.id_col = None
        self.question_col = None
        self.answer_col = None
        self.relevant_ids_col = None
        
        print(f"📁 输入目录: {self.input_dir.absolute()}")
    
    def _get_column_name(self, preferred_names: List[str]) -> str:
        """
        获取数据中实际存在的列名
        
        Args:
            preferred_names: 按优先级排序的列名列表
            
        Returns:
            str: 实际存在的列名
        """
        for name in preferred_names:
            if name in self.qa_df.columns:
                return name
        return preferred_names[0]  # 如果都不存在，返回第一个作为默认值
    
    def load_data(self) -> bool:
        """
        加载QA和语料库数据
        
        Returns:
            bool: 加载是否成功
        """
        try:
            # 检查文件是否存在
            if not self.qa_file.exists():
                print(f"❌ QA文件不存在: {self.qa_file}")
                return False
            
            if not self.corpus_file.exists():
                print(f"❌ 语料库文件不存在: {self.corpus_file}")
                return False
            
            # 加载QA数据
            print(f"🔄 加载QA数据: {self.qa_file}")
            self.qa_df = pd.read_parquet(self.qa_file)
            print(f"   ✅ QA数据加载成功! 行数: {len(self.qa_df):,}")
            print(f"   列名: {list(self.qa_df.columns)}")
            
            # 初始化列名映射
            self.id_col = self._get_column_name(['qid', 'id'])
            self.question_col = self._get_column_name(['query', 'question'])
            self.answer_col = self._get_column_name(['generation_gt', 'answer'])
            self.relevant_ids_col = self._get_column_name(['retrieval_gt', 'relevant_passage_ids'])
            
            print(f"   使用列名映射: id='{self.id_col}', question='{self.question_col}', answer='{self.answer_col}', relevant_ids='{self.relevant_ids_col}'")
            
            # 加载语料库数据
            print(f"🔄 加载语料库数据: {self.corpus_file}")
            self.corpus_df = pd.read_parquet(self.corpus_file)
            print(f"   ✅ 语料库数据加载成功! 行数: {len(self.corpus_df):,}")
            print(f"   列名: {list(self.corpus_df.columns)}")
            
            # 构建语料库ID映射
            self._build_corpus_id_map()
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            return False
    
    def _build_corpus_id_map(self):
        """构建语料库ID到索引的映射"""
        print("🔄 构建语料库ID映射...")
        
        # 检查语料库的ID列名
        id_column = None
        possible_id_columns = ['doc_id', 'id', 'passage_id']
        
        for col in possible_id_columns:
            if col in self.corpus_df.columns:
                id_column = col
                break
        
        if id_column is None:
            print(f"⚠️  未找到ID列，可用列: {list(self.corpus_df.columns)}")
            # 如果没有ID列，使用索引作为ID
            self.corpus_df['doc_id'] = self.corpus_df.index.astype(str)
            id_column = 'doc_id'
        
        print(f"   使用ID列: {id_column}")
        
        # 构建映射
        for idx, row in self.corpus_df.iterrows():
            passage_id = str(row[id_column])
            self.corpus_id_map[passage_id] = idx
        
        print(f"   ✅ 构建完成! 语料库ID数量: {len(self.corpus_id_map):,}")
    
    def debug_qa_parsing(self, num_samples: int = 5):
        """
        调试QA数据解析过程
        
        Args:
            num_samples: 要调试的样本数量
        """
        print(f"\n🔍 调试前{num_samples}个QA的解析过程...")
        
        for i in range(min(num_samples, len(self.qa_df))):
            row = self.qa_df.iloc[i]
            print(f"\n--- 样本 {i+1} ---")
            print(f"QA ID: {row.get(self.id_col, 'N/A')}")
            print(f"Query: {str(row[self.question_col])[:50]}...")
            
            # 显示原始的retrieval_gt数据
            raw_retrieval_gt = row[self.relevant_ids_col]
            print(f"原始retrieval_gt: {raw_retrieval_gt}")
            print(f"数据类型: {type(raw_retrieval_gt)}")
            
            # 解析并显示结果
            parsed_ids = self._parse_relevant_passage_ids(raw_retrieval_gt)
            print(f"解析后的ID: {parsed_ids}")
            
            # 检查在corpus中是否存在
            found_ids = []
            missing_ids = []
            for pid in parsed_ids:
                if pid in self.corpus_id_map:
                    found_ids.append(pid)
                else:
                    missing_ids.append(pid)
            
            print(f"在corpus中找到的ID: {found_ids}")
            print(f"在corpus中缺失的ID: {missing_ids}")
            
            # 显示corpus中的前几个ID作为参考
            if i == 0:
                print(f"\n语料库中的前10个ID: {list(self.corpus_id_map.keys())[:10]}")
    
    def _parse_relevant_passage_ids(self, relevant_ids) -> List[str]:
        """
        解析relevant_passage_ids字段或retrieval_gt字段
        
        Args:
            relevant_ids: 原始的relevant_passage_ids或retrieval_gt数据
            
        Returns:
            List[str]: 解析后的passage ID列表
        """
        if pd.isna(relevant_ids):
            return []
        
        # 如果是numpy数组，需要特殊处理
        if isinstance(relevant_ids, np.ndarray):
            result = []
            # 遍历数组中的每个元素
            for item in relevant_ids:
                if isinstance(item, np.ndarray):
                    # 如果元素也是numpy数组，递归处理
                    result.extend([str(pid) for pid in item.tolist()])
                elif isinstance(item, (list, tuple)):
                    result.extend([str(pid) for pid in item])
                else:
                    result.append(str(item))
            return result
        
        # 如果是字符串，尝试解析
        elif isinstance(relevant_ids, str):
            try:
                # 处理包含numpy array字符串表示的格式
                # 例如: "[array(['2WikiMultihopQA_C0_0', '2WikiMultihopQA_C0_1'], dtype=object)]"
                if 'array([' in relevant_ids:
                    # 使用正则表达式提取引号内的ID
                    pattern = r"'([^']+)'"
                    matches = re.findall(pattern, relevant_ids)
                    return [str(match) for match in matches]
                
                # 尝试解析标准JSON格式
                elif relevant_ids.startswith('['):
                    parsed = json.loads(relevant_ids)
                    if isinstance(parsed, list):
                        result = []
                        for item in parsed:
                            if isinstance(item, (list, tuple)):
                                result.extend([str(pid) for pid in item])
                            else:
                                result.append(str(item))
                        return result
                    else:
                        return [str(parsed)]
                else:
                    # 单个ID
                    return [relevant_ids]
            except Exception as e:
                print(f"⚠️  解析相关文档ID失败: {relevant_ids}, 错误: {e}")
                return [relevant_ids]
        
        # 如果是列表，可能是retrieval_gt格式: [array(['id1', 'id2'], dtype=object)]
        elif isinstance(relevant_ids, (list, tuple)):
            result = []
            for item in relevant_ids:
                if isinstance(item, np.ndarray):
                    result.extend([str(pid) for pid in item.tolist()])
                elif isinstance(item, (list, tuple)):
                    result.extend([str(pid) for pid in item])
                else:
                    result.append(str(item))
            return result
        
        # 其他类型，转换为字符串
        else:
            return [str(relevant_ids)]
    
    def analyze_data_coverage(self) -> Dict:
        """
        分析数据覆盖情况
        
        Returns:
            Dict: 覆盖情况统计
        """
        print("\n🔍 分析数据覆盖情况...")
        
        all_referenced_ids = set()
        qa_with_missing_passages = []
        
        for idx, row in self.qa_df.iterrows():
            relevant_ids = self._parse_relevant_passage_ids(row[self.relevant_ids_col])
            all_referenced_ids.update(relevant_ids)
            
            # 检查这个QA的passage是否都存在
            missing_ids = []
            for pid in relevant_ids:
                if pid not in self.corpus_id_map:
                    missing_ids.append(pid)
            
            if missing_ids:
                qa_with_missing_passages.append({
                    'qa_idx': idx,
                    'qa_id': row.get(self.id_col, idx),
                    'missing_ids': missing_ids,
                    'total_ids': len(relevant_ids),
                    'missing_count': len(missing_ids)
                })
        
        corpus_ids = set(self.corpus_id_map.keys())
        found_ids = all_referenced_ids & corpus_ids
        missing_ids = all_referenced_ids - corpus_ids
        
        coverage_stats = {
            'total_qa': len(self.qa_df),
            'total_corpus': len(self.corpus_df),
            'referenced_passage_ids': len(all_referenced_ids),
            'found_passage_ids': len(found_ids),
            'missing_passage_ids': len(missing_ids),
            'coverage_rate': len(found_ids) / len(all_referenced_ids) * 100 if all_referenced_ids else 0,
            'qa_with_missing_passages': len(qa_with_missing_passages),
            'qa_coverage_rate': (len(self.qa_df) - len(qa_with_missing_passages)) / len(self.qa_df) * 100
        }
        
        print(f"   QA总数: {coverage_stats['total_qa']:,}")
        print(f"   语料库总数: {coverage_stats['total_corpus']:,}")
        print(f"   引用的passage ID总数: {coverage_stats['referenced_passage_ids']:,}")
        print(f"   找到的passage ID: {coverage_stats['found_passage_ids']:,}")
        print(f"   缺失的passage ID: {coverage_stats['missing_passage_ids']:,}")
        print(f"   passage覆盖率: {coverage_stats['coverage_rate']:.2f}%")
        print(f"   有缺失passage的QA: {coverage_stats['qa_with_missing_passages']:,}")
        print(f"   QA完整性: {coverage_stats['qa_coverage_rate']:.2f}%")
        
        return coverage_stats
    
    def sample_qa_with_corpus(self, sample_size: int, require_complete: bool = True, 
                             random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        采样QA数据并获取对应的语料库
        
        Args:
            sample_size: 采样数量
            require_complete: 是否要求QA的所有passage都存在
            random_seed: 随机种子
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (采样的QA数据, 对应的语料库数据)
        """
        print(f"\n🎯 开始采样 {sample_size} 个QA...")
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 筛选可用的QA
        valid_qa_indices = []
        all_needed_passage_ids = set()
        
        for idx, row in self.qa_df.iterrows():
            relevant_ids = self._parse_relevant_passage_ids(row[self.relevant_ids_col])
            
            if require_complete:
                # 检查所有passage是否都存在
                all_exist = all(pid in self.corpus_id_map for pid in relevant_ids)
                if all_exist and relevant_ids:  # 确保至少有一个passage
                    valid_qa_indices.append(idx)
                    all_needed_passage_ids.update(relevant_ids)
            else:
                # 只要有至少一个passage存在就可以
                any_exist = any(pid in self.corpus_id_map for pid in relevant_ids)
                if any_exist:
                    valid_qa_indices.append(idx)
                    # 只添加存在的passage ID
                    existing_ids = [pid for pid in relevant_ids if pid in self.corpus_id_map]
                    all_needed_passage_ids.update(existing_ids)
        
        print(f"   可用的QA数量: {len(valid_qa_indices):,}")
        
        if len(valid_qa_indices) < sample_size:
            print(f"⚠️  可用QA数量({len(valid_qa_indices)})少于请求的采样数量({sample_size})")
            sample_size = len(valid_qa_indices)
        
        # 随机采样
        sampled_indices = random.sample(valid_qa_indices, sample_size)
        sampled_qa = self.qa_df.iloc[sampled_indices].copy()
        
        # 收集所有需要的passage ID
        final_needed_passage_ids = set()
        for idx in sampled_indices:
            row = self.qa_df.iloc[idx]
            relevant_ids = self._parse_relevant_passage_ids(row[self.relevant_ids_col])
            
            if require_complete:
                final_needed_passage_ids.update(relevant_ids)
            else:
                # 只添加存在的passage ID
                existing_ids = [pid for pid in relevant_ids if pid in self.corpus_id_map]
                final_needed_passage_ids.update(existing_ids)
        
        # 获取对应的语料库数据
        corpus_indices = []
        for pid in final_needed_passage_ids:
            if pid in self.corpus_id_map:
                corpus_indices.append(self.corpus_id_map[pid])
        
        sampled_corpus = self.corpus_df.iloc[corpus_indices].copy()
        
        print(f"   ✅ 采样完成!")
        print(f"   采样的QA数量: {len(sampled_qa):,}")
        print(f"   对应的语料库文档数量: {len(sampled_corpus):,}")
        
        return sampled_qa, sampled_corpus
    
    def convert_to_downstream_format(self, sampled_qa: pd.DataFrame, sampled_corpus: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        转换为下游系统要求的格式
        
        Args:
            sampled_qa: 原始采样的QA数据
            sampled_corpus: 原始采样的语料库数据
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (转换后的QA数据, 转换后的语料库数据)
        """
        print("\n🔄 转换为下游系统格式...")
        
        # 转换QA数据格式
        converted_qa = pd.DataFrame()
        
        # 基本列映射
        converted_qa['qid'] = sampled_qa.get(self.id_col, sampled_qa.index).astype(str)
        converted_qa['query'] = sampled_qa[self.question_col]
        
        # 将answer字段转换为generation_gt格式
        generation_gt_list = []
        for idx, row in sampled_qa.iterrows():
            answer = row.get(self.answer_col, '')
            if pd.isna(answer) or answer == '':
                generation_gt_list.append([])
            else:
                # 如果已经是列表格式，直接使用；否则包装为列表
                if isinstance(answer, (list, tuple)):
                    generation_gt_list.append(list(answer))
                else:
                    generation_gt_list.append([str(answer)])
        
        converted_qa['generation_gt'] = generation_gt_list
        
        # 转换 retrieval_gt 格式
        retrieval_gt_list = []
        for idx, row in sampled_qa.iterrows():
            relevant_ids = self._parse_relevant_passage_ids(row[self.relevant_ids_col])
            # 转换为numpy数组格式，然后包装在列表中
            if relevant_ids:
                retrieval_gt = [np.array(relevant_ids, dtype=object)]
            else:
                retrieval_gt = [np.array([], dtype=object)]
            retrieval_gt_list.append(retrieval_gt)
        
        converted_qa['retrieval_gt'] = retrieval_gt_list
        
        # 转换语料库数据格式
        converted_corpus = pd.DataFrame()
        
        # 确定ID列
        id_column = None
        for col in ['doc_id', 'id', 'passage_id']:
            if col in sampled_corpus.columns:
                id_column = col
                break
        
        if id_column is None:
            # 如果没有ID列，使用索引
            converted_corpus['doc_id'] = sampled_corpus.index.astype(str)
        else:
            converted_corpus['doc_id'] = sampled_corpus[id_column].astype(str)
        
        # 确定内容列
        content_column = None
        for col in ['contents', 'passage', 'text', 'content']:
            if col in sampled_corpus.columns:
                content_column = col
                break
        
        if content_column is None:
            print("⚠️  未找到内容列，使用空字符串")
            converted_corpus['contents'] = ["" for _ in range(len(sampled_corpus))]
        else:
            converted_corpus['contents'] = sampled_corpus[content_column].fillna("")
        
        # 生成metadata
        current_time = datetime.now()
        metadata_list = []
        for i in range(len(sampled_corpus)):
            metadata = {
                'last_modified_datetime': current_time,
                'source': 'rag-mini-bioasq',
                'index': i
            }
            metadata_list.append(metadata)
        
        converted_corpus['metadata'] = metadata_list
        
        print(f"   ✅ 格式转换完成!")
        print(f"   QA列名: {list(converted_qa.columns)}")
        print(f"   语料库列名: {list(converted_corpus.columns)}")
        
        # 显示转换示例
        if len(converted_qa) > 0:
            print(f"\n   QA数据示例:")
            example_qa = converted_qa.iloc[0]
            print(f"     qid: {example_qa['qid']}")
            print(f"     query: {example_qa['query'][:50]}...")
            print(f"     retrieval_gt: {example_qa['retrieval_gt']}")
            print(f"     generation_gt: {example_qa['generation_gt']}")
        
        if len(converted_corpus) > 0:
            print(f"\n   语料库数据示例:")
            example_corpus = converted_corpus.iloc[0]
            print(f"     doc_id: {example_corpus['doc_id']}")
            print(f"     contents: {str(example_corpus['contents'])[:50]}...")
            print(f"     metadata: {example_corpus['metadata']}")
        
        return converted_qa, converted_corpus
    
    def save_sampled_data(self, sampled_qa: pd.DataFrame, sampled_corpus: pd.DataFrame, 
                         output_prefix: str = "sampled", convert_format: bool = False) -> bool:
        """
        保存采样的数据
        
        Args:
            sampled_qa: 采样的QA数据
            sampled_corpus: 采样的语料库数据
            output_prefix: 输出文件前缀
            convert_format: 是否转换为下游系统格式
            
        Returns:
            bool: 保存是否成功
        """
        try:
            if convert_format:
                # 转换格式
                sampled_qa, sampled_corpus = self.convert_to_downstream_format(sampled_qa, sampled_corpus)
                qa_output_file = self.input_dir / f"{output_prefix}_qa_formatted.parquet"
                corpus_output_file = self.input_dir / f"{output_prefix}_corpus_formatted.parquet"
            else:
                qa_output_file = self.input_dir / f"{output_prefix}_qa.parquet"
                corpus_output_file = self.input_dir / f"{output_prefix}_corpus.parquet"
            
            print(f"\n💾 保存采样数据...")
            print(f"   QA文件: {qa_output_file}")
            print(f"   语料库文件: {corpus_output_file}")
            
            # 保存数据
            sampled_qa.to_parquet(qa_output_file, index=False)
            sampled_corpus.to_parquet(corpus_output_file, index=False)
            
            # 验证保存的文件
            qa_size = qa_output_file.stat().st_size / (1024 * 1024)  # MB
            corpus_size = corpus_output_file.stat().st_size / (1024 * 1024)  # MB
            
            print(f"   ✅ 保存成功!")
            print(f"   QA文件大小: {qa_size:.2f} MB")
            print(f"   语料库文件大小: {corpus_size:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 保存失败: {str(e)}")
            return False
    
    def show_sample_preview(self, sampled_qa: pd.DataFrame, sampled_corpus: pd.DataFrame, 
                           num_examples: int = 3):
        """
        显示采样数据的预览
        
        Args:
            sampled_qa: 采样的QA数据
            sampled_corpus: 采样的语料库数据
            num_examples: 显示的示例数量
        """
        print(f"\n👀 数据预览 (前{num_examples}个示例):")
        print("=" * 80)
        
        for i in range(min(num_examples, len(sampled_qa))):
            qa_row = sampled_qa.iloc[i]
            print(f"\n📝 示例 {i+1}:")
            print(f"   QA ID: {qa_row.get(self.id_col, 'N/A')}")
            print(f"   问题: {str(qa_row[self.question_col])[:100]}...")
            
            # 处理答案显示
            answer = qa_row.get(self.answer_col, 'N/A')
            if isinstance(answer, (list, tuple)) and len(answer) > 0:
                answer_str = str(answer[0])
            else:
                answer_str = str(answer)
            print(f"   答案: {answer_str[:100]}...")
            
            # 显示相关的passages
            relevant_ids = self._parse_relevant_passage_ids(qa_row[self.relevant_ids_col])
            print(f"   相关passage数量: {len(relevant_ids)}")
            
            for j, pid in enumerate(relevant_ids[:2]):  # 只显示前2个
                if pid in self.corpus_id_map:
                    corpus_idx = self.corpus_id_map[pid]
                    if corpus_idx < len(sampled_corpus):
                        corpus_row = sampled_corpus.iloc[corpus_idx]
                        passage_text = corpus_row.get('contents', corpus_row.get('passage', corpus_row.get('text', 'N/A')))
                        print(f"     Passage {j+1} (ID: {pid}): {str(passage_text)[:80]}...")

    def show_data_overview(self, data_path: str):
        """
        展示指定路径数据的详细信息
        
        Args:
            data_path: 数据文件路径
        """
        data_file = Path(data_path)
        
        if not data_file.exists():
            print(f"❌ 数据文件不存在: {data_file}")
            return
        
        try:
            print(f"\n📊 数据概览: {data_file}")
            print("=" * 80)
            
            # 读取数据
            if data_file.suffix.lower() == '.parquet':
                df = pd.read_parquet(data_file)
            elif data_file.suffix.lower() == '.csv':
                df = pd.read_csv(data_file)
            else:
                print(f"❌ 不支持的文件格式: {data_file.suffix}")
                return
            
            # 基本信息
            print(f"📁 文件路径: {data_file.absolute()}")
            print(f"📏 数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
            print(f"💾 内存使用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            print(f"🗂️  文件大小: {data_file.stat().st_size / 1024 / 1024:.2f} MB")
            
            # 列信息
            print(f"\n📋 列信息:")
            for i, col in enumerate(df.columns):
                col_type = str(df[col].dtype)  # 转换为字符串避免格式化错误
                null_count = df[col].isnull().sum()
                null_percent = (null_count / len(df)) * 100 if len(df) > 0 else 0
                print(f"   {i+1:2d}. {col:<20} | 类型: {col_type:<15} | 空值: {null_count:>6} ({null_percent:>5.1f}%)")
            
            # 数据类型统计
            print(f"\n📊 数据类型分布:")
            type_counts = df.dtypes.value_counts()
            for dtype, count in type_counts.items():
                print(f"   {str(dtype)}: {count} 列")  # 转换为字符串
            
            # 显示前几行数据
            print(f"\n👀 前5行数据预览:")
            print("-" * 80)
            for i in range(min(5, len(df))):
                row = df.iloc[i]
                print(f"\n第 {i+1} 行:")
                for col in df.columns:
                    value = row[col]
                    if pd.isna(value):
                        value_str = "null"
                    elif isinstance(value, str) and len(value) > 50:
                        value_str = value[:50] + "..."
                    elif isinstance(value, np.ndarray):
                        # 特殊处理numpy数组
                        value_str = f"数组[长度:{len(value)}] {str(value)[:50]}..."
                    else:
                        value_str = str(value)
                        if len(value_str) > 50:
                            value_str = value_str[:50] + "..."
                    print(f"   {col}: {value_str}")
            
            # 如果是QA数据，尝试解析retrieval_gt
            if 'retrieval_gt' in df.columns:
                print(f"\n🔍 retrieval_gt 数据解析示例:")
                print("-" * 50)
                for i in range(min(3, len(df))):
                    row = df.iloc[i]
                    print(f"\n样本 {i+1}:")
                    retrieval_gt = row['retrieval_gt']
                    print(f"   原始数据: {retrieval_gt}")
                    print(f"   数据类型: {type(retrieval_gt)}")
                    
                    # 尝试解析
                    try:
                        parsed_ids = self._parse_relevant_passage_ids(retrieval_gt)
                        print(f"   解析结果: {parsed_ids}")
                    except Exception as e:
                        print(f"   解析失败: {e}")
            
            # 如果是语料库数据，显示内容长度分布
            if 'contents' in df.columns:
                print(f"\n📝 contents 内容长度分布:")
                print("-" * 50)
                try:
                    contents_lengths = df['contents'].astype(str).str.len()  # 确保转为字符串
                    print(f"   最短: {contents_lengths.min()} 字符")
                    print(f"   最长: {contents_lengths.max()} 字符")
                    print(f"   平均: {contents_lengths.mean():.1f} 字符")
                    print(f"   中位数: {contents_lengths.median():.1f} 字符")
                    
                    # 显示几个不同长度的内容样本
                    print(f"\n📄 内容样本:")
                    sorted_indices = contents_lengths.argsort()
                    sample_indices = [
                        sorted_indices[0],  # 最短
                        sorted_indices[len(sorted_indices)//4],  # 25%
                        sorted_indices[len(sorted_indices)//2],  # 50%
                        sorted_indices[len(sorted_indices)*3//4],  # 75%
                        sorted_indices[-1]  # 最长
                    ]
                    
                    for i, idx in enumerate(sample_indices):
                        content = str(df.iloc[idx]['contents'])
                        length = len(content)
                        doc_id = df.iloc[idx].get('doc_id', f'行{idx}')
                        print(f"\n   样本 {i+1} (ID: {doc_id}, 长度: {length}):")
                        print(f"   {content[:100]}...")
                except Exception as e:
                    print(f"   ⚠️ 内容长度分析失败: {e}")
                    
        except Exception as e:
            print(f"❌ 读取数据失败: {str(e)}")
            import traceback
            print(f"详细错误信息:\n{traceback.format_exc()}")

    def explore_directory(self, dir_path: str):
        """
        探索目录中的数据文件
        
        Args:
            dir_path: 目录路径
        """
        directory = Path(dir_path)
        
        if not directory.exists():
            print(f"❌ 目录不存在: {directory}")
            return
        
        if not directory.is_dir():
            print(f"❌ 不是目录: {directory}")
            return
        
        print(f"\n📁 目录探索: {directory.absolute()}")
        print("=" * 80)
        
        # 查找数据文件
        data_files = []
        for pattern in ['*.parquet', '*.csv']:
            data_files.extend(directory.glob(pattern))
        
        if not data_files:
            print("❌ 未找到数据文件 (.parquet 或 .csv)")
            return
        
        print(f"📊 找到 {len(data_files)} 个数据文件:")
        for i, file in enumerate(data_files):
            file_size = file.stat().st_size / 1024 / 1024  # MB
            print(f"   {i+1}. {file.name} ({file_size:.2f} MB)")
        
        # 让用户选择要查看的文件
        while True:
            try:
                choice = input(f"\n请选择要查看的文件 (1-{len(data_files)}) 或输入 'q' 退出: ").strip()
                if choice.lower() == 'q':
                    break
                
                file_index = int(choice) - 1
                if 0 <= file_index < len(data_files):
                    selected_file = data_files[file_index]
                    self.show_data_overview(str(selected_file))
                    break
                else:
                    print(f"❌ 请输入 1-{len(data_files)} 之间的数字")
            except ValueError:
                print("❌ 请输入有效的数字")

def main():
    """主函数"""
    print("🔬 QA数据采样和语料库匹配工具")
    print("=" * 60)
    
    # 获取输入目录
    while True:
        input_dir = input("\n请输入包含parquet文件的目录 (直接回车使用默认 './rag-mini-bioasq'): ").strip()
        if not input_dir:
            input_dir = "./rag-mini-bioasq"
        
        input_path = Path(input_dir)
        if input_path.exists():
            break
        else:
            print(f"❌ 目录不存在: {input_dir}")
    
    # 创建采样器
    sampler = QACorpusSampler(input_dir)
    
    # 加载数据
    if not sampler.load_data():
        print("❌ 数据加载失败，程序退出")
        return
    
    while True:
        print("\n📋 可用操作:")
        print("1. 分析数据覆盖情况")
        print("2. 采样QA数据")
        print("3. 调试QA数据解析")
        print("4. 展示数据详情")
        print("5. 探索目录")
        print("6. 退出")
        
        choice = input("\n请选择操作 (1-6): ").strip()
        
        if choice == '1':
            sampler.analyze_data_coverage()
        
        elif choice == '2':
            # 获取采样参数
            try:
                sample_size = int(input("请输入采样数量: ").strip())
            except ValueError:
                print("❌ 请输入有效的数字")
                continue
            
            require_complete = input("是否要求QA的所有passage都存在? (y/n, 默认y): ").strip().lower()
            require_complete = require_complete != 'n'
            
            output_prefix = input("请输入输出文件前缀 (默认 'sampled'): ").strip()
            if not output_prefix:
                output_prefix = "sampled"
            
            # 是否转换格式
            convert_format = input("是否转换为下游系统格式? (y/n, 默认y): ").strip().lower()
            convert_format = convert_format != 'n'
            
            # 执行采样
            sampled_qa, sampled_corpus = sampler.sample_qa_with_corpus(
                sample_size=sample_size,
                require_complete=require_complete
            )
            
            # 显示预览
            sampler.show_sample_preview(sampled_qa, sampled_corpus)
            
            # 保存数据
            save_choice = input("\n是否保存采样数据? (y/n): ").strip().lower()
            if save_choice == 'y':
                sampler.save_sampled_data(sampled_qa, sampled_corpus, output_prefix, convert_format)
        
        elif choice == '3':
            # 调试QA数据解析
            try:
                num_samples = int(input("请输入要调试的样本数量 (默认5): ").strip() or "5")
            except ValueError:
                num_samples = 5
            sampler.debug_qa_parsing(num_samples)
        
        elif choice == '4':
            # 展示数据详情
            data_path = input("请输入数据文件路径 (.parquet 或 .csv): ").strip()
            if data_path:
                sampler.show_data_overview(data_path)
        
        elif choice == '5':
            # 探索目录
            dir_path = input("请输入目录路径: ").strip()
            if dir_path:
                sampler.explore_directory(dir_path)
        
        elif choice == '6':
            print("👋 再见!")
            break
        
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    main()