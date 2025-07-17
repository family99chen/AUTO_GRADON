import pandas as pd
import numpy as np
from typing import Set, List, Dict, Any
import os
from collections import defaultdict

class QACorpusAnalyzer:
    def __init__(self):
        self.qa_df = None
        self.corpus_df = None
        self.qa_path = None
        self.corpus_path = None
        self.all_qa_doc_ids = set()
        self.corpus_doc_ids = set()
        
    def load_data(self, qa_path: str, corpus_path: str):
        """加载QA和语料库数据"""
        try:
            print(f"正在加载QA数据: {qa_path}")
            self.qa_df = pd.read_parquet(qa_path)
            self.qa_path = qa_path
            
            print(f"正在加载语料库数据: {corpus_path}")
            self.corpus_df = pd.read_parquet(corpus_path)
            self.corpus_path = corpus_path
            
            print(f"✅ 数据加载成功!")
            print(f"   QA数据: {len(self.qa_df)} 条")
            print(f"   语料库数据: {len(self.corpus_df)} 条")
            
            # 提取所有文档ID
            self._extract_all_doc_ids()
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            return False
    
    def _extract_all_doc_ids(self):
        """从QA数据中提取所有需要的文档ID"""
        print("\n正在提取QA数据中的文档ID...")
        self.all_qa_doc_ids = set()
        
        for i, row in self.qa_df.iterrows():
            retrieval_gt = row['retrieval_gt']
            doc_ids = self._parse_retrieval_gt(retrieval_gt)
            self.all_qa_doc_ids.update(doc_ids)
        
        # 提取语料库中的所有文档ID
        self.corpus_doc_ids = set(self.corpus_df['doc_id'].values)
        
        print(f"✅ ID提取完成!")
        print(f"   QA数据需要的文档ID: {len(self.all_qa_doc_ids)} 个")
        print(f"   语料库包含的文档ID: {len(self.corpus_doc_ids)} 个")
    
    def _parse_retrieval_gt(self, retrieval_gt) -> Set[str]:
        """解析retrieval_gt字段，提取文档ID"""
        doc_ids = set()
        
        if isinstance(retrieval_gt, np.ndarray):
            for item in retrieval_gt:
                if isinstance(item, np.ndarray):
                    for doc_id in item:
                        if isinstance(doc_id, str):
                            doc_ids.add(doc_id)
                elif isinstance(item, str):
                    doc_ids.add(item)
        elif isinstance(retrieval_gt, list):
            for item in retrieval_gt:
                if isinstance(item, np.ndarray):
                    for doc_id in item:
                        if isinstance(doc_id, str):
                            doc_ids.add(doc_id)
                elif isinstance(item, str):
                    doc_ids.add(item)
        elif isinstance(retrieval_gt, str):
            doc_ids.add(retrieval_gt)
            
        return doc_ids
    
    def analyze_coverage(self):
        """分析ID覆盖情况"""
        if not self.qa_df is not None or not self.corpus_df is not None:
            print("❌ 请先加载数据!")
            return
        
        found_ids = self.all_qa_doc_ids & self.corpus_doc_ids
        missing_ids = self.all_qa_doc_ids - self.corpus_doc_ids
        extra_ids = self.corpus_doc_ids - self.all_qa_doc_ids
        
        print("\n" + "="*60)
        print("📊 ID匹配分析报告")
        print("="*60)
        print(f"QA数据需要的文档ID总数: {len(self.all_qa_doc_ids)}")
        print(f"语料库包含的文档ID总数: {len(self.corpus_doc_ids)}")
        print(f"匹配成功的ID数量: {len(found_ids)}")
        print(f"缺失的ID数量: {len(missing_ids)}")
        print(f"多余的ID数量: {len(extra_ids)}")
        print(f"覆盖率: {len(found_ids)/len(self.all_qa_doc_ids)*100:.2f}%")
        
        return {
            'total_qa_ids': len(self.all_qa_doc_ids),
            'total_corpus_ids': len(self.corpus_doc_ids),
            'found_ids': found_ids,
            'missing_ids': missing_ids,
            'extra_ids': extra_ids,
            'coverage_rate': len(found_ids)/len(self.all_qa_doc_ids)*100
        }
    
    def show_missing_ids(self, limit: int = 20):
        """显示缺失的ID"""
        missing_ids = self.all_qa_doc_ids - self.corpus_doc_ids
        
        print(f"\n❌ 缺失的文档ID (显示前{min(limit, len(missing_ids))}个):")
        print("-" * 50)
        
        for i, doc_id in enumerate(sorted(missing_ids)):
            if i >= limit:
                print(f"... 还有 {len(missing_ids) - limit} 个缺失ID")
                break
            print(f"{i+1:3d}. {doc_id}")
    
    def find_similar_ids(self, missing_id: str, max_results: int = 5):
        """为缺失的ID寻找相似的ID"""
        similar_ids = []
        
        # 策略1: 寻找扩展ID (例如 "doc_123" -> "doc_123_0")
        potential_ids = [id for id in self.corpus_doc_ids if id.startswith(missing_id + "_")]
        similar_ids.extend(potential_ids[:max_results])
        
        # 策略2: 寻找截断ID (例如 "doc_123_0" -> "doc_123")
        if "_" in missing_id:
            base_id = "_".join(missing_id.split("_")[:-1])
            if base_id in self.corpus_doc_ids:
                similar_ids.append(base_id)
        
        # 策略3: 模糊匹配
        if len(similar_ids) < max_results:
            base_pattern = missing_id.split("_")[0] + "_" + missing_id.split("_")[1] if "_" in missing_id else missing_id
            fuzzy_matches = [id for id in self.corpus_doc_ids if base_pattern in id]
            similar_ids.extend(fuzzy_matches[:max_results-len(similar_ids)])
        
        return similar_ids[:max_results]
    
    def analyze_missing_patterns(self):
        """分析缺失ID的模式"""
        missing_ids = self.all_qa_doc_ids - self.corpus_doc_ids
        
        # 按数据集分组
        dataset_patterns = defaultdict(list)
        for missing_id in missing_ids:
            dataset = missing_id.split("_")[0] if "_" in missing_id else "unknown"
            dataset_patterns[dataset].append(missing_id)
        
        print(f"\n🔍 缺失ID模式分析:")
        print("-" * 50)
        for dataset, ids in dataset_patterns.items():
            print(f"{dataset}: {len(ids)} 个缺失ID")
            if len(ids) <= 5:
                for id in ids:
                    print(f"  - {id}")
            else:
                for id in ids[:3]:
                    print(f"  - {id}")
                print(f"  ... 还有 {len(ids)-3} 个")
    
    def check_qa_completeness(self):
        """检查每个QA是否都有完整的文档支持"""
        print(f"\n🔍 检查QA完整性...")
        incomplete_qas = []
        
        for i, row in self.qa_df.iterrows():
            retrieval_gt = row['retrieval_gt']
            needed_ids = self._parse_retrieval_gt(retrieval_gt)
            missing_for_qa = needed_ids - self.corpus_doc_ids
            
            if missing_for_qa:
                incomplete_qas.append({
                    'qa_index': i,
                    'qid': row.get('qid', f'QA_{i}'),
                    'needed_ids': needed_ids,
                    'missing_ids': missing_for_qa,
                    'missing_count': len(missing_for_qa)
                })
        
        print(f"完整的QA: {len(self.qa_df) - len(incomplete_qas)}/{len(self.qa_df)}")
        print(f"不完整的QA: {len(incomplete_qas)}/{len(self.qa_df)}")
        
        if incomplete_qas:
            print(f"\n前5个不完整的QA:")
            for qa in incomplete_qas[:5]:
                print(f"  QA {qa['qid']}: 缺失 {qa['missing_count']} 个文档")
                for missing_id in list(qa['missing_ids'])[:3]:
                    print(f"    - {missing_id}")
        
        return incomplete_qas

def main():
    analyzer = QACorpusAnalyzer()
    
    print("🔍 QA数据与语料库匹配分析工具")
    print("=" * 50)
    
    while True:
        print("\n📋 可用命令:")
        print("1. load    - 加载数据文件")
        print("2. analyze - 分析ID匹配情况")
        print("3. missing - 显示缺失的ID")
        print("4. pattern - 分析缺失ID模式")
        print("5. complete - 检查QA完整性")
        print("6. search  - 搜索特定ID")
        print("7. similar - 为缺失ID寻找相似ID")
        print("8. quit    - 退出程序")
        
        command = input("\n请输入命令 (1-8): ").strip().lower()
        
        if command in ['1', 'load']:
            print("\n请输入文件路径 (或按回车使用默认路径):")
            qa_path = input("QA文件路径: ").strip()
            if not qa_path:
                qa_path = "/home/cz/AUTO_GRADON/data/5dataset_100/qa_last200.parquet"
            
            corpus_path = input("语料库文件路径: ").strip()
            if not corpus_path:
                corpus_path = "/home/cz/AUTO_GRADON/data/5dataset_100/corpus.parquet"
            
            analyzer.load_data(qa_path, corpus_path)
            
        elif command in ['2', 'analyze']:
            if analyzer.qa_df is None:
                print("❌ 请先加载数据!")
                continue
            analyzer.analyze_coverage()
            
        elif command in ['3', 'missing']:
            if analyzer.qa_df is None:
                print("❌ 请先加载数据!")
                continue
            limit = input("显示多少个缺失ID? (默认20): ").strip()
            limit = int(limit) if limit.isdigit() else 20
            analyzer.show_missing_ids(limit)
            
        elif command in ['4', 'pattern']:
            if analyzer.qa_df is None:
                print("❌ 请先加载数据!")
                continue
            analyzer.analyze_missing_patterns()
            
        elif command in ['5', 'complete']:
            if analyzer.qa_df is None:
                print("❌ 请先加载数据!")
                continue
            analyzer.check_qa_completeness()
            
        elif command in ['6', 'search']:
            if analyzer.qa_df is None:
                print("❌ 请先加载数据!")
                continue
            search_id = input("请输入要搜索的ID: ").strip()
            if search_id in analyzer.corpus_doc_ids:
                print(f"✅ ID '{search_id}' 在语料库中找到")
            elif search_id in analyzer.all_qa_doc_ids:
                print(f"❌ ID '{search_id}' 在QA数据中需要，但语料库中缺失")
            else:
                print(f"❓ ID '{search_id}' 在QA数据中未找到")
                
        elif command in ['7', 'similar']:
            if analyzer.qa_df is None:
                print("❌ 请先加载数据!")
                continue
            missing_id = input("请输入缺失的ID: ").strip()
            similar_ids = analyzer.find_similar_ids(missing_id)
            if similar_ids:
                print(f"为 '{missing_id}' 找到的相似ID:")
                for i, sim_id in enumerate(similar_ids, 1):
                    print(f"  {i}. {sim_id}")
            else:
                print(f"未找到与 '{missing_id}' 相似的ID")
                
        elif command in ['8', 'quit']:
            print("👋 再见!")
            break
            
        else:
            print("❌ 无效命令，请重新输入")

if __name__ == "__main__":
    main()
