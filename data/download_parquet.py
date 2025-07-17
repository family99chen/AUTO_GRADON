#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace RAG-Mini-BioASQ 数据集下载脚本

从 HuggingFace 下载 rag-mini-bioasq 数据集的 parquet 文件
包含两个子集：question-answer-passages 和 text-corpus
"""

import pandas as pd
import os
from pathlib import Path
from typing import Dict, List
import time

class HuggingFaceDatasetDownloader:
    def __init__(self, output_dir: str = "./rag-mini-bioasq"):
        """
        初始化下载器
        
        Args:
            output_dir: 输出目录，默认为当前目录下的 rag-mini-bioasq 文件夹
        """
        self.dataset_name = "rag-datasets/rag-mini-bioasq"
        self.output_dir = Path(output_dir)
        
        # 数据集配置 - 使用datasets库的方式，包含正确的split信息
        self.subsets = {
            "question-answer-passages": {
                "description": "问答对数据，包含问题、答案和相关文档ID",
                "rows": "4.72k",
                "split": "test"
            },
            "text-corpus": {
                "description": "文本语料库，包含文档内容",
                "rows": "40.2k",
                "split": "passages"
            }
        }
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 输出目录: {self.output_dir.absolute()}")

    def download_with_datasets_library(self, subset_name: str) -> bool:
        """
        使用datasets库下载数据
        
        Args:
            subset_name: 子集名称
        
        Returns:
            bool: 下载是否成功
        """
        try:
            from datasets import load_dataset
            
            subset_config = self.subsets[subset_name]
            split_name = subset_config["split"]
            
            print(f"\n🔄 使用datasets库下载: {subset_name}")
            print(f"   使用split: {split_name}")
            
            # 使用datasets库加载数据，使用正确的split
            dataset = load_dataset(self.dataset_name, subset_name, split=split_name)
            
            # 转换为pandas DataFrame
            df = dataset.to_pandas()
            
            # 保存到本地
            output_file = self.output_dir / f"{subset_name}.parquet"
            df.to_parquet(output_file, index=False)
            
            print(f"   ✅ 下载成功! 行数: {len(df):,}")
            print(f"   列名: {list(df.columns)}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ datasets库下载失败: {str(e)}")
            return False

    def download_with_direct_url(self, subset_name: str) -> bool:
        """
        使用直接URL下载数据
        
        Args:
            subset_name: 子集名称
        
        Returns:
            bool: 下载是否成功
        """
        try:
            print(f"\n🔄 使用直接URL下载: {subset_name}")
            
            subset_config = self.subsets[subset_name]
            split_name = subset_config["split"]
            
            # 尝试不同的URL格式，使用正确的split名称
            possible_urls = [
                f"hf://datasets/{self.dataset_name}/{subset_name}/{split_name}.parquet",
                f"hf://datasets/{self.dataset_name}/data/{subset_name}/{split_name}.parquet",
                f"hf://datasets/{self.dataset_name}/data/{split_name}-{subset_name}.parquet",
                f"hf://datasets/{self.dataset_name}/resolve/main/{subset_name}/{split_name}.parquet",
            ]
            
            for url in possible_urls:
                try:
                    print(f"   尝试URL: {url}")
                    df = pd.read_parquet(url)
                    
                    # 保存到本地
                    output_file = self.output_dir / f"{subset_name}.parquet"
                    df.to_parquet(output_file, index=False)
                    
                    print(f"   ✅ 下载成功! 行数: {len(df):,}")
                    print(f"   列名: {list(df.columns)}")
                    
                    return True
                    
                except Exception as e:
                    print(f"   ❌ 此URL失败: {str(e)}")
                    continue
            
            return False
            
        except Exception as e:
            print(f"   ❌ 直接URL下载失败: {str(e)}")
            return False

    def download_subset(self, subset_name: str) -> bool:
        """
        下载指定的数据子集
        
        Args:
            subset_name: 子集名称 ('question-answer-passages' 或 'text-corpus')
        
        Returns:
            bool: 下载是否成功
        """
        if subset_name not in self.subsets:
            print(f"❌ 未知的子集名称: {subset_name}")
            print(f"可用子集: {list(self.subsets.keys())}")
            return False
        
        subset_config = self.subsets[subset_name]
        output_file = self.output_dir / f"{subset_name}.parquet"
        
        print(f"\n🔄 开始下载: {subset_name}")
        print(f"   描述: {subset_config['description']}")
        print(f"   预计行数: {subset_config['rows']}")
        print(f"   split: {subset_config['split']}")
        print(f"   保存位置: {output_file}")
        
        start_time = time.time()
        
        # 方法1: 尝试使用datasets库
        if self.download_with_datasets_library(subset_name):
            elapsed_time = time.time() - start_time
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            print(f"   文件大小: {file_size:.2f} MB")
            print(f"   耗时: {elapsed_time:.2f} 秒")
            return True
        
        # 方法2: 尝试直接URL
        if self.download_with_direct_url(subset_name):
            elapsed_time = time.time() - start_time
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            print(f"   文件大小: {file_size:.2f} MB")
            print(f"   耗时: {elapsed_time:.2f} 秒")
            return True
        
        print(f"   ❌ 所有下载方法都失败了")
        return False

    def download_all(self) -> Dict[str, bool]:
        """
        下载所有子集
        
        Returns:
            Dict[str, bool]: 每个子集的下载结果
        """
        print("🚀 开始下载 RAG-Mini-BioASQ 数据集")
        print("=" * 60)
        
        results = {}
        for subset_name in self.subsets.keys():
            results[subset_name] = self.download_subset(subset_name)
        
        # 显示总结
        print("\n" + "=" * 60)
        print("📊 下载结果总结:")
        successful = 0
        for subset_name, success in results.items():
            status = "✅ 成功" if success else "❌ 失败"
            print(f"   {subset_name}: {status}")
            if success:
                successful += 1
        
        print(f"\n🎯 总体结果: {successful}/{len(self.subsets)} 个文件下载成功")
        
        if successful == len(self.subsets):
            print("🎉 所有文件下载完成!")
            self.show_usage_example()
        
        return results

    def show_usage_example(self):
        """显示使用示例"""
        print("\n" + "=" * 60)
        print("💡 使用示例:")
        print("=" * 60)
        
        for subset_name in self.subsets.keys():
            file_path = self.output_dir / f"{subset_name}.parquet"
            print(f"\n# 读取 {subset_name}")
            print(f"import pandas as pd")
            print(f"df = pd.read_parquet('{file_path}')")
            print(f"print(f'数据形状: {{df.shape}}')")
            print(f"print(df.head())")

    def verify_downloads(self) -> Dict[str, Dict]:
        """
        验证已下载的文件
        
        Returns:
            Dict: 每个文件的验证信息
        """
        print("\n🔍 验证已下载的文件:")
        print("=" * 50)
        
        verification_results = {}
        
        for subset_name in self.subsets.keys():
            file_path = self.output_dir / f"{subset_name}.parquet"
            
            if not file_path.exists():
                print(f"❌ {subset_name}: 文件不存在")
                verification_results[subset_name] = {"exists": False}
                continue
            
            try:
                # 读取文件验证
                df = pd.read_parquet(file_path)
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                
                verification_results[subset_name] = {
                    "exists": True,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns),
                    "file_size_mb": round(file_size, 2)
                }
                
                print(f"✅ {subset_name}:")
                print(f"   行数: {len(df):,}")
                print(f"   列数: {len(df.columns)}")
                print(f"   文件大小: {file_size:.2f} MB")
                print(f"   列名: {list(df.columns)}")
                
                # 显示数据预览
                if len(df) > 0:
                    print("   数据预览:")
                    print(f"   {df.head(2).to_string()}")
                
            except Exception as e:
                print(f"❌ {subset_name}: 文件损坏 - {str(e)}")
                verification_results[subset_name] = {"exists": True, "valid": False, "error": str(e)}
        
        return verification_results

    def test_datasets_library(self) -> bool:
        """
        测试datasets库是否可用
        
        Returns:
            bool: 是否可用
        """
        try:
            from datasets import load_dataset
            print("✅ datasets库可用")
            return True
        except ImportError:
            print("⚠️  datasets库未安装")
            return False

    def show_dataset_info(self):
        """显示数据集信息"""
        print("\n📋 数据集信息:")
        print("=" * 50)
        
        for subset_name, config in self.subsets.items():
            print(f"\n📂 {subset_name}:")
            print(f"   描述: {config['description']}")
            print(f"   预计行数: {config['rows']}")
            print(f"   split: {config['split']}")

def main():
    """主函数"""
    print("🤗 HuggingFace RAG-Mini-BioASQ 数据集下载工具")
    print("=" * 60)
    
    # 获取用户输入的输出目录
    while True:
        output_dir = input("\n请输入输出目录 (直接回车使用默认 './rag-mini-bioasq'): ").strip()
        if not output_dir:
            output_dir = "./rag-mini-bioasq"
        
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            break
        except Exception as e:
            print(f"❌ 无法创建目录 {output_dir}: {e}")
            continue
    
    # 创建下载器
    downloader = HuggingFaceDatasetDownloader(output_dir)
    
    # 测试datasets库
    downloader.test_datasets_library()
    
    # 显示数据集信息
    downloader.show_dataset_info()
    
    while True:
        print("\n📋 可用操作:")
        print("1. 下载所有数据集")
        print("2. 下载 question-answer-passages")
        print("3. 下载 text-corpus")
        print("4. 验证已下载的文件")
        print("5. 显示使用示例")
        print("6. 显示数据集信息")
        print("7. 退出")
        
        choice = input("\n请选择操作 (1-7): ").strip()
        
        if choice == '1':
            downloader.download_all()
        elif choice == '2':
            downloader.download_subset('question-answer-passages')
        elif choice == '3':
            downloader.download_subset('text-corpus')
        elif choice == '4':
            downloader.verify_downloads()
        elif choice == '5':
            downloader.show_usage_example()
        elif choice == '6':
            downloader.show_dataset_info()
        elif choice == '7':
            print("👋 再见!")
            break
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    # 检查依赖
    try:
        import pandas as pd
        print("✅ pandas 已安装")
        print(f"   pandas 版本: {pd.__version__}")
    except ImportError:
        print("❌ 请先安装 pandas: pip install pandas")
        exit(1)
    
    # 检查其他依赖
    try:
        import fsspec
        print(f"✅ fsspec 已安装，版本: {fsspec.__version__}")
    except ImportError:
        print("⚠️  建议安装 fsspec: pip install fsspec[http]")
    
    try:
        import huggingface_hub
        print(f"✅ huggingface_hub 已安装，版本: {huggingface_hub.__version__}")
    except ImportError:
        print("⚠️  建议安装 huggingface_hub: pip install huggingface_hub")
    
    try:
        import datasets
        print(f"✅ datasets 已安装，版本: {datasets.__version__}")
    except ImportError:
        print("⚠️  建议安装 datasets: pip install datasets")
        print("   这是HuggingFace的官方库，可以更可靠地下载数据集")
    
    main()
