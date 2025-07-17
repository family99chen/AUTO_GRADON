import os
import torch
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time
from tqdm import tqdm, trange
import logging
from datetime import datetime
from contextlib import contextmanager
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import inspect

# 设置CUDA环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # 明确指定GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 修改CUDA架构设置
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;9.0"  # 使用兼容的架构
os.environ["CUDA_MODULE_LOADING"] = "EAGER"  # 改为EAGER加载
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

# 修改检查CUDA环境的函数
def check_cuda_setup():
    if torch.cuda.is_available():
        # 打印CUDA信息
        print(f"CUDA 是否可用: {torch.cuda.is_available()}")
        print(f"当前CUDA版本: {torch.version.cuda}")
        print(f"当前设备数量: {torch.cuda.device_count()}")
        print(f"PyTorch版本: {torch.__version__}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"设备 {i}: {torch.cuda.get_device_name(i)}")
            print(f"设备 {i} 计算能力: {props.major}.{props.minor}")
            print(f"设备 {i} 内存: {props.total_memory/1e9:.2f}GB")
        
        # 测试CUDA功能
        try:
            # 使用CPU创建张量然后移动到GPU
            x = torch.randn(10)
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                x = x.to(device)
                y = x + x
                # 确保计算完成
                torch.cuda.synchronize()
            print("CUDA 基本运算测试通过")
            return True
        except Exception as e:
            print(f"CUDA 测试失败: {str(e)}")
            print(f"详细错误信息: {torch.cuda.get_device_properties(0)}")
            return False
    return False

# 在程序开始时检查CUDA设置，但不立即退出
if not check_cuda_setup():
    print("警告：CUDA 环境检查失败，将尝试使用CPU运行")
    # 不抛出异常，而是继续运行

# 导入自定义模块
from configuration.promptmaker import PromptMakerConfiguration
from create_config import create_config, load_and_generate_nodes
from evaluator import TestEvaluator
from AUTO_GRADON.ragas.grpo_policy_based_on_env import GRPOTrainer

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing
# 设置Python标准多进程为spawn
multiprocessing.set_start_method('spawn', force=True)

# 添加新的CUDA设置函数
def setup_cuda_device(gpu_id: int):
    """设置CUDA设备并进行必要的初始化"""
    if torch.cuda.is_available():
        try:
            device_id = gpu_id % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            # 设置CUDA性能优化
            torch.backends.cudnn.benchmark = False  # 关闭自动优化
            torch.backends.cudnn.deterministic = True  # 确保结果可重现
            torch.backends.cuda.matmul.allow_tf32 = False  # 禁用TF32
            torch.backends.cudnn.allow_tf32 = False  # 禁用TF32
            
            return device_id
        except Exception as e:
            print(f"CUDA设备{gpu_id}初始化失败: {str(e)}")
            return None
    return None

class RAGOptimizer:
    """使用GRPO优化RAG系统的多组件配置"""
    
    # 所有可能的组件列表
    ALL_COMPONENTS = [
        "vectordb", 
        "query_expansion",
        "retrieval",
        "passage_augmenter",
        "passage_reranker",
        "passage_filter",
        "passage_compressor",
        "prompt_maker",
        "generator"
    ]
    
    @contextmanager
    def timer(self, name):
        start = time.time()
        yield
        end = time.time()
        self.logger.info(f"{name} 耗时: {end - start:.2f} 秒")

    def __init__(self, 
                 qa_data_path: str, 
                 corpus_data_path: str, 
                 project_dir: str,
                 config_path: str = "/home/cz/AUTO_GRADON/ragas/configuration/config204.yaml",
                 trial_name: str = "rag_optimization",
                 num_gpus: int = 1,
                 target_components: Optional[List[str]] = None,
                 fixed_components: Optional[Dict[str, str]] = None,
                 use_cache: bool = True):
        """
        初始化优化器
        
        Args:
            qa_data_path: 问答数据路径
            corpus_data_path: 语料库数据路径
            project_dir: 项目目录
            config_path: 配置文件路径
            trial_name: 试验名称
            num_gpus: 可用GPU数量
            target_components: 需要优化的组件列表，如果为None则优化除generator外的所有组件
            fixed_components: 固定组件的配置，格式为 {"组件名": "方法名"}
            use_cache: 是否使用评估缓存
        """
        # 设置GPU数量
        self.num_gpus = num_gpus
        self.logger_initialized = False
        
        # 初始化评估器
        self.evaluator = TestEvaluator(qa_data_path, corpus_data_path, project_dir)
        self.evaluator.init_trial(trial_name)
        
        # 保存路径 
        self.config_path = config_path
        self.project_dir = project_dir
        self.trial_dir = os.path.join(project_dir, trial_name)
        os.makedirs(os.path.join(self.trial_dir, "configs"), exist_ok=True)
        
        # 设置日志
        self._setup_logger(trial_name)
        
        # 加载基础配置
        self.base_config = self._load_base_config()
        
        # 设置目标组件和固定组件
        self.fixed_components = fixed_components or {}
        self._setup_target_components(target_components)
        
        # 初始化节点的配置
        self._initialize_node_configs()
        
        # 设置缓存
        self.use_cache = use_cache
        if self.use_cache:
            self._initialize_cache()
        
    def _setup_logger(self, trial_name):
        """设置日志器"""
        log_dir = os.path.join(self.trial_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'grpo_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"日志文件保存在: {log_file}")
        self.logger_initialized = True
        
    def _load_base_config(self) -> Dict:
        """加载基础配置"""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)
            
    def _setup_target_components(self, target_components: Optional[List[str]]):
        """设置目标优化组件"""
        if target_components is None:
            # 默认优化除generator外的所有组件
            self.components = [c for c in self.ALL_COMPONENTS if c != "generator" and c in self.base_config]
        else:
            # 验证用户提供的组件是否有效
            valid_components = []
            for component in target_components:
                if component in self.ALL_COMPONENTS and component in self.base_config:
                    valid_components.append(component)
                else:
                    if self.logger_initialized:
                        self.logger.warning(f"组件 '{component}' 无效或不在配置文件中，将被忽略")
            
            self.components = valid_components
            
        # 从组件列表中移除固定组件
        self.components = [c for c in self.components if c not in self.fixed_components]
        
        if self.logger_initialized:
            self.logger.info(f"将优化以下 {len(self.components)} 个组件: {self.components}")
            if self.fixed_components:
                self.logger.info(f"以下组件将使用固定配置: {self.fixed_components}")
    
    def _initialize_node_configs(self):
        """初始化所有节点的配置选项"""
        # 初始化节点配置
        self.nodes_config = {}
        self.method_counts = {}  # 存储每个节点的方法数量
        
        for idx, component in enumerate(self.components):
            node_key = f"node{idx+1}"
            
            # 获取组件的方法列表
            methods = self.base_config[component].get("method", [])
            
            # 确保是列表
            if not isinstance(methods, list):
                methods = [methods]
                
            # 记录方法数量
            self.method_counts[node_key] = len(methods)
            
            # 创建节点配置映射
            self.nodes_config[node_key] = {
                str(i): method for i, method in enumerate(methods)
            }
        
        self.logger.info(f"初始化了{len(self.components)}个节点的配置:")
        for node, methods in self.nodes_config.items():
            self.logger.info(f"  {node}: {methods}")
    
    def setup_grpo_trainer(self, d_model: int = 256, nhead: int = 8, num_layers: int = 3):
        """
        设置GRPO训练器
        
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
        """
        # 确定每个节点的操作数量
        operations = [self.method_counts[f"node{i+1}"] for i in range(len(self.components))]
        
        # 取最大操作数量作为通用操作数
        #max_operation = max(operations)
        max_operation = 4
        
        self.logger.info(f"设置GRPO训练器，共{len(self.components)}个节点，操作维度: {operations}")
        self.logger.info(f"使用最大操作数 {max_operation} 作为统一操作数")
        
        self.trainer = GRPOTrainer(
            num_process=len(self.components),  # 优化指定数量的节点
            d_model=d_model,
            nhead=nhead, 
            num_layers=num_layers,
            operation=max_operation,  # 使用最大操作数
            kl_coeff=0.01,
            clip_eps=0.2
        )
        
        # 存储每个节点的实际操作数以便后续处理
        self.node_operations = operations
        
        self.logger.info(f"GRPO训练器初始化完成，策略网络结构: d_model={d_model}, nhead={nhead}, layers={num_layers}")
        return self.trainer
    
    def generate_config(self, actions: np.ndarray) -> str:
        """
        根据动作生成配置文件
        
        Args:
            actions: 动作数组，长度为组件数量
            
        Returns:
            配置文件路径
        """
        # 🔧 调试：打印 create_config 函数的签名
        sig = inspect.signature(create_config)
        self.logger.info(f"[DEBUG] create_config 函数签名: {sig}")
        
        # 🔧 添加详细调试信息
        self.logger.info(f"[DEBUG generate_config] 开始生成配置")
        self.logger.info(f"[DEBUG generate_config] 输入actions: {actions}")
        self.logger.info(f"[DEBUG generate_config] actions类型: {type(actions)}")
        self.logger.info(f"[DEBUG generate_config] actions长度: {len(actions)}")
        self.logger.info(f"[DEBUG generate_config] 组件列表: {self.components}")
        self.logger.info(f"[DEBUG generate_config] 组件数量: {len(self.components)}")
        
        # 创建一个唯一的配置文件名
        config_id = "_".join([str(a) for a in actions])
        config_file = os.path.join(self.trial_dir, "configs", f"config_{config_id}.yaml")
        
        self.logger.info(f"[DEBUG generate_config] 配置文件路径: {config_file}")
        
        # 获取每个节点选择的方法
        selected_methods = {}
        
        # 🔧 只遍历前len(actions)个组件，跳过最后一个（generator）
        components_to_process = self.components[:len(actions)]
        self.logger.info(f"[DEBUG generate_config] 实际处理的组件: {components_to_process}")
        
        for i, component in enumerate(components_to_process):
            # 🔧 添加详细调试信息
            self.logger.info(f"[DEBUG generate_config] 处理组件 {i}: {component}")
            
            if i >= len(actions):
                self.logger.error(f"[DEBUG generate_config] 动作索引 {i} 超出范围，actions长度: {len(actions)}")
                return None
            
            action_idx = actions[i]
            self.logger.info(f"[DEBUG generate_config] 组件 {component} 的动作索引: {action_idx}")
            
            node_key = f"node{i+1}"
            self.logger.info(f"[DEBUG generate_config] 节点键: {node_key}")
            
            # 🔧 检查节点配置是否存在
            if node_key not in self.nodes_config:
                self.logger.error(f"[DEBUG generate_config] 节点 {node_key} 不在 nodes_config 中")
                self.logger.error(f"[DEBUG generate_config] 可用节点: {list(self.nodes_config.keys())}")
                return None
            
            node_config = self.nodes_config[node_key]
            self.logger.info(f"[DEBUG generate_config] 节点 {node_key} 的配置: {node_config}")
            
            action_str = str(action_idx)
            self.logger.info(f"[DEBUG generate_config] 查找动作字符串: '{action_str}'")
            
            # 🔧 检查动作索引是否在节点配置中
            if action_str not in node_config:
                self.logger.error(f"[DEBUG generate_config] 动作 '{action_str}' 不在节点 {node_key} 的配置中")
                self.logger.error(f"[DEBUG generate_config] 可用动作: {list(node_config.keys())}")
                return None
            
            selected_method = node_config[action_str]
            selected_methods[component] = selected_method
            self.logger.info(f"[DEBUG generate_config] 组件 {component} 选择方法: {selected_method}")
        
        # 添加固定组件配置
        for component, method in self.fixed_components.items():
            selected_methods[component] = method
            self.logger.info(f"[DEBUG generate_config] 添加固定组件 {component}: {method}")
        
        self.logger.info(f"[DEBUG generate_config] 最终选择的方法: {selected_methods}")
        
        # 准备生成配置所需的node_lines
        node_lines_list = []
        vectordb_line = None
        retriever_cfg = None
        
        # 首先处理vectordb，因为它需要作为create_config的第一个参数
        try:
            self.logger.info(f"[DEBUG generate_config] 开始生成vectordb配置")
            vectordb_lines, _ = load_and_generate_nodes(
                self.config_path, 
                "vectordb", 
                size=1, 
                exhaustive=False
            )
            vectordb_line = vectordb_lines[0]  # 使用第一个vectordb配置
            self.logger.info(f"[DEBUG generate_config] vectordb配置生成成功")
        except Exception as e:
            self.logger.error(f"[DEBUG generate_config] 生成vectordb配置失败: {str(e)}")
            return None  # 如果vectordb配置失败，无法继续
        
        # 为每个组件生成配置
        for component in self.ALL_COMPONENTS:
            if component == "vectordb":  # 已经处理过
                continue
            
            if component not in self.base_config:
                self.logger.info(f"[DEBUG generate_config] 组件 {component} 不在基础配置中，跳过")
                continue
            
            # 获取该组件选择的方法
            method = selected_methods.get(component)
            self.logger.info(f"[DEBUG generate_config] 处理组件 {component}，选择的方法: {method}")
            
            try:
                # 生成多个配置以增加找到匹配方法的机会
                self.logger.info(f"[DEBUG generate_config] 为组件 {component} 生成配置")
                lines, cfg = load_and_generate_nodes(
                    self.config_path,
                    component,
                    size=10,  # 生成更多配置以增加匹配成功率
                    exhaustive=True  # 使用穷举模式
                )
                
                # 如果是retrieval组件，保存配置对象以便后续提取bm25_tokenizer
                if component == "retrieval":
                    retriever_cfg = cfg
                
                # 确保lines是列表
                if not isinstance(lines, list):
                    lines = [lines]
                
                self.logger.info(f"[DEBUG generate_config] 组件 {component} 生成了 {len(lines)} 个配置")
                
                # 如果指定了方法，尝试找到匹配的配置行
                if method:
                    matched_line = None
                    
                    for idx, line in enumerate(lines):
                        # 尝试将line转为字符串以进行匹配检查
                        line_str = str(line)
                        
                        # 根据不同组件的配置格式，检查方法名是否在配置中
                        if f"module_type: {method}" in line_str or \
                           f"\"{method}\"" in line_str or \
                           f"'{method}'" in line_str:
                            matched_line = line
                            self.logger.info(f"[DEBUG generate_config] 为组件 {component} 找到匹配方法 {method} 的配置 (索引 {idx})")
                            break
                    
                    # 如果找到匹配，使用它；否则使用第一个配置
                    if matched_line:
                        node_lines_list.append(matched_line)
                    else:
                        node_lines_list.append(lines[0])
                        self.logger.warning(f"[DEBUG generate_config] 无法为组件 {component} 找到匹配方法 {method} 的配置，使用默认配置")
                else:
                    # 如果没有指定方法，使用第一个配置
                    node_lines_list.append(lines[0])
                    self.logger.info(f"[DEBUG generate_config] 组件 {component} 没有指定方法，使用第一个配置")
                
            except Exception as e:
                self.logger.error(f"[DEBUG generate_config] 为组件 {component} 生成配置失败: {str(e)}")
                # 尝试获取一个基本配置
                try:
                    basic_lines, _ = load_and_generate_nodes(
                        self.config_path,
                        component,
                        size=1,
                        exhaustive=False
                    )
                    if isinstance(basic_lines, list) and basic_lines:
                        node_lines_list.append(basic_lines[0])
                    elif basic_lines:
                        node_lines_list.append(basic_lines)
                    self.logger.info(f"[DEBUG generate_config] 为组件 {component} 使用基本配置")
                except Exception as inner_e:
                    self.logger.error(f"[DEBUG generate_config] 无法为组件 {component} 获取基本配置: {str(inner_e)}")
                    return None
        
        self.logger.info(f"[DEBUG generate_config] 总共生成了 {len(node_lines_list)} 个组件配置")
        
        # 准备额外参数
        extra_params = {
            'strategies': {'metrics': ['meteor', 'rouge', 'bert_score']},
            'bm25_tokenizer_list': ['porter_stemmer', 'space']
        }
        
        # 处理bm25_tokenizer特殊情况
        if retriever_cfg and "bm25" in selected_methods.get("retrieval", ""):
            if hasattr(retriever_cfg, 'cs') and '[bm25]bm25_tokenizer' in retriever_cfg.cs:
                extra_params['bm25_tokenizer_list'] = retriever_cfg.cs.get('[bm25]bm25_tokenizer').choices
            else:
                self.logger.warning(f"[DEBUG generate_config] 无法获取bm25_tokenizer配置")
        
        try:
            self.logger.info(f"[DEBUG generate_config] 开始调用create_config")
            self.logger.info(f"[DEBUG generate_config] vectordb_line类型: {type(vectordb_line)}")
            self.logger.info(f"[DEBUG generate_config] node_lines_list长度: {len(node_lines_list)}")
            
            # 生成最终配置
            create_config(
                vectordb_line,
                *node_lines_list,
                extra_params=extra_params,
                save_path=config_file
            )
            
            self.logger.info(f"[DEBUG generate_config] 配置文件生成成功: {config_file}")
            return config_file
            
        except Exception as e:
            self.logger.error(f"[DEBUG generate_config] 调用create_config失败: {str(e)}")
            self.logger.error(f"[DEBUG generate_config] 错误详情: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"[DEBUG generate_config] 完整错误堆栈:\n{traceback.format_exc()}")
            return None
    
    def _verify_config_file(self, config_file, selected_methods):
        """验证生成的配置文件是否包含预期的方法"""
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                
            missing_methods = []
            for component, method in selected_methods.items():
                if method not in content:
                    missing_methods.append(f"{component}:{method}")
                
            if missing_methods:
                self.logger.warning(f"配置文件可能未正确包含以下组件的方法: {', '.join(missing_methods)}")
            else:
                self.logger.info("配置文件验证通过：所有选定的方法都在配置中")
                
        except Exception as e:
            self.logger.error(f"验证配置文件时出错: {str(e)}")
    
    def _evaluate_batch(self, actions_batch, gpu_id: int) -> List[float]:
        """在特定GPU上评估一批配置"""
        rewards = []
        
        # 🔧 添加调试信息
        self.logger.info(f"[DEBUG] actions_batch 形状: {np.array(actions_batch).shape}")
        self.logger.info(f"[DEBUG] actions_batch 内容: {actions_batch}")
        
        try:
            # 设置CUDA设备
            if torch.cuda.is_available():
                device_id = gpu_id % torch.cuda.device_count()
                torch.cuda.set_device(device_id)
                
                # 测试CUDA设备
                test_tensor = torch.zeros(1, device=f'cuda:{device_id}')
                del test_tensor
                
                self.logger.info(f"[GPU {gpu_id}] 成功初始化 CUDA:{device_id}")
        except Exception as e:
            self.logger.error(f"[GPU {gpu_id}] 设备初始化失败: {str(e)}")
            return [0.0] * len(actions_batch)

        for i, actions in enumerate(actions_batch):
            try:
                # 🔧 添加调试信息
                self.logger.info(f"[DEBUG] 处理第 {i+1} 个配置")
                self.logger.info(f"[DEBUG] 当前actions: {actions}, 类型: {type(actions)}")
                
                # 🔧 动作映射逻辑
                mapped_actions = actions
                
                if mapped_actions is None:
                    self.logger.error(f"[GPU {gpu_id}] 配置 {i+1} 映射失败")
                    rewards.append(0.0)
                    continue
                
                # 生成配置文件
                config_file = self.generate_config(mapped_actions)
                self.logger.info(f"[DEBUG] 生成的配置文件: {config_file}")
                
                # 确保GPU内存清理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device_id)
                
                # 使用try-except包装每个可能使用CUDA的操作
                try:
                    # 初始化评估器
                    self.evaluator.init_runner_from_yaml(config_file)
                    
                    # 评估配置
                    yaml_name = f"config_{'_'.join([str(a) for a in mapped_actions])}.yaml"
                    with torch.cuda.amp.autocast(enabled=False):
                        summary_df = self.evaluator.run_with_qa_eval(yaml_name=yaml_name)
                    
                    # 计算奖励
                    rouge_score = summary_df["rouge"].values[0]
                    meteor_score = summary_df["meteor"].values[0]
                    bert_score = summary_df.get("bert_score", pd.Series([0])).values[0]
                    
                    raw_reward = 0.4 * rouge_score + 0.4 * meteor_score + 0.2 * bert_score
                    transformed_reward = np.exp(20 * raw_reward)
                    rewards.append(transformed_reward)
                    
                except RuntimeError as cuda_e:
                    if "CUDA error" in str(cuda_e):
                        self.logger.error(f"[GPU {gpu_id}] CUDA运行时错误: {str(cuda_e)}")
                        # 尝试重置设备
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device_id)
                        torch.cuda.set_device(device_id)
                    rewards.append(0.0)
                
            except Exception as e:
                self.logger.error(f"[GPU {gpu_id}] 评估配置 {i+1} 时出错: {str(e)}")
                rewards.append(0.0)
                
                # 尝试恢复GPU状态
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(device_id)
                        torch.cuda.set_device(device_id)
                    except:
                        pass
        
        return rewards
    
    def _initialize_cache(self):
        """初始化评估结果缓存"""
        # 缓存文件路径
        self.cache_file = os.path.join(self.trial_dir, "evaluation_cache.json")
        
        # 尝试加载现有缓存
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.evaluation_cache = json.load(f)
                self.logger.info(f"加载评估缓存，包含 {len(self.evaluation_cache)} 条记录")
            except Exception as e:
                self.logger.warning(f"加载缓存失败: {str(e)}，将创建新缓存")
                self.evaluation_cache = {}
        else:
            self.evaluation_cache = {}
        
        # 缓存命中统计
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 在_initialize_cache方法中添加
        self.use_raw_rewards_in_cache = False  # 标记缓存中存储的是变换后的奖励

        # 如果是旧缓存，可能需要添加迁移逻辑
        if os.path.exists(self.cache_file) and not self.use_raw_rewards_in_cache:
            try:
                # 迁移旧缓存
                with open(self.cache_file, 'r') as f:
                    old_cache = json.load(f)
                
                # 转换奖励
                new_cache = {}
                for key, value in old_cache.items():
                    new_cache[key] = np.exp(20 * float(value))
                
                self.evaluation_cache = new_cache
                self.logger.info(f"已将{len(old_cache)}条缓存记录从原始奖励转换为非线性奖励")
            except Exception as e:
                self.logger.error(f"缓存迁移失败: {str(e)}")

    def _save_cache(self):
        """保存评估缓存到文件"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.evaluation_cache, f)
            self.logger.info(f"评估缓存已保存，共 {len(self.evaluation_cache)} 条记录")
        except Exception as e:
            self.logger.error(f"保存缓存失败: {str(e)}")

    def evaluate_configs(self, actions_batch) -> List[float]:
        """评估多个配置，支持多GPU并行"""
        total_start = time.time()
        
        # 将动作转换为可哈希的格式用于缓存查找
        actions_tuples = [tuple(map(int, action)) for action in actions_batch]
        
        # 检查哪些配置需要评估，哪些可以从缓存中获取
        cache_keys = []
        need_evaluate_indices = []
        need_evaluate_actions = []
        
        # 打印每个配置对应的策略
        for i, action_tuple in enumerate(actions_tuples):
            # 获取当前动作对应的策略
            selected_methods = {}
            for j, component_idx in enumerate(action_tuple):
                if j < len(self.components):
                    component = self.components[j]
                    node_key = f"node{j+1}"
                    action_idx = str(component_idx)
                    if action_idx in self.nodes_config[node_key]:
                        selected_methods[component] = self.nodes_config[node_key][action_idx]
            
            # 添加固定组件配置
            for component, method in self.fixed_components.items():
                selected_methods[component] = method
            
            # 生成缓存键
            cache_key = str(action_tuple)
            cache_keys.append(cache_key)
            
            # 检查是否命中缓存
            if cache_key in self.evaluation_cache:
                self.cache_hits += 1
                # 打印命中缓存的策略和奖励值
                reward = self.evaluation_cache[cache_key]
                self.logger.info(f"配置 {i+1} 命中缓存: 策略={selected_methods}, 奖励={reward:.4f}")
            else:
                self.cache_misses += 1
                need_evaluate_indices.append(i)
                need_evaluate_actions.append(actions_batch[i])
        
        self.logger.info(f"开始评估 {len(actions_batch)} 个配置，其中 {len(need_evaluate_actions)} 个需要评估，{len(actions_batch) - len(need_evaluate_actions)} 个使用缓存")
        
        # 如果有需要评估的配置
        if need_evaluate_actions:
            need_evaluate_actions = np.array(need_evaluate_actions)
            
            if self.num_gpus <= 1:
                new_rewards = self._evaluate_batch(need_evaluate_actions, 0)
            else:
                # 减小每个GPU的批次大小
                batch_size = max(1, len(need_evaluate_actions) // (self.num_gpus * 2))
                batches = [need_evaluate_actions[i:i + batch_size] 
                          for i in range(0, len(need_evaluate_actions), batch_size)]
                
                new_rewards = []
                # 串行处理每个GPU的批次
                for i, batch in enumerate(batches):
                    gpu_id = i % self.num_gpus
                    batch_rewards = self._evaluate_batch(batch, gpu_id)
                    new_rewards.extend(batch_rewards)
                    
                    # 每个批次后强制同步和清理
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
            
            # 将新评估的结果添加到缓存
            for i, idx in enumerate(need_evaluate_indices):
                self.evaluation_cache[cache_keys[idx]] = new_rewards[i]
            
            # 定期保存缓存
            if self.cache_misses % 10 == 0:
                self._save_cache()
        
        # 组装所有结果（从缓存中获取和新评估的）
        rewards = []
        for cache_key in cache_keys:
            rewards.append(self.evaluation_cache[cache_key])
        
        total_time = time.time() - total_start
        self.logger.info(f"批次评估完成，总耗时: {total_time:.2f} 秒，平均每个配置耗时: {total_time/len(actions_batch):.2f} 秒")
        self.logger.info(f"缓存统计 - 命中: {self.cache_hits}, 未命中: {self.cache_misses}, 命中率: {self.cache_hits/(self.cache_hits+self.cache_misses):.2%}")
        
        # 汇总评估结果，按照奖励值从高到低排序并显示策略
        self._summarize_batch_results(actions_batch, rewards)
        
        return rewards

    def _summarize_batch_results(self, actions_batch, rewards):
        """汇总批次评估结果，显示每个策略的奖励值"""
        # 创建策略-奖励对
        strategy_reward_pairs = []
        
        for i, actions in enumerate(actions_batch):
            # 获取策略
            selected_methods = {}
            for j, action in enumerate(actions):
                if j < len(self.components):
                    component = self.components[j]
                    node_key = f"node{j+1}"
                    action_idx = str(int(action))
                    if action_idx in self.nodes_config[node_key]:
                        selected_methods[component] = self.nodes_config[node_key][action_idx]
            
            # 添加固定组件配置
            for component, method in self.fixed_components.items():
                selected_methods[component] = method
            
            strategy_reward_pairs.append((selected_methods, rewards[i]))
        
        # 按奖励值从高到低排序
        strategy_reward_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 输出排序后的结果
        self.logger.info(f"本批次评估结果汇总 (共{len(strategy_reward_pairs)}个配置):")
        for i, (strategy, reward) in enumerate(strategy_reward_pairs[:5]):  # 只显示前5个
            self.logger.info(f"  排名 {i+1}: 奖励={reward:.4f}, 策略={strategy}")
        
        # 如果有更多配置，简单提示
        if len(strategy_reward_pairs) > 5:
            self.logger.info(f"  ... 以及 {len(strategy_reward_pairs)-5} 个更多配置")
    
    def generate_actions_with_elite(self, num_samples, epsilon):
        # 如果已经有最佳动作，生成(num_samples-1)个新样本+1个精英样本
        if self.trainer.best_actions is not None:
            actions = self.trainer.generate_actions(num_samples-1, epsilon)
            actions_np = actions.numpy()
            best_action = np.array([self.trainer.best_actions])
            actions_np = np.vstack([actions_np, best_action])
        # 否则生成全部num_samples个样本
        else:
            actions = self.trainer.generate_actions(num_samples, epsilon)
            actions_np = actions.numpy()
        
        # 裁剪动作到有效范围
        for sample_idx in range(actions_np.shape[0]):
            for node_idx in range(actions_np.shape[1]):
                # 确保动作不超过该节点的实际操作数
                actions_np[sample_idx, node_idx] = min(
                    actions_np[sample_idx, node_idx],
                    self.node_operations[node_idx] - 1
                )
        
        return actions_np

    def run_optimization(self, num_epochs: int = 20, batch_size: int = 20):
        """
        运行优化流程
        
        Args:
            num_epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            tuple: (最佳配置, 最佳奖励)
        """
        optimization_start = time.time()
        self.logger.info(f"开始GRPO优化，{len(self.components)}个组件，{num_epochs}轮训练，批量大小{batch_size}")
        
        with self.timer("设置GRPO训练器"):
            self.setup_grpo_trainer()
        
        history = {
            "epoch": [],
            "avg_reward": [],
            "max_reward": [],
            "best_actions": [],
            "best_config": [],
            "epoch_time": []
        }
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            self.logger.info(f"\n{'='*20} Epoch {epoch+1}/{num_epochs} {'='*20}")
            
            # 自适应探索率：从大到小逐渐衰减
            epsilon = max(0.7 * (1.0 - epoch / num_epochs), 0.05)
            self.logger.info(f"当前探索率 (epsilon): {epsilon:.3f}")
            
            with self.timer(f"Epoch {epoch+1} 生成动作样本"):
                actions_np = self.generate_actions_with_elite(batch_size, epsilon)
            
            with self.timer(f"Epoch {epoch+1} 评估配置"):
                rewards = self.evaluate_configs(actions_np)
                if rewards:
                    self.logger.info(f"本批次奖励值统计: 最小={min(rewards):.4f}, 最大={max(rewards):.4f}, 平均={np.mean(rewards):.4f}")
                else:
                    self.logger.warning("本批次没有有效的奖励值")
                    continue
            
            with self.timer(f"Epoch {epoch+1} 更新策略"):
                metrics = self.trainer.update_policy(actions_np, rewards)
            
            epoch_time = time.time() - epoch_start
            history["epoch"].append(epoch)
            history["avg_reward"].append(metrics["avg_reward"])
            history["max_reward"].append(metrics["max_reward"])
            
            # 修复这里，检查是列表还是numpy数组，适当处理
            if isinstance(metrics["best_actions"], np.ndarray):
                history["best_actions"].append(metrics["best_actions"].tolist())
            else:
                # 如果已经是列表，直接添加
                history["best_actions"].append(metrics["best_actions"])
            
            # 创建最佳配置的描述
            best_config = {}
            for i, component in enumerate(self.components):
                action_idx = str(metrics["best_actions"][i])
                node_key = f"node{i+1}"
                if action_idx in self.nodes_config[node_key]:
                    best_config[component] = self.nodes_config[node_key][action_idx]
            
            # 添加固定组件配置
            for component, method in self.fixed_components.items():
                best_config[component] = method
            
            history["best_config"].append(best_config)
            history["epoch_time"].append(epoch_time)
            
            self.logger.info(f"Epoch {epoch+1} 统计:")
            self.logger.info(f"  - 平均奖励: {metrics['avg_reward']:.4f}")
            self.logger.info(f"  - 最大奖励: {metrics['max_reward']:.4f}")
            self.logger.info(f"  - 当前最佳动作: {metrics['best_actions']}")
            self.logger.info(f"  - 当前最佳配置: {best_config}")
            self.logger.info(f"  - 本轮耗时: {epoch_time:.2f} 秒")
            
            # 打印当前策略分布
            current_policy = self.trainer.generate_actions(1, 0).tolist()
            self.logger.info(f"  - 当前策略分布: {current_policy}")
            
            # 保存当前最佳模型
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                checkpoint_path = os.path.join(self.trial_dir, f"checkpoint_epoch_{epoch}.pt")
                state_dict = {
                    'policy_state': self.trainer.policy.state_dict(),
                    'old_policy_state': self.trainer.old_policy.state_dict(),
                    'best_actions': self.trainer.best_actions,
                    'best_reward': self.trainer.best_reward
                }
                torch.save(state_dict, checkpoint_path)
                self.logger.info(f"保存检查点到: {checkpoint_path}")
        
        total_time = time.time() - optimization_start
        self.logger.info(f"\n优化完成!")
        self.logger.info(f"总耗时: {total_time:.2f} 秒")
        self.logger.info(f"平均每轮耗时: {total_time/num_epochs:.2f} 秒")
        
        # 获取最终的最佳配置
        best_actions = self.trainer.best_actions
        best_reward = self.trainer.best_reward
        
        best_config = {}
        for i, component in enumerate(self.components):
            action_idx = str(best_actions[i])
            node_key = f"node{i+1}"
            if action_idx in self.nodes_config[node_key]:
                best_config[component] = self.nodes_config[node_key][action_idx]
        
        # 添加固定组件配置
        for component, method in self.fixed_components.items():
            best_config[component] = method
        
        self.logger.info(f"最佳动作: {best_actions}")
        self.logger.info(f"最佳配置: {best_config}")
        self.logger.info(f"最佳奖励: {best_reward:.4f}")
        
        # 保存优化历史
        history_df = pd.DataFrame(history)
        history_path = os.path.join(self.trial_dir, "optimization_history.csv")
        history_df.to_csv(history_path, index=False)
        self.logger.info(f"优化历史已保存至: {history_path}")
        
        # 保存最佳配置文件
        best_config_file = self.generate_config(best_actions)
        best_config_path = os.path.join(self.trial_dir, "best_config.yaml")
        import shutil
        shutil.copy2(best_config_file, best_config_path)
        self.logger.info(f"最佳配置文件已保存至: {best_config_path}")
        
        # 保存最终评估缓存
        if self.use_cache:
            self._save_cache()
            self.logger.info(f"评估缓存统计 - 总记录数: {len(self.evaluation_cache)}, 命中率: {self.cache_hits/(self.cache_hits+self.cache_misses):.2%}")
        
        return best_config, best_reward

    def _map_actions_to_components(self, actions):
        """
        将动作映射到组件配置
        actions: 动作向量
        return: 映射后的组件动作
        """
        # 🔧 添加调试信息
        self.logger.info(f"[DEBUG] 输入动作形状: {np.array(actions).shape}")
        self.logger.info(f"[DEBUG] 输入动作内容: {actions}")
        self.logger.info(f"[DEBUG] 组件数量: {len(self.components)}")
        self.logger.info(f"[DEBUG] 组件列表: {self.components}")
        
        try:
            # 确保actions是列表或数组
            if hasattr(actions, 'tolist'):
                actions = actions.tolist()
            elif not isinstance(actions, list):
                actions = list(actions)
            
            # 🔧 确保只使用前8维（如果有的话）
            if len(actions) >= 8:
                actions = actions[:8]
            else:
                self.logger.error(f"[DEBUG] 动作长度不足: {len(actions)}, 期望至少8")
                return None
            
            mapped_actions = []
            
            # 前两个位置直接映射 (retrieval, query_expansion)
            mapped_actions.append(actions[0])  # retrieval: 4个选项 (0-3)
            mapped_actions.append(actions[1])  # query_expansion: 4个选项 (0-3)
            
            # 🔧 第三个位置：组合两列表示8个选项 (passage_reranker)
            third_component = actions[2] + (actions[3] + 1)
            third_component = min(third_component, 7)  # 确保不超过范围
            mapped_actions.append(third_component)  # passage_reranker: 8个选项 (0-7)
            
            # 第四个位置直接映射 (passage_filter)
            mapped_actions.append(actions[4])  # passage_filter: 4个选项 (0-3)
            
            # 第五个位置直接映射 (passage_compressor)
            mapped_actions.append(actions[5])  # passage_compressor: 4个选项 (0-3)
            
            # 🔧 第六个位置：3个选项，但用4维表示 (prompt_maker)
            sixth_component = actions[6]
            if sixth_component >= 3:  # 2和3都映射到2
                sixth_component = 2
            mapped_actions.append(sixth_component)  # prompt_maker: 3个选项 (0-2)
            
            # 🔧 第七个位置：2个选项，但用4维表示 (passage_augmenter)
            seventh_component = actions[7]
            if seventh_component >= 2:  # 2和3都映射到1
                seventh_component = 1
            mapped_actions.append(seventh_component)  # passage_augmenter: 2个选项 (0-1)
            
            self.logger.info(f"[DEBUG] 映射后动作: {mapped_actions}")
            return mapped_actions
            
        except Exception as e:
            self.logger.error(f"[DEBUG] 映射过程出错: {str(e)}")
            self.logger.error(f"[DEBUG] 错误位置的actions: {actions}")
            return None

# 使用示例
if __name__ == "__main__":
    # 配置路径
    qa_data_path = "../data/5dataset_100/qa100.parquet"
    corpus_data_path = "../data/5dataset_100/corpus.parquet"
    project_dir = "../experiments/204-re-qe_new"
    
    # 示例1: 优化所有组件
    # optimizer = RAGOptimizer(
    #     qa_data_path=qa_data_path,
    #     corpus_data_path=corpus_data_path,
    #     project_dir=project_dir,
    #     trial_name="full_rag_opt",
    #     num_gpus=4 
    # )
    
    # 示例2: 只优化特定组件
    # optimizer = RAGOptimizer(
    #     qa_data_path=qa_data_path,
    #     corpus_data_path=corpus_data_path,
    #     project_dir=project_dir,
    #     trial_name="retrieval_prompt_opt",
    #     num_gpus=4,
    #     target_components=["retrieval", "prompt_maker"]  # 只优化这两个组件
    # )
    
    # 示例3: 固定某些组件的配置，优化其他组件
    optimizer = RAGOptimizer(
        qa_data_path=qa_data_path,
        corpus_data_path=corpus_data_path,
        project_dir=project_dir,
        trial_name="strategy_opt",
        num_gpus=3,
        target_components=["retrieval", "query_expansion"],
        fixed_components={
            # 如果有些组件你想固定为特定方法，在这里指定
            # 例如: "vectordb": "chroma"
        },
        use_cache=True  # 启用评估缓存
    )
    
    # 运行优化，使用较小的batch size
    best_config, best_reward = optimizer.run_optimization(num_epochs=100, batch_size=4)