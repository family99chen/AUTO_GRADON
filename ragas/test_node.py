from create_config import load_and_generate_nodes
from pprint import pprint

def simple_test():
    """简单测试 load_and_generate_nodes 的两个输出"""
    
    config_path = "/home/cz/AUTO_GRADON/ragas/configuration/config.yaml"
    
    print("🔍 测试 load_and_generate_nodes 的输出格式")
    print("=" * 60)
    
    # 测试几个主要组件
    components = ["query_expansion", "retrieval", "passage_reranker"]
    
    for component in components:
        print(f"\n📋 组件: {component}")
        print("-" * 30)
        
        try:
            # 调用函数
            lines, cfg = load_and_generate_nodes(
                config_path=config_path,
                key=component, 
                size=4,
                exhaustive=False
            )
            
            print(f"✅ 第一个输出 (lines) 类型: {type(lines)}")
            print(f"   长度: {len(lines) if hasattr(lines, '__len__') else '不是列表'}")
            print(f"   内容预览:")
            if isinstance(lines, list):
                pprint(lines[0] if lines else "空列表", width=200, depth=10)  # 增大width和depth
            else:
                pprint(lines, width=200, depth=10)
            
            print(f"\n✅ 第二个输出 (cfg) 类型: {type(cfg)}")
            print(f"   cfg 的主要属性:")
            if hasattr(cfg, '__dict__'):
                for key, value in list(cfg.__dict__.items())[:3]:  # 只显示前3个属性
                    print(f"     {key}: {type(value)}")
            
        except Exception as e:
            print(f"❌ 出错: {e}")

if __name__ == "__main__":
    simple_test()
