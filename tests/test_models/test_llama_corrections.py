#!/usr/bin/env python3
"""
测试修正后的Llama FLOPS和内存计算

验证基于meta-llama官方实现的修正计算公式
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from llm_estimate.models.registry import model_registry

# 可选导入pytest
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    print("警告: pytest未安装，跳过单元测试类")


if PYTEST_AVAILABLE:
    class TestLlamaCorrections:
        """测试Llama模型FLOPS和内存计算的修正"""
        
        @pytest.fixture
        def test_models(self):
            """测试模型列表"""
            return [
                "llama-2-7b",
                "llama-3.1-8b",
            ]
        
        def test_model_creation(self, test_models):
            """测试模型创建"""
            for model_name in test_models:
                model = model_registry.create_model(model_name)
                assert model is not None
                assert model.specs.name == model_name
        
        def test_flops_calculation(self, test_models):
            """测试FLOPS计算"""
            for model_name in test_models:
                model = model_registry.create_model(model_name)
                
                # 详细FLOPS计算
                detailed_flops = model.estimate_flops_per_token()
                assert detailed_flops > 0
                
                # 简化FLOPS计算 (OpenAI scaling laws)
                simplified_flops = model.estimate_model_flops_per_token_simplified()
                assert simplified_flops > 0
                
                # 两种计算结果应该在同一个数量级
                ratio = detailed_flops / simplified_flops
                assert 0.5 <= ratio <= 3.0, f"FLOPS计算比率异常: {ratio}"
        
        def test_memory_calculation(self, test_models):
            """测试内存计算"""
            for model_name in test_models:
                model = model_registry.create_model(model_name)
                
                memory_usage = model.calculate_memory_usage("fp16")
                
                # 检查所有内存组件都大于0
                assert memory_usage["model_memory_gb"] > 0
                assert memory_usage["kv_cache_memory_gb"] >= 0  # 可能为0（如果禁用KV缓存）
                assert memory_usage["total_activation_memory_gb"] > 0
                assert memory_usage["total_inference_gb"] > 0
                assert memory_usage["total_training_gb"] > 0
                
                # 训练内存应该大于推理内存
                assert memory_usage["total_training_gb"] > memory_usage["total_inference_gb"]
        
        def test_gqa_optimization(self):
            """测试GQA优化的效果"""
            # 比较相似规模的MHA和GQA模型
            model_mha = model_registry.create_model("llama-2-7b")  # MHA
            model_gqa = model_registry.create_model("llama-3.1-8b")  # GQA
            
            # 使用相同配置
            config = {"context_length": 4096, "batch_size": 1, "precision": "fp16"}
            model_mha.update_config(**config)
            model_gqa.update_config(**config)
            
            # 计算KV缓存使用
            mem_mha = model_mha.calculate_memory_usage()
            mem_gqa = model_gqa.calculate_memory_usage()
            
            kv_mha = mem_mha['kv_cache_memory_gb']
            kv_gqa = mem_gqa['kv_cache_memory_gb']
            
            # GQA应该显著减少KV缓存使用
            assert kv_gqa < kv_mha, "GQA应该减少KV缓存使用"
            
            reduction = ((kv_mha - kv_gqa) / kv_mha) * 100
            assert reduction > 50, f"GQA KV缓存节省应该超过50%，实际: {reduction:.1f}%"
        
        def test_context_length_scaling(self):
            """测试上下文长度对内存的影响"""
            model = model_registry.create_model("llama-3.1-8b")
            
            context_lengths = [1024, 4096, 8192, 16384]
            kv_cache_sizes = []
            
            for ctx_len in context_lengths:
                model.update_config(context_length=ctx_len)
                memory_usage = model.calculate_memory_usage()
                kv_cache_sizes.append(memory_usage['kv_cache_memory_gb'])
            
            # KV缓存应该随上下文长度线性增长
            for i in range(1, len(context_lengths)):
                ratio_ctx = context_lengths[i] / context_lengths[i-1]
                ratio_mem = kv_cache_sizes[i] / kv_cache_sizes[i-1]
                
                # 允许一定的误差范围
                assert abs(ratio_ctx - ratio_mem) < 0.1, f"KV缓存不是线性缩放: {ratio_ctx} vs {ratio_mem}"
        
        def test_precision_scaling(self):
            """测试不同精度对内存的影响"""
            model = model_registry.create_model("llama-2-7b")
            
            precisions = ["fp32", "fp16", "int8", "int4"]
            expected_bytes = [4, 2, 1, 0.5]
            
            memory_usages = []
            for precision in precisions:
                memory_usage = model.calculate_memory_usage(precision)
                memory_usages.append(memory_usage['model_memory_gb'])
            
            # 内存使用应该按精度比例缩放
            for i in range(1, len(precisions)):
                expected_ratio = expected_bytes[0] / expected_bytes[i]
                actual_ratio = memory_usages[0] / memory_usages[i]
                
                # 允许5%的误差
                assert abs(expected_ratio - actual_ratio) / expected_ratio < 0.05, \
                    f"精度 {precisions[i]} 内存缩放不正确: 预期 {expected_ratio}, 实际 {actual_ratio}"
        
        def test_performance_analysis(self):
            """测试性能分析功能"""
            model = model_registry.create_model("llama-2-7b")
            analysis = model.get_performance_analysis()
            
            # 检查分析结果结构
            assert "model_info" in analysis
            assert "flops_analysis" in analysis
            assert "memory_analysis" in analysis
            assert "efficiency_metrics" in analysis
            
            # 检查FLOPS分布合理性
            flops_info = analysis["flops_analysis"]
            attn_pct = flops_info["attention_flops_percentage"]
            ffn_pct = flops_info["ffn_flops_percentage"]
            
            # 注意力和FFN应该占据大部分FLOPS
            assert attn_pct + ffn_pct > 80, "注意力和FFN应该占据大部分FLOPS"
            
            # FFN通常比注意力消耗更多FLOPS（特别是对于较大的模型）
            if model.specs.parameters >= 13:  # 13B及以上模型
                assert ffn_pct > attn_pct, "对于大模型，FFN FLOPS应该大于注意力FLOPS"


def run_interactive_test():
    """交互式测试函数，用于手动运行和查看详细结果"""
    
    test_models = [
        "llama-2-7b",
        "llama-3.1-8b",
    ]
    
    print("=" * 80)
    print("Llama模型FLOPS和内存计算测试 (基于meta-llama官方实现修正)")
    print("=" * 80)
    
    for model_name in test_models:
        try:
            print(f"\n🔍 测试模型: {model_name}")
            print("-" * 50)
            
            # 创建模型实例
            model = model_registry.create_model(model_name)
            
            # 获取完整性能分析
            analysis = model.get_performance_analysis()
            
            # 显示模型基本信息
            model_info = analysis["model_info"]
            print(f"📊 模型信息:")
            print(f"   参数量: {model_info['parameters']}")
            print(f"   层数: {model_info['layers']}")
            print(f"   隐藏维度: {model_info['hidden_size']}")
            print(f"   注意力头数: {model_info['attention_heads']}")
            print(f"   KV头数: {model_info['kv_heads']} (GQA比率: {model_info['attention_heads']//model_info['kv_heads']:.1f})")
            
            # FLOPS分析
            flops_info = analysis["flops_analysis"]
            print(f"\n⚡ FLOPS分析 (每token):")
            print(f"   总FLOPS: {flops_info['total_flops_per_token']:,.0f}")
            print(f"   简化FLOPS (OpenAI): {flops_info['simplified_flops_per_token']:,.0f}")
            print(f"   注意力FLOPS: {flops_info['attention_flops_per_token']:,.0f} ({flops_info['attention_flops_percentage']:.1f}%)")
            print(f"   FFN FLOPS: {flops_info['ffn_flops_per_token']:,.0f} ({flops_info['ffn_flops_percentage']:.1f}%)")
            
            # 内存分析
            memory_info = analysis["memory_analysis"]
            print(f"\n💾 内存使用 (精度: {memory_info['precision']}):")
            print(f"   模型权重: {memory_info['model_memory_gb']:.2f} GB")
            print(f"   KV缓存: {memory_info['kv_cache_memory_gb']:.2f} GB")
            print(f"   注意力分数: {memory_info['attention_scores_memory_gb']:.2f} GB")
            print(f"   总激活值: {memory_info['total_activation_memory_gb']:.2f} GB")
            print(f"   推理总内存: {memory_info['total_inference_gb']:.2f} GB")
            print(f"   训练总内存: {memory_info['total_training_gb']:.2f} GB")
            
            # 效率指标
            efficiency = analysis["efficiency_metrics"]
            print(f"\n📈 效率指标:")
            print(f"   每参数内存: {efficiency['memory_per_parameter_bytes']:.2f} bytes")
            print(f"   每参数FLOPS: {efficiency['flops_per_parameter']:.0f}")
            print(f"   KV缓存开销: {efficiency['kv_cache_overhead_percentage']:.1f}%")
            
            # 上下文长度分析 (仅对支持长上下文的模型)
            if model_name.startswith("llama-3.1"):
                print(f"\n🔄 长上下文分析:")
                print("   测试不同上下文长度的KV缓存使用...")
                
                for ctx_len in [4096, 8192, 32768, 131072]:
                    model.update_config(context_length=ctx_len)
                    mem_usage = model.calculate_memory_usage()
                    kv_cache_gb = mem_usage['kv_cache_memory_gb']
                    print(f"     {ctx_len:>6} tokens: KV缓存 {kv_cache_gb:.2f} GB")
                
                # 恢复默认上下文长度
                model.update_config(context_length=4096)
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            continue
    
    # 比较注意力机制
    print("\n" + "=" * 80)
    print("注意力机制效率对比 (MHA vs GQA)")
    print("=" * 80)
    
    comparisons = [
        ("llama-2-7b", "llama-3.1-8b"),   # 7B vs 8B
    ]
    
    for mha_model, gqa_model in comparisons:
        print(f"\n🔄 对比: {mha_model} (MHA) vs {gqa_model} (GQA)")
        print("-" * 60)
        
        try:
            model_mha = model_registry.create_model(mha_model)
            model_gqa = model_registry.create_model(gqa_model)
            
            config = {"context_length": 4096, "batch_size": 1, "precision": "fp16"}
            model_mha.update_config(**config)
            model_gqa.update_config(**config)
            
            mem_mha = model_mha.calculate_memory_usage()
            mem_gqa = model_gqa.calculate_memory_usage()
            
            kv_mha = mem_mha['kv_cache_memory_gb']
            kv_gqa = mem_gqa['kv_cache_memory_gb']
            
            reduction = ((kv_mha - kv_gqa) / kv_mha) * 100
            
            print(f"   MHA KV缓存: {kv_mha:.2f} GB")
            print(f"   GQA KV缓存: {kv_gqa:.2f} GB")
            print(f"   内存节省: {reduction:.1f}%")
            
        except Exception as e:
            print(f"   ❌ 对比失败: {e}")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    # 如果直接运行此文件，执行交互式测试
    run_interactive_test() 