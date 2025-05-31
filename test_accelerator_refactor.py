#!/usr/bin/env python3
"""
测试新的统一加速器架构

验证GPU和CPU统一为加速器概念后的功能是否正常。
"""

import sys
sys.path.insert(0, '.')

from llm_estimate.hardware.accelerator import (
    create_accelerator, 
    list_supported_accelerators,
    get_accelerator_by_type,
    compare_accelerators
)
from llm_estimate.hardware.base import SystemSpec
from llm_estimate.estimator.base import PerformanceEstimator


def test_create_accelerator():
    """测试创建加速器"""
    print("=== 测试创建加速器 ===")
    
    # 测试创建GPU加速器
    gpu = create_accelerator("rtx-4090")
    print(f"GPU: {gpu.name}")
    print(f"算力: {gpu.compute_capability_tflops} TFLOPS")
    print(f"内存带宽: {gpu.memory_bandwidth_gb_s} GB/s")
    print(f"内存容量: {gpu.memory_capacity_gb} GB")
    
    # 测试创建CPU加速器
    cpu = create_accelerator("i9-13900k")
    print(f"\nCPU: {cpu.name}")
    print(f"算力: {cpu.compute_capability_tflops} TFLOPS")
    print(f"内存带宽: {cpu.memory_bandwidth_gb_s} GB/s")
    print(f"内存容量: {cpu.memory_capacity_gb} GB")


def test_list_accelerators():
    """测试列出加速器"""
    print("\n=== 测试列出加速器 ===")
    
    all_accelerators = list_supported_accelerators()
    print(f"总共支持 {len(all_accelerators)} 个加速器")
    
    # 按类型筛选
    gpus = get_accelerator_by_type("gpu")
    cpus = get_accelerator_by_type("cpu")
    
    print(f"GPU: {len(gpus)} 个")
    for name, info in list(gpus.items())[:3]:  # 只显示前3个
        print(f"  {info['name']}: {info['compute_capability_tflops']} TFLOPS")
    
    print(f"CPU: {len(cpus)} 个")
    for name, info in cpus.items():
        print(f"  {info['name']}: {info['compute_capability_tflops']} TFLOPS")


def test_system_spec():
    """测试系统规格"""
    print("\n=== 测试系统规格 ===")
    
    # 创建多加速器系统
    system = SystemSpec()
    system.add_accelerator(create_accelerator("rtx-4090"))
    system.add_accelerator(create_accelerator("i9-13900k"))
    
    print(f"总算力: {system.get_total_compute_capability():.1f} TFLOPS")
    print(f"总内存: {system.get_total_memory_capacity():.0f} GB")
    print(f"总带宽: {system.get_total_memory_bandwidth():.0f} GB/s")
    print(f"总功耗: {system.estimate_total_power():.0f} W")


def test_performance_estimation():
    """测试性能估算"""
    print("\n=== 测试性能估算 ===")
    
    estimator = PerformanceEstimator()
    
    # 测试单加速器配置
    try:
        result = estimator.estimate(
            model_name="llama-2-7b",
            hardware_config={"accelerator": "rtx-4090"},
            model_config={"precision": "fp16"}
        )
        
        print("单加速器估算结果:")
        print(f"  吞吐量: {result['throughput_tokens_per_sec']:.1f} tokens/s")
        print(f"  内存使用: {result['memory_usage_gb']:.2f} GB")
        print(f"  延迟: {result['latency_ms']:.1f} ms")
        print(f"  瓶颈: {result['bottleneck']}")
        
    except Exception as e:
        print(f"单加速器估算失败: {e}")
    
    # 测试多加速器配置
    try:
        result = estimator.estimate(
            model_name="llama-2-7b",
            hardware_config={"accelerators": ["rtx-4090", "a100-40gb"]},
            model_config={"precision": "fp16"}
        )
        
        print("\n多加速器估算结果:")
        print(f"  吞吐量: {result['throughput_tokens_per_sec']:.1f} tokens/s")
        print(f"  内存使用: {result['memory_usage_gb']:.2f} GB")
        print(f"  延迟: {result['latency_ms']:.1f} ms")
        print(f"  瓶颈: {result['bottleneck']}")
        
    except Exception as e:
        print(f"多加速器估算失败: {e}")


def test_compatibility():
    """测试兼容性"""
    print("\n=== 测试向后兼容性 ===")
    
    estimator = PerformanceEstimator()
    
    # 测试旧的GPU格式
    try:
        result = estimator.estimate(
            model_name="llama-2-7b",
            hardware_config={"gpu": "rtx-4090"}
        )
        print("旧GPU格式兼容 ✓")
        
    except Exception as e:
        print(f"旧GPU格式不兼容: {e}")
    
    # 测试旧的CPU格式
    try:
        result = estimator.estimate(
            model_name="llama-2-7b",
            hardware_config={"cpu": "i9-13900k"}
        )
        print("旧CPU格式兼容 ✓")
        
    except Exception as e:
        print(f"旧CPU格式不兼容: {e}")


def test_compare_accelerators():
    """测试加速器比较"""
    print("\n=== 测试加速器比较 ===")
    
    comparison = compare_accelerators(["rtx-4090", "a100-40gb", "i9-13900k"])
    
    print("算力排名:")
    for i, acc in enumerate(comparison["rankings"]["compute_capability"][:3], 1):
        print(f"  {i}. {acc}")
    
    print("内存带宽排名:")
    for i, acc in enumerate(comparison["rankings"]["memory_bandwidth"][:3], 1):
        print(f"  {i}. {acc}")


def main():
    """主测试函数"""
    print("测试统一加速器架构\n")
    
    test_create_accelerator()
    test_list_accelerators() 
    test_system_spec()
    test_performance_estimation()
    test_compatibility()
    test_compare_accelerators()
    
    print("\n测试完成!")


if __name__ == "__main__":
    main() 