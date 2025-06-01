#!/usr/bin/env python3
"""
LLM-Estimate 基本使用示例

演示如何使用LLM-Estimate进行性能估算。
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from llm_estimate import PerformanceEstimator, ModelRegistry


def main():
    """主函数"""
    print("=== LLM-Estimate 基本使用示例 ===\n")
    
    # 1. 查看支持的模型
    registry = ModelRegistry()
    models = registry.list_models()
    print("支持的模型:")
    for model in models:
        print(f"  - {model}")
    print()
    
    # 2. 创建性能估算器
    estimator = PerformanceEstimator()
    
    # 3. 估算Llama-2-7B在RTX-4090上的性能
    print("估算 Llama-2-7B 在 RTX-4090 上的性能...")
    try:
        result = estimator.estimate(
            model_name="llama-2-7b",
            hardware_config={
                "gpu": "rtx-4090",
                "memory": "32GB"
            },
            model_config={
                "batch_size": 1,
                "precision": "fp16",
                "context_length": 4096
            }
        )
        
        # 4. 显示结果
        print(f"模型: {result['model_name']}")
        print(f"内存使用: {result['memory_usage_gb']:.2f} GB")
        print(f"吞吐量: {result['throughput_tokens_per_sec']:.1f} tokens/s")
        print(f"延迟: {result['latency_ms']:.1f} ms")
        print(f"算力利用率: {result['compute_utilization_percent']:.1f}%")
        print(f"存储带宽利用率: {result['memory_bandwidth_utilization_percent']:.1f}%")
        print(f"存储容量利用率: {result['memory_capacity_utilization_percent']:.1f}%")
        print(f"性能瓶颈: {result['bottleneck']}")
        
        if result.get('recommendations'):
            print("\n优化建议:")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"  {i}. {rec}")
                
    except Exception as e:
        print(f"估算失败: {e}")
    
    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    main() 