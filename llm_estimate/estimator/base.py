"""
基础估算器类

提供性能估算的核心接口和基础实现。
使用统一的加速器抽象进行性能计算。
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..models.registry import model_registry
from ..models.base import BaseModel
from ..hardware.base import SystemSpec
from ..hardware.accelerator import create_accelerator


@dataclass
class EstimationResult:
    """估算结果数据类"""
    model_name: str
    hardware_config: Dict[str, Any]
    throughput_tokens_per_sec: float
    latency_ms: float
    memory_usage_gb: float
    utilization_percent: float
    bottleneck: str
    additional_metrics: Dict[str, Any]


class PerformanceEstimator:
    """性能估算器主类"""
    
    def __init__(self):
        self.model_registry = model_registry
    
    def estimate(self, model_name: str, 
                hardware_config: Dict[str, Any],
                model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行性能估算
        
        Args:
            model_name: 模型名称
            hardware_config: 硬件配置，支持单加速器或多加速器
                格式: {"accelerator": "rtx-4090"} 或 {"accelerators": ["rtx-4090", "a100-40gb"]}
            model_config: 模型配置
            
        Returns:
            估算结果字典
        """
        # 创建模型实例
        model = self._create_model(model_name, model_config)
        
        # 创建系统规格
        system_spec = self._create_system_spec(hardware_config)
        
        # 执行各项估算
        memory_usage = self._estimate_memory_usage(model, system_spec)
        throughput = self._estimate_throughput(model, system_spec)
        latency = self._estimate_latency(model, system_spec)
        utilization = self._estimate_utilization(model, system_spec)
        bottleneck = self._identify_bottleneck(model, system_spec)
        
        # 组装结果
        result = {
            "model_name": model_name,
            "model_info": model.get_model_info(),
            "system_info": system_spec.get_system_info(),
            "memory_usage_gb": memory_usage,
            "throughput_tokens_per_sec": throughput,
            "latency_ms": latency,
            "utilization_percent": utilization,
            "bottleneck": bottleneck,
            "recommendations": self._generate_recommendations(model, system_spec),
            "compatibility": system_spec.check_compatibility(memory_usage)
        }
        
        return result
    
    def _create_model(self, model_name: str, 
                     model_config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """创建模型实例"""
        config_kwargs = model_config or {}
        return self.model_registry.create_model(model_name, **config_kwargs)
    
    def _create_system_spec(self, hardware_config: Dict[str, Any]) -> SystemSpec:
        """
        创建系统硬件规格
        
        Args:
            hardware_config: 硬件配置，支持多种格式：
                - {"accelerator": "rtx-4090"}  # 单加速器
                - {"accelerators": ["rtx-4090", "a100-40gb"]}  # 多加速器
                - {"gpu": "rtx-4090"}  # 兼容旧格式
                - {"cpu": "i9-13900k"}  # 兼容旧格式
        """
        system_spec = SystemSpec()
        
        # 处理单加速器配置
        if "accelerator" in hardware_config:
            accelerator = create_accelerator(hardware_config["accelerator"])
            system_spec.add_accelerator(accelerator)
        
        # 处理多加速器配置
        elif "accelerators" in hardware_config:
            for acc_name in hardware_config["accelerators"]:
                accelerator = create_accelerator(acc_name)
                system_spec.add_accelerator(accelerator)
        
        # 兼容旧的GPU/CPU格式
        else:
            if "gpu" in hardware_config:
                gpu = create_accelerator(hardware_config["gpu"])
                system_spec.add_accelerator(gpu)
            
            if "cpu" in hardware_config:
                cpu = create_accelerator(hardware_config["cpu"])
                system_spec.add_accelerator(cpu)
        
        # 如果没有指定任何加速器，使用默认配置
        if not system_spec.accelerators:
            default_accelerator = create_accelerator("rtx-4090")
            system_spec.add_accelerator(default_accelerator)
        
        return system_spec
    
    def _estimate_memory_usage(self, model: BaseModel, system_spec: SystemSpec) -> float:
        """估算内存使用量"""
        # 获取模型的内存需求
        precision = "fp16"  # 默认精度
        if system_spec.accelerators:
            precision = system_spec.accelerators[0].config.precision
        
        memory_info = model.calculate_memory_usage(precision)
        return memory_info["total_inference_gb"]
    
    def _estimate_throughput(self, model: BaseModel, system_spec: SystemSpec) -> float:
        """估算吞吐量 (tokens/second)"""
        if not system_spec.accelerators:
            return 0.0
        
        # 计算模型的工作负载特征
        flops_per_token = model.estimate_flops_per_token()
        memory_per_token = model.estimate_memory_per_token()
        
        workload = {
            "flops_per_token": flops_per_token,
            "memory_per_token": memory_per_token
        }
        
        # 计算每个加速器的吞吐量并汇总
        total_throughput = 0.0
        for accelerator in system_spec.accelerators:
            acc_throughput = accelerator.calculate_throughput(workload)
            total_throughput += acc_throughput
        
        return total_throughput
    
    def _estimate_latency(self, model: BaseModel, system_spec: SystemSpec) -> float:
        """估算延迟 (milliseconds)"""
        if not system_spec.accelerators:
            return 1000.0  # 默认1秒延迟
        
        # 使用性能最强的加速器计算延迟
        primary_accelerator = max(
            system_spec.accelerators,
            key=lambda acc: acc.compute_capability_tflops
        )
        
        flops_per_token = model.estimate_flops_per_token()
        workload = {
            "flops_per_token": flops_per_token,
            "memory_overhead_ms": 5.0  # 基础内存访问开销
        }
        
        return primary_accelerator.estimate_latency(workload)
    
    def _estimate_utilization(self, model: BaseModel, system_spec: SystemSpec) -> float:
        """估算硬件利用率"""
        if not system_spec.accelerators:
            return 0.0
        
        memory_usage = self._estimate_memory_usage(model, system_spec)
        total_memory = system_spec.get_total_memory_capacity()
        
        # 内存利用率
        memory_utilization = min(memory_usage / total_memory * 100, 100) if total_memory > 0 else 0
        
        # 计算利用率（这里简化为内存利用率，实际应该考虑算力利用率）
        return memory_utilization
    
    def _identify_bottleneck(self, model: BaseModel, system_spec: SystemSpec) -> str:
        """识别性能瓶颈"""
        if not system_spec.accelerators:
            return "no_accelerator"
        
        memory_usage = self._estimate_memory_usage(model, system_spec)
        total_memory = system_spec.get_total_memory_capacity()
        
        # 检查内存瓶颈
        if memory_usage > total_memory * 0.9:
            return "memory_capacity"
        
        # 检查内存带宽 vs 算力瓶颈
        flops_per_token = model.estimate_flops_per_token()
        memory_per_token = model.estimate_memory_per_token()
        
        total_compute = system_spec.get_total_compute_capability() * 1e12
        total_bandwidth = system_spec.get_total_memory_bandwidth() * 1e9
        
        compute_limited_tps = total_compute / flops_per_token
        memory_limited_tps = total_bandwidth / memory_per_token
        
        if compute_limited_tps < memory_limited_tps:
            return "compute"
        else:
            return "memory_bandwidth"
    
    def _generate_recommendations(self, model: BaseModel, 
                                system_spec: SystemSpec) -> list:
        """生成优化建议"""
        recommendations = []
        
        memory_usage = self._estimate_memory_usage(model, system_spec)
        total_memory = system_spec.get_total_memory_capacity()
        bottleneck = self._identify_bottleneck(model, system_spec)
        
        # 基于瓶颈类型生成建议
        if bottleneck == "memory_capacity":
            recommendations.append("内存容量不足，建议使用更低精度 (int8/int4) 减少内存使用")
            recommendations.append("考虑使用模型并行分布到多个设备")
            recommendations.append("考虑使用梯度检查点等内存优化技术")
        
        elif bottleneck == "memory_bandwidth":
            recommendations.append("内存带宽成为瓶颈，建议使用更高带宽的加速器")
            recommendations.append("考虑增加batch_size以提高内存带宽利用率")
        
        elif bottleneck == "compute":
            recommendations.append("算力不足，建议使用更强的加速器")
            recommendations.append("考虑使用多卡并行提升算力")
        
        # 基于利用率生成建议
        utilization = self._estimate_utilization(model, system_spec)
        if utilization < 50:
            recommendations.append("硬件利用率较低，可以考虑增加batch_size或使用更小的模型")
        
        # 基于模型配置生成建议
        if hasattr(model.config, 'batch_size') and model.config.batch_size == 1:
            recommendations.append("当前batch_size=1，可以考虑增加以提高吞吐量")
        
        return recommendations 