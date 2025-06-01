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
    compute_utilization_percent: float
    memory_bandwidth_utilization_percent: float
    memory_capacity_utilization_percent: float
    bottleneck: str
    bottleneck_details: str
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
        detailed_utilization = self._estimate_detailed_utilization(model, system_spec)
        utilization_analysis = self._analyze_utilization_and_bottleneck(model, system_spec)
        
        # 组装结果
        result = {
            "model_name": model_name,
            "model_info": model.get_model_info(),
            "system_info": system_spec.get_system_info(),
            "memory_usage_gb": memory_usage,
            "throughput_tokens_per_sec": throughput,
            "latency_ms": latency,
            "utilization_percent": detailed_utilization["overall"],
            "compute_utilization_percent": detailed_utilization["compute"],
            "memory_bandwidth_utilization_percent": detailed_utilization["memory_bandwidth"], 
            "memory_capacity_utilization_percent": detailed_utilization["memory_capacity"],
            "bottleneck": utilization_analysis["bottleneck"],
            "bottleneck_details": utilization_analysis["bottleneck_details"],
            "recommendations": self._generate_recommendations(model, system_spec, detailed_utilization, utilization_analysis),
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
    
    def _estimate_detailed_utilization(self, model: BaseModel, system_spec: SystemSpec) -> Dict[str, float]:
        """估算详细的硬件利用率"""
        if not system_spec.accelerators:
            return {
                "overall": 0.0,
                "compute": 0.0,
                "memory_bandwidth": 0.0,
                "memory_capacity": 0.0
            }
        
        memory_usage = self._estimate_memory_usage(model, system_spec)
        total_memory = system_spec.get_total_memory_capacity()
        
        # 计算各种利用率指标
        flops_per_token = model.estimate_flops_per_token()
        memory_per_token = model.estimate_memory_per_token()
        total_compute = system_spec.get_total_compute_capability() * 1e12  # TFLOPS转换为FLOPS
        total_bandwidth = system_spec.get_total_memory_bandwidth() * 1e9   # GB/s转换为B/s
        
        # 1. 存储容量利用率：已使用内存 / 总内存容量
        memory_capacity_utilization = min(memory_usage / total_memory * 100, 100) if total_memory > 0 else 0
        
        # 2. 计算实际吞吐量限制
        compute_limited_tps = total_compute / flops_per_token if flops_per_token > 0 else float('inf')
        memory_limited_tps = total_bandwidth / memory_per_token if memory_per_token > 0 else float('inf')
        actual_tps = min(compute_limited_tps, memory_limited_tps)
        
        # 3. 算力利用率：实际使用的算力 / 总算力
        compute_utilization = min((actual_tps * flops_per_token) / total_compute * 100, 100) if total_compute > 0 else 0
        
        # 4. 存储带宽利用率：基于实际测得的吞吐量计算
        actual_throughput = self._estimate_throughput(model, system_spec)
        theoretical_bandwidth_tps = total_bandwidth / memory_per_token if memory_per_token > 0 else float('inf')
        memory_bandwidth_utilization = min((actual_throughput * memory_per_token) / total_bandwidth * 100, 100) if total_bandwidth > 0 else 0
        
        # 5. 整体利用率：取各项利用率的加权平均
        # 这里使用存储容量利用率作为主要指标，因为它最直观
        overall_utilization = memory_capacity_utilization
        
        return {
            "overall": overall_utilization,
            "compute": compute_utilization,
            "memory_bandwidth": memory_bandwidth_utilization,
            "memory_capacity": memory_capacity_utilization
        }
    
    def _analyze_utilization_and_bottleneck(self, model: BaseModel, system_spec: SystemSpec) -> Dict[str, Any]:
        """分析硬件利用率和识别性能瓶颈"""
        if not system_spec.accelerators:
            return {
                "compute_utilization": 0.0,
                "memory_utilization": 0.0,
                "bottleneck": "no_accelerator",
                "bottleneck_details": "系统中没有可用的加速器"
            }
        
        memory_usage = self._estimate_memory_usage(model, system_spec)
        total_memory = system_spec.get_total_memory_capacity()
        
        # 内存利用率 (存储利用率)
        memory_utilization = min(memory_usage / total_memory * 100, 100) if total_memory > 0 else 0
        
        # 计算算力利用率
        flops_per_token = model.estimate_flops_per_token()
        memory_per_token = model.estimate_memory_per_token()
        total_compute = system_spec.get_total_compute_capability() * 1e12  # TFLOPS转换为FLOPS
        total_bandwidth = system_spec.get_total_memory_bandwidth() * 1e9   # GB/s转换为B/s
        
        # 计算理论最大吞吐量
        compute_limited_tps = total_compute / flops_per_token if flops_per_token > 0 else float('inf')
        memory_limited_tps = total_bandwidth / memory_per_token if memory_per_token > 0 else float('inf')
        
        # 实际吞吐量受限于较小的那个
        actual_tps = min(compute_limited_tps, memory_limited_tps)
        
        # 算力利用率：实际使用的算力 / 总算力
        compute_utilization = min((actual_tps * flops_per_token) / total_compute * 100, 100) if total_compute > 0 else 0
        
        # 确定瓶颈类型和详细信息
        if memory_usage > total_memory * 0.9:
            bottleneck = "memory_capacity"
            bottleneck_details = f"内存容量不足: 需要 {memory_usage:.2f} GB，可用 {total_memory:.2f} GB (利用率 {memory_utilization:.1f}%)"
        elif compute_limited_tps < memory_limited_tps:
            bottleneck = "compute"
            bottleneck_details = f"算力受限: 算力限制吞吐量 {compute_limited_tps:.0f} tokens/s，内存带宽限制 {memory_limited_tps:.0f} tokens/s"
        else:
            bottleneck = "memory_bandwidth"
            bottleneck_details = f"内存带宽受限: 内存带宽限制吞吐量 {memory_limited_tps:.0f} tokens/s，算力限制 {compute_limited_tps:.0f} tokens/s"
        
        return {
            "compute_utilization": compute_utilization,
            "memory_utilization": memory_utilization,
            "bottleneck": bottleneck,
            "bottleneck_details": bottleneck_details
        }
    
    def _generate_recommendations(self, model: BaseModel, 
                                system_spec: SystemSpec,
                                detailed_utilization: Dict[str, float],
                                utilization_analysis: Dict[str, Any]) -> list:
        """生成优化建议"""
        recommendations = []
        
        memory_usage = self._estimate_memory_usage(model, system_spec)
        total_memory = system_spec.get_total_memory_capacity()
        bottleneck = utilization_analysis["bottleneck"]
        
        # 提取详细利用率指标
        compute_utilization = detailed_utilization["compute"]
        memory_bandwidth_utilization = detailed_utilization["memory_bandwidth"]
        memory_capacity_utilization = detailed_utilization["memory_capacity"]
        
        # 基于瓶颈类型生成建议
        if bottleneck == "memory_capacity":
            recommendations.append("内存容量不足，建议使用更低精度 (int8/int4) 减少内存使用")
            recommendations.append("考虑使用模型并行分布到多个设备")
            recommendations.append("考虑使用梯度检查点等内存优化技术")
        
        elif bottleneck == "memory_bandwidth":
            recommendations.append("内存带宽成为瓶颈，建议使用更高带宽的加速器")
            recommendations.append("考虑增加batch_size以提高内存带宽利用率")
            if memory_bandwidth_utilization < 80:
                recommendations.append(f"当前存储带宽利用率仅 {memory_bandwidth_utilization:.1f}%，可以考虑增加batch_size")
        
        elif bottleneck == "compute":
            recommendations.append("算力不足，建议使用更强的加速器")
            recommendations.append("考虑使用多卡并行提升算力")
            if compute_utilization < 80:
                recommendations.append(f"当前算力利用率仅 {compute_utilization:.1f}%，可以优化计算效率")
        
        # 基于算力利用率生成建议
        if compute_utilization < 30:
            recommendations.append(f"算力利用率较低 ({compute_utilization:.1f}%)，考虑：")
            recommendations.append("  - 增加batch_size提高并行度")
            recommendations.append("  - 使用更复杂的模型充分利用算力")
            recommendations.append("  - 检查是否存在计算瓶颈")
        elif compute_utilization > 90:
            recommendations.append(f"算力利用率很高 ({compute_utilization:.1f}%)，性能接近理论上限")
        
        # 基于存储带宽利用率生成建议
        if memory_bandwidth_utilization < 30:
            recommendations.append(f"存储带宽利用率较低 ({memory_bandwidth_utilization:.1f}%)，考虑：")
            recommendations.append("  - 增加batch_size提高数据传输效率")
            recommendations.append("  - 优化数据加载和预处理流程")
        elif memory_bandwidth_utilization > 90:
            recommendations.append(f"存储带宽利用率很高 ({memory_bandwidth_utilization:.1f}%)，接近带宽上限")
        
        # 基于存储容量利用率生成建议
        if memory_capacity_utilization < 30:
            recommendations.append(f"存储容量利用率较低 ({memory_capacity_utilization:.1f}%)，考虑：")
            recommendations.append("  - 增加batch_size或使用更大的模型")
            recommendations.append("  - 当前配置可能过于保守")
        elif memory_capacity_utilization > 85:
            recommendations.append(f"存储容量利用率较高 ({memory_capacity_utilization:.1f}%)，注意：")
            recommendations.append("  - 留意内存不足的风险")
            recommendations.append("  - 考虑使用内存优化技术")
        
        # 基于模型配置生成建议
        if hasattr(model.config, 'batch_size') and model.config.batch_size == 1:
            recommendations.append("当前batch_size=1，可以考虑增加以提高吞吐量")
        
        # 平衡性建议
        utilization_values = [compute_utilization, memory_bandwidth_utilization, memory_capacity_utilization]
        max_util = max(utilization_values)
        min_util = min(utilization_values)
        
        if max_util - min_util > 40:
            if compute_utilization == max_util:
                recommendations.append("算力利用率远高于存储利用率，系统配置不平衡，考虑增加内存容量或带宽")
            elif memory_bandwidth_utilization == max_util:
                recommendations.append("存储带宽利用率过高，考虑增加内存带宽或优化数据访问模式")
            elif memory_capacity_utilization == max_util:
                recommendations.append("存储容量利用率过高，考虑增加内存容量或使用内存优化技术")
        
        return recommendations 