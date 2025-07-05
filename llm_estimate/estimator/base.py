"""
基础估算器类

提供性能估算的核心接口和基础实现。
使用统一的加速器抽象进行性能计算。
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..models.registry import model_registry
from ..models.base import BaseModel
from ..hardware.base import SystemSpec
from ..hardware.accelerator import create_accelerator
from .op_level_estimator import OpLevelEstimator


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
        self.op_level_estimator = OpLevelEstimator()
    
    def estimate(self, model_name: str, 
                hardware_config: Dict[str, Any],
                model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行性能估算
        
        Args:
            model_name: 模型名称
            hardware_config: 硬件配置，支持单加速器
                格式: {"accelerator": "rtx-4090"}
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
    
    def estimate_op_level(self, model_name: str, 
                         hardware_config: Dict[str, Any],
                         model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行操作级别的详细性能估算
        
        基于矩阵乘法运算的算力和带宽利用率，
        细化到每个transformer操作的耗时估算。
        
        Args:
            model_name: 模型名称
            hardware_config: 硬件配置
            model_config: 模型配置
            
        Returns:
            详细的操作级别估算结果
        """
        # 创建模型实例
        model = self._create_model(model_name, model_config)
        
        # 创建系统规格
        system_spec = self._create_system_spec(hardware_config)
        
        # 执行操作级别估算
        op_level_result = self.op_level_estimator.estimate_model_ops(model, system_spec)
        
        # 添加主要瓶颈操作到结果中
        op_level_result["major_bottleneck_ops"] = self._identify_major_bottleneck_ops(op_level_result)
        
        # 组合结果
        combined_result = {
            "estimation_type": "op_level_detailed",
            "model_name": model_name,
            "model_info": model.get_model_info(),
            "system_info": system_spec.get_system_info(),
            
            # 操作级别详细结果
            "op_level_analysis": op_level_result,
            
            # 综合建议
            "comprehensive_recommendations": self._generate_comprehensive_recommendations(
                op_level_result, model, system_spec
            )
        }
        
        return combined_result
    
    def _identify_major_bottleneck_ops(self, op_level_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别主要的瓶颈操作"""
        detailed_ops = op_level_result.get("detailed_ops", [])
        
        # 按耗时排序，取前5个最耗时的操作
        sorted_ops = sorted(detailed_ops, key=lambda x: x["time_ms"], reverse=True)
        top_ops = sorted_ops[:5]
        
        total_time = sum(op["time_ms"] for op in detailed_ops)
        
        bottleneck_ops = []
        for op in top_ops:
            bottleneck_ops.append({
                "name": op["name"],
                "type": op["type"],
                "time_ms": op["time_ms"],
                "percentage": (op["time_ms"] / total_time * 100) if total_time > 0 else 0,
                "bottleneck_type": "compute" if op["compute_util"] > op["memory_util"] else "memory",
                "gemm_shape": op.get("gemm_shape"),
                "arithmetic_intensity": op["arithmetic_intensity"]
            })
        
        return bottleneck_ops
    
    def _generate_comprehensive_recommendations(self, op_level_result: Dict[str, Any],
                                              model: BaseModel, 
                                              system_spec: SystemSpec) -> List[str]:
        """基于操作级别分析生成综合优化建议"""
        recommendations = []
        
        # 基于操作级别分析的建议
        op_suggestions = op_level_result.get("optimization_suggestions", [])
        recommendations.extend(["=== 基于操作级别分析的建议 ==="])
        recommendations.extend(op_suggestions)
        
        # 基于主要瓶颈操作的建议
        major_bottleneck_ops = self._identify_major_bottleneck_ops(op_level_result)
        
        if major_bottleneck_ops:
            recommendations.append("\n=== 主要瓶颈操作优化建议 ===")
            for i, op in enumerate(major_bottleneck_ops[:3]):  # 只显示前3个
                recommendations.append(f"{i+1}. {op['name']} (占总时间 {op['percentage']:.1f}%)")
                
                if op["type"].startswith("gemm"):
                    if op["bottleneck_type"] == "compute":
                        recommendations.append(f"   - 算力受限的GEMM操作，矩阵形状: {op['gemm_shape']}")
                        recommendations.append(f"   - 建议：使用Tensor Core优化，调整tile大小")
                    else:
                        recommendations.append(f"   - 内存受限的GEMM操作，算术强度: {op['arithmetic_intensity']:.2f}")
                        recommendations.append(f"   - 建议：优化数据布局，使用更高带宽内存")
                elif "softmax" in op["type"]:
                    recommendations.append(f"   - Softmax操作通常内存受限")
                    recommendations.append(f"   - 建议：使用fused kernel，减少内存访问")
                elif "layer_norm" in op["type"]:
                    recommendations.append(f"   - 层归一化操作")
                    recommendations.append(f"   - 建议：使用fused layer norm实现")
        
        # 基于硬件特性的建议
        recommendations.append("\n=== 硬件优化建议 ===")
        if system_spec.accelerators:
            primary_acc = system_spec.accelerators[0]
            recommendations.append(f"当前使用: {primary_acc.name}")
            
            # 根据算力/带宽比例给出建议
            compute_memory_ratio = primary_acc.compute_capability_tflops / (primary_acc.memory_bandwidth_gb_s / 1000)
            if compute_memory_ratio > 100:  # 算力相对过强
                recommendations.append("硬件算力相对较强，建议优化内存访问：")
                recommendations.append("  - 增大batch size提高内存利用率")
                recommendations.append("  - 使用内存优化的算子实现")
            elif compute_memory_ratio < 50:  # 带宽相对过强
                recommendations.append("硬件内存带宽充足，建议充分利用算力：")
                recommendations.append("  - 使用更高精度计算") 
                recommendations.append("  - 考虑更复杂的模型结构")
        
        # 模型特定建议
        recommendations.append("\n=== 模型配置建议 ===")
        batch_size = model.config.batch_size
        context_length = model.config.context_length
        
        if batch_size == 1:
            recommendations.append("当前batch_size=1，可能限制了硬件利用率：")
            recommendations.append("  - 考虑增加batch_size到2-8提高吞吐量")
            recommendations.append("  - 注意：会增加延迟和内存使用")
        
        if context_length > 8192:
            recommendations.append(f"长上下文({context_length})会显著影响注意力计算：")
            recommendations.append("  - 注意力操作时间复杂度为O(n²)")
            recommendations.append("  - 考虑使用稀疏注意力或滑动窗口注意力")
        
        return recommendations
    
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

        """
        system_spec = SystemSpec()
        
        # 处理单加速器配置
        if "accelerator" in hardware_config:
            accelerator = create_accelerator(hardware_config["accelerator"])
            system_spec.add_accelerator(accelerator)
        
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
    
    def _calculate_performance_limits(self, model: BaseModel, system_spec: SystemSpec) -> Dict[str, float]:
        """计算性能限制的统一方法，考虑算力和带宽限制"""
        if not system_spec.accelerators:
            return {
                "compute_limited_tps": 0.0,
                "memory_limited_tps": 0.0,
                "actual_tps": 0.0,
                "total_compute_tflops": 0.0,
                "total_bandwidth_gbps": 0.0
            }
        
        # 计算模型的工作负载特征
        flops_per_token = model.estimate_flops_per_token()
        memory_per_token = model.estimate_memory_per_token()
        
        # 计算系统总的算力和带宽
        total_compute = system_spec.get_total_compute_capability() * 1e12  # TFLOPS转换为FLOPS
        total_bandwidth = system_spec.get_total_memory_bandwidth() * 1e9   # GB/s转换为B/s
        
        # 计算理论最大吞吐量限制
        compute_limited_tps = total_compute / flops_per_token if flops_per_token > 0 else float('inf')
        memory_limited_tps = total_bandwidth / memory_per_token if memory_per_token > 0 else float('inf')
        
        # 实际吞吐量受限于较小的那个
        actual_tps = min(compute_limited_tps, memory_limited_tps)
        
        return {
            "compute_limited_tps": compute_limited_tps,
            "memory_limited_tps": memory_limited_tps,
            "actual_tps": actual_tps,
            "total_compute_tflops": system_spec.get_total_compute_capability(),
            "total_bandwidth_gbps": system_spec.get_total_memory_bandwidth()
        }
    
    def _estimate_throughput(self, model: BaseModel, system_spec: SystemSpec) -> float:
        """估算吞吐量 (tokens/second)"""
        if not system_spec.accelerators:
            return 0.0
        
        # 使用统一的性能限制计算
        performance_limits = self._calculate_performance_limits(model, system_spec)
        
        # 多加速器情况下，返回基于系统总体能力的吞吐量
        return performance_limits["actual_tps"]
    
    def _estimate_latency(self, model: BaseModel, system_spec: SystemSpec) -> float:
        """估算延迟 (milliseconds)"""
        if not system_spec.accelerators:
            return 1000.0  # 默认1秒延迟
        
        # 使用统一的性能限制计算
        performance_limits = self._calculate_performance_limits(model, system_spec)
        
        # 延迟计算：基于单个token的处理时间
        # 对于多加速器系统，延迟主要由最强的加速器决定（推理时的流水线特性）
        primary_accelerator = max(
            system_spec.accelerators,
            key=lambda acc: acc.compute_capability_tflops
        )
        
        # 计算单个加速器的性能限制
        flops_per_token = model.estimate_flops_per_token()
        memory_per_token = model.estimate_memory_per_token()
        
        primary_compute = primary_accelerator.compute_capability_tflops * 1e12  # TFLOPS转换为FLOPS
        primary_bandwidth = primary_accelerator.memory_bandwidth_gb_s * 1e9    # GB/s转换为B/s
        
        # 计算主加速器的限制
        primary_compute_limited_tps = primary_compute / flops_per_token if flops_per_token > 0 else float('inf')
        primary_memory_limited_tps = primary_bandwidth / memory_per_token if memory_per_token > 0 else float('inf')
        primary_actual_tps = min(primary_compute_limited_tps, primary_memory_limited_tps)
        
        # 延迟 = 1000ms / tps （转换为毫秒）
        if primary_actual_tps > 0:
            base_latency = 1000.0 / primary_actual_tps
        else:
            base_latency = 1000.0
        
        # 添加基础开销（内存访问、上下文切换等）
        overhead_ms = 5.0
        
        return base_latency + overhead_ms
    
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
        
        # 使用统一的性能限制计算
        performance_limits = self._calculate_performance_limits(model, system_spec)
        
        # 1. 存储容量利用率：已使用内存 / 总内存容量
        memory_capacity_utilization = min(memory_usage / total_memory * 100, 100) if total_memory > 0 else 0
        
        # 2. 算力利用率：实际使用的算力 / 总算力
        flops_per_token = model.estimate_flops_per_token()
        total_compute = performance_limits["total_compute_tflops"] * 1e12  # TFLOPS转换为FLOPS
        actual_tps = performance_limits["actual_tps"]
        compute_utilization = min((actual_tps * flops_per_token) / total_compute * 100, 100) if total_compute > 0 else 0
        
        # 3. 存储带宽利用率：基于实际测得的吞吐量计算
        memory_per_token = model.estimate_memory_per_token()
        total_bandwidth = performance_limits["total_bandwidth_gbps"] * 1e9   # GB/s转换为B/s
        actual_throughput = self._estimate_throughput(model, system_spec)
        memory_bandwidth_utilization = min((actual_throughput * memory_per_token) / total_bandwidth * 100, 100) if total_bandwidth > 0 else 0
        
        # 4. 整体利用率：取各项利用率的加权平均
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
        
        # 使用统一的性能限制计算
        performance_limits = self._calculate_performance_limits(model, system_spec)
        
        compute_limited_tps = performance_limits["compute_limited_tps"]
        memory_limited_tps = performance_limits["memory_limited_tps"]
        actual_tps = performance_limits["actual_tps"]
        
        # 算力利用率：实际使用的算力 / 总算力
        flops_per_token = model.estimate_flops_per_token()
        total_compute = performance_limits["total_compute_tflops"] * 1e12  # TFLOPS转换为FLOPS
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