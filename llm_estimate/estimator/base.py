"""
基础估算器类

提供性能估算的核心接口和基础实现。
使用统一的加速器抽象进行性能计算。
"""

from typing import Dict, Any, Optional, List

from ..models.registry import model_registry
from ..models.base import BaseModel
from ..hardware.base import SystemSpec
from ..hardware.accelerator import create_accelerator
from .op_level_estimator import OpLevelEstimator


class PerformanceEstimator:
    """性能估算器主类"""
    
    def __init__(self):
        self.model_registry = model_registry
        self.op_level_estimator = OpLevelEstimator()
    
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
