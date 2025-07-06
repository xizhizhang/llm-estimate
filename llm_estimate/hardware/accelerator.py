"""
统一加速器实现

将GPU、CPU、TPU、NPU等计算设备统一抽象为加速器。
"""

from typing import Dict, Any, Optional
from .base import AcceleratorSpec, AcceleratorSpecs, AcceleratorConfig


class GenericAccelerator(AcceleratorSpec):
    """通用加速器实现"""
    
    def __init__(self, specs: AcceleratorSpecs, config: Optional[AcceleratorConfig] = None):
        super().__init__(specs, config)


# 预定义的常见加速器规格
ACCELERATOR_SPECS = {
    # NVIDIA GPU系列
    "rtx-4090": AcceleratorSpecs(
        name="RTX-4090",
        manufacturer="NVIDIA",
        device_type="gpu",
        compute_capability_tflops={
            "fp16": 330.0,     # FP16算力
            "bf16": 330.0,     # BF16算力
            "fp8": 660.0,      # FP8算力（稠密）
        },
        memory_bandwidth_gb_s=1008.0,
        memory_capacity_gb=24.0,
        release_year=2022,
        price_usd=1599,
        power_consumption_w=450
    ),

    "rtx-6001": AcceleratorSpecs(
        name="RTX-6001",
        manufacturer="NVIDIA",
        device_type="gpu",
        compute_capability_tflops={
            "fp16": 500.0,     # FP16算力
            "bf16": 500.0,     # BF16算力
            "fp8": 1000.0,     # FP8算力（稠密）
            "fp4": 2000.0,     # FP4算力（稠密）
        },
        memory_bandwidth_gb_s=1800.0,
        memory_capacity_gb=96.0,
        release_year=2025,
        price_usd=10000,
        power_consumption_w=600
    ),
    
    # Apple Silicon系列
    "m2-ultra": AcceleratorSpecs(
        name="M2-Ultra",
        manufacturer="Apple",
        device_type="soc",  # System on Chip
        compute_capability_tflops={
            "fp32": 27.2,      # FP32算力（GPU部分）
            "fp16": 54.4,      # FP16算力
            "bf16": 54.4,      # BF16算力
            "int8": 108.8,     # INT8算力
            "int16": 54.4,     # INT16算力
        },
        memory_bandwidth_gb_s=800.0,
        memory_capacity_gb=192.0,
        release_year=2023,
        price_usd=5000,  # 估算
        power_consumption_w=100
    ),
}


def create_accelerator(accelerator_name: str, 
                      config: Optional[AcceleratorConfig] = None) -> GenericAccelerator:
    """
    创建加速器实例
    
    Args:
        accelerator_name: 加速器名称
        config: 可选的配置参数
        
    Returns:
        加速器实例
        
    Raises:
        ValueError: 不支持的加速器型号
    """
    accelerator_name_lower = accelerator_name.lower()
    
    if accelerator_name_lower not in ACCELERATOR_SPECS:
        raise ValueError(f"不支持的加速器型号: {accelerator_name}")
    
    specs = ACCELERATOR_SPECS[accelerator_name_lower]
    return GenericAccelerator(specs, config)


def list_supported_accelerators() -> Dict[str, Dict[str, Any]]:
    """
    列出所有支持的加速器
    
    Returns:
        加速器信息字典
    """
    result = {}
    
    for name, specs in ACCELERATOR_SPECS.items():
        result[name] = {
            "name": specs.name,
            "manufacturer": specs.manufacturer,
            "device_type": specs.device_type,
            "compute_capability_tflops": specs.compute_capability_tflops,
            "memory_bandwidth_gb_s": specs.memory_bandwidth_gb_s,
            "memory_capacity_gb": specs.memory_capacity_gb,
            "release_year": specs.release_year,
            "price_usd": specs.price_usd,
            "power_consumption_w": specs.power_consumption_w
        }
    
    return result


def get_accelerator_by_type(device_type: str) -> Dict[str, Dict[str, Any]]:
    """
    按设备类型筛选加速器
    
    Args:
        device_type: 设备类型 ("gpu", "cpu", "tpu", "soc")
        
    Returns:
        筛选后的加速器信息
    """
    all_accelerators = list_supported_accelerators()
    
    return {
        name: info for name, info in all_accelerators.items()
        if info["device_type"] == device_type
    }


def compare_accelerators(accelerator_names: list) -> Dict[str, Any]:
    """
    比较多个加速器的性能
    
    Args:
        accelerator_names: 加速器名称列表
        
    Returns:
        比较结果
    """
    comparison = {
        "accelerators": [],
        "rankings": {
            "compute_capability_fp32": [],
            "compute_capability_fp16": [],
            "compute_capability_bf16": [],
            "compute_capability_int8": [],
            "memory_bandwidth": [],
            "memory_capacity": [],
            "performance_per_watt": []
        }
    }
    
    accelerators_data = []
    
    for name in accelerator_names:
        try:
            acc = create_accelerator(name)
            info = acc.get_accelerator_info()
            accelerators_data.append((name, info))
            comparison["accelerators"].append(info)
        except ValueError:
            continue
    
    # 按各项指标排序
    if accelerators_data:
        # 按FP32算力排序
        sorted_by_fp32 = sorted(
            accelerators_data, 
            key=lambda x: x[1]["compute_capability_tflops"].get("fp32", 0), 
            reverse=True
        )
        comparison["rankings"]["compute_capability_fp32"] = [x[0] for x in sorted_by_fp32]
        
        # 按FP16算力排序
        sorted_by_fp16 = sorted(
            accelerators_data, 
            key=lambda x: x[1]["compute_capability_tflops"].get("fp16", 0), 
            reverse=True
        )
        comparison["rankings"]["compute_capability_fp16"] = [x[0] for x in sorted_by_fp16]
        
        # 按BF16算力排序
        sorted_by_bf16 = sorted(
            accelerators_data, 
            key=lambda x: x[1]["compute_capability_tflops"].get("bf16", 0), 
            reverse=True
        )
        comparison["rankings"]["compute_capability_bf16"] = [x[0] for x in sorted_by_bf16]
        
        # 按INT8算力排序
        sorted_by_int8 = sorted(
            accelerators_data, 
            key=lambda x: x[1]["compute_capability_tflops"].get("int8", 0), 
            reverse=True
        )
        comparison["rankings"]["compute_capability_int8"] = [x[0] for x in sorted_by_int8]
        
        # 按内存带宽排序
        sorted_by_bandwidth = sorted(
            accelerators_data,
            key=lambda x: x[1]["memory_bandwidth_gb_s"],
            reverse=True
        )
        comparison["rankings"]["memory_bandwidth"] = [x[0] for x in sorted_by_bandwidth]
        
        # 按内存容量排序
        sorted_by_memory = sorted(
            accelerators_data,
            key=lambda x: x[1]["memory_capacity_gb"],
            reverse=True
        )
        comparison["rankings"]["memory_capacity"] = [x[0] for x in sorted_by_memory]
        
        # 按能效比排序（FP16算力/功耗）
        sorted_by_efficiency = sorted(
            accelerators_data,
            key=lambda x: x[1]["compute_capability_tflops"].get("fp16", 0) / (x[1]["power_consumption_w"] or 1),
            reverse=True
        )
        comparison["rankings"]["performance_per_watt"] = [x[0] for x in sorted_by_efficiency]
    
    return comparison 


def get_accelerator_by_precision(precision: str) -> Dict[str, Dict[str, Any]]:
    """
    按精度筛选并排序加速器
    
    Args:
        precision: 精度类型 ("fp32", "fp16", "bf16", "int8", "int4", "tf32", "fp8")
        
    Returns:
        按指定精度算力排序的加速器信息
    """
    all_accelerators = list_supported_accelerators()
    
    # 筛选支持指定精度的加速器并排序
    supported_accelerators = {}
    for name, info in all_accelerators.items():
        if precision in info["compute_capability_tflops"]:
            supported_accelerators[name] = info
    
    # 按指定精度的算力排序
    sorted_items = sorted(
        supported_accelerators.items(),
        key=lambda x: x[1]["compute_capability_tflops"][precision],
        reverse=True
    )
    
    return dict(sorted_items)


def get_best_accelerator_for_precision(precision: str, device_type: str = None) -> Dict[str, Any]:
    """
    获取指定精度下最佳的加速器
    
    Args:
        precision: 精度类型
        device_type: 可选的设备类型限制
        
    Returns:
        最佳加速器信息
    """
    candidates = get_accelerator_by_precision(precision)
    
    if device_type:
        candidates = {
            name: info for name, info in candidates.items()
            if info["device_type"] == device_type
        }
    
    if not candidates:
        return {}
    
    # 返回算力最高的加速器
    best_name = next(iter(candidates))
    return {
        "name": best_name,
        "info": candidates[best_name],
        "compute_capability_tflops": candidates[best_name]["compute_capability_tflops"][precision]
    }


def compare_precision_performance(accelerator_name: str) -> Dict[str, Any]:
    """
    比较单个加速器在不同精度下的性能
    
    Args:
        accelerator_name: 加速器名称
        
    Returns:
        精度性能比较结果
    """
    try:
        acc = create_accelerator(accelerator_name)
        info = acc.get_accelerator_info()
        
        compute_capabilities = info["compute_capability_tflops"]
        
        # 计算各精度相对于FP32的性能倍数
        fp32_performance = compute_capabilities.get("fp32", 1.0)
        precision_ratios = {}
        
        for precision, tflops in compute_capabilities.items():
            if precision != "fp32":
                precision_ratios[precision] = tflops / fp32_performance
        
        # 计算能效比（TFLOPS/W）
        power_w = info["power_consumption_w"] or 1
        efficiency_ratios = {}
        
        for precision, tflops in compute_capabilities.items():
            efficiency_ratios[precision] = tflops / power_w
        
        return {
            "accelerator": accelerator_name,
            "compute_capabilities": compute_capabilities,
            "precision_ratios": precision_ratios,
            "efficiency_ratios": efficiency_ratios,
            "supported_precisions": list(compute_capabilities.keys()),
            "best_precision": max(compute_capabilities.items(), key=lambda x: x[1])
        }
    
    except ValueError:
        return {"error": f"Unsupported accelerator: {accelerator_name}"}


def estimate_model_performance(accelerator_name: str, model_precision: str, 
                             model_size_gb: float, batch_size: int = 1) -> Dict[str, Any]:
    """
    估算模型在指定加速器和精度下的性能
    
    Args:
        accelerator_name: 加速器名称
        model_precision: 模型精度
        model_size_gb: 模型大小（GB）
        batch_size: 批次大小
        
    Returns:
        性能估算结果
    """
    try:
        acc = create_accelerator(accelerator_name)
        
        # 检查内存容量
        if not acc.check_memory_fit(model_size_gb):
            return {
                "error": f"Model size ({model_size_gb:.1f}GB) exceeds accelerator memory capacity ({acc.memory_capacity_gb:.1f}GB)"
            }
        
        # 获取指定精度的算力
        compute_tflops = acc.get_compute_capability_by_precision(model_precision)
        
        if compute_tflops == 0:
            return {
                "error": f"Precision {model_precision} not supported by {accelerator_name}"
            }
        
        # 估算性能指标
        memory_utilization = model_size_gb / acc.memory_capacity_gb
        
        # 简单估算（实际性能会更复杂）
        estimated_throughput_tokens_per_s = compute_tflops * 1000 / model_size_gb  # 简化计算
        estimated_latency_ms = (model_size_gb / acc.memory_bandwidth_gb_s) * 1000  # 简化计算
        
        return {
            "accelerator": accelerator_name,
            "model_precision": model_precision,
            "model_size_gb": model_size_gb,
            "batch_size": batch_size,
            "compute_capability_tflops": compute_tflops,
            "memory_utilization": memory_utilization,
            "estimated_throughput_tokens_per_s": estimated_throughput_tokens_per_s,
            "estimated_latency_ms": estimated_latency_ms,
            "memory_bandwidth_gb_s": acc.memory_bandwidth_gb_s,
            "memory_capacity_gb": acc.memory_capacity_gb
        }
    
    except ValueError:
        return {"error": f"Unsupported accelerator: {accelerator_name}"} 