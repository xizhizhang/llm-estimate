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
        compute_capability_tflops=660.0,  # Tensor Core FP16/BF16算力（稠密）
        memory_bandwidth_gb_s=1008.0,
        memory_capacity_gb=24.0,
        release_year=2022,
        price_usd=1599,
        power_consumption_w=450
    ),

    "h100": AcceleratorSpecs(
        name="H100-80GB",
        manufacturer="NVIDIA",
        device_type="gpu",
        compute_capability_tflops=1979.0,  # Tensor Core BF16/FP16算力（稠密）
        memory_bandwidth_gb_s=2039.0,
        memory_capacity_gb=80.0,
        release_year=2022,
        price_usd=25000,
        power_consumption_w=700
    ),
    
    # Intel CPU系列（作为加速器）
    "i9-13900k": AcceleratorSpecs(
        name="i9-13900K",
        manufacturer="Intel",
        device_type="cpu",
        compute_capability_tflops=1.2,  # 估算值，基于AVX-512
        memory_bandwidth_gb_s=76.8,    # DDR5-4800双通道
        memory_capacity_gb=128.0,      # 最大支持内存
        release_year=2022,
        price_usd=589,
        power_consumption_w=125
    ),
    
    # AMD CPU系列
    "ryzen-9-7950x": AcceleratorSpecs(
        name="Ryzen-9-7950X",
        manufacturer="AMD",
        device_type="cpu",
        compute_capability_tflops=1.1,
        memory_bandwidth_gb_s=83.2,    # DDR5-5200双通道
        memory_capacity_gb=128.0,
        release_year=2022,
        price_usd=699,
        power_consumption_w=170
    ),
    
    # Apple Silicon系列
    "m2-ultra": AcceleratorSpecs(
        name="M2-Ultra",
        manufacturer="Apple",
        device_type="soc",  # System on Chip
        compute_capability_tflops=27.2,  # GPU部分算力
        memory_bandwidth_gb_s=800.0,
        memory_capacity_gb=192.0,
        release_year=2023,
        price_usd=5000,  # 估算
        power_consumption_w=100
    ),
    
    # Google TPU系列
    "tpu-v4": AcceleratorSpecs(
        name="TPU-v4",
        manufacturer="Google",
        device_type="tpu",
        compute_capability_tflops=275.0,  # BF16算力
        memory_bandwidth_gb_s=1200.0,
        memory_capacity_gb=32.0,
        release_year=2021,
        price_usd=None,  # 云服务
        power_consumption_w=200
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
            "compute_capability": [],
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
        # 按算力排序
        sorted_by_compute = sorted(
            accelerators_data, 
            key=lambda x: x[1]["compute_capability_tflops"], 
            reverse=True
        )
        comparison["rankings"]["compute_capability"] = [x[0] for x in sorted_by_compute]
        
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
        
        # 按能效比排序（算力/功耗）
        sorted_by_efficiency = sorted(
            accelerators_data,
            key=lambda x: x[1]["compute_capability_tflops"] / (x[1]["power_consumption_w"] or 1),
            reverse=True
        )
        comparison["rankings"]["performance_per_watt"] = [x[0] for x in sorted_by_efficiency]
    
    return comparison 