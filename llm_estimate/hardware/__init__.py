"""
硬件管理模块

提供统一的加速器抽象，将GPU、CPU、TPU等计算设备统一为加速器概念。
专注于算力（FLOPS）和存储带宽（GB/s）的核心性能指标。
"""

from .base import AcceleratorSpec, AcceleratorSpecs, AcceleratorConfig, SystemSpec
from .accelerator import (
    GenericAccelerator, 
    create_accelerator,
    list_supported_accelerators,
    get_accelerator_by_type,
    compare_accelerators,
    ACCELERATOR_SPECS
)

__all__ = [
    # 基础类
    "AcceleratorSpec",
    "AcceleratorSpecs", 
    "AcceleratorConfig",
    "SystemSpec",
    
    # 加速器实现
    "GenericAccelerator",
    "create_accelerator",
    "list_supported_accelerators",
    "get_accelerator_by_type",
    "compare_accelerators",
    "ACCELERATOR_SPECS",
] 