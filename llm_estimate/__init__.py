"""
LLM-Estimate: 大语言模型性能估算工具

这是一个用于估算大语言模型在不同硬件配置下性能表现的工具包。
统一GPU、CPU、TPU等计算设备为加速器概念，专注于算力和内存带宽。
"""

__version__ = "0.1.0"
__author__ = "Zhang Weimin"
__email__ = "zhangwm@example.com"

from .estimator.base import PerformanceEstimator
from .estimator.op_level_estimator import OpLevelEstimator, OpType, OpProfile, LayerProfile
from .models.registry import ModelRegistry
from .hardware.base import AcceleratorSpec, SystemSpec
from .hardware.accelerator import create_accelerator, list_supported_accelerators

__all__ = [
    "PerformanceEstimator",
    "OpLevelEstimator",
    "OpType", 
    "OpProfile",
    "LayerProfile",
    "ModelRegistry", 
    "AcceleratorSpec",
    "SystemSpec",
    "create_accelerator",
    "list_supported_accelerators",
] 