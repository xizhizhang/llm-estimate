"""
基础硬件类定义

定义所有硬件的基础结构和通用属性。
将GPU、CPU等统一抽象为加速器，专注于算力和存储带宽。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pydantic import BaseModel as PydanticModel, Field


@dataclass 
class AcceleratorSpecs:
    """加速器规格数据类"""
    name: str
    manufacturer: str
    device_type: str  # "gpu", "cpu", "tpu", "npu" 等，仅用于分类
    # 核心性能指标
    compute_capability_tflops: float  # 计算能力 (TFLOPS)
    memory_bandwidth_gb_s: float     # 内存带宽 (GB/s)
    memory_capacity_gb: float        # 内存容量 (GB)
    # 辅助信息
    release_year: int
    price_usd: Optional[float] = None
    power_consumption_w: Optional[float] = None


class AcceleratorConfig(PydanticModel):
    """加速器配置类"""
    power_limit: Optional[int] = Field(default=None, description="功耗限制(W)")
    cooling_solution: str = Field(default="air", description="散热方案")
    overclock_enabled: bool = Field(default=False, description="是否超频")
    utilization_target: float = Field(default=0.8, description="目标利用率")
    precision: str = Field(default="fp16", description="计算精度")


class AcceleratorSpec(ABC):
    """统一的加速器基础类"""
    
    def __init__(self, specs: AcceleratorSpecs, config: Optional[AcceleratorConfig] = None):
        self.specs = specs
        self.config = config or AcceleratorConfig()
    
    @property
    def name(self) -> str:
        """加速器名称"""
        return self.specs.name
    
    @property
    def device_type(self) -> str:
        """设备类型（仅用于分类）"""
        return self.specs.device_type
    
    @property
    def compute_capability_tflops(self) -> float:
        """计算能力 (TFLOPS)"""
        base_tflops = self.specs.compute_capability_tflops
        
        # 根据精度调整算力
        precision_multiplier = {
            "fp32": 1.0,
            "fp16": 2.0,
            "bf16": 2.0,
            "int8": 4.0,
            "int4": 8.0
        }.get(self.config.precision, 1.0)
        
        # 考虑超频
        overclock_multiplier = 1.1 if self.config.overclock_enabled else 1.0
        
        return base_tflops * precision_multiplier * overclock_multiplier
    
    @property
    def memory_bandwidth_gb_s(self) -> float:
        """内存带宽 (GB/s)"""
        base_bandwidth = self.specs.memory_bandwidth_gb_s
        
        # 考虑超频对内存带宽的影响
        overclock_multiplier = 1.05 if self.config.overclock_enabled else 1.0
        
        return base_bandwidth * overclock_multiplier
    
    @property
    def memory_capacity_gb(self) -> float:
        """内存容量 (GB)"""
        return self.specs.memory_capacity_gb
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        获取核心性能指标
        
        Returns:
            性能指标字典，包含算力、内存带宽、内存容量等
        """
        return {
            "compute_capability_tflops": self.compute_capability_tflops,
            "memory_bandwidth_gb_s": self.memory_bandwidth_gb_s,
            "memory_capacity_gb": self.memory_capacity_gb,
            "power_consumption_w": self.specs.power_consumption_w or 0,
            "utilization_target": self.config.utilization_target,
            "precision": self.config.precision
        }
    
    def calculate_throughput(self, workload: Dict[str, Any]) -> float:
        """
        计算特定工作负载下的吞吐量
        
        Args:
            workload: 工作负载描述，包含 flops_per_token, memory_per_token 等
            
        Returns:
            吞吐量 (tokens/s)
        """
        flops_per_token = workload.get("flops_per_token", 0)
        memory_per_token = workload.get("memory_per_token", 0)
        
        # 计算受限因子
        compute_limited_tps = (self.compute_capability_tflops * 1e12 * self.config.utilization_target) / flops_per_token
        memory_limited_tps = (self.memory_bandwidth_gb_s * 1e9 * self.config.utilization_target) / memory_per_token
        
        # 返回瓶颈限制的吞吐量
        return min(compute_limited_tps, memory_limited_tps)
    
    def estimate_latency(self, workload: Dict[str, Any]) -> float:
        """
        估算首token延迟
        
        Args:
            workload: 工作负载描述，应包含:
                - flops_per_token: 每个token的浮点运算量
                - model_size_gb: 模型大小(GB)，用于计算内存读取时间
                - memory_overhead_ms: 其他内存开销(ms)，默认5ms
            
        Returns:
            延迟 (ms)
        """
        flops_per_token = workload.get("flops_per_token", 0)
        model_size_gb = workload.get("model_size_gb", 0)
        
        # 计算时间（秒）- 受算力限制
        compute_time = flops_per_token / (self.compute_capability_tflops * 1e12)
        
        # 内存读取时间（秒）- 受内存带宽限制
        # 首token需要读取完整模型权重
        memory_read_time = model_size_gb / self.memory_bandwidth_gb_s
        
        # 取计算时间和内存读取时间的较大值（瓶颈）
        bottleneck_time = max(compute_time, memory_read_time)
        
        # 加上其他开销
        memory_overhead = workload.get("memory_overhead_ms", 5.0) / 1000  # 默认5ms开销
        
        total_latency_s = bottleneck_time + memory_overhead
        
        return total_latency_s * 1000  # 转换为毫秒
    
    def check_memory_fit(self, required_memory_gb: float) -> bool:
        """检查内存是否足够"""
        return required_memory_gb <= self.memory_capacity_gb
    
    def get_accelerator_info(self) -> Dict[str, Any]:
        """获取加速器基本信息"""
        return {
            "name": self.specs.name,
            "manufacturer": self.specs.manufacturer,
            "device_type": self.specs.device_type,
            "compute_capability_tflops": self.compute_capability_tflops,
            "memory_bandwidth_gb_s": self.memory_bandwidth_gb_s,
            "memory_capacity_gb": self.memory_capacity_gb,
            "release_year": self.specs.release_year,
            "price_usd": self.specs.price_usd,
            "power_consumption_w": self.specs.power_consumption_w,
            "config": self.config.model_dump()
        }
    
    def update_config(self, **kwargs) -> None:
        """更新加速器配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")


class SystemSpec:
    """系统硬件规格组合"""
    
    def __init__(self, accelerators: List[AcceleratorSpec] = None):
        """
        初始化系统规格
        
        Args:
            accelerators: 加速器列表，支持多卡/多设备配置
        """
        self.accelerators = accelerators or []
    
    def add_accelerator(self, accelerator: AcceleratorSpec) -> None:
        """添加加速器"""
        self.accelerators.append(accelerator)
    
    def get_total_compute_capability(self) -> float:
        """获取总算力 (TFLOPS)"""
        return sum(acc.compute_capability_tflops for acc in self.accelerators)
    
    def get_total_memory_capacity(self) -> float:
        """获取总内存容量 (GB)"""
        return sum(acc.memory_capacity_gb for acc in self.accelerators)
    
    def get_total_memory_bandwidth(self) -> float:
        """获取总内存带宽 (GB/s)"""
        return sum(acc.memory_bandwidth_gb_s for acc in self.accelerators)
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            "accelerator_count": len(self.accelerators),
            "accelerators": [acc.get_accelerator_info() for acc in self.accelerators],
            "total_compute_capability_tflops": self.get_total_compute_capability(),
            "total_memory_capacity_gb": self.get_total_memory_capacity(),
            "total_memory_bandwidth_gb_s": self.get_total_memory_bandwidth()
        }
        return info
    
    def estimate_total_power(self) -> float:
        """估算总功耗"""
        accelerator_power = sum(
            acc.specs.power_consumption_w or 0 
            for acc in self.accelerators
        )
        
        # 其他系统组件功耗估算
        system_overhead = 50  # 主板、内存、存储等
        
        return accelerator_power + system_overhead
    
    def check_compatibility(self, required_memory_gb: float) -> Dict[str, Any]:
        """检查系统兼容性"""
        total_memory = self.get_total_memory_capacity()
        total_power = self.estimate_total_power()
        
        compatibility = {
            "memory_sufficient": total_memory >= required_memory_gb,
            "power_manageable": total_power <= 1500,  # 假设电源限制1500W
            "cooling_adequate": all(
                (acc.specs.power_consumption_w or 0) <= 400 
                for acc in self.accelerators
            ),
            "details": {
                "required_memory_gb": required_memory_gb,
                "available_memory_gb": total_memory,
                "total_power_w": total_power
            }
        }
        
        return compatibility 