"""
基础模型类定义

定义所有LLM模型的基础结构和通用属性。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from pydantic import BaseModel as PydanticModel, Field

# 避免循环导入
if TYPE_CHECKING:
    from ..estimator.op_level_estimator import LayerProfile


@dataclass
class ModelSpecs:
    """模型规格数据类"""
    name: str
    parameters: int  # 参数量（单位：B）
    layers: int  # 层数
    hidden_size: int  # 隐藏维度
    intermediate_size: int  # FFN 中间层维度
    attention_heads: int  # 注意力头数
    head_dim: int  # 注意力头维度
    num_key_value_heads: int  # 键值对注意力头数
    vocab_size: int  # 词汇表大小
    max_position_embeddings: int  # 最大位置编码
    model_type: str  # 模型类型
    # MoE related parameters
    num_experts: Optional[int] = None  # 专家数量
    experts_per_token: Optional[int] = None  # 每个token激活的专家数
    expert_capacity: Optional[int] = None  # 专家容量限制
    moe_layers: Optional[List[int]] = None  # MoE层位置，如果None则所有FFN层都是MoE


class ModelConfig(PydanticModel):
    """模型配置类"""
    context_length: int = Field(default=4096, description="上下文长度")
    batch_size: int = Field(default=1, description="批次大小")
    precision: str = Field(default="fp16", description="精度类型")
    use_kv_cache: bool = Field(default=True, description="是否使用KV缓存")
    inference_mode: str = Field(default="prefill", description="推理模式：prefill或decode")
    temperature: float = Field(default=0.7, description="温度参数")
    top_p: float = Field(default=0.9, description="top-p采样参数")
    max_new_tokens: int = Field(default=512, description="最大生成token数")


class BaseModel(ABC):
    """所有LLM模型的基础类"""
    
    def __init__(self, specs: ModelSpecs, config: Optional[ModelConfig] = None, **kwargs):
        self.specs = specs
        self.config = config or ModelConfig()
        
        # 使用kwargs更新配置
        if kwargs:
            self.update_config(**kwargs)
    
    @property
    def name(self) -> str:
        """模型名称"""
        return self.specs.name
    
    @property 
    def parameters(self) -> int:
        """参数量（单位：B）"""
        return self.specs.parameters
    
    @abstractmethod
    def calculate_memory_usage(self, precision: str = "fp16") -> Dict[str, float]:
        """
        计算模型内存使用量
        
        Args:
            precision: 精度类型 ("fp32", "fp16", "int8", "int4")
            
        Returns:
            包含各种内存使用量的字典
        """
        pass
    
    @abstractmethod
    def estimate_flops_per_token(self) -> float:
        """
        估算每个token的计算量（FLOPS）
        
        Returns:
            每个token的FLOPS数
        """
        pass
    
    @abstractmethod
    def decompose_to_ops(self) -> List['LayerProfile']:
        """
        将模型分解为具体的操作
        
        Returns:
            包含所有层操作分解的列表
        """
        pass
    
    def estimate_memory_per_token(self) -> float:
        """
        估算每个token的内存访问量（bytes）
        
        Returns:
            每个token的内存访问量
        """
        # 简化估算：主要是权重访问
        precision_bytes = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "int8": 1,
            "int4": 0.5
        }.get(self.config.precision, 2)
        
        # 估算每个token需要访问的参数量
        params_per_token = self.specs.parameters * 1e9  # 转换为实际参数数量
        memory_per_token = params_per_token * precision_bytes
        
        return memory_per_token
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型基本信息"""
        return {
            "name": self.specs.name,
            "parameters": f"{self.specs.parameters}B",
            "layers": self.specs.layers,
            "hidden_size": self.specs.hidden_size,
            "attention_heads": self.specs.attention_heads,
            "vocab_size": self.specs.vocab_size,
            "max_position_embeddings": self.specs.max_position_embeddings,
            "model_type": self.specs.model_type,
            "config": self.config.model_dump()
        }
    
    def update_config(self, **kwargs) -> None:
        """更新模型配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}") 