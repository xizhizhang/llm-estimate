"""
模型配置管理

定义各种模型的默认配置和预设参数。
"""

from typing import Dict, Any
from .base import ModelConfig

# Qwen3系列默认配置
QWEN3_DEFAULT_PARAMS = {
    "context_length": 32768,
    "batch_size": 1,
    "precision": "bf16",
    "use_kv_cache": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 512
}

# Qwen3-MoE系列默认配置
QWEN3_MOE_DEFAULT_PARAMS = {
    "context_length": 32768,
    "batch_size": 1,
    "precision": "bf16",
    "use_kv_cache": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 512
}

# 模型特定配置覆盖
MODEL_SPECIFIC_CONFIGS = {
    "qwen3-0.6b": {
        "context_length": 4096,
        "max_new_tokens": 1024
    },
    "qwen3-1.7b": {
        "context_length": 4096,
        "max_new_tokens": 1024
    },
}

# 精度相关配置
PRECISION_CONFIGS = {
    "fp32": {
        "bytes_per_param": 4,
        "description": "32位浮点数，最高精度"
    },
    "fp16": {
        "bytes_per_param": 2,
        "description": "16位浮点数，平衡精度和性能"
    },
    "int8": {
        "bytes_per_param": 1,
        "description": "8位整数，大幅减少内存使用"
    },
    "int4": {
        "bytes_per_param": 0.5,
        "description": "4位整数，极大减少内存使用"
    }
}


class ModelConfigManager:
    """模型配置管理器"""
    
    @staticmethod
    def get_default_config(model_type: str) -> ModelConfig:
        """
        获取模型类型的默认配置
        
        Args:
            model_type: 模型类型 ("llama", "qwen")
            
        Returns:
            默认配置对象
        """
        if model_type == "qwen3":
            return ModelConfig(**QWEN3_DEFAULT_PARAMS)
        elif model_type == "qwen3-moe":
            return ModelConfig(**QWEN3_MOE_DEFAULT_PARAMS)
        else:
            # 使用通用默认配置
            return ModelConfig()
    
    @staticmethod
    def get_model_config(model_name: str) -> ModelConfig:
        """
        获取特定模型的配置
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型配置对象
        """
        # 确定模型类型
        if "qwen3" in model_name.lower():
            base_config = QWEN3_DEFAULT_PARAMS.copy()
        elif "qwen3-moe" in model_name.lower():
            base_config = QWEN3_MOE_DEFAULT_PARAMS.copy()
        else:
            base_config = {}
        
        # 应用模型特定配置
        if model_name in MODEL_SPECIFIC_CONFIGS:
            base_config.update(MODEL_SPECIFIC_CONFIGS[model_name])
        
        return ModelConfig(**base_config)
    
    @staticmethod
    def get_precision_info(precision: str) -> Dict[str, Any]:
        """
        获取精度相关信息
        
        Args:
            precision: 精度类型
            
        Returns:
            精度信息字典
        """
        if precision not in PRECISION_CONFIGS:
            raise ValueError(f"Unsupported precision: {precision}")
        
        return PRECISION_CONFIGS[precision]
    
    @staticmethod
    def list_supported_precisions() -> list:
        """获取支持的精度类型列表"""
        return list(PRECISION_CONFIGS.keys())
    
    @staticmethod
    def validate_config(config: ModelConfig) -> bool:
        """
        验证配置有效性
        
        Args:
            config: 模型配置
            
        Returns:
            配置是否有效
        """
        # 检查精度类型
        if config.precision not in PRECISION_CONFIGS:
            return False
        
        # 检查数值范围
        if config.context_length <= 0:
            return False
        
        if config.batch_size <= 0:
            return False
        
        if not (0.0 <= config.temperature <= 2.0):
            return False
        
        if not (0.0 <= config.top_p <= 1.0):
            return False
        
        if config.max_new_tokens <= 0:
            return False
        
        return True 