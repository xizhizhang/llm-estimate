"""
模型配置管理

定义各种模型的默认配置和预设参数。
"""

from typing import Dict, Any
from .base import ModelConfig


# Llama系列默认配置
LLAMA_DEFAULT_PARAMS = {
    "context_length": 4096,
    "batch_size": 1,
    "precision": "fp16",
    "use_kv_cache": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 512
}

# Qwen系列默认配置
QWEN_DEFAULT_PARAMS = {
    "context_length": 8192,
    "batch_size": 1,
    "precision": "fp16",
    "use_kv_cache": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 512
}

# 模型特定配置覆盖
MODEL_SPECIFIC_CONFIGS = {
    # Llama-2系列
    "llama-2-7b": {
        "context_length": 4096,
        "max_new_tokens": 512
    },
    
    # Qwen系列
    "qwen-7b": {
        "context_length": 8192,
        "max_new_tokens": 1024
    },
    "qwen-14b": {
        "context_length": 8192,
        "max_new_tokens": 1024
    },
    "qwen-72b": {
        "context_length": 32768,
        "max_new_tokens": 2048,
        "batch_size": 1
    }
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
        if model_type == "llama":
            return ModelConfig(**LLAMA_DEFAULT_PARAMS)
        elif model_type == "qwen":
            return ModelConfig(**QWEN_DEFAULT_PARAMS)
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
        if "llama" in model_name.lower():
            base_config = LLAMA_DEFAULT_PARAMS.copy()
        elif "qwen" in model_name.lower():
            base_config = QWEN_DEFAULT_PARAMS.copy()
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