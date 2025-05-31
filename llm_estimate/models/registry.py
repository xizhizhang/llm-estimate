"""
模型注册表

统一管理所有支持的模型，提供模型查询、注册和管理功能。
"""

from typing import Dict, List, Optional, Type
from .base import BaseModel, ModelSpecs
from .llama import LlamaModel
from .qwen import QwenModel


class ModelRegistry:
    """模型注册表类"""
    
    def __init__(self):
        self._models: Dict[str, Type[BaseModel]] = {}
        self._model_specs: Dict[str, ModelSpecs] = {}
        self._register_built_in_models()
    
    def _register_built_in_models(self) -> None:
        """注册内置模型"""
        # 注册Llama系列模型
        self.register("llama", LlamaModel)
        
        # 注册Qwen系列模型
        self.register("qwen", QwenModel)
        
        # 添加模型规格数据
        self._load_model_specs()
    
    def _load_model_specs(self) -> None:
        """加载模型规格数据"""
        # Llama 2 系列
        self._model_specs.update({
            "llama-2-7b": ModelSpecs(
                name="llama-2-7b",
                parameters=7,
                layers=32,
                hidden_size=4096,
                attention_heads=32,
                vocab_size=32000,
                max_position_embeddings=4096,
                model_type="llama"
            ),
            # Qwen 系列
            "qwen3-8b": ModelSpecs(
                name="qwen3-8b",
                parameters=8,
                layers=36,
                hidden_size=4096,
                attention_heads=32,
                vocab_size=151936,
                max_position_embeddings=40960,
                model_type="qwen3"
            ),
        })
    
    def register(self, model_type: str, model_class: Type[BaseModel]) -> None:
        """
        注册模型类
        
        Args:
            model_type: 模型类型
            model_class: 模型类
        """
        self._models[model_type] = model_class
    
    def create_model(self, model_name: str, **config_kwargs) -> BaseModel:
        """
        创建模型实例
        
        Args:
            model_name: 模型名称
            **config_kwargs: 配置参数
            
        Returns:
            模型实例
        """
        if model_name not in self._model_specs:
            raise ValueError(f"Unsupported model: {model_name}")
        
        specs = self._model_specs[model_name]
        model_type = specs.model_type
        
        if model_type not in self._models:
            raise ValueError(f"Model type not registered: {model_type}")
        
        model_class = self._models[model_type]
        return model_class(specs, **config_kwargs)
    
    def list_models(self) -> List[str]:
        """获取所有支持的模型列表"""
        return list(self._model_specs.keys())
    
    def list_model_types(self) -> List[str]:
        """获取所有支持的模型类型"""
        return list(self._models.keys())
    
    def get_model_info(self, model_name: str) -> Dict:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型信息字典
        """
        if model_name not in self._model_specs:
            raise ValueError(f"Unsupported model: {model_name}")
        
        specs = self._model_specs[model_name]
        return {
            "name": specs.name,
            "parameters": f"{specs.parameters}B",
            "layers": specs.layers,
            "hidden_size": specs.hidden_size,
            "attention_heads": specs.attention_heads,
            "vocab_size": specs.vocab_size,
            "max_position_embeddings": specs.max_position_embeddings,
            "model_type": specs.model_type
        }
    
    def search_models(self, query: str) -> List[str]:
        """
        搜索模型
        
        Args:
            query: 搜索关键词
            
        Returns:
            匹配的模型名称列表
        """
        query = query.lower()
        return [
            name for name in self._model_specs.keys()
            if query in name.lower()
        ]


# 全局模型注册表实例
model_registry = ModelRegistry() 