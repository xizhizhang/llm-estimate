"""
全局系统设置

定义系统级配置参数和默认值。
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """系统设置类"""
    
    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    log_file: Optional[str] = Field(default=None, description="日志文件路径")
    
    # 输出配置
    default_output_format: str = Field(default="table", description="默认输出格式")
    
    # 缓存配置
    cache_enabled: bool = Field(default=True, description="是否启用缓存")
    cache_dir: str = Field(default=".cache", description="缓存目录")
    cache_ttl: int = Field(default=3600, description="缓存过期时间(秒)")
    
    # 数据目录
    data_dir: str = Field(default="data", description="数据目录")
    
    # 性能配置
    parallel_estimation: bool = Field(default=True, description="是否启用并行估算")
    max_workers: int = Field(default=4, description="最大工作线程数")
    
    # 模型默认配置
    default_model_precision: str = Field(default="fp16", description="默认模型精度")
    default_batch_size: int = Field(default=1, description="默认批次大小")
    default_context_length: int = Field(default=4096, description="默认上下文长度")
    
    # 硬件配置
    auto_detect_hardware: bool = Field(default=True, description="是否自动检测硬件")
    
    # API配置 (未来扩展)
    api_base_url: Optional[str] = Field(default=None, description="API基础URL")
    api_timeout: int = Field(default=30, description="API超时时间(秒)")
    
    class Config:
        env_prefix = "LLM_ESTIMATE_"
        env_file = ".env"


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self._settings: Optional[Settings] = None
    
    def get_settings(self) -> Settings:
        """获取设置实例（单例模式）"""
        if self._settings is None:
            self._settings = Settings()
            self._ensure_directories()
        return self._settings
    
    def _ensure_directories(self) -> None:
        """确保必要的目录存在"""
        settings = self._settings
        
        # 创建缓存目录
        if settings.cache_enabled:
            cache_path = Path(settings.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
        
        # 创建数据目录
        data_path = Path(settings.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
    
    def update_settings(self, **kwargs) -> None:
        """更新设置"""
        if self._settings is None:
            self._settings = Settings()
        
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
    
    def get_data_path(self, *args) -> Path:
        """获取数据文件路径"""
        settings = self.get_settings()
        return Path(settings.data_dir) / Path(*args)
    
    def get_cache_path(self, *args) -> Path:
        """获取缓存文件路径"""
        settings = self.get_settings()
        return Path(settings.cache_dir) / Path(*args)


# 全局配置管理器实例
config_manager = ConfigManager()


def get_settings() -> Settings:
    """获取全局设置"""
    return config_manager.get_settings()


def get_data_path(*args) -> Path:
    """获取数据文件路径"""
    return config_manager.get_data_path(*args)


def get_cache_path(*args) -> Path:
    """获取缓存文件路径"""
    return config_manager.get_cache_path(*args) 