"""
pytest配置文件

定义测试的全局配置和fixture。
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_model_config():
    """示例模型配置"""
    return {
        "batch_size": 1,
        "context_length": 4096,
        "precision": "fp16"
    }


@pytest.fixture
def sample_hardware_config():
    """示例硬件配置"""
    return {
        "gpu": "rtx-4090",
        "memory": "32GB"
    } 