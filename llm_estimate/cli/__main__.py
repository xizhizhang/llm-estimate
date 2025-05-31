#!/usr/bin/env python3
"""
CLI模块的主入口

支持使用 python -m llm_estimate.cli 方式运行
"""

import sys
import os

# 确保项目根目录在Python路径中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .commands import main

if __name__ == "__main__":
    sys.exit(main()) 