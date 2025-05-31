#!/usr/bin/env python3
"""
LLM-Estimate 主程序入口
"""

import sys
import os

# 如果作为独立脚本运行，添加项目根目录到路径
if __name__ == "__main__":
    # 获取项目根目录（当前文件的上级目录）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

try:
    from llm_estimate.cli.commands import main
except ImportError:
    # 如果导入失败，尝试相对导入
    try:
        from .cli.commands import main
    except ImportError:
        print("错误: 无法导入命令模块")
        print("请确保在正确的目录下运行，或者使用项目根目录的 llm_estimate.py 脚本")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main()) 