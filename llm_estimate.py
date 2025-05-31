#!/usr/bin/env python3
"""
LLM-Estimate 项目入口脚本

支持不安装包直接运行项目，自动处理模块路径。
"""

import sys
import os

# 将项目根目录添加到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入并运行CLI
try:
    from llm_estimate.cli.commands import main
    
    if __name__ == "__main__":
        sys.exit(main())
        
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保您在项目根目录下运行此脚本")
    sys.exit(1)
except Exception as e:
    print(f"运行错误: {e}")
    sys.exit(1) 