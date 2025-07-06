# LLM-Estimate 项目结构设计

## 目录结构

```
llm-estimate/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包
├── setup.py                     # 项目安装配置
├── pyproject.toml              # 项目配置文件
├── .gitignore                  # Git忽略文件
├── .env.example                # 环境变量示例
│
├── llm_estimate/               # 主要代码目录
│   ├── __init__.py
│   ├── main.py                 # 主程序入口
│   │
│   ├── models/                 # 模型管理模块
│   │   ├── __init__.py
│   │   ├── base.py             # 基础模型类
│   │   ├── qwen.py             # Qwen系列模型
│   │   ├── registry.py         # 模型注册表
│   │   └── configs.py          # 模型配置管理
│   │
│   ├── hardware/               # 硬件管理模块
│   │   ├── __init__.py
│   │   ├── base.py             # 基础硬件类
│   │   ├── gpu.py              # GPU规格管理
│   │   ├── cpu.py              # CPU规格管理
│   │   ├── memory.py           # 内存规格管理
│   │   ├── specs.py            # 硬件规格数据
│   │   └── configs.py          # 硬件配置管理
│   │
│   ├── estimator/              # 估算引擎模块
│   │   ├── __init__.py
│   │   ├── base.py             # 基础估算器
│   │   ├── throughput.py       # 吞吐量估算
│   │   ├── latency.py          # 延迟估算
│   │   ├── memory_usage.py     # 内存使用估算
│   │   └── algorithms.py       # 核心估算算法
│   │
│   ├── config/                 # 全局配置模块
│   │   ├── __init__.py
│   │   └── settings.py         # 全局系统设置
│   │
│   ├── utils/                  # 工具模块
│   │   ├── __init__.py
│   │   ├── validators.py       # 参数验证
│   │   ├── formatters.py       # 数据格式化
│   │   ├── calculators.py      # 计算工具
│   │   └── exceptions.py       # 自定义异常
│   │
│   └── cli/                    # 命令行接口
│       ├── __init__.py
│       ├── commands.py         # CLI命令
│       └── interactive.py      # 交互式界面
│
├── tests/                      # 测试模块
│   ├── __init__.py
│   ├── conftest.py             # pytest配置
│   ├── test_models/            # 模型测试
│   ├── test_hardware/          # 硬件测试
│   ├── test_estimator/         # 估算器测试
│   ├── test_cli/               # CLI测试
│   └── test_integration/       # 集成测试
│
├── data/                       # 数据目录
│   ├── models/                 # 模型数据
│   │   ├── qwen_specs.json
│   │   └── ...
│   ├── hardware/               # 硬件数据
│   │   ├── gpu_specs.json
│   │   ├── cpu_specs.json
│   │   └── ...
│   └── benchmarks/             # 基准测试数据
│       ├── performance_data.json
│       └── validation_results.json
│
├── docs/                       # 文档目录
│   ├── user_guide.md           # 用户指南
│   ├── developer_guide.md      # 开发者指南
│   ├── model_support.md        # 支持的模型列表
│   └── examples/               # 使用示例
│       ├── basic_usage.py
│       ├── batch_estimation.py
│       └── custom_model.py
│
└── scripts/                    # 脚本目录
    ├── setup.sh                # 环境设置脚本
    ├── data_collector.py       # 数据收集脚本
    └── benchmark.py            # 基准测试脚本
```

## 模块划分详解

### 1. 模型管理模块 (models/)
**职责**: 管理各种LLM模型的规格、参数和配置
- `base.py`: 定义基础模型类，包含通用属性和方法
- `qwen.py`: Qwen系列模型实现（Qwen-7B、Qwen-14B等）
- `registry.py`: 模型注册表，统一管理所有支持的模型
- `configs.py`: 模型配置管理，包括默认参数、变体配置等

**核心功能**:
- 模型参数规格定义（参数量、层数、隐藏维度等）
- 模型配置管理（默认设置、推理参数等）
- 模型内存需求计算
- 模型推理复杂度分析

### 2. 硬件管理模块 (hardware/)
**职责**: 管理硬件配置和规格信息
- `gpu.py`: GPU规格管理（型号、显存、算力等）
- `cpu.py`: CPU规格管理（核心数、频率、架构等）
- `memory.py`: 内存规格管理（容量、带宽、类型等）
- `specs.py`: 硬件规格数据库
- `configs.py`: 硬件配置管理，包括性能参数、兼容性配置等

**核心功能**:
- 硬件规格数据库维护
- 硬件配置管理（性能调优参数等）
- 硬件性能参数计算
- 硬件兼容性检查

### 3. 估算引擎模块 (estimator/)
**职责**: 核心算法实现，估算性能指标
- `throughput.py`: 吞吐量估算算法
- `latency.py`: 延迟估算算法
- `memory_usage.py`: 内存使用估算
- `algorithms.py`: 核心数学模型和算法

**核心功能**:
- QPS (Queries Per Second) 估算
- Token/s 吞吐量估算
- 内存占用预测
- 延迟时间估算

### 4. 全局配置模块 (config/)
**职责**: 管理系统级全局配置
- `settings.py`: 全局系统设置（日志级别、缓存配置、默认参数等）

**核心功能**:
- 应用程序全局设置
- 环境变量管理
- 系统默认值配置

### 5. 工具模块 (utils/)
**职责**: 提供通用工具和辅助函数
- `validators.py`: 输入参数验证
- `formatters.py`: 数据格式化
- `calculators.py`: 通用计算工具
- `exceptions.py`: 自定义异常类

### 6. CLI模块 (cli/)
**职责**: 命令行界面，作为主要用户交互方式
- `commands.py`: CLI命令实现
- `interactive.py`: 交互式用户界面

**核心功能**:
- 参数解析和验证
- 估算结果展示和格式化
- 交互式配置向导
- 批量处理支持

## 配置管理策略

### 分层配置设计
```python
# config/settings.py - 全局设置
DEFAULT_OUTPUT_FORMAT = "table"
LOG_LEVEL = "INFO"
CACHE_ENABLED = True

# models/configs.py - 模型特定配置
QWEN3_DEFAULT_PARAMS = {
    "context_length": 8192,
    "batch_size": 1,
    "precision": "fp16"
}

# hardware/configs.py - 硬件特定配置  
GPU_PERFORMANCE_FACTORS = {
    "rtx-4090": {"memory_efficiency": 0.9, "compute_efficiency": 0.95},
    "v100": {"memory_efficiency": 0.85, "compute_efficiency": 0.90}
}
```

## 使用方式设计

### 命令行使用
```bash
# 基本使用
llm-estimate --model qwen-7b --accelerator rtx-4090 --memory 32GB

# 交互式模式
llm-estimate interactive

# 批量估算
llm-estimate batch --config configs/hardware_list.yaml

# 输出详细信息
llm-estimate --model qwen-14b --accelerator v100 --verbose --output report.json
```

### Python API 使用
```python
from llm_estimate import PerformanceEstimator

estimator = PerformanceEstimator()
result = estimator.estimate(
    model='qwen-7b',
    hardware={'accelerator': 'rtx-4090', 'memory': '32GB'}
)
print(f"预估QPS: {result.qps}")
```

## 数据流设计

1. **输入阶段**: 用户通过CLI输入模型和硬件规格
2. **验证阶段**: 参数验证模块检查输入的有效性
3. **查询阶段**: 从模型和硬件数据库中获取详细规格
4. **计算阶段**: 估算引擎执行性能计算
5. **输出阶段**: 格式化并展示估算结果

## 扩展性设计

- **模型扩展**: 通过继承基础模型类，可轻松添加新的模型支持
- **硬件扩展**: 采用插件化设计，支持新硬件类型
- **算法扩展**: 估算算法模块化，便于算法优化和替换
- **输出格式扩展**: 支持多种输出格式（JSON、CSV、表格等） 