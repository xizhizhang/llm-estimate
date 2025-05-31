# LLM-Estimate

大语言模型性能估算工具 - 估算LLM在不同硬件配置下的性能表现。

## 功能特性

- 🚀 支持多种主流LLM模型（Llama、Qwen等）
- 💻 统一加速器抽象（GPU、CPU、TPU、NPU等）
- 📊 估算关键性能指标（吞吐量、延迟、内存使用）
- 🔧 提供优化建议和瓶颈分析
- 📋 支持多种输出格式（表格、JSON、CSV）
- 🖥️ 命令行工具和Python API
- ⚡ 专注算力（FLOPS）和内存带宽核心指标

## 核心概念

本项目将GPU、CPU、TPU等计算设备统一抽象为**加速器**，不再区分设备类型，只关注：
- **算力**: 计算能力（TFLOPS）
- **内存带宽**: 存储带宽（GB/s）
- **内存容量**: 可用内存（GB）

这种统一抽象简化了硬件配置，使性能估算更加直观和准确。

## 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/zhangwm/llm-estimate.git
cd llm-estimate

# 安装依赖
pip install -r requirements.txt

# 可选：安装项目（用于全局命令）
pip install -e .
```

### 运行方式

#### 方式1: 直接运行（推荐，无需安装）

```bash
# 使用项目根目录的入口脚本
python3 llm_estimate.py --help

# 或者给脚本执行权限后直接运行
chmod +x llm_estimate.py
./llm_estimate.py --help
```

#### 方式2: 模块方式运行

```bash
# 在项目根目录下运行CLI模块
python3 -m llm_estimate.cli --help

# 或者运行模块内的main.py
python3 llm_estimate/main.py --help
```

#### 方式3: 安装后全局命令

```bash
# 先安装项目
pip install -e .

# 然后可以全局使用
llm-estimate --help
```

**注意**: 方式1是最简单的方法，只需要克隆项目和安装依赖即可运行，无需安装包。

### 基本使用

#### 命令行工具

```bash
# 估算Llama-2-7B在RTX-4090上的性能
python3 llm_estimate.py estimate --model llama-2-7b --accelerator rtx-4090

# 使用多个加速器
python3 llm_estimate.py estimate --model llama-2-7b --accelerators rtx-4090,a100-40gb

# 指定精度和批次大小
python3 llm_estimate.py estimate --model llama-2-7b --accelerator rtx-4090 --precision fp16 --batch-size 4

# 列出支持的模型
python3 llm_estimate.py list-models

# 列出支持的加速器
python3 llm_estimate.py list-accelerators

# 按类型筛选加速器
python3 llm_estimate.py list-accelerators --type gpu
python3 llm_estimate.py list-accelerators --type cpu

# 比较多个模型
python3 llm_estimate.py compare --models llama-2-7b,qwen-7b --accelerator rtx-4090

# 基准测试多个加速器
python3 llm_estimate.py benchmark --accelerators rtx-4090,a100-40gb,h100 --model llama-2-7b

# 交互式模式
python3 llm_estimate.py interactive
```

#### Python API

```python
from llm_estimate import PerformanceEstimator, create_accelerator

# 创建估算器
estimator = PerformanceEstimator()

# 单加速器估算
result = estimator.estimate(
    model_name="llama-2-7b",
    hardware_config={"accelerator": "rtx-4090"},
    model_config={"batch_size": 1, "precision": "fp16"}
)

# 多加速器估算
result = estimator.estimate(
    model_name="llama-2-7b",
    hardware_config={"accelerators": ["rtx-4090", "a100-40gb"]},
    model_config={"batch_size": 4, "precision": "fp16"}
)

print(f"吞吐量: {result['throughput_tokens_per_sec']:.1f} tokens/s")
print(f"内存使用: {result['memory_usage_gb']:.2f} GB")
print(f"延迟: {result['latency_ms']:.1f} ms")
print(f"瓶颈: {result['bottleneck']}")

# 直接创建加速器
accelerator = create_accelerator("rtx-4090")
print(f"算力: {accelerator.compute_capability_tflops} TFLOPS")
print(f"内存带宽: {accelerator.memory_bandwidth_gb_s} GB/s")
```

## 项目结构

```
llm-estimate/
├── llm_estimate/           # 主要代码目录
│   ├── models/            # 模型管理模块
│   ├── hardware/          # 硬件管理模块（统一加速器）
│   ├── estimator/         # 估算引擎模块
│   ├── config/            # 全局配置模块
│   ├── utils/             # 工具模块
│   └── cli/               # 命令行接口
├── tests/                 # 测试模块
├── data/                  # 数据目录
├── docs/                  # 文档目录
└── scripts/               # 脚本目录
```

## 支持的模型

- **Llama系列**: Llama-2-7B, Llama-2-13B, Llama-2-70B
- **Qwen系列**: Qwen-7B, Qwen-14B, Qwen-72B
- 更多模型持续添加中...

## 支持的加速器

### GPU加速器
- **NVIDIA**: RTX-4090, RTX-4080, RTX-3090, A100, H100, V100
- **AMD**: (规划中)

### CPU加速器
- **Intel**: i9-13900K, i7-13700K
- **AMD**: Ryzen-9-7950X

### 专用加速器
- **Apple**: M1-Ultra, M2-Ultra
- **Google**: TPU-v4

## 兼容性说明

为保持向后兼容，仍支持旧的`--gpu`和`--cpu`参数，但建议使用新的`--accelerator`参数。

```bash
# 旧格式（仍然支持）
llm-estimate estimate --model llama-2-7b --gpu rtx-4090

# 新格式（推荐）
llm-estimate estimate --model llama-2-7b --accelerator rtx-4090
```

## 开发

### 环境设置

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
black llm_estimate/

# 类型检查
mypy llm_estimate/
```

### 添加新模型

1. 在 `llm_estimate/models/` 中创建新的模型类
2. 继承 `BaseModel` 并实现必要方法
3. 在 `registry.py` 中注册新模型

### 添加新加速器

1. 在 `llm_estimate/hardware/accelerator.py` 的 `ACCELERATOR_SPECS` 中添加规格
2. 提供算力（TFLOPS）、内存带宽（GB/s）、内存容量（GB）等关键参数
3. 可选择性添加功耗、价格等辅助信息

示例：
```python
"new-accelerator": AcceleratorSpecs(
    name="New-Accelerator",
    manufacturer="Vendor",
    device_type="gpu",  # 或 "cpu", "tpu", "soc"
    compute_capability_tflops=100.0,
    memory_bandwidth_gb_s=1500.0,
    memory_capacity_gb=48.0,
    release_year=2024,
    price_usd=5000,
    power_consumption_w=400
)
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！
