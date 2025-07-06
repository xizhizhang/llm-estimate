[中文] | [English](README.md)

# LLM-Estimate

大语言模型性能估算工具 - 估算LLM在不同硬件配置下的性能表现。

## 功能特性

- 🚀 支持多种主流LLM模型（Qwen、MoE等）
- 💻 统一加速器抽象（GPU、CPU、TPU、SOC等）
- 📊 估算关键性能指标（TTFT、TPOT、吞吐量、内存使用）
- 🔧 操作级别详细分析和瓶颈识别
- 📋 支持多种输出格式（表格、JSON、CSV）
- 🖥️ 命令行工具和Python API
- ⚡ 专注算力（FLOPS）和内存带宽核心指标

## 核心概念

本项目将GPU、CPU、TPU、SOC等计算设备统一抽象为**加速器**，不再区分设备类型，只关注：
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
# 估算Qwen-8B在RTX-4090上的性能
python3 llm_estimate.py estimate --model qwen3-8b --accelerator rtx-4090

# 指定精度、批次大小和序列长度
python3 llm_estimate.py estimate --model qwen3-8b --accelerator rtx-4090 --precision fp16 --batch-size 4 --input-length 1024 --output-length 256

# 详细分析，包含操作级别分解
python3 llm_estimate.py estimate --model qwen3-8b --accelerator rtx-4090 --verbose

# 显示详细的操作分解
python3 llm_estimate.py estimate --model qwen3-8b --accelerator rtx-4090 --show-ops --top-ops 20 --detailed

# 列出支持的模型
python3 llm_estimate.py list-models

# 列出支持的加速器
python3 llm_estimate.py list-accelerators

# 按类型筛选加速器
python3 llm_estimate.py list-accelerators --type gpu
python3 llm_estimate.py list-accelerators --type cpu

# 跨不同序列长度的性能基准测试
python3 llm_estimate.py benchmark --model qwen3-8b --accelerator rtx-4090 --input-lengths 512,1024,2048,4096 --output-lengths 128,256,512

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
    model_name="qwen3-8b",
    hardware_config={"accelerator": "rtx-4090"},
    model_config={"batch_size": 1, "precision": "fp16"}
)

print(f"吞吐量: {result['throughput_tokens_per_sec']:.1f} tokens/s")
print(f"内存使用: {result['memory_usage_gb']:.2f} GB")
print(f"首Token时间: {result['ttft_ms']:.1f} ms")
print(f"每Token时间: {result['tpot_ms']:.1f} ms")
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

### Qwen系列
- **qwen3-8b**: 8B参数，36层，40K上下文，GQA架构

### 混合专家模型（MoE）
- **qwen3-235b-a22b**: 235B总参数，94层，128个专家，每token激活8个专家

## 支持的加速器

### GPU加速器
- **RTX-4090**: 660 TFLOPS，1008 GB/s，24 GB显存
- **H100-80GB**: 1979 TFLOPS，2039 GB/s，80 GB显存

### CPU加速器
- **i9-13900K**: 1.2 TFLOPS，77 GB/s，最大128 GB内存
- **Ryzen-9-7950X**: 1.1 TFLOPS，83 GB/s，最大128 GB内存

### Apple Silicon
- **M2-Ultra**: 27.2 TFLOPS，800 GB/s，192 GB统一内存

### Google TPU
- **TPU-v4**: 275 TFLOPS，1200 GB/s，32 GB内存

## 核心功能

### 操作级别分析
- 详细分解Transformer操作（注意力、FFN、归一化等）
- 每个操作的FLOPS和内存带宽分析
- 瓶颈识别和优化建议

### 性能指标
- **TTFT (Time To First Token)**: 首Token生成时间
- **TPOT (Time Per Output Token)**: 后续Token平均生成时间
- **吞吐量**: 每秒处理的总Token数
- **内存使用**: 模型和激活值内存需求

### 高级CLI选项
- `--verbose`: 启用详细的操作级别分析
- `--show-ops`: 显示操作分解
- `--top-ops N`: 显示前N个最耗时的操作
- `--detailed`: 显示包括瓶颈的综合分析
- `--format`: 以表格、JSON或CSV格式输出

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

## 输出示例

```
=== 性能估算 ===
模型: qwen3-8b
加速器: RTX-4090
精度: fp16
批次大小: 1
输入长度: 1024
输出长度: 256

=== 核心指标 ===
• TTFT (首Token时间): 45.2 ms
• TPOT (每Token时间): 18.7 ms
• 总延迟: 4.83 s
• 吞吐量: 265 tokens/s
• 内存使用: 14.8 GB
• 瓶颈: memory_bandwidth (89% 利用率)

=== 性能分析 ===
• 计算利用率: 72%
• 内存带宽利用率: 89%
• 内存容量利用率: 62%
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！ 