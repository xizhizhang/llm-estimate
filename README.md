[English] | [中文](README_cn.md)

# LLM-Estimate

LLM Performance Estimation Tool - Estimate the performance of Large Language Models on different hardware configurations.

## Features

- 🚀 Support for mainstream LLM models (Llama, Qwen, etc.)
- 💻 Unified accelerator abstraction (GPU, CPU, TPU, NPU, etc.)
- 📊 Estimate key performance metrics (throughput, latency, memory usage)
- 🔧 Provide optimization suggestions and bottleneck analysis
- 📋 Support multiple output formats (table, JSON, CSV)
- 🖥️ Command-line tool and Python API
- ⚡ Focus on core metrics: computation (FLOPS) and memory bandwidth

## Core Concepts

This project unifies GPU, CPU, TPU and other computing devices as **accelerators**, no longer distinguishing device types, focusing only on:
- **Compute**: Computational capability (TFLOPS)
- **Memory Bandwidth**: Storage bandwidth (GB/s)
- **Memory Capacity**: Available memory (GB)

This unified abstraction simplifies hardware configuration, making performance estimation more intuitive and accurate.

## Quick Start

### Installation

```bash
# Clone the project
git clone https://github.com/zhangwm/llm-estimate.git
cd llm-estimate

# Install dependencies
pip install -r requirements.txt

# Optional: Install the project (for global commands)
pip install -e .
```

### Running Methods

#### Method 1: Direct Execution (Recommended, no installation required)

```bash
# Use the entry script in the project root directory
python3 llm_estimate.py --help

# Or give the script execute permission and run directly
chmod +x llm_estimate.py
./llm_estimate.py --help
```

#### Method 2: Module Execution

```bash
# Run CLI module in project root directory
python3 -m llm_estimate.cli --help

# Or run main.py in the module
python3 llm_estimate/main.py --help
```

#### Method 3: Global Command After Installation

```bash
# First install the project
pip install -e .

# Then use globally
llm-estimate --help
```

**Note**: Method 1 is the simplest approach, requiring only cloning the project and installing dependencies.

### Basic Usage

#### Command Line Tool

```bash
# Estimate Llama-2-7B performance on RTX-4090
python3 llm_estimate.py estimate --model llama-2-7b --accelerator rtx-4090

# Use multiple accelerators
python3 llm_estimate.py estimate --model llama-2-7b --accelerators rtx-4090,a100-40gb

# Specify precision and batch size
python3 llm_estimate.py estimate --model llama-2-7b --accelerator rtx-4090 --precision fp16 --batch-size 4

# List supported models
python3 llm_estimate.py list-models

# List supported accelerators
python3 llm_estimate.py list-accelerators

# Filter accelerators by type
python3 llm_estimate.py list-accelerators --type gpu
python3 llm_estimate.py list-accelerators --type cpu

# Compare multiple models
python3 llm_estimate.py compare --models llama-2-7b,qwen-7b --accelerator rtx-4090

# Benchmark multiple accelerators
python3 llm_estimate.py benchmark --accelerators rtx-4090,a100-40gb,h100 --model llama-2-7b

# Interactive mode
python3 llm_estimate.py interactive
```

#### Python API

```python
from llm_estimate import PerformanceEstimator, create_accelerator

# Create estimator
estimator = PerformanceEstimator()

# Single accelerator estimation
result = estimator.estimate(
    model_name="llama-2-7b",
    hardware_config={"accelerator": "rtx-4090"},
    model_config={"batch_size": 1, "precision": "fp16"}
)

# Multi-accelerator estimation
result = estimator.estimate(
    model_name="llama-2-7b",
    hardware_config={"accelerators": ["rtx-4090", "a100-40gb"]},
    model_config={"batch_size": 4, "precision": "fp16"}
)

print(f"Throughput: {result['throughput_tokens_per_sec']:.1f} tokens/s")
print(f"Memory Usage: {result['memory_usage_gb']:.2f} GB")
print(f"Latency: {result['latency_ms']:.1f} ms")
print(f"Bottleneck: {result['bottleneck']}")

# Create accelerator directly
accelerator = create_accelerator("rtx-4090")
print(f"Compute: {accelerator.compute_capability_tflops} TFLOPS")
print(f"Memory Bandwidth: {accelerator.memory_bandwidth_gb_s} GB/s")
```

## Project Structure

```
llm-estimate/
├── llm_estimate/           # Main code directory
│   ├── models/            # Model management module
│   ├── hardware/          # Hardware management module (unified accelerators)
│   ├── estimator/         # Estimation engine module
│   ├── config/            # Global configuration module
│   ├── utils/             # Utilities module
│   └── cli/               # Command line interface
├── tests/                 # Test module
├── data/                  # Data directory
├── docs/                  # Documentation directory
└── scripts/               # Scripts directory
```

## Supported Models

- **Llama Series**: Llama-2-7B, Llama-2-13B, Llama-2-70B
- **Qwen Series**: Qwen-7B, Qwen-14B, Qwen-72B
- More models being added continuously...

## Supported Accelerators

### GPU Accelerators
- **NVIDIA**: RTX-4090, RTX-4080, RTX-3090, A100, H100, V100
- **AMD**: (Planned)

### CPU Accelerators
- **Intel**: i9-13900K, i7-13700K
- **AMD**: Ryzen-9-7950X

### Specialized Accelerators
- **Apple**: M1-Ultra, M2-Ultra
- **Google**: TPU-v4

## Compatibility

For backward compatibility, the old `--gpu` and `--cpu` parameters are still supported, but the new `--accelerator` parameter is recommended.

```bash
# Old format (still supported)
llm-estimate estimate --model llama-2-7b --gpu rtx-4090

# New format (recommended)
llm-estimate estimate --model llama-2-7b --accelerator rtx-4090
```

## Development

### Environment Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black llm_estimate/

# Type checking
mypy llm_estimate/
```

### Adding New Models

1. Create a new model class in `llm_estimate/models/`
2. Inherit from `BaseModel` and implement necessary methods
3. Register the new model in `registry.py`

### Adding New Accelerators

1. Add specifications to `ACCELERATOR_SPECS` in `llm_estimate/hardware/accelerator.py`
2. Provide key parameters: compute (TFLOPS), memory bandwidth (GB/s), memory capacity (GB)
3. Optionally add auxiliary information like power consumption, price

Example:
```python
"new-accelerator": AcceleratorSpecs(
    name="New-Accelerator",
    manufacturer="Vendor",
    device_type="gpu",  # or "cpu", "tpu", "soc"
    compute_capability_tflops=100.0,
    memory_bandwidth_gb_s=1500.0,
    memory_capacity_gb=48.0,
    release_year=2024,
    price_usd=5000,
    power_consumption_w=400
)
```

## License

MIT License

## Contributing

Issues and Pull Requests are welcome!
