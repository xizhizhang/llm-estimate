[English] | [中文](README_cn.md)

# LLM-Estimate

LLM Performance Estimation Tool - Estimate the performance of Large Language Models on different hardware configurations.

## Features

- 🚀 Support for mainstream LLM models (Qwen, MoE, etc.)
- 💻 Unified accelerator abstraction (GPU, CPU, TPU, SOC, etc.)
- 📊 Estimate key performance metrics (TTFT, TPOT, throughput, memory usage)
- 🔧 Operation-level detailed analysis and bottleneck identification
- 📋 Support multiple output formats (table, JSON, CSV)
- 🖥️ Command-line tool and Python API
- ⚡ Focus on core metrics: computation (FLOPS) and memory bandwidth

## Core Concepts

This project unifies GPU, CPU, TPU, SOC and other computing devices as **accelerators**, no longer distinguishing device types, focusing only on:
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
# Estimate Qwen-8B performance on RTX-4090
python3 llm_estimate.py estimate --model qwen3-8b --accelerator rtx-4090

# Specify precision, batch size, and sequence lengths
python3 llm_estimate.py estimate --model qwen3-8b --accelerator rtx-4090 --precision fp16 --batch-size 4 --input-length 1024 --output-length 256

# Detailed analysis with operation-level breakdown
python3 llm_estimate.py estimate --model qwen3-8b --accelerator rtx-4090 --verbose

# Show detailed operations breakdown
python3 llm_estimate.py estimate --model qwen3-8b --accelerator rtx-4090 --show-ops --top-ops 20 --detailed

# List supported models
python3 llm_estimate.py list-models

# List supported accelerators
python3 llm_estimate.py list-accelerators

# Filter accelerators by type
python3 llm_estimate.py list-accelerators --type gpu
python3 llm_estimate.py list-accelerators --type cpu

# Benchmark performance across different sequence lengths
python3 llm_estimate.py benchmark --model qwen3-8b --accelerator rtx-4090 --input-lengths 512,1024,2048,4096 --output-lengths 128,256,512

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
    model_name="qwen3-8b",
    hardware_config={"accelerator": "rtx-4090"},
    model_config={"batch_size": 1, "precision": "fp16"}
)

print(f"Throughput: {result['throughput_tokens_per_sec']:.1f} tokens/s")
print(f"Memory Usage: {result['memory_usage_gb']:.2f} GB")
print(f"TTFT: {result['ttft_ms']:.1f} ms")
print(f"TPOT: {result['tpot_ms']:.1f} ms")
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

### Qwen Series
- **qwen3-8b**: 8B parameters, 36 layers, 40K context, GQA

### Mixture of Experts (MoE)
- **qwen3-235b-a22b**: 235B total parameters, 94 layers, 128 experts, 8 experts per token

## Supported Accelerators

### GPU Accelerators
- **RTX-4090**: 660 TFLOPS, 1008 GB/s, 24 GB
- **H100-80GB**: 1979 TFLOPS, 2039 GB/s, 80 GB

### CPU Accelerators
- **i9-13900K**: 1.2 TFLOPS, 77 GB/s, 128 GB max
- **Ryzen-9-7950X**: 1.1 TFLOPS, 83 GB/s, 128 GB max

### Apple Silicon
- **M2-Ultra**: 27.2 TFLOPS, 800 GB/s, 192 GB unified memory

### Google TPU
- **TPU-v4**: 275 TFLOPS, 1200 GB/s, 32 GB

## Key Features

### Operation-Level Analysis
- Detailed breakdown of transformer operations (attention, FFN, norm, etc.)
- FLOPS and memory bandwidth analysis for each operation
- Bottleneck identification and optimization suggestions

### Performance Metrics
- **TTFT (Time To First Token)**: Time to generate the first token
- **TPOT (Time Per Output Token)**: Average time per subsequent token
- **Throughput**: Total tokens processed per second
- **Memory Usage**: Model and activation memory requirements

### Advanced CLI Options
- `--verbose`: Enable detailed operation-level analysis
- `--show-ops`: Show operation breakdown
- `--top-ops N`: Display top N most time-consuming operations
- `--detailed`: Show comprehensive analysis including bottlenecks
- `--format`: Output in table, JSON, or CSV format

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

## Example Output

```
=== Performance Estimate ===
Model: qwen3-8b
Accelerator: RTX-4090
Precision: fp16
Batch Size: 1
Input Length: 1024
Output Length: 256

=== Core Metrics ===
• TTFT (Time to First Token): 45.2 ms
• TPOT (Time Per Output Token): 18.7 ms
• Total Latency: 4.83 s
• Throughput: 265 tokens/s
• Memory Usage: 14.8 GB
• Bottleneck: memory_bandwidth (89% utilization)

=== Performance Analysis ===
• Compute Utilization: 72%
• Memory Bandwidth Utilization: 89%
• Memory Capacity Utilization: 62%
```

## License

MIT License

## Contributing

Issues and Pull Requests are welcome!
