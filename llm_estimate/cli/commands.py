"""
CLI命令实现

提供命令行界面的具体命令实现。
支持统一的加速器概念，简化硬件配置。
"""

import click
import json
from typing import Optional, Dict, Any
from tabulate import tabulate

from ..models.registry import model_registry
from ..estimator.base import PerformanceEstimator
from ..hardware.accelerator import list_supported_accelerators, get_accelerator_by_type
from ..utils.formatters import format_results


@click.group()
@click.version_option(version="0.1.0", prog_name="llm-estimate")
def cli():
    """LLM性能估算工具
    
    估算大语言模型在不同硬件配置下的性能表现。
    统一GPU、CPU等为加速器概念，专注于算力和内存带宽。
    """
    pass


@cli.command()
@click.option("--model", "-m", required=True, help="模型名称")
@click.option("--accelerator", "-a", help="加速器型号 (如: rtx-4090, a100-40gb, i9-13900k)")
@click.option("--accelerators", help="多个加速器，逗号分隔 (如: rtx-4090,a100-40gb)")
@click.option("--batch-size", "-b", type=int, default=1, help="批次大小")
@click.option("--context-length", "-l", type=int, help="上下文长度")
@click.option("--precision", "-p", default="fp16", help="精度类型 (fp32/fp16/bf16/int8/int4)")
@click.option("--output", "-o", type=click.Path(), help="输出文件路径")
@click.option("--format", "-f", default="table", type=click.Choice(["table", "json", "csv"]), help="输出格式")
@click.option("--verbose", "-v", is_flag=True, help="详细输出")
def estimate(model: str, accelerator: Optional[str], accelerators: Optional[str],
            batch_size: int, context_length: Optional[int], precision: str,
            output: Optional[str], format: str, verbose: bool):
    """估算模型性能"""
    
    try:
        # 创建估算器
        estimator = PerformanceEstimator()
        
        # 构建硬件配置
        hardware_config = {}
        
        # 使用加速器参数
        if accelerator:
            hardware_config["accelerator"] = accelerator
        elif accelerators:
            acc_list = [acc.strip() for acc in accelerators.split(",")]
            hardware_config["accelerators"] = acc_list
        else:
            raise click.BadParameter("必须指定 --accelerator 或 --accelerators 参数")
            
        # 构建模型配置
        model_config = {
            "batch_size": batch_size,
            "precision": precision
        }
        if context_length:
            model_config["context_length"] = context_length
        
        # 执行估算
        result = estimator.estimate(
            model_name=model,
            hardware_config=hardware_config,
            model_config=model_config
        )
        
        # 格式化输出
        formatted_result = format_results(result, format, verbose)
        
        # 输出结果
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(formatted_result)
            click.echo(f"结果已保存到: {output}")
        else:
            click.echo(formatted_result)
            
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        raise click.Abort()


@cli.command()
def list_models():
    """列出支持的模型"""
    models = model_registry.list_models()
    
    data = []
    for model_name in models:
        try:
            info = model_registry.get_model_info(model_name)
            data.append([
                model_name,
                info["parameters"],
                info["model_type"],
                info["layers"],
                info["max_position_embeddings"]
            ])
        except Exception:
            continue
    
    headers = ["模型名称", "参数量", "类型", "层数", "最大长度"]
    table = tabulate(data, headers=headers, tablefmt="grid")
    click.echo(table)


@cli.command()
@click.option("--type", "-t", help="设备类型筛选 (gpu/cpu/tpu/soc)")
def list_accelerators(type: Optional[str]):
    """列出支持的加速器"""
    
    if type:
        accelerators = get_accelerator_by_type(type)
        click.echo(f"支持的{type.upper()}加速器:")
    else:
        accelerators = list_supported_accelerators()
        click.echo("支持的所有加速器:")
    
    # 按设备类型分组显示
    by_type = {}
    for name, info in accelerators.items():
        device_type = info["device_type"]
        if device_type not in by_type:
            by_type[device_type] = []
        by_type[device_type].append((name, info))
    
    for device_type, items in by_type.items():
        click.echo(f"\n=== {device_type.upper()} ===")
        
        data = []
        for name, info in items:
            data.append([
                info["name"],
                info["manufacturer"],
                f"{info['compute_capability_tflops']:.1f} TFLOPS",
                f"{info['memory_bandwidth_gb_s']:.0f} GB/s",
                f"{info['memory_capacity_gb']:.0f} GB",
                f"{info['power_consumption_w']}W" if info['power_consumption_w'] else "N/A"
            ])
        
        headers = ["型号", "厂商", "算力", "内存带宽", "内存容量", "功耗"]
        table = tabulate(data, headers=headers, tablefmt="grid")
        click.echo(table)


@cli.command()
@click.option("--model", "-m", help="模型名称（用于筛选）")
@click.option("--accelerator", "-a", help="加速器型号（用于筛选）")
def interactive(model: Optional[str], accelerator: Optional[str]):
    """交互式模式"""
    click.echo("欢迎使用LLM性能估算工具交互模式!")
    click.echo("输入 'help' 查看帮助，输入 'quit' 退出")
    
    while True:
        try:
            command = click.prompt("\nllm-estimate", type=str)
            
            if command.lower() in ['quit', 'exit', 'q']:
                click.echo("再见!")
                break
            elif command.lower() in ['help', 'h']:
                show_interactive_help()
            elif command.lower().startswith('list'):
                if 'models' in command:
                    list_models.callback()
                elif 'accelerators' in command or 'hardware' in command:
                    list_accelerators.callback(None)
            else:
                click.echo("未知命令，输入 'help' 查看帮助")
                
        except (KeyboardInterrupt, EOFError):
            click.echo("\n再见!")
            break
        except Exception as e:
            click.echo(f"错误: {e}")


def show_interactive_help():
    """显示交互模式帮助"""
    help_text = """
可用命令:
  list models        - 列出支持的模型
  list accelerators  - 列出支持的加速器
  help              - 显示此帮助信息
  quit              - 退出程序
  
示例:
  estimate --model llama-2-7b --accelerator rtx-4090
  estimate --model qwen-7b --accelerators rtx-4090,a100-40gb
  """
    click.echo(help_text)


@cli.command()
@click.option("--models", "-m", required=True, help="模型列表，逗号分隔")
@click.option("--accelerator", "-a", required=True, help="加速器型号")
@click.option("--output", "-o", type=click.Path(), help="输出文件路径")
def compare(models: str, accelerator: str, output: Optional[str]):
    """比较多个模型的性能"""
    
    model_list = [m.strip() for m in models.split(",")]
    estimator = PerformanceEstimator()
    
    results = []
    for model_name in model_list:
        try:
            result = estimator.estimate(
                model_name=model_name,
                hardware_config={"accelerator": accelerator}
            )
            results.append({
                "model": model_name,
                "result": result
            })
        except Exception as e:
            click.echo(f"估算 {model_name} 时出错: {e}")
    
    # 格式化比较结果
    if results:
        comparison_table = format_comparison_results(results)
        
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(comparison_table)
            click.echo(f"比较结果已保存到: {output}")
        else:
            click.echo(comparison_table)
    else:
        click.echo("没有成功估算的模型")


@cli.command()
@click.option("--accelerators", "-a", required=True, help="加速器列表，逗号分隔")
@click.option("--model", "-m", default="llama-2-7b", help="基准模型")
@click.option("--output", "-o", type=click.Path(), help="输出文件路径")
def benchmark(accelerators: str, model: str, output: Optional[str]):
    """对比多个加速器在同一模型下的性能"""
    
    acc_list = [acc.strip() for acc in accelerators.split(",")]
    estimator = PerformanceEstimator()
    
    results = []
    for acc_name in acc_list:
        try:
            result = estimator.estimate(
                model_name=model,
                hardware_config={"accelerator": acc_name}
            )
            results.append({
                "accelerator": acc_name,
                "result": result
            })
        except Exception as e:
            click.echo(f"测试 {acc_name} 时出错: {e}")
    
    # 格式化基准测试结果
    if results:
        benchmark_table = format_benchmark_results(results)
        
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(benchmark_table)
            click.echo(f"基准测试结果已保存到: {output}")
        else:
            click.echo(benchmark_table)
    else:
        click.echo("没有成功测试的加速器")


def format_comparison_results(results) -> str:
    """格式化模型比较结果"""
    lines = []
    lines.append("=== 模型性能比较 ===\n")
    
    data = []
    for item in results:
        model_name = item["model"]
        result = item["result"]
        
        data.append([
            model_name,
            f"{result.get('memory_usage_gb', 0):.2f} GB",
            f"{result.get('throughput_tokens_per_sec', 0):.1f} tokens/s",
            f"{result.get('latency_ms', 0):.1f} ms",
            result.get('bottleneck', 'N/A')
        ])
    
    headers = ["模型", "内存使用", "吞吐量", "延迟", "瓶颈"]
    table = tabulate(data, headers=headers, tablefmt="grid")
    lines.append(table)
    
    return "\n".join(lines)


def format_benchmark_results(results) -> str:
    """格式化加速器基准测试结果"""
    lines = []
    lines.append("=== 加速器性能基准测试 ===\n")
    
    data = []
    for item in results:
        acc_name = item["accelerator"]
        result = item["result"]
        
        data.append([
            acc_name,
            f"{result.get('memory_usage_gb', 0):.2f} GB",
            f"{result.get('throughput_tokens_per_sec', 0):.1f} tokens/s",
            f"{result.get('latency_ms', 0):.1f} ms",
            f"{result.get('utilization_percent', 0):.1f}%",
            result.get('bottleneck', 'N/A')
        ])
    
    headers = ["加速器", "内存使用", "吞吐量", "延迟", "利用率", "瓶颈"]
    table = tabulate(data, headers=headers, tablefmt="grid")
    lines.append(table)
    
    return "\n".join(lines)


def main():
    """主程序入口"""
    cli()


if __name__ == "__main__":
    main() 