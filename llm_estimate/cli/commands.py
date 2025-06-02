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
  benchmark --model llama-2-7b --accelerator rtx-4090 --input-lengths 512,1024 --output-lengths 128,256
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
@click.option("--model", "-m", required=True, help="模型名称")
@click.option("--accelerator", "-a", required=True, help="加速器型号")
@click.option("--input-lengths", "-i", default="128,512,1024,2048", help="输入长度列表，逗号分隔")
@click.option("--output-lengths", "-o", default="128,256,512", help="输出长度列表，逗号分隔")
@click.option("--batch-size", "-b", type=int, default=1, help="批次大小")
@click.option("--precision", "-p", default="fp16", help="精度类型")
@click.option("--output-file", type=click.Path(), help="输出文件路径")
@click.option("--format", "-f", default="table", type=click.Choice(["table", "json", "csv"]), help="输出格式")
def benchmark(model: str, accelerator: str, input_lengths: str, output_lengths: str, 
              batch_size: int, precision: str, output_file: Optional[str], format: str):
    """预估模型在不同输入/输出长度下的 TTFT 和 TPOT 指标
    
    TTFT (Time To First Token): 从开始推理到产生第一个token的时间
    TPOT (Time Per Output Token): 每个输出token的平均生成时间
    """
    
    input_length_list = [int(x.strip()) for x in input_lengths.split(",")]
    output_length_list = [int(x.strip()) for x in output_lengths.split(",")]
    
    estimator = PerformanceEstimator()
    
    results = []
    total_tests = len(input_length_list) * len(output_length_list)
    current_test = 0
    
    click.echo(f"开始基准测试: {model} @ {accelerator}")
    click.echo(f"总共需要测试 {total_tests} 种配置...")
    
    for input_len in input_length_list:
        for output_len in output_length_list:
            current_test += 1
            click.echo(f"测试 [{current_test}/{total_tests}]: 输入长度={input_len}, 输出长度={output_len}")
            
            try:
                # 构建配置
                hardware_config = {"accelerator": accelerator}
                model_config = {
                    "batch_size": batch_size,
                    "precision": precision,
                    "context_length": input_len,
                    "max_new_tokens": output_len
                }
                
                # 执行估算
                result = estimator.estimate(
                    model_name=model,
                    hardware_config=hardware_config,
                    model_config=model_config
                )
                
                # 提取 TTFT 和 TPOT 指标
                ttft_ms = result.get('ttft_ms', 0)  # Time To First Token
                tpot_ms = result.get('tpot_ms', 0)  # Time Per Output Token
                
                # 如果没有直接的 TTFT/TPOT，从其他指标计算
                if ttft_ms == 0 or tpot_ms == 0:
                    latency_ms = result.get('latency_ms', 0)
                    throughput = result.get('throughput_tokens_per_sec', 0)
                    
                    if throughput > 0:
                        tpot_ms = 1000 / throughput  # 每个token的时间(ms)
                    
                    # TTFT 计算：基于输入长度和模型推理特性
                    # TTFT 主要由 prefill 阶段决定，与输入长度相关
                    # 对于自回归模型，prefill 时间通常比单个 decode 步骤稍长
                    if input_len > 0 and tpot_ms > 0:
                        # TTFT ≈ prefill_time，通常比 TPOT 高 20-50%
                        # 这里用简化模型：TTFT = base_time + input_processing_time
                        base_prefill_overhead = tpot_ms * 1.3  # prefill 比 decode 慢 30%
                        input_processing_factor = max(1.0, input_len / 1024)  # 长输入需要更多时间
                        ttft_ms = base_prefill_overhead * input_processing_factor
                    elif latency_ms > 0 and output_len > 1:
                        # 备选方案：假设总延迟 = TTFT + (output_len-1) * TPOT
                        # 重新计算 TTFT
                        estimated_decode_time = (output_len - 1) * tpot_ms
                        ttft_ms = max(tpot_ms * 0.5, latency_ms - estimated_decode_time)
                    else:
                        ttft_ms = tpot_ms  # 最后的备选方案
                
                # 重新计算总延迟：TTFT + (output_len - 1) * TPOT
                # 这表示生成第一个token的时间 + 生成剩余token的时间
                total_latency_ms = ttft_ms + (output_len - 1) * tpot_ms
                
                # 重新计算吞吐量：总token数 / 总耗时
                # 总token数 = 输入长度 + 输出长度
                # 总耗时 = 总延迟（转换为秒）
                total_tokens = input_len + output_len
                total_latency_s = total_latency_ms / 1000  # 转换为秒
                actual_throughput = total_tokens / total_latency_s if total_latency_s > 0 else 0
                
                results.append({
                    "input_length": input_len,
                    "output_length": output_len,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "total_latency_ms": total_latency_ms,
                    "throughput_tokens_per_sec": actual_throughput,
                    "memory_usage_gb": result.get('memory_usage_gb', 0),
                    "utilization_percent": result.get('utilization_percent', 0)
                })
                
            except Exception as e:
                click.echo(f"  错误: {e}")
                results.append({
                    "input_length": input_len,
                    "output_length": output_len,
                    "ttft_ms": 0,
                    "tpot_ms": 0,
                    "total_latency_ms": 0,
                    "throughput_tokens_per_sec": 0,
                    "memory_usage_gb": 0,
                    "utilization_percent": 0,
                    "error": str(e)
                })
    
    # 格式化基准测试结果
    if results:
        benchmark_table = format_ttft_tpot_results(results, model, accelerator, format)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(benchmark_table)
            click.echo(f"基准测试结果已保存到: {output_file}")
        else:
            click.echo(benchmark_table)
    else:
        click.echo("没有成功的测试结果")


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


def format_ttft_tpot_results(results, model, accelerator, output_format) -> str:
    """格式化 TTFT 和 TPOT 结果"""
    
    if output_format == "json":
        return json.dumps({
            "model": model,
            "accelerator": accelerator,
            "test_results": results
        }, indent=2, ensure_ascii=False)
    
    elif output_format == "csv":
        lines = []
        lines.append("模型,加速器,输入长度,输出长度,TTFT(ms),TPOT(ms),总延迟(ms),吞吐量(tokens/s),内存使用(GB),利用率(%)")
        
        for item in results:
            if "error" in item:
                continue
            lines.append(f"{model},{accelerator},{item['input_length']},{item['output_length']},"
                        f"{item['ttft_ms']:.1f},{item['tpot_ms']:.1f},{item['total_latency_ms']:.1f},"
                        f"{item['throughput_tokens_per_sec']:.1f},{item['memory_usage_gb']:.2f},"
                        f"{item['utilization_percent']:.1f}")
        return "\n".join(lines)
    
    else:  # table format
        lines = []
        lines.append(f"=== TTFT & TPOT 基准测试结果 ===")
        lines.append(f"模型: {model}")
        lines.append(f"加速器: {accelerator}")
        lines.append("")
        
        data = []
        for item in results:
            if "error" in item:
                data.append([
                    f"{item['input_length']}",
                    f"{item['output_length']}",
                    "错误",
                    "错误", 
                    "错误",
                    "错误",
                    "错误",
                    "错误"
                ])
            else:
                data.append([
                    f"{item['input_length']}",
                    f"{item['output_length']}",
                    f"{item['ttft_ms']:.1f} ms",
                    f"{item['tpot_ms']:.1f} ms",
                    f"{item['total_latency_ms']:.1f} ms",
                    f"{item['throughput_tokens_per_sec']:.1f} tokens/s",
                    f"{item['memory_usage_gb']:.2f} GB",
                    f"{item['utilization_percent']:.1f}%"
                ])
        
        headers = ["输入长度", "输出长度", "TTFT", "TPOT", "总延迟", "吞吐量", "内存使用", "利用率"]
        table = tabulate(data, headers=headers, tablefmt="grid")
        lines.append(table)
        
        # 添加统计摘要
        successful_results = [r for r in results if "error" not in r]
        if successful_results:
            lines.append("\n=== 统计摘要 ===")
            avg_ttft = sum(r['ttft_ms'] for r in successful_results) / len(successful_results)
            avg_tpot = sum(r['tpot_ms'] for r in successful_results) / len(successful_results)
            avg_throughput = sum(r['throughput_tokens_per_sec'] for r in successful_results) / len(successful_results)
            
            lines.append(f"平均 TTFT: {avg_ttft:.1f} ms")
            lines.append(f"平均 TPOT: {avg_tpot:.1f} ms")  
            lines.append(f"平均吞吐量: {avg_throughput:.1f} tokens/s")
        
        return "\n".join(lines)


def main():
    """主程序入口"""
    cli()


if __name__ == "__main__":
    main() 