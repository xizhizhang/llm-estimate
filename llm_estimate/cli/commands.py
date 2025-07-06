"""
CLI命令实现

提供命令行界面的具体命令实现。
支持统一的加速器概念，简化硬件配置。
"""

import click
import json
from typing import Optional, Dict, Any, Tuple
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


def calculate_performance_metrics(estimator: PerformanceEstimator, 
                                model_name: str, 
                                hardware_config: Dict[str, Any],
                                batch_size: int,
                                precision: str,
                                input_length: int,
                                output_length: int = 1) -> Dict[str, Any]:
    """统一的性能计算函数
    
    Args:
        estimator: 性能估算器实例
        model_name: 模型名称
        hardware_config: 硬件配置
        batch_size: 批次大小
        precision: 精度类型
        input_length: 输入序列长度
        output_length: 输出序列长度
    
    Returns:
        包含性能指标的字典
    """
    
    # 计算 TTFT: prefill阶段处理完整输入序列的时间
    # 设置为 prefill 模式，处理整个输入序列
    prefill_result = estimator.estimate_op_level(
        model_name=model_name,
        hardware_config=hardware_config,
        model_config={
            "batch_size": batch_size,
            "precision": precision,
            "context_length": input_length,
            "inference_mode": "prefill",  # 明确设置为 prefill 模式
            "use_kv_cache": True,
            "max_new_tokens": 1
        }
    )
    prefill_analysis = prefill_result["op_level_analysis"]
    
    # TTFT: prefill阶段处理整个输入序列的时间
    ttft_ms = prefill_analysis["total_time_per_token_ms"]
    
    # 计算 TPOT：decode阶段生成单个token的时间
    if output_length == 1:
        # 只有一个输出token，使用decode阶段的时间
        decode_result_single = estimator.estimate_op_level(
            model_name=model_name,
            hardware_config=hardware_config,
            model_config={
                "batch_size": batch_size,
                "precision": precision,
                "context_length": input_length,
                "inference_mode": "decode",  # 明确设置为 decode 模式
                "use_kv_cache": True,
                "max_new_tokens": 1
            }
        )
        tpot_ms = decode_result_single["op_level_analysis"]["total_time_per_token_ms"]
        total_latency_ms = ttft_ms
        total_decode_time = 0
    else:
        # 计算在输入序列长度基础上的第一个token生成时间
        start_decode_result = estimator.estimate_op_level(
            model_name=model_name,
            hardware_config=hardware_config,
            model_config={
                "batch_size": batch_size,
                "precision": precision,
                "context_length": input_length,
                "inference_mode": "decode",  # 明确设置为 decode 模式
                "use_kv_cache": True,
                "max_new_tokens": 1
            }
        )
        start_tpot_ms = start_decode_result["op_level_analysis"]["total_time_per_token_ms"]
        
        # 计算在最终序列长度基础上的token生成时间
        end_decode_result = estimator.estimate_op_level(
            model_name=model_name,
            hardware_config=hardware_config,
            model_config={
                "batch_size": batch_size,
                "precision": precision,
                "context_length": input_length + output_length - 1,
                "inference_mode": "decode",  # 明确设置为 decode 模式
                "use_kv_cache": True,
                "max_new_tokens": 1
            }
        )
        end_tpot_ms = end_decode_result["op_level_analysis"]["total_time_per_token_ms"]
        
        # TPOT: 取起始和结束token生成时间的平均值
        tpot_ms = (start_tpot_ms + end_tpot_ms) / 2
        
        # 计算总解码时间
        total_decode_time = 0
        for i in range(output_length - 1):
            if output_length > 2:
                progress = i / (output_length - 2)
            else:
                progress = 0
            current_tpot = start_tpot_ms + progress * (end_tpot_ms - start_tpot_ms)
            total_decode_time += current_tpot
        
        # 总延迟 = TTFT + 后续token的解码时间
        total_latency_ms = ttft_ms + total_decode_time
    
    # 使用decode阶段的分析结果作为主要指标
    decode_result = estimator.estimate_op_level(
        model_name=model_name,
        hardware_config=hardware_config,
        model_config={
            "batch_size": batch_size,
            "precision": precision,
            "context_length": input_length + output_length // 2,  # 中等序列长度
            "inference_mode": "decode",  # 明确设置为 decode 模式
            "use_kv_cache": True,
            "max_new_tokens": 1
        }
    )
    decode_analysis = decode_result["op_level_analysis"]
    
    # 计算总吞吐量
    total_tokens = input_length + output_length
    actual_throughput = total_tokens / (total_latency_ms / 1000) if total_latency_ms > 0 else 0
    
    # 获取瓶颈分析
    bottleneck_analysis = decode_analysis["bottleneck_analysis"]
    compute_utilization = bottleneck_analysis["compute_bound_percentage"]
    memory_utilization = bottleneck_analysis["memory_bound_percentage"]
    overall_utilization = max(compute_utilization, memory_utilization)
    
    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "total_latency_ms": total_latency_ms,
        "total_decode_time_ms": total_decode_time,
        "throughput_tokens_per_sec": actual_throughput,
        "memory_usage_gb": decode_analysis["memory_usage_gb"],
        "bottleneck": bottleneck_analysis["major_bottleneck"],
        "utilization_percent": overall_utilization,
        "compute_utilization_percent": compute_utilization,
        "memory_bandwidth_utilization_percent": memory_utilization,
        "memory_capacity_utilization_percent": memory_utilization,
        "compute_util": compute_utilization,
        "memory_util": memory_utilization,
        "op_level_analysis": decode_analysis,
        "model_info": decode_result["model_info"],
        "system_info": decode_result["system_info"]
    }


@cli.command()
@click.option("--model", "-m", required=True, help="模型名称")
@click.option("--accelerator", "-a", help="加速器型号 (如: rtx-4090, a100-40gb, i9-13900k)")
@click.option("--batch-size", "-b", type=int, default=1, help="批次大小")
@click.option("--input-length", "-i", type=int, default=512, help="输入序列长度")
@click.option("--output-length", "-o", type=int, default=128, help="输出序列长度")
@click.option("--precision", "-p", default="fp16", help="精度类型 (fp32/fp16/bf16/int8/int4)")
@click.option("--output-file", type=click.Path(), help="输出文件路径")
@click.option("--format", "-f", default="table", type=click.Choice(["table", "json", "csv"]), help="输出格式")
@click.option("--verbose", "-v", is_flag=True, help="详细输出（等同于 --show-ops --detailed --top-ops=15）")
@click.option("--show-ops", is_flag=True, help="显示详细的操作分解")
@click.option("--top-ops", type=int, default=10, help="显示前N个最耗时的操作")
@click.option("--detailed", is_flag=True, help="显示完整的详细分析")
def estimate(model: str, accelerator: Optional[str],
            batch_size: int, input_length: int, output_length: int, precision: str,
            output_file: Optional[str], format: str, verbose: bool, show_ops: bool, top_ops: int, detailed: bool):
    """估算模型性能（基于操作级别的详细分析）
    
    支持多种输出级别：
    - 默认：简化的主要性能指标
    - --verbose：详细的操作级别分析（等同于 --show-ops --detailed --top-ops=15）
    - --show-ops：显示详细的操作分解
    - --detailed：显示完整的详细分析
    - --top-ops N：显示前N个最耗时的操作
    """
    
    try:
        # 创建估算器
        estimator = PerformanceEstimator()
        
        # 构建硬件配置
        hardware_config = {}
        
        # 使用加速器参数
        if accelerator:
            hardware_config["accelerator"] = accelerator
        else:
            raise click.BadParameter("必须指定 --accelerator 参数")
        
        # 使用统一的计算函数
        metrics = calculate_performance_metrics(
            estimator=estimator,
            model_name=model,
            hardware_config=hardware_config,
            batch_size=batch_size,
            precision=precision,
            input_length=input_length,
            output_length=output_length
        )
        
        # 处理verbose选项：verbose等同于--show-ops --detailed --top-ops=15
        if verbose:
            show_ops = True
            detailed = True
            top_ops = 15
        
        # 构建简化结果
        simplified_result = {
            "model_name": model,
            "model_info": metrics["model_info"],
            "system_info": metrics["system_info"],
            "input_length": input_length,
            "output_length": output_length,
            "ttft_ms": metrics["ttft_ms"],
            "tpot_ms": metrics["tpot_ms"],
            "total_latency_ms": metrics["total_latency_ms"],
            "throughput_tokens_per_sec": metrics["throughput_tokens_per_sec"],
            "memory_usage_gb": metrics["memory_usage_gb"],
            "bottleneck": metrics["bottleneck"],
            "utilization_percent": metrics["utilization_percent"],
            "compute_utilization_percent": metrics["compute_utilization_percent"],
            "memory_bandwidth_utilization_percent": metrics["memory_bandwidth_utilization_percent"],
            "memory_capacity_utilization_percent": metrics["memory_capacity_utilization_percent"],
            "compute_util": metrics["compute_util"],
            "memory_util": metrics["memory_util"]
        }
        
        # 格式化输出
        if format == "json":
            # 根据是否需要详细输出来决定JSON格式
            if show_ops or detailed:
                # 构建完整结果用于详细输出
                full_result = {
                    "model_info": metrics["model_info"],
                    "system_info": metrics["system_info"],
                    "op_level_analysis": metrics["op_level_analysis"]
                }
                formatted_result = json.dumps(full_result, indent=2, ensure_ascii=False)
            else:
                formatted_result = json.dumps(simplified_result, indent=2, ensure_ascii=False)
        elif format == "csv":
            formatted_result = format_results(simplified_result, format, verbose)
        else:  # table format
            if show_ops or detailed:
                # 详细输出：显示操作级别分析
                full_result = {
                    "model_info": metrics["model_info"],
                    "system_info": metrics["system_info"],
                    "op_level_analysis": metrics["op_level_analysis"]
                }
                formatted_result = format_op_level_results(full_result, show_ops, top_ops, detailed)
            else:
                # 简化输出：显示主要指标和TTFT/TPOT
                formatted_result = format_estimate_results(simplified_result, format, verbose)
        
        # 输出结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_result)
            click.echo(f"结果已保存到: {output_file}")
        else:
            click.echo(formatted_result)
            
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        raise click.Abort()


def format_estimate_results(result: Dict[str, Any], format: str, verbose: bool) -> str:
    """格式化estimate命令的结果，包含TTFT/TPOT信息"""
    lines = []
    lines.append("=" * 60)
    lines.append("模型性能估算结果")
    lines.append("=" * 60)
    
    model_info = result["model_info"]
    system_info = result["system_info"]
    
    # 模型信息
    lines.append(f"\n模型: {model_info['name']}")
    lines.append(f"参数量: {model_info['parameters']}")
    lines.append(f"层数: {model_info['layers']}")
    lines.append(f"配置: batch_size={model_info['config']['batch_size']}, "
                 f"precision={model_info['config']['precision']}")
    
    # 硬件信息
    lines.append(f"\n硬件配置:")
    for acc in system_info["accelerators"]:
        # Handle compute_capability_tflops as dictionary
        compute_cap = acc['compute_capability_tflops']
        if isinstance(compute_cap, dict):
            # Display FP16 if available, otherwise FP32, otherwise first available
            tflops = compute_cap.get('fp16', compute_cap.get('fp32', next(iter(compute_cap.values()))))
            precision = 'fp16' if 'fp16' in compute_cap else ('fp32' if 'fp32' in compute_cap else next(iter(compute_cap.keys())))
            compute_str = f"{tflops:.1f} TFLOPS ({precision})"
        else:
            compute_str = f"{compute_cap:.1f} TFLOPS"
        
        lines.append(f"  - {acc['name']}: {compute_str}, "
                     f"{acc['memory_bandwidth_gb_s']:.0f} GB/s, {acc['memory_capacity_gb']:.0f} GB")
    
    # 序列长度配置
    lines.append(f"\n序列配置:")
    lines.append(f"  输入长度: {result['input_length']} tokens")
    lines.append(f"  输出长度: {result['output_length']} tokens")
    lines.append(f"  总长度: {result['input_length'] + result['output_length']} tokens")
    
    # 性能指标
    lines.append(f"\n性能指标:")
    lines.append(f"  TTFT (Time To First Token): {result['ttft_ms']:.1f} ms")
    lines.append(f"  TPOT (Time Per Output Token): {result['tpot_ms']:.1f} ms")
    lines.append(f"  总延迟: {result['total_latency_ms']:.1f} ms")
    lines.append(f"  吞吐量: {result['throughput_tokens_per_sec']:.1f} tokens/s")
    lines.append(f"  内存使用: {result['memory_usage_gb']:.2f} GB")
    
    # 利用率分析
    lines.append(f"\n利用率分析:")
    lines.append(f"  整体利用率: {result['utilization_percent']:.1f}%")
    lines.append(f"  算力利用率: {result['compute_util']:.1f}%")
    lines.append(f"  内存利用率: {result['memory_util']:.1f}%")
    lines.append(f"  主要瓶颈: {result['bottleneck']}")
    
    return "\n".join(lines)


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
            # Handle compute_capability_tflops as dictionary
            compute_cap = info['compute_capability_tflops']
            if isinstance(compute_cap, dict):
                # Display FP16 if available, otherwise FP32, otherwise first available
                tflops = compute_cap.get('fp16', compute_cap.get('fp32', next(iter(compute_cap.values()))))
                precision = 'fp16' if 'fp16' in compute_cap else ('fp32' if 'fp32' in compute_cap else next(iter(compute_cap.keys())))
                compute_str = f"{tflops:.1f} TFLOPS ({precision})"
            else:
                compute_str = f"{compute_cap:.1f} TFLOPS"
            
            data.append([
                info["name"],
                info["manufacturer"],
                compute_str,
                f"{info['memory_bandwidth_gb_s']:.0f} GB/s",
                f"{info['memory_capacity_gb']:.0f} GB",
                f"{info['power_consumption_w']}W" if info['power_consumption_w'] else "N/A"
            ])
        
        headers = ["型号", "厂商", "算力", "内存带宽", "内存容量", "功耗"]
        table = tabulate(data, headers=headers, tablefmt="grid")
        click.echo(table)


@cli.command()
@click.option("--model", "-m", required=True, help="模型名称")
@click.option("--accelerator", "-a", required=True, help="加速器型号")
@click.option("--input-lengths", "-i", default="1024,2048,4096,8192,16384", help="输入长度列表，逗号分隔")
@click.option("--output-lengths", "-o", default="1024,2048,4096", help="输出长度列表，逗号分隔")
@click.option("--batch-size", "-b", type=int, default=1, help="批次大小")
@click.option("--precision", "-p", default="fp16", help="精度类型")
@click.option("--output-file", type=click.Path(), help="输出文件路径")
@click.option("--format", "-f", default="table", type=click.Choice(["table", "json", "csv"]), help="输出格式")
def benchmark(model: str, accelerator: str, input_lengths: str, output_lengths: str, 
              batch_size: int, precision: str, output_file: Optional[str], format: str):
    """预估模型在不同输入/输出长度下的 TTFT 和 TPOT 指标（基于操作级别的详细分析）
    
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
                # 构建硬件配置
                hardware_config = {"accelerator": accelerator}
                
                # 使用统一的计算函数
                metrics = calculate_performance_metrics(
                    estimator=estimator,
                    model_name=model,
                    hardware_config=hardware_config,
                    batch_size=batch_size,
                    precision=precision,
                    input_length=input_len,
                    output_length=output_len
                )
                
                results.append({
                    "input_length": input_len,
                    "output_length": output_len,
                    "ttft_ms": metrics["ttft_ms"],
                    "tpot_ms": metrics["tpot_ms"],
                    "total_latency_ms": metrics["total_latency_ms"],
                    "throughput_tokens_per_sec": metrics["throughput_tokens_per_sec"],
                    "memory_usage_gb": metrics["memory_usage_gb"],
                    "utilization_percent": metrics["utilization_percent"],
                    "bottleneck": metrics["bottleneck"],
                    "compute_util": metrics["compute_util"],
                    "memory_util": metrics["memory_util"]
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
    lines.append("=== 模型性能比较（操作级别分析）===\n")
    
    data = []
    for item in results:
        model_name = item["model"]
        result = item["result"]
        
        data.append([
            model_name,
            f"{result.get('memory_usage_gb', 0):.2f} GB",
            f"{result.get('throughput_tokens_per_sec', 0):.1f} tokens/s",
            f"{result.get('latency_ms', 0):.1f} ms",
            f"{result.get('utilization_percent', 0):.1f}%",
            result.get('bottleneck', 'N/A'),
            f"{result.get('compute_util', 0):.1f}%",
            f"{result.get('memory_util', 0):.1f}%"
        ])
    
    headers = ["模型", "内存使用", "吞吐量", "延迟", "整体利用率", "瓶颈", "算力利用率", "内存利用率"]
    table = tabulate(data, headers=headers, tablefmt="grid")
    lines.append(table)
    
    # 添加性能排名
    if len(results) > 1:
        lines.append("\n=== 性能排名 ===")
        
        # 按吞吐量排序
        throughput_ranking = sorted(results, key=lambda x: x["result"].get('throughput_tokens_per_sec', 0), reverse=True)
        lines.append("\n吞吐量排名:")
        for i, item in enumerate(throughput_ranking, 1):
            result = item["result"]
            lines.append(f"  {i}. {item['model']}: {result.get('throughput_tokens_per_sec', 0):.1f} tokens/s")
        
        # 按延迟排序
        latency_ranking = sorted(results, key=lambda x: x["result"].get('latency_ms', float('inf')))
        lines.append("\n延迟排名 (越低越好):")
        for i, item in enumerate(latency_ranking, 1):
            result = item["result"]
            lines.append(f"  {i}. {item['model']}: {result.get('latency_ms', 0):.1f} ms")
        
        # 按内存使用排序
        memory_ranking = sorted(results, key=lambda x: x["result"].get('memory_usage_gb', 0))
        lines.append("\n内存使用排名 (越低越好):")
        for i, item in enumerate(memory_ranking, 1):
            result = item["result"]
            lines.append(f"  {i}. {item['model']}: {result.get('memory_usage_gb', 0):.2f} GB")
    
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
        lines.append("模型,加速器,输入长度,输出长度,TTFT(ms),TPOT(ms),总延迟(ms),吞吐量(tokens/s),内存使用(GB),利用率(%),瓶颈类型,算力利用率(%),内存利用率(%)")
        
        for item in results:
            if "error" in item:
                continue
            lines.append(f"{model},{accelerator},{item['input_length']},{item['output_length']},"
                        f"{item['ttft_ms']:.1f},{item['tpot_ms']:.1f},{item['total_latency_ms']:.1f},"
                        f"{item['throughput_tokens_per_sec']:.1f},{item['memory_usage_gb']:.2f},"
                        f"{item['utilization_percent']:.1f},{item.get('bottleneck', 'N/A')},"
                        f"{item.get('compute_util', 0):.1f},{item.get('memory_util', 0):.1f}")
        return "\n".join(lines)
    
    else:  # table format
        lines = []
        lines.append(f"=== TTFT & TPOT 基准测试结果（操作级别分析）===")
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
                    f"{item['utilization_percent']:.1f}%",
                    f"{item.get('bottleneck', 'N/A')}"
                ])
        
        headers = ["输入长度", "输出长度", "TTFT", "TPOT", "总延迟", "吞吐量", "内存使用", "利用率", "瓶颈"]
        table = tabulate(data, headers=headers, tablefmt="grid")
        lines.append(table)
        
        # 添加统计摘要
        successful_results = [r for r in results if "error" not in r]
        if successful_results:
            lines.append("\n=== 统计摘要 ===")
            avg_ttft = sum(r['ttft_ms'] for r in successful_results) / len(successful_results)
            avg_tpot = sum(r['tpot_ms'] for r in successful_results) / len(successful_results)
            avg_throughput = sum(r['throughput_tokens_per_sec'] for r in successful_results) / len(successful_results)
            avg_compute_util = sum(r.get('compute_util', 0) for r in successful_results) / len(successful_results)
            avg_memory_util = sum(r.get('memory_util', 0) for r in successful_results) / len(successful_results)
            
            lines.append(f"平均 TTFT: {avg_ttft:.1f} ms")
            lines.append(f"平均 TPOT: {avg_tpot:.1f} ms")  
            lines.append(f"平均吞吐量: {avg_throughput:.1f} tokens/s")
            lines.append(f"平均算力利用率: {avg_compute_util:.1f}%")
            lines.append(f"平均内存利用率: {avg_memory_util:.1f}%")
            
            # 瓶颈分析统计
            bottleneck_counts = {}
            for r in successful_results:
                bottleneck = r.get('bottleneck', 'Unknown')
                bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
            
            lines.append(f"\n瓶颈分布:")
            for bottleneck, count in bottleneck_counts.items():
                percentage = (count / len(successful_results)) * 100
                lines.append(f"  {bottleneck}: {count} 次 ({percentage:.1f}%)")
        
        return "\n".join(lines)


def format_op_level_results(result: Dict[str, Any], show_ops: bool = False, 
                           top_ops: int = 10, detailed: bool = False) -> str:
    """格式化操作级别估算结果"""
    output = []
    
    # 基本信息
    output.append("=" * 60)
    output.append("操作级别性能估算结果")
    output.append("=" * 60)
    
    model_info = result["model_info"]
    system_info = result["system_info"]
    op_analysis = result["op_level_analysis"]
    
    # 模型信息
    output.append(f"\n模型: {model_info['name']}")
    output.append(f"参数量: {model_info['parameters']}")
    output.append(f"层数: {model_info['layers']}")
    output.append(f"配置: batch_size={model_info['config']['batch_size']}, "
                 f"context_length={model_info['config']['context_length']}, "
                 f"precision={model_info['config']['precision']}")
    
    # 硬件信息
    output.append(f"\n硬件配置:")
    for acc in system_info["accelerators"]:
        # Handle compute_capability_tflops as dictionary
        compute_cap = acc['compute_capability_tflops']
        if isinstance(compute_cap, dict):
            # Display FP16 if available, otherwise FP32, otherwise first available
            tflops = compute_cap.get('fp16', compute_cap.get('fp32', next(iter(compute_cap.values()))))
            precision = 'fp16' if 'fp16' in compute_cap else ('fp32' if 'fp32' in compute_cap else next(iter(compute_cap.keys())))
            compute_str = f"{tflops:.1f} TFLOPS ({precision})"
        else:
            compute_str = f"{compute_cap:.1f} TFLOPS"
        
        output.append(f"  - {acc['name']}: {compute_str}, "
                     f"{acc['memory_bandwidth_gb_s']:.0f} GB/s, {acc['memory_capacity_gb']:.0f} GB")
    
    # 性能指标
    output.append(f"\n性能指标:")
    output.append(f"  吞吐量: {op_analysis['throughput_tokens_per_sec']:.1f} tokens/s")
    output.append(f"  延迟: {op_analysis['total_time_per_token_ms']:.1f} ms/token")
    
    # 瓶颈分析
    bottleneck = op_analysis["bottleneck_analysis"]
    output.append(f"\n瓶颈分析:")
    output.append(f"  主要瓶颈: {bottleneck['major_bottleneck']}")
    output.append(f"  算力受限操作: {bottleneck['compute_bound_ops_count']} 个 "
                 f"({bottleneck['compute_bound_percentage']:.1f}%)")
    output.append(f"  内存受限操作: {bottleneck['memory_bound_ops_count']} 个 "
                 f"({bottleneck['memory_bound_percentage']:.1f}%)")
    
    # 操作类型分解
    output.append(f"\n操作类型分解:")
    op_breakdown = op_analysis["op_breakdown"]
    
    # 准备表格数据
    table_data = []
    for op_type, stats in op_breakdown.items():
        percentage = (stats["total_time_ms"] / op_analysis["total_time_per_token_ms"] * 100) if op_analysis["total_time_per_token_ms"] > 0 else 0
        table_data.append([
            op_type,
            stats["count"],
            f"{stats['total_time_ms']:.3f}",
            f"{percentage:.1f}%",
            f"{stats['total_flops'] / 1e9:.3f} G",
            f"{stats['total_memory_bytes'] / 1024**2:.3f} MB",
            f"{stats['avg_compute_util']:.1f}%",
            f"{stats['avg_memory_util']:.1f}%"
        ])
    
    # 按时间排序
    table_data.sort(key=lambda x: float(x[2]), reverse=True)
    
    headers = ["操作类型", "数量", "总时间(ms)", "占比", "计算量(FLOPs)", "内存读写", "算力利用率", "内存利用率"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    output.append(table)
    
    # 主要瓶颈操作
    if op_analysis.get("major_bottleneck_ops"):
        output.append(f"\n主要瓶颈操作 (前{min(top_ops, len(op_analysis['major_bottleneck_ops']))}个):")
        
        bottleneck_data = []
        for i, op in enumerate(op_analysis["major_bottleneck_ops"][:top_ops]):
            bottleneck_data.append([
                i + 1,
                op["name"],
                op["type"],
                f"{op['time_ms']:.3f}",
                f"{op['percentage']:.1f}%",
                op["bottleneck_type"],
                op.get("gemm_shape", "N/A")
            ])
        
        headers = ["#", "操作名称", "类型", "时间(ms)", "占比", "瓶颈", "矩阵形状"]
        table = tabulate(bottleneck_data, headers=headers, tablefmt="grid")
        output.append(table)
    
    # 详细操作列表
    if show_ops:
        output.append(f"\n详细操作列表 (前{top_ops}个最耗时):")
        detailed_ops = sorted(op_analysis["detailed_ops"], 
                            key=lambda x: x["time_ms"], reverse=True)[:top_ops]
        
        ops_data = []
        for op in detailed_ops:
            ops_data.append([
                op["name"],
                op["type"],
                f"{op['time_ms']:.4f}",
                f"{op['arithmetic_intensity']:.2f}",
                f"{op['compute_util']:.1f}%",
                f"{op['memory_util']:.1f}%",
                "✓" if op["is_gemm"] else "✗",
                op.get("gemm_shape", "N/A")
            ])
        
        headers = ["操作名称", "类型", "时间(ms)", "算术强度", "算力%", "内存%", "GEMM", "形状"]
        table = tabulate(ops_data, headers=headers, tablefmt="grid")
        output.append(table)
    
    # 优化建议
    output.append(f"\n优化建议:")
    recommendations = result.get("comprehensive_recommendations", [])
    for rec in recommendations:
        output.append(f"  {rec}")
    
    # 详细分析
    if detailed:
        output.append(f"\n详细性能分析:")
        output.append(f"  总计算量: {op_analysis['total_flops']:.2e} FLOPs")
        output.append(f"  总内存访问: {op_analysis['total_memory_bytes'] / 1024**3:.2f} GB")
        output.append(f"  单token时间: {op_analysis['total_time_per_token_ms']:.3f} ms")
        
        # 层级分解
        output.append(f"\n层级时间分解:")
        layer_data = []
        for layer in op_analysis["layer_breakdown"]:
            layer_data.append([
                layer["layer_idx"] if layer["layer_idx"] >= 0 else "LM_Head",
                layer["layer_type"],
                layer["op_count"],
                f"{layer['time_ms']:.3f}",
                f"{layer['percentage']:.1f}%"
            ])
        
        headers = ["层索引", "层类型", "操作数", "时间(ms)", "占比"]
        table = tabulate(layer_data, headers=headers, tablefmt="grid")
        output.append(table)
    
    return "\n".join(output)


def main():
    """主程序入口"""
    cli()


if __name__ == "__main__":
    main() 