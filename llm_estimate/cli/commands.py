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
@click.option("--batch-size", "-b", type=int, default=1, help="批次大小")
@click.option("--context-length", "-l", type=int, help="上下文长度")
@click.option("--precision", "-p", default="fp16", help="精度类型 (fp32/fp16/bf16/int8/int4)")
@click.option("--output", "-o", type=click.Path(), help="输出文件路径")
@click.option("--format", "-f", default="table", type=click.Choice(["table", "json", "csv"]), help="输出格式")
@click.option("--verbose", "-v", is_flag=True, help="详细输出")
def estimate(model: str, accelerator: Optional[str],
            batch_size: int, context_length: Optional[int], precision: str,
            output: Optional[str], format: str, verbose: bool):
    """估算模型性能（基于操作级别的详细分析）"""
    
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
            
        # 构建模型配置
        model_config = {
            "batch_size": batch_size,
            "precision": precision
        }
        if context_length:
            model_config["context_length"] = context_length
        
        # 执行操作级别估算
        result = estimator.estimate_op_level(
            model_name=model,
            hardware_config=hardware_config,
            model_config=model_config
        )
        
        # 提取主要性能指标用于格式化
        op_analysis = result["op_level_analysis"]
        
        # 从操作级别分析计算各种利用率
        bottleneck_analysis = op_analysis["bottleneck_analysis"]
        compute_utilization = bottleneck_analysis["compute_bound_percentage"]
        memory_utilization = bottleneck_analysis["memory_bound_percentage"]
        overall_utilization = max(compute_utilization, memory_utilization)
        
        simplified_result = {
            "model_name": model,
            "model_info": result["model_info"],
            "system_info": result["system_info"],
            "throughput_tokens_per_sec": op_analysis["throughput_tokens_per_sec"],
            "latency_ms": op_analysis["total_time_per_token_ms"],
            "memory_usage_gb": op_analysis["memory_usage_gb"],
            "bottleneck": bottleneck_analysis["major_bottleneck"],
            "utilization_percent": overall_utilization,
            "compute_utilization_percent": compute_utilization,
            "memory_bandwidth_utilization_percent": memory_utilization,
            "memory_capacity_utilization_percent": memory_utilization,
            "compute_util": compute_utilization,
            "memory_util": memory_utilization
        }
        
        # 格式化输出
        if format == "json":
            formatted_result = json.dumps(simplified_result, indent=2, ensure_ascii=False)
        elif format == "csv":
            formatted_result = format_results(simplified_result, format, verbose)
        else:  # table format
            if verbose:
                # 详细输出：显示操作级别分析
                formatted_result = format_op_level_results(result, show_ops=True, top_ops=15, detailed=True)
            else:
                # 简化输出：只显示主要指标
                formatted_result = format_results(simplified_result, format, verbose)
        
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


@cli.command("estimate-ops")
@click.option("--model", "-m", required=True, help="模型名称")
@click.option("--accelerator", "-a", help="加速器型号 (如: rtx-4090, a100-40gb, i9-13900k)")
@click.option("--batch-size", "-b", type=int, default=1, help="批次大小")
@click.option("--context-length", "-l", type=int, help="上下文长度")
@click.option("--precision", "-p", default="fp16", help="精度类型 (fp32/fp16/bf16/int8/int4)")
@click.option("--output", "-o", type=click.Path(), help="输出文件路径")
@click.option("--format", "-f", default="table", type=click.Choice(["table", "json", "csv"]), help="输出格式")
@click.option("--show-ops", is_flag=True, help="显示详细的操作分解")
@click.option("--top-ops", type=int, default=10, help="显示前N个最耗时的操作")
@click.option("--detailed", is_flag=True, help="显示完整的详细分析")
def estimate_ops(model: str, accelerator: Optional[str],
                batch_size: int, context_length: Optional[int], precision: str,
                output: Optional[str], format: str, show_ops: bool, top_ops: int, detailed: bool):
    """
    执行操作级别的详细性能估算
    
    基于矩阵乘法运算的算力和带宽利用率，
    细化到每个transformer操作的耗时估算。
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
            
        # 构建模型配置
        model_config = {
            "batch_size": batch_size,
            "precision": precision
        }
        if context_length:
            model_config["context_length"] = context_length
        
        # 执行操作级别估算
        result = estimator.estimate_op_level(
            model_name=model,
            hardware_config=hardware_config,
            model_config=model_config
        )
        
        # 格式化输出
        if format == "json":
            formatted_result = json.dumps(result, indent=2, ensure_ascii=False)
        else:
            formatted_result = format_op_level_results(result, show_ops, top_ops, detailed)
        
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
        output.append(f"  - {acc['name']}: {acc['compute_capability_tflops']:.1f} TFLOPS, "
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
            f"{stats['avg_compute_util']:.1f}%",
            f"{stats['avg_memory_util']:.1f}%"
        ])
    
    # 按时间排序
    table_data.sort(key=lambda x: float(x[2]), reverse=True)
    
    headers = ["操作类型", "数量", "总时间(ms)", "占比", "算力利用率", "内存利用率"]
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
  benchmark --model llama-2-7b --accelerator rtx-4090 --input-lengths 512,1024 --output-lengths 128,256
  """
    click.echo(help_text)


@cli.command()
@click.option("--models", "-m", required=True, help="模型列表，逗号分隔")
@click.option("--accelerator", "-a", required=True, help="加速器型号")
@click.option("--output", "-o", type=click.Path(), help="输出文件路径")
def compare(models: str, accelerator: str, output: Optional[str]):
    """比较多个模型的性能（基于操作级别的详细分析）"""
    
    model_list = [m.strip() for m in models.split(",")]
    estimator = PerformanceEstimator()
    
    results = []
    for model_name in model_list:
        try:
            # 使用操作级别估算
            result = estimator.estimate_op_level(
                model_name=model_name,
                hardware_config={"accelerator": accelerator}
            )
            
            # 提取主要指标
            op_analysis = result["op_level_analysis"]
            
            # 从操作级别分析计算各种利用率
            bottleneck_analysis = op_analysis["bottleneck_analysis"]
            compute_utilization = bottleneck_analysis["compute_bound_percentage"]
            memory_utilization = bottleneck_analysis["memory_bound_percentage"]
            overall_utilization = max(compute_utilization, memory_utilization)
            
            simplified_result = {
                "model_name": model_name,
                "model_info": result["model_info"],
                "system_info": result["system_info"],
                "throughput_tokens_per_sec": op_analysis["throughput_tokens_per_sec"],
                "latency_ms": op_analysis["total_time_per_token_ms"],
                "memory_usage_gb": op_analysis["memory_usage_gb"],
                "bottleneck": bottleneck_analysis["major_bottleneck"],
                "utilization_percent": overall_utilization,
                "compute_utilization_percent": compute_utilization,
                "memory_bandwidth_utilization_percent": memory_utilization,
                "memory_capacity_utilization_percent": memory_utilization,
                "compute_util": compute_utilization,
                "memory_util": memory_utilization
            }
            
            results.append({
                "model": model_name,
                "result": simplified_result
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
                # 构建配置
                hardware_config = {"accelerator": accelerator}
                model_config = {
                    "batch_size": batch_size,
                    "precision": precision,
                    "context_length": input_len,
                    "max_new_tokens": output_len
                }
                
                # 执行操作级别估算
                result = estimator.estimate_op_level(
                    model_name=model,
                    hardware_config=hardware_config,
                    model_config=model_config
                )
                
                # 从操作级别分析中提取指标
                op_analysis = result["op_level_analysis"]
                
                # 重新计算真实的 TPOT 和 TTFT：
                # 1. TTFT = prefill阶段处理完整输入序列的时间
                # 2. TPOT = 基于当前序列长度的单token decode时间（会随序列增长而增加）
                
                # 获取prefill阶段的时间
                # prefill时间 = 处理完整输入序列的时间
                prefill_result = estimator.estimate_op_level(
                    model_name=model,
                    hardware_config=hardware_config,
                    model_config={
                        "batch_size": batch_size,
                        "precision": precision,
                        "context_length": input_len,  # 完整输入序列
                        "max_new_tokens": 1
                    }
                )
                prefill_analysis = prefill_result["op_level_analysis"]
                
                # TTFT: prefill阶段处理整个输入序列的时间
                # 注意：这里的 total_time_per_token_ms 实际上是"每个token的平均时间"
                # 所以 prefill 总时间 = 平均时间 * 序列长度
                ttft_ms = prefill_analysis["total_time_per_token_ms"] * input_len
                
                # 计算 TPOT：更准确的方法是考虑序列长度随时间的增长
                # 在解码过程中，序列长度从 input_len 增长到 input_len + output_len
                # 我们计算起始和结束时的token生成时间，然后取平均值
                
                # 计算在输入序列长度基础上的第一个token生成时间
                start_decode_result = estimator.estimate_op_level(
                    model_name=model,
                    hardware_config=hardware_config,
                    model_config={
                        "batch_size": batch_size,
                        "precision": precision,
                        "context_length": input_len,  # 刚开始解码时的序列长度
                        "max_new_tokens": 1
                    }
                )
                start_tpot_ms = start_decode_result["op_level_analysis"]["total_time_per_token_ms"]
                
                # 计算在最终序列长度基础上的token生成时间
                end_decode_result = estimator.estimate_op_level(
                    model_name=model,
                    hardware_config=hardware_config,
                    model_config={
                        "batch_size": batch_size,
                        "precision": precision,
                        "context_length": input_len + output_len - 1,  # 解码结束时的序列长度
                        "max_new_tokens": 1
                    }
                )
                end_tpot_ms = end_decode_result["op_level_analysis"]["total_time_per_token_ms"]
                
                # TPOT: 取起始和结束token生成时间的平均值
                # 这更准确地反映了整个解码过程中的平均token生成时间
                tpot_ms = (start_tpot_ms + end_tpot_ms) / 2
                
                # 使用更精确的总延迟计算：
                # 对于不同的输出长度，采用不同的计算策略
                if output_len == 1:
                    # 只有一个输出token，总延迟就是 TTFT
                    total_latency_ms = ttft_ms
                    total_decode_time = 0
                else:
                    # 多个输出token，需要计算后续token的解码时间
                    total_decode_time = 0
                    # 线性插值计算每个输出token的生成时间
                    for i in range(output_len - 1):
                        # 当前序列长度：input_len + i + 1 (因为已经生成了i个token)
                        if output_len > 2:
                            progress = i / (output_len - 2)
                        else:
                            progress = 0
                        current_tpot = start_tpot_ms + progress * (end_tpot_ms - start_tpot_ms)
                        total_decode_time += current_tpot
                    
                    # 总延迟 = TTFT + 后续token的解码时间
                    total_latency_ms = ttft_ms + total_decode_time
                
                # 使用decode过程中的典型利用率（基于中等序列长度）
                decode_analysis = start_decode_result["op_level_analysis"]
                
                # 计算总延迟和吞吐量
                total_tokens = input_len + output_len
                actual_throughput = total_tokens / (total_latency_ms / 1000) if total_latency_ms > 0 else 0
                
                # 获取其他指标
                memory_usage_gb = op_analysis["memory_usage_gb"]
                
                # 使用decode阶段的利用率作为主要指标（因为这是推理的常态）
                utilization_percent = max(
                    decode_analysis["bottleneck_analysis"]["compute_bound_percentage"],
                    decode_analysis["bottleneck_analysis"]["memory_bound_percentage"]
                )
                
                results.append({
                    "input_length": input_len,
                    "output_length": output_len,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "total_latency_ms": total_latency_ms,
                    "throughput_tokens_per_sec": actual_throughput,
                    "memory_usage_gb": memory_usage_gb,
                    "utilization_percent": utilization_percent,
                    "bottleneck": decode_analysis["bottleneck_analysis"]["major_bottleneck"],
                    "compute_util": decode_analysis["bottleneck_analysis"]["compute_bound_percentage"],
                    "memory_util": decode_analysis["bottleneck_analysis"]["memory_bound_percentage"]
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


def main():
    """主程序入口"""
    cli()


if __name__ == "__main__":
    main() 