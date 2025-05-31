"""
数据格式化工具

提供各种数据格式化功能。
"""

import json
from typing import Dict, Any
from tabulate import tabulate


def format_results(result: Dict[str, Any], format_type: str = "table", 
                  verbose: bool = False) -> str:
    """
    格式化估算结果
    
    Args:
        result: 估算结果字典
        format_type: 输出格式 ("table", "json", "csv")
        verbose: 是否显示详细信息
        
    Returns:
        格式化后的字符串
    """
    if format_type == "json":
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    elif format_type == "csv":
        return format_results_csv(result)
    
    else:  # table format
        return format_results_table(result, verbose)


def format_results_table(result: Dict[str, Any], verbose: bool = False) -> str:
    """格式化为表格形式"""
    lines = []
    
    # 基本信息
    lines.append("=== LLM性能估算结果 ===\n")
    
    # 模型信息
    lines.append(f"模型: {result['model_name']}")
    if verbose and 'model_info' in result:
        model_info = result['model_info']
        lines.append(f"参数量: {model_info.get('parameters', 'N/A')}")
        lines.append(f"层数: {model_info.get('layers', 'N/A')}")
        lines.append(f"隐藏维度: {model_info.get('hidden_size', 'N/A')}")
    
    lines.append("")
    
    # 硬件配置
    lines.append("硬件配置:")
    hw_config = result.get('hardware_config', {})
    for key, value in hw_config.items():
        lines.append(f"  {key}: {value}")
    
    lines.append("")
    
    # 性能指标
    metrics_data = [
        ["内存使用", f"{result.get('memory_usage_gb', 0):.2f} GB"],
        ["吞吐量", f"{result.get('throughput_tokens_per_sec', 0):.1f} tokens/s"],
        ["延迟", f"{result.get('latency_ms', 0):.1f} ms"],
        ["硬件利用率", f"{result.get('utilization_percent', 0):.1f}%"],
        ["性能瓶颈", result.get('bottleneck', 'N/A')]
    ]
    
    table = tabulate(metrics_data, headers=["指标", "值"], tablefmt="grid")
    lines.append("性能指标:")
    lines.append(table)
    
    # 建议
    if 'recommendations' in result and result['recommendations']:
        lines.append("\n优化建议:")
        for i, rec in enumerate(result['recommendations'], 1):
            lines.append(f"{i}. {rec}")
    
    return "\n".join(lines)


def format_results_csv(result: Dict[str, Any]) -> str:
    """格式化为CSV形式"""
    csv_lines = []
    
    # CSV头部
    headers = [
        "model_name", "memory_usage_gb", "throughput_tokens_per_sec",
        "latency_ms", "utilization_percent", "bottleneck"
    ]
    csv_lines.append(",".join(headers))
    
    # 数据行
    values = [
        result.get('model_name', ''),
        str(result.get('memory_usage_gb', 0)),
        str(result.get('throughput_tokens_per_sec', 0)),
        str(result.get('latency_ms', 0)),
        str(result.get('utilization_percent', 0)),
        result.get('bottleneck', '')
    ]
    csv_lines.append(",".join(values))
    
    return "\n".join(csv_lines)


def format_memory_size(bytes_size: float) -> str:
    """
    格式化内存大小
    
    Args:
        bytes_size: 字节大小
        
    Returns:
        格式化的内存大小字符串
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(bytes_size)
    
    for unit in units:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    
    return f"{size:.2f} PB"


def format_number_with_units(value: float, unit: str) -> str:
    """
    格式化数字并添加单位
    
    Args:
        value: 数值
        unit: 单位
        
    Returns:
        格式化后的字符串
    """
    if value >= 1e12:
        return f"{value/1e12:.2f}T{unit}"
    elif value >= 1e9:
        return f"{value/1e9:.2f}G{unit}"
    elif value >= 1e6:
        return f"{value/1e6:.2f}M{unit}"
    elif value >= 1e3:
        return f"{value/1e3:.2f}K{unit}"
    else:
        return f"{value:.2f}{unit}"


def format_performance_summary(results: list) -> str:
    """
    格式化性能对比摘要
    
    Args:
        results: 多个估算结果列表
        
    Returns:
        格式化的摘要表格
    """
    if not results:
        return "无数据"
    
    # 构建对比表格数据
    table_data = []
    for result in results:
        row = [
            result.get('model_name', 'N/A'),
            f"{result.get('memory_usage_gb', 0):.1f}",
            f"{result.get('throughput_tokens_per_sec', 0):.1f}",
            f"{result.get('latency_ms', 0):.1f}",
            f"{result.get('utilization_percent', 0):.1f}%"
        ]
        table_data.append(row)
    
    headers = ["模型", "内存(GB)", "吞吐量(token/s)", "延迟(ms)", "利用率"]
    
    return tabulate(table_data, headers=headers, tablefmt="grid") 