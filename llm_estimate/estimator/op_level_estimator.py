"""
操作级别性能估算器

基于矩阵乘法运算的算力和带宽利用率，
细化到每个transformer操作的耗时估算。
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import math

from ..models.base import BaseModel
from ..hardware.base import SystemSpec


class OpType(Enum):
    """操作类型枚举"""
    # 矩阵乘法操作
    GEMM_QKV_PROJ = "gemm_qkv_projection"    # Q/K/V投影
    GEMM_ATTENTION = "gemm_attention"        # 注意力计算
    GEMM_OUTPUT_PROJ = "gemm_output_proj"    # 注意力输出投影
    GEMM_FFN_GATE = "gemm_ffn_gate"          # FFN门投影
    GEMM_FFN_UP = "gemm_ffn_up"              # FFN上投影
    GEMM_FFN_DOWN = "gemm_ffn_down"          # FFN下投影
    GEMM_LM_HEAD = "gemm_lm_head"            # 语言模型头
    
    # 元素级操作
    ELEMENTWISE_SOFTMAX = "elementwise_softmax"     # Softmax
    ELEMENTWISE_ACTIVATION = "elementwise_activation" # 激活函数
    ELEMENTWISE_LAYER_NORM = "elementwise_layer_norm" # 层归一化
    ELEMENTWISE_ADD = "elementwise_add"              # 残差连接
    
    # 内存操作
    MEMORY_COPY = "memory_copy"              # 内存拷贝
    MEMORY_KV_CACHE = "memory_kv_cache"      # KV缓存读写


@dataclass
class OpProfile:
    """单个操作的性能特征"""
    op_type: OpType
    op_name: str
    
    # 计算特征
    flops: float                # 浮点运算量
    memory_bytes: float         # 内存访问量
    
    # 矩阵乘法特征 (如果适用)
    is_gemm: bool = False
    m: int = 0                  # 矩阵维度 M
    n: int = 0                  # 矩阵维度 N  
    k: int = 0                  # 矩阵维度 K
    
    # 计算强度 (FLOPs/Byte)
    arithmetic_intensity: float = 0.0
    
    # 估算耗时
    compute_time_ms: float = 0.0      # 计算受限耗时
    memory_time_ms: float = 0.0       # 内存受限耗时
    actual_time_ms: float = 0.0       # 实际耗时
    
    # 利用率
    compute_utilization: float = 0.0   # 算力利用率
    memory_utilization: float = 0.0    # 带宽利用率


@dataclass 
class LayerProfile:
    """单层的操作分解"""
    layer_idx: int
    layer_type: str  # "attention" 或 "ffn"
    ops: List[OpProfile]
    total_time_ms: float = 0.0
    

class OpLevelEstimator:
    """操作级别性能估算器"""
    
    def __init__(self):
        # GEMM性能参数 - 基于实际GPU测试数据（进一步优化）
        self.gemm_efficiency = {
            "large": 0.90,      # 大矩阵效率 - 稍微保守一些，考虑实际瓶颈
            "medium": 0.85,     # 中等矩阵效率  
            "small": 0.75,      # 小矩阵效率 - 进一步提高
        }
        
        # 内存带宽效率 - 考虑实际访问模式
        self.memory_efficiency = {
            "sequential": 0.95,  # 顺序访问
            "random": 0.80,      # 随机访问 - 进一步提高
            "broadcast": 0.90,   # 广播访问
        }
        
        # GPU overlap factor - 考虑计算和内存访问的重叠
        self.compute_memory_overlap = 0.85  # 85%的重叠度 - 现代GPU重叠能力很强
        
        # Tensor Core加速因子 (对于支持的数据类型和矩阵形状)
        self.tensor_core_speedup = 1.5  # 1.5x加速
    
    def estimate_model_ops(self, model: BaseModel, system_spec: SystemSpec) -> Dict[str, Any]:
        """
        估算整个模型的操作级别性能
        
        Args:
            model: 模型实例
            system_spec: 系统硬件规格
            
        Returns:
            详细的操作级别估算结果
        """
        # 分解模型为层级操作
        layer_profiles = self._decompose_model_to_ops(model)
        
        # 估算每个操作的性能
        for layer_profile in layer_profiles:
            for op in layer_profile.ops:
                self._estimate_op_performance(op, system_spec)
            
            # 计算层总时间
            layer_profile.total_time_ms = sum(op.actual_time_ms for op in layer_profile.ops)
        
        # 汇总结果
        return self._aggregate_results(layer_profiles, model, system_spec)
    
    def _decompose_model_to_ops(self, model: BaseModel) -> List[LayerProfile]:
        """将模型分解为具体的操作"""
        layer_profiles = []
        
        batch_size = model.config.batch_size
        seq_len = model.config.context_length
        hidden_size = model.specs.hidden_size
        intermediate_size = model.specs.intermediate_size
        num_heads = model.specs.attention_heads
        num_kv_heads = model.specs.num_key_value_heads
        head_dim = model.specs.head_dim
        vocab_size = model.specs.vocab_size
        
        # 为每一层生成操作分解
        for layer_idx in range(model.specs.layers):
            
            # === Attention层操作 ===
            attention_ops = []
            
            # 1. Q/K/V投影 - 3个GEMM操作
            # Q投影: [batch*seq, hidden] @ [hidden, hidden] = [batch*seq, hidden]
            q_proj = OpProfile(
                op_type=OpType.GEMM_QKV_PROJ,
                op_name=f"layer_{layer_idx}_q_proj",
                flops=2 * batch_size * seq_len * hidden_size * hidden_size,
                memory_bytes=(batch_size * seq_len * hidden_size + hidden_size * hidden_size + 
                             batch_size * seq_len * hidden_size) * 2,  # fp16
                is_gemm=True,
                m=batch_size * seq_len,
                n=hidden_size,
                k=hidden_size
            )
            attention_ops.append(q_proj)
            
            # K投影: [batch*seq, hidden] @ [hidden, kv_hidden] = [batch*seq, kv_hidden]
            kv_hidden = num_kv_heads * head_dim
            k_proj = OpProfile(
                op_type=OpType.GEMM_QKV_PROJ,
                op_name=f"layer_{layer_idx}_k_proj",
                flops=2 * batch_size * seq_len * hidden_size * kv_hidden,
                memory_bytes=(batch_size * seq_len * hidden_size + hidden_size * kv_hidden + 
                             batch_size * seq_len * kv_hidden) * 2,
                is_gemm=True,
                m=batch_size * seq_len,
                n=kv_hidden,
                k=hidden_size
            )
            attention_ops.append(k_proj)
            
            # V投影: 同K投影
            v_proj = OpProfile(
                op_type=OpType.GEMM_QKV_PROJ,
                op_name=f"layer_{layer_idx}_v_proj",
                flops=2 * batch_size * seq_len * hidden_size * kv_hidden,
                memory_bytes=(batch_size * seq_len * hidden_size + hidden_size * kv_hidden + 
                             batch_size * seq_len * kv_hidden) * 2,
                is_gemm=True,
                m=batch_size * seq_len,
                n=kv_hidden,
                k=hidden_size
            )
            attention_ops.append(v_proj)
            
            # 2. 注意力计算 - Q@K^T
            # [batch, heads, seq, head_dim] @ [batch, heads, head_dim, seq] = [batch, heads, seq, seq]
            attention_qk = OpProfile(
                op_type=OpType.GEMM_ATTENTION,
                op_name=f"layer_{layer_idx}_attention_qk",
                flops=2 * batch_size * num_heads * seq_len * seq_len * head_dim,
                memory_bytes=(batch_size * num_heads * seq_len * head_dim * 2 + 
                             batch_size * num_heads * seq_len * seq_len) * 2,
                is_gemm=True,
                m=seq_len,
                n=seq_len,
                k=head_dim
            )
            attention_ops.append(attention_qk)
            
            # 3. Softmax - 元素级操作
            softmax_op = OpProfile(
                op_type=OpType.ELEMENTWISE_SOFTMAX,
                op_name=f"layer_{layer_idx}_softmax",
                flops=3 * batch_size * num_heads * seq_len * seq_len,  # exp + sum + div
                memory_bytes=batch_size * num_heads * seq_len * seq_len * 2 * 2  # 读写两次
            )
            attention_ops.append(softmax_op)
            
            # 4. 注意力输出 - Attention@V  
            attention_v = OpProfile(
                op_type=OpType.GEMM_ATTENTION,
                op_name=f"layer_{layer_idx}_attention_v",
                flops=2 * batch_size * num_heads * seq_len * seq_len * head_dim,
                memory_bytes=(batch_size * num_heads * seq_len * seq_len + 
                             batch_size * num_heads * seq_len * head_dim + 
                             batch_size * num_heads * seq_len * head_dim) * 2,
                is_gemm=True,
                m=seq_len,
                n=head_dim,
                k=seq_len
            )
            attention_ops.append(attention_v)
            
            # 5. 输出投影
            output_proj = OpProfile(
                op_type=OpType.GEMM_OUTPUT_PROJ,
                op_name=f"layer_{layer_idx}_output_proj",
                flops=2 * batch_size * seq_len * hidden_size * hidden_size,
                memory_bytes=(batch_size * seq_len * hidden_size + hidden_size * hidden_size + 
                             batch_size * seq_len * hidden_size) * 2,
                is_gemm=True,
                m=batch_size * seq_len,
                n=hidden_size,
                k=hidden_size
            )
            attention_ops.append(output_proj)
            
            # 6. 层归一化 + 残差连接
            layer_norm_1 = OpProfile(
                op_type=OpType.ELEMENTWISE_LAYER_NORM,
                op_name=f"layer_{layer_idx}_layer_norm_1",
                flops=3 * batch_size * seq_len * hidden_size,  # mean + var + normalize
                memory_bytes=batch_size * seq_len * hidden_size * 2 * 2  # 读写两次
            )
            attention_ops.append(layer_norm_1)
            
            attention_layer = LayerProfile(
                layer_idx=layer_idx,
                layer_type="attention",
                ops=attention_ops
            )
            layer_profiles.append(attention_layer)
            
            # === FFN层操作 ===
            ffn_ops = []
            
            # 1. Gate投影
            gate_proj = OpProfile(
                op_type=OpType.GEMM_FFN_GATE,
                op_name=f"layer_{layer_idx}_ffn_gate",
                flops=2 * batch_size * seq_len * hidden_size * intermediate_size,
                memory_bytes=(batch_size * seq_len * hidden_size + 
                             hidden_size * intermediate_size + 
                             batch_size * seq_len * intermediate_size) * 2,
                is_gemm=True,
                m=batch_size * seq_len,
                n=intermediate_size,
                k=hidden_size
            )
            ffn_ops.append(gate_proj)
            
            # 2. Up投影
            up_proj = OpProfile(
                op_type=OpType.GEMM_FFN_UP,
                op_name=f"layer_{layer_idx}_ffn_up",
                flops=2 * batch_size * seq_len * hidden_size * intermediate_size,
                memory_bytes=(batch_size * seq_len * hidden_size + 
                             hidden_size * intermediate_size + 
                             batch_size * seq_len * intermediate_size) * 2,
                is_gemm=True,
                m=batch_size * seq_len,
                n=intermediate_size,
                k=hidden_size
            )
            ffn_ops.append(up_proj)
            
            # 3. SwiGLU激活
            activation_op = OpProfile(
                op_type=OpType.ELEMENTWISE_ACTIVATION,
                op_name=f"layer_{layer_idx}_swiglu",
                flops=batch_size * seq_len * intermediate_size,  # 简化估算
                memory_bytes=batch_size * seq_len * intermediate_size * 2 * 2  # gate * up
            )
            ffn_ops.append(activation_op)
            
            # 4. Down投影
            down_proj = OpProfile(
                op_type=OpType.GEMM_FFN_DOWN,
                op_name=f"layer_{layer_idx}_ffn_down",
                flops=2 * batch_size * seq_len * intermediate_size * hidden_size,
                memory_bytes=(batch_size * seq_len * intermediate_size + 
                             intermediate_size * hidden_size + 
                             batch_size * seq_len * hidden_size) * 2,
                is_gemm=True,
                m=batch_size * seq_len,
                n=hidden_size,
                k=intermediate_size
            )
            ffn_ops.append(down_proj)
            
            # 5. 层归一化 + 残差连接  
            layer_norm_2 = OpProfile(
                op_type=OpType.ELEMENTWISE_LAYER_NORM,
                op_name=f"layer_{layer_idx}_layer_norm_2",
                flops=3 * batch_size * seq_len * hidden_size,
                memory_bytes=batch_size * seq_len * hidden_size * 2 * 2
            )
            ffn_ops.append(layer_norm_2)
            
            ffn_layer = LayerProfile(
                layer_idx=layer_idx,
                layer_type="ffn",
                ops=ffn_ops
            )
            layer_profiles.append(ffn_layer)
        
        # === 添加LM Head ===
        lm_head_op = OpProfile(
            op_type=OpType.GEMM_LM_HEAD,
            op_name="lm_head",
            flops=2 * batch_size * seq_len * hidden_size * vocab_size,
            memory_bytes=(batch_size * seq_len * hidden_size + 
                         hidden_size * vocab_size + 
                         batch_size * seq_len * vocab_size) * 2,
            is_gemm=True,
            m=batch_size * seq_len,
            n=vocab_size,
            k=hidden_size
        )
        
        lm_head_layer = LayerProfile(
            layer_idx=-1,
            layer_type="lm_head",
            ops=[lm_head_op]
        )
        layer_profiles.append(lm_head_layer)
        
        return layer_profiles
    
    def _estimate_op_performance(self, op: OpProfile, system_spec: SystemSpec) -> None:
        """估算单个操作的性能"""
        if not system_spec.accelerators:
            return
        
        # 使用最强的加速器进行估算
        primary_accelerator = max(
            system_spec.accelerators,
            key=lambda acc: acc.compute_capability_tflops
        )
        
        peak_compute_flops = primary_accelerator.compute_capability_tflops * 1e12  # TFLOPS转FLOPS
        peak_memory_bw = primary_accelerator.memory_bandwidth_gb_s * 1e9          # GB/s转B/s
        
        # 计算算术强度
        op.arithmetic_intensity = op.flops / op.memory_bytes if op.memory_bytes > 0 else float('inf')
        
        if op.is_gemm:
            # 对于GEMM操作，使用实际效率估算
            base_efficiency = self._get_gemm_efficiency(op.m, op.n, op.k)
            
            # 检查是否可以使用Tensor Core加速 (fp16/bf16 + 合适的矩阵维度)
            efficiency = base_efficiency
            if self._can_use_tensor_core(op.m, op.n, op.k, system_spec):
                efficiency = min(0.95, base_efficiency * self.tensor_core_speedup)  # 上限95%
            
            effective_compute = peak_compute_flops * efficiency
            
            # 计算受限时间
            op.compute_time_ms = (op.flops / effective_compute) * 1000
            
            # 内存受限时间 (考虑内存访问效率)
            memory_efficiency = self.memory_efficiency["sequential"]  # GEMM通常是顺序访问
            effective_bandwidth = peak_memory_bw * memory_efficiency
            op.memory_time_ms = (op.memory_bytes / effective_bandwidth) * 1000
            
            # 考虑计算和内存访问的重叠 - 现代GPU可以同时进行计算和内存访问
            # 实际时间 = max(compute_time, memory_time) * (1 - overlap_factor) + min(compute_time, memory_time)
            max_time = max(op.compute_time_ms, op.memory_time_ms)
            min_time = min(op.compute_time_ms, op.memory_time_ms)
            
            # 对于高算术强度的GEMM，主要受算力限制，重叠效果更好
            if op.arithmetic_intensity > 20:  # 很高算术强度
                overlap_factor = self.compute_memory_overlap * 1.1  # 更好的重叠
            elif op.arithmetic_intensity > 10:  # 高算术强度
                overlap_factor = self.compute_memory_overlap
            else:  # 低算术强度，更多受内存限制
                overlap_factor = self.compute_memory_overlap * 0.6
            
            # 确保重叠因子不超过1
            overlap_factor = min(overlap_factor, 0.95)
                
            op.actual_time_ms = max_time * (1 - overlap_factor) + min_time
            
        else:
            # 对于元素级操作，效率较低但考虑现代GPU的向量化能力
            efficiency = 0.7  # 进一步提高元素级操作效率
            effective_compute = peak_compute_flops * efficiency
            
            op.compute_time_ms = (op.flops / effective_compute) * 1000
            
            # 元素级操作通常是内存受限的
            memory_efficiency = self.memory_efficiency["random"]
            effective_bandwidth = peak_memory_bw * memory_efficiency
            op.memory_time_ms = (op.memory_bytes / effective_bandwidth) * 1000
            
            # 元素级操作重叠度较低
            op.actual_time_ms = max(op.compute_time_ms, op.memory_time_ms) * 0.9  # 考虑一些优化
        
        # 计算利用率
        if op.actual_time_ms > 0:
            op.compute_utilization = (op.compute_time_ms / (op.actual_time_ms + 1e-9)) * 100
            op.memory_utilization = (op.memory_time_ms / (op.actual_time_ms + 1e-9)) * 100
            
            # 确保利用率不超过100%
            op.compute_utilization = min(op.compute_utilization, 100.0)
            op.memory_utilization = min(op.memory_utilization, 100.0)
    
    def _can_use_tensor_core(self, m: int, n: int, k: int, system_spec: SystemSpec) -> bool:
        """判断是否可以使用Tensor Core加速"""
        if not system_spec.accelerators:
            return False
            
        # 检查加速器是否支持Tensor Core (简化判断，主要是RTX/A系列GPU)
        primary_accelerator = system_spec.accelerators[0]
        accelerator_name = primary_accelerator.name.lower()
        
        # 支持Tensor Core的GPU
        tensor_core_gpus = ['rtx', 'a100', 'h100', 'v100', 'a10', 'a30', 'a40']
        supports_tensor_core = any(gpu in accelerator_name for gpu in tensor_core_gpus)
        
        if not supports_tensor_core:
            return False
        
        # Tensor Core对矩阵维度有要求，通常需要是8或16的倍数
        # 这里简化判断：矩阵足够大且维度合适
        min_size_for_tensor_core = 128
        dimension_alignment = 16  # Tensor Core通常要求16的倍数
        
        # 检查矩阵是否足够大和对齐
        large_enough = (m >= min_size_for_tensor_core and 
                       n >= min_size_for_tensor_core and 
                       k >= min_size_for_tensor_core)
        
        # 检查关键维度是否对齐（放宽要求，现代Tensor Core比较灵活）
        aligned = (k % dimension_alignment == 0 or 
                  n % dimension_alignment == 0 or
                  m >= 512)  # 大矩阵即使不完全对齐也能受益
        
        return large_enough and aligned
    
    def _get_gemm_efficiency(self, m: int, n: int, k: int) -> float:
        """根据矩阵规模估算GEMM效率"""
        # 计算总的计算量作为矩阵"大小"的指标
        total_ops = m * n * k
        
        # 同时考虑矩阵的各个维度，避免某个维度过小导致效率低下
        min_dim = min(m, n, k)
        
        # 基于总计算量和最小维度综合判断
        if total_ops > 64 * 1024 * 1024 and min_dim >= 512:  # 大矩阵且各维度都足够大
            return self.gemm_efficiency["large"]
        elif total_ops > 4 * 1024 * 1024 and min_dim >= 128:  # 中等矩阵
            return self.gemm_efficiency["medium"]
        else:  # 小矩阵
            # 对于非常小的矩阵，效率会进一步降低
            if min_dim < 32:
                return self.gemm_efficiency["small"] * 0.7
            else:
                return self.gemm_efficiency["small"]
    
    def _aggregate_results(self, layer_profiles: List[LayerProfile], 
                          model: BaseModel, system_spec: SystemSpec) -> Dict[str, Any]:
        """汇总操作级别的估算结果"""
        # 统计所有操作
        all_ops = []
        for layer_profile in layer_profiles:
            all_ops.extend(layer_profile.ops)
        
        # 按操作类型分组统计
        op_stats = {}
        for op in all_ops:
            op_type = op.op_type.value
            if op_type not in op_stats:
                op_stats[op_type] = {
                    "count": 0,
                    "total_time_ms": 0.0,
                    "total_flops": 0.0,
                    "total_memory_bytes": 0.0,
                    "avg_compute_util": 0.0,
                    "avg_memory_util": 0.0,
                }
            
            stats = op_stats[op_type]
            stats["count"] += 1
            stats["total_time_ms"] += op.actual_time_ms
            stats["total_flops"] += op.flops
            stats["total_memory_bytes"] += op.memory_bytes
            stats["avg_compute_util"] += op.compute_utilization
            stats["avg_memory_util"] += op.memory_utilization
        
        # 计算平均值
        for stats in op_stats.values():
            if stats["count"] > 0:
                stats["avg_compute_util"] /= stats["count"]
                stats["avg_memory_util"] /= stats["count"]
        
        # 总体统计
        total_time_ms = sum(op.actual_time_ms for op in all_ops)
        total_flops = sum(op.flops for op in all_ops)
        total_memory_bytes = sum(op.memory_bytes for op in all_ops)
        
        # 生成性能分析
        bottleneck_analysis = self._analyze_bottlenecks(all_ops)
        optimization_suggestions = self._generate_optimization_suggestions(op_stats, bottleneck_analysis)
        
        # 计算内存使用量 (基于模型参数和激活值)
        memory_usage_gb = self._estimate_memory_usage(model)
        
        return {
            "model_name": model.name,
            "total_time_per_token_ms": total_time_ms,
            "throughput_tokens_per_sec": 1000.0 / total_time_ms if total_time_ms > 0 else 0,
            "total_flops": total_flops,
            "total_memory_bytes": total_memory_bytes,
            "memory_usage_gb": memory_usage_gb,
            "op_breakdown": op_stats,
            "layer_breakdown": [
                {
                    "layer_idx": lp.layer_idx,
                    "layer_type": lp.layer_type,
                    "time_ms": lp.total_time_ms,
                    "percentage": (lp.total_time_ms / total_time_ms * 100) if total_time_ms > 0 else 0,
                    "op_count": len(lp.ops)
                }
                for lp in layer_profiles
            ],
            "bottleneck_analysis": bottleneck_analysis,
            "optimization_suggestions": optimization_suggestions,
            "detailed_ops": [
                {
                    "name": op.op_name,
                    "type": op.op_type.value,
                    "time_ms": op.actual_time_ms,
                    "flops": op.flops,
                    "memory_bytes": op.memory_bytes,
                    "arithmetic_intensity": op.arithmetic_intensity,
                    "compute_util": op.compute_utilization,
                    "memory_util": op.memory_utilization,
                    "is_gemm": op.is_gemm,
                    "gemm_shape": f"({op.m}, {op.n}, {op.k})" if op.is_gemm else None
                }
                for op in all_ops
            ]
        }
    
    def _analyze_bottlenecks(self, ops: List[OpProfile]) -> Dict[str, Any]:
        """分析性能瓶颈"""
        compute_bound_ops = [op for op in ops if op.compute_time_ms > op.memory_time_ms]
        memory_bound_ops = [op for op in ops if op.memory_time_ms >= op.compute_time_ms]
        
        compute_bound_time = sum(op.actual_time_ms for op in compute_bound_ops)
        memory_bound_time = sum(op.actual_time_ms for op in memory_bound_ops)
        total_time = compute_bound_time + memory_bound_time
        
        return {
            "compute_bound_percentage": (compute_bound_time / total_time * 100) if total_time > 0 else 0,
            "memory_bound_percentage": (memory_bound_time / total_time * 100) if total_time > 0 else 0,
            "compute_bound_ops_count": len(compute_bound_ops),
            "memory_bound_ops_count": len(memory_bound_ops),
            "major_bottleneck": "compute" if compute_bound_time > memory_bound_time else "memory"
        }
    
    def _generate_optimization_suggestions(self, op_stats: Dict[str, Any], 
                                         bottleneck_analysis: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        # 基于瓶颈类型的建议
        if bottleneck_analysis["major_bottleneck"] == "compute":
            suggestions.append("系统主要受算力限制，建议：")
            suggestions.append("  - 使用更高算力的加速器")
            suggestions.append("  - 考虑模型量化降低计算需求")
            suggestions.append("  - 优化GEMM操作的tile大小")
        else:
            suggestions.append("系统主要受内存带宽限制，建议：")
            suggestions.append("  - 使用更高带宽的加速器")
            suggestions.append("  - 优化内存访问模式")
            suggestions.append("  - 考虑减少激活值的内存占用")
        
        # 基于操作类型的建议
        gemm_types = ["gemm_qkv_projection", "gemm_attention", "gemm_ffn_gate", "gemm_ffn_up", "gemm_ffn_down"]
        total_gemm_time = sum(op_stats.get(op_type, {}).get("total_time_ms", 0) for op_type in gemm_types)
        total_time = sum(stats["total_time_ms"] for stats in op_stats.values())
        
        if total_time > 0:
            gemm_percentage = total_gemm_time / total_time * 100
            if gemm_percentage > 80:
                suggestions.append(f"GEMM操作占总时间的{gemm_percentage:.1f}%，建议：")
                suggestions.append("  - 优化矩阵乘法库(如使用cuBLAS/cutlass)")
                suggestions.append("  - 调整batch size提高GEMM效率") 
                suggestions.append("  - 考虑使用Tensor Core加速")
        
        return suggestions 
    
    def _estimate_memory_usage(self, model: BaseModel) -> float:
        """估算模型的内存使用量 (GB)"""
        # 使用模型的内存计算方法
        memory_info = model.calculate_memory_usage("fp16")  # 默认使用fp16
        return memory_info.get("total_inference_gb", 0)