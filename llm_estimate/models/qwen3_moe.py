"""
Qwen3-MoE模型实现

包含Qwen3-MoE模型的具体实现，支持混合专家架构。
参考 Qwen3-MoE 官方实现和优化指南修正 FLOPS 和内存计算。
"""

from typing import Dict
from .base import BaseModel, ModelSpecs, ModelConfig


class Qwen3MoEModel(BaseModel):
    """Qwen3-MoE混合专家模型实现"""
    
    def __init__(self, specs: ModelSpecs, config: ModelConfig = None, **kwargs):
        """
        初始化Qwen3-MoE模型
        
        Args:
            specs: 模型规格
            config: 模型配置
            **kwargs: 其他配置参数，包括MoE特有参数
        """
        super().__init__(specs, config, **kwargs)
        
        # MoE特有参数
        self.num_experts = kwargs.get('num_experts', 64)  # 专家数量
        self.experts_per_token = kwargs.get('experts_per_token', 4)  # 每个token激活的专家数
        self.expert_capacity = kwargs.get('expert_capacity', None)  # 专家容量限制
        self.moe_layers = kwargs.get('moe_layers', None)  # MoE层位置，如果None则所有FFN层都是MoE
        
        # 计算MoE层数，如果未指定则假设所有FFN层都是MoE
        if self.moe_layers is None:
            self.num_moe_layers = self.specs.layers
        else:
            self.num_moe_layers = len(self.moe_layers)
        
        # 计算激活比例
        self.activation_ratio = self.experts_per_token / self.num_experts
        
        # 路由器开销因子
        self.router_overhead_factor = kwargs.get('router_overhead_factor', 0.1)
    
    def calculate_memory_usage(self, precision: str = "fp16") -> Dict[str, float]:
        """
        计算Qwen3-MoE模型内存使用量
        
        MoE模型的内存计算需要考虑：
        1. 共享层（非MoE层）的内存
        2. 专家层的内存（通常只加载激活的专家）
        3. 路由器的内存开销
        4. 专家间通信的额外内存
        
        Args:
            precision: 精度类型 ("fp32", "fp16", "bf16", "int8", "int4")
            
        Returns:
            包含各种内存使用量的字典
        """
        # 精度对应的字节数
        precision_bytes = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "int8": 1,
            "int4": 0.5
        }
        
        if precision not in precision_bytes:
            raise ValueError(f"Unsupported precision: {precision}")
        
        bytes_per_param = precision_bytes[precision]
        
        # 1. 共享层内存计算
        hidden_size = self.specs.hidden_size
        layers = self.specs.layers
        vocab_size = self.specs.vocab_size
        
        # 注意力层参数（所有层都有）
        attention_params = layers * (
            hidden_size * hidden_size +  # Q projection
            2 * hidden_size * (self.specs.num_key_value_heads * self.specs.head_dim) +  # K,V projections (GQA)
            hidden_size * hidden_size  # output projection
        )
        
        # 层归一化参数
        layer_norm_params = layers * 2 * hidden_size  # 每层2个RMSNorm
        
        # 词嵌入参数
        embedding_params = vocab_size * hidden_size
        
        # 共享层内存
        shared_memory = (attention_params + layer_norm_params + embedding_params) * bytes_per_param
        
        # 2. MoE专家层内存计算
        # 每个专家的参数数量 (SwiGLU: gate, up, down三个投影)
        expert_params = 3 * hidden_size * self.specs.intermediate_size
        
        # 所有专家的总参数数量
        total_expert_params = self.num_moe_layers * self.num_experts * expert_params
        
        # 路由器参数 (hidden_size -> num_experts)
        router_params = self.num_moe_layers * hidden_size * self.num_experts
        
        # MoE层总内存（包括所有专家）
        moe_total_memory = (total_expert_params + router_params) * bytes_per_param
        
        # 3. 激活专家内存（运行时实际加载的专家）
        active_expert_params = self.num_moe_layers * self.experts_per_token * expert_params
        active_expert_memory = active_expert_params * bytes_per_param
        
        # 4. 路由器和专家选择的额外内存开销
        router_overhead_memory = moe_total_memory * self.router_overhead_factor
        
        # 5. 模型总内存（设计时）
        model_memory_total = shared_memory + moe_total_memory
        
        # 6. 运行时内存（只加载激活的专家）
        model_memory_runtime = shared_memory + active_expert_memory + router_overhead_memory
        
        # 7. KV缓存内存计算
        if self.config.use_kv_cache:
            kv_cache_memory = (
                2 * self.specs.layers * self.specs.num_key_value_heads * 
                self.specs.head_dim * self.config.context_length * 
                self.config.batch_size * bytes_per_param
            )
        else:
            kv_cache_memory = 0
        
        # 8. 激活值内存 (基于 Qwen3-MoE transformer 架构)
        sequence_length = self.config.context_length
        batch_size = self.config.batch_size
        
        # 注意力分数矩阵
        attention_scores_memory = (
            batch_size * self.specs.attention_heads * 
            sequence_length * sequence_length * bytes_per_param
        )
        
        # 隐藏状态激活
        hidden_activations_memory = (
            batch_size * sequence_length * hidden_size * 
            self.specs.layers * bytes_per_param
        )
        
        # MoE激活内存（只考虑激活的专家）
        moe_activations_memory = (
            batch_size * sequence_length * self.specs.intermediate_size * 
            self.num_moe_layers * self.experts_per_token * bytes_per_param
        )
        
        # 路由器激活内存
        router_activations_memory = (
            batch_size * sequence_length * self.num_experts * 
            self.num_moe_layers * bytes_per_param
        )
        
        # RoPE相关激活
        rope_activations_memory = (
            batch_size * sequence_length * self.specs.attention_heads * 
            self.specs.head_dim * self.specs.layers * bytes_per_param * 0.1
        )
        
        total_activation_memory = (
            attention_scores_memory + hidden_activations_memory + 
            moe_activations_memory + router_activations_memory + rope_activations_memory
        )
        
        # 9. 训练时额外内存
        gradient_memory = model_memory_runtime  # 只对激活的专家计算梯度
        optimizer_memory = model_memory_runtime * 2  # Adam优化器状态
        
        # 10. 总内存计算
        total_inference = model_memory_runtime + kv_cache_memory + total_activation_memory
        total_training = total_inference + gradient_memory + optimizer_memory
        
        return {
            "model_memory_total_gb": model_memory_total / (1024**3),
            "model_memory_runtime_gb": model_memory_runtime / (1024**3),
            "shared_memory_gb": shared_memory / (1024**3),
            "moe_total_memory_gb": moe_total_memory / (1024**3),
            "active_expert_memory_gb": active_expert_memory / (1024**3),
            "router_overhead_memory_gb": router_overhead_memory / (1024**3),
            "kv_cache_memory_gb": kv_cache_memory / (1024**3),
            "attention_scores_memory_gb": attention_scores_memory / (1024**3),
            "hidden_activations_memory_gb": hidden_activations_memory / (1024**3),
            "moe_activations_memory_gb": moe_activations_memory / (1024**3),
            "router_activations_memory_gb": router_activations_memory / (1024**3),
            "rope_activations_memory_gb": rope_activations_memory / (1024**3),
            "total_activation_memory_gb": total_activation_memory / (1024**3),
            "gradient_memory_gb": gradient_memory / (1024**3),
            "optimizer_memory_gb": optimizer_memory / (1024**3),
            "total_inference_gb": total_inference / (1024**3),
            "total_training_gb": total_training / (1024**3),
            "precision": precision,
            "moe_memory_efficiency": {
                "memory_reduction_ratio": model_memory_runtime / model_memory_total,
                "expert_utilization": self.activation_ratio,
                "memory_savings_gb": (model_memory_total - model_memory_runtime) / (1024**3)
            }
        }
    
    def estimate_flops_per_token(self) -> float:
        """
        估算Qwen3-MoE模型每个token的计算量（FLOPS）
        
        MoE模型的FLOPS计算需要考虑：
        1. 共享层的FLOPS（注意力层、层归一化等）
        2. 路由器的FLOPS（专家选择）
        3. 激活专家的FLOPS（只计算被激活的专家）
        4. 专家间通信和负载均衡的开销
        
        Returns:
            每个token的FLOPS数
        """
        hidden_size = self.specs.hidden_size
        layers = self.specs.layers
        vocab_size = self.specs.vocab_size
        num_heads = self.specs.attention_heads
        num_kv_heads = self.specs.num_key_value_heads
        head_dim = self.specs.head_dim
        context_length = self.config.context_length
        
        # === 1. 共享层FLOPS计算 ===
        
        # 1.1 注意力机制 FLOPS (所有层都有)
        # QKV投影
        q_projection_flops = hidden_size * hidden_size
        kv_projection_flops = 2 * hidden_size * (num_kv_heads * head_dim)
        
        # RoPE计算
        rope_flops = num_heads * head_dim * context_length * 4
        
        # 注意力计算
        attention_scores_flops = num_heads * head_dim * context_length
        softmax_flops = 3 * num_heads * context_length
        attention_output_flops = num_heads * context_length * head_dim
        
        # 输出投影
        output_projection_flops = hidden_size * hidden_size
        
        # 单层注意力总FLOPS
        attention_flops_per_layer = (
            q_projection_flops + kv_projection_flops + rope_flops +
            attention_scores_flops + softmax_flops + 
            attention_output_flops + output_projection_flops
        )
        
        # 所有层的注意力FLOPS
        total_attention_flops = layers * attention_flops_per_layer
        
        # 1.2 层归一化 FLOPS
        layer_norm_flops = layers * 2 * 3 * hidden_size  # 每层2个RMSNorm
        
        # === 2. MoE层FLOPS计算 ===
        
        # 2.1 路由器FLOPS
        # 路由器计算：hidden_size -> num_experts (线性投影 + softmax)
        router_projection_flops = hidden_size * self.num_experts
        router_softmax_flops = 3 * self.num_experts  # softmax计算
        router_selection_flops = self.num_experts * 2  # top-k选择的近似开销
        
        router_flops_per_layer = (
            router_projection_flops + router_softmax_flops + router_selection_flops
        )
        total_router_flops = self.num_moe_layers * router_flops_per_layer
        
        # 2.2 激活专家FLOPS
        # 每个激活专家的FLOPS (SwiGLU: gate, up, down三个投影)
        gate_projection_flops = hidden_size * self.specs.intermediate_size
        up_projection_flops = hidden_size * self.specs.intermediate_size
        down_projection_flops = self.specs.intermediate_size * hidden_size
        
        # SwiGLU激活函数FLOPS
        swish_flops = self.specs.intermediate_size * 3  # sigmoid + multiply
        activation_flops = self.specs.intermediate_size + swish_flops
        
        # 单个专家的总FLOPS
        expert_flops = (
            gate_projection_flops + up_projection_flops + 
            activation_flops + down_projection_flops
        )
        
        # 所有激活专家的FLOPS
        total_expert_flops = self.num_moe_layers * self.experts_per_token * expert_flops
        
        # 2.3 专家输出组合FLOPS
        # 将多个专家的输出按权重组合
        expert_combination_flops = (
            self.num_moe_layers * self.experts_per_token * hidden_size
        )
        
        # === 3. 词嵌入和输出投影 FLOPS ===
        embedding_flops = hidden_size
        output_head_flops = hidden_size * vocab_size
        
        # === 4. 总FLOPS计算 ===
        total_flops = (
            total_attention_flops + layer_norm_flops + total_router_flops +
            total_expert_flops + expert_combination_flops + 
            embedding_flops + output_head_flops
        )
        
        # 乘以2以考虑乘法和加法操作 (multiply-accumulate)
        total_flops *= 2
        
        return total_flops
    
    def estimate_model_flops_per_token_simplified(self) -> float:
        """
        使用简化的公式估算MoE模型的FLOPS
        
        基于激活参数数量和MoE效率的简化计算
        
        Returns:
            每个token的FLOPS数 (简化估算)
        """
        hidden_size = self.specs.hidden_size
        layers = self.specs.layers
        vocab_size = self.specs.vocab_size
        
        # 共享层参数
        attention_params = layers * (4 * hidden_size * hidden_size)  # QKV + output
        layer_norm_params = layers * 2 * hidden_size
        embedding_params = vocab_size * hidden_size
        
        # MoE层参数（只计算激活的专家）
        active_expert_params = (
            self.num_moe_layers * self.experts_per_token * 
            3 * hidden_size * self.specs.intermediate_size
        )
        
        # 路由器参数
        router_params = self.num_moe_layers * hidden_size * self.num_experts
        
        # 激活参数总数
        total_active_params = (
            attention_params + layer_norm_params + 
            active_expert_params + router_params
        )
        
        # 简化公式：forward pass ≈ 2N FLOPs (N为激活参数数量)
        # 对于MoE，考虑路由器开销，系数稍微增加
        forward_flops = 2.2 * total_active_params
        
        return forward_flops
    
    def get_performance_analysis(self) -> Dict[str, any]:
        """
        获取Qwen3-MoE模型性能分析报告
        
        Returns:
            包含FLOPS分解和性能指标的字典
        """
        total_flops = self.estimate_flops_per_token()
        simplified_flops = self.estimate_model_flops_per_token_simplified()
        memory_usage = self.calculate_memory_usage(self.config.precision)
        
        # FLOPS分解分析
        hidden_size = self.specs.hidden_size
        layers = self.specs.layers
        context_length = self.config.context_length
        
        # 按组件分解FLOPS
        # 注意力FLOPS
        attention_flops_per_layer = (
            4 * hidden_size * hidden_size +  # projections
            self.specs.attention_heads * self.specs.head_dim * context_length * 4 +  # RoPE
            self.specs.attention_heads * self.specs.head_dim * context_length +  # attention
            3 * self.specs.attention_heads * context_length  # softmax
        ) * 2
        total_attention_flops = layers * attention_flops_per_layer
        
        # 路由器FLOPS
        router_flops_per_layer = (
            hidden_size * self.num_experts + 3 * self.num_experts + self.num_experts * 2
        ) * 2
        total_router_flops = self.num_moe_layers * router_flops_per_layer
        
        # 激活专家FLOPS
        expert_flops_per_layer = (
            3 * hidden_size * self.specs.intermediate_size +  # projections
            self.specs.intermediate_size * 4  # activation
        ) * 2
        total_expert_flops = self.num_moe_layers * self.experts_per_token * expert_flops_per_layer
        
        # 词汇表FLOPS
        vocab_flops = hidden_size * self.specs.vocab_size * 2
        
        # MoE效率分析
        # 如果是Dense模型，FFN FLOPS会是多少
        dense_ffn_flops = layers * expert_flops_per_layer
        moe_efficiency = total_expert_flops / dense_ffn_flops if dense_ffn_flops > 0 else 0
        
        return {
            "model_info": {
                "name": self.specs.name,
                "parameters": f"{self.specs.parameters}B",
                "layers": layers,
                "hidden_size": hidden_size,
                "attention_heads": self.specs.attention_heads,
                "kv_heads": self.specs.num_key_value_heads,
                "vocab_size": self.specs.vocab_size,
                "max_context_length": context_length,
                "model_features": ["RoPE", "SwiGLU", "RMSNorm", "GQA", "MoE"],
                "moe_config": {
                    "num_experts": self.num_experts,
                    "experts_per_token": self.experts_per_token,
                    "moe_layers": self.num_moe_layers,
                    "activation_ratio": self.activation_ratio,
                    "router_overhead_factor": self.router_overhead_factor
                }
            },
            "flops_analysis": {
                "total_flops_per_token": total_flops,
                "simplified_flops_per_token": simplified_flops,
                "attention_flops_percentage": (total_attention_flops / total_flops) * 100,
                "router_flops_percentage": (total_router_flops / total_flops) * 100,
                "expert_flops_percentage": (total_expert_flops / total_flops) * 100,
                "vocab_flops_percentage": (vocab_flops / total_flops) * 100,
                "attention_flops_per_token": total_attention_flops,
                "router_flops_per_token": total_router_flops,
                "expert_flops_per_token": total_expert_flops,
                "vocab_flops_per_token": vocab_flops,
                "moe_efficiency": {
                    "compute_efficiency": moe_efficiency,
                    "expert_utilization": self.activation_ratio,
                    "routing_overhead_percentage": (total_router_flops / total_flops) * 100
                }
            },
            "memory_analysis": memory_usage,
            "efficiency_metrics": {
                "memory_per_parameter_bytes": (memory_usage["model_memory_runtime_gb"] * 1024**3) / (self.specs.parameters * 1e9),
                "flops_per_parameter": total_flops / (self.specs.parameters * 1e9),
                "kv_cache_overhead_percentage": (memory_usage["kv_cache_memory_gb"] / memory_usage["total_inference_gb"]) * 100 if memory_usage["total_inference_gb"] > 0 else 0,
                "context_scaling_factor": context_length / 2048,
                "vocab_overhead_percentage": (vocab_flops / total_flops) * 100,
                "moe_memory_efficiency": memory_usage["moe_memory_efficiency"]["memory_reduction_ratio"],
                "expert_activation_efficiency": self.activation_ratio
            },
            "moe_specific_features": {
                "expert_selection_strategy": f"Top-{self.experts_per_token}",
                "routing_mechanism": "Learned Router",
                "expert_capacity": self.expert_capacity,
                "load_balancing": "Auxiliary Loss",
                "sparsity_ratio": 1 - self.activation_ratio,
                "memory_savings": {
                    "runtime_vs_total_ratio": memory_usage["moe_memory_efficiency"]["memory_reduction_ratio"],
                    "memory_savings_gb": memory_usage["moe_memory_efficiency"]["memory_savings_gb"],
                    "expert_parameter_ratio": (self.experts_per_token / self.num_experts)
                },
                "computational_advantages": {
                    "compute_flops_reduction": f"{(1 - moe_efficiency) * 100:.1f}%",
                    "parameter_efficiency": f"{self.activation_ratio * 100:.1f}% active parameters",
                    "scaling_benefits": "Constant compute with increased capacity"
                }
            }
        } 