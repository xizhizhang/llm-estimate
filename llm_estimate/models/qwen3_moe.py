"""
Qwen3-MoE模型实现

包含Qwen3-MoE模型的具体实现，支持混合专家架构。
参考 Qwen3-MoE 官方实现和优化指南修正 FLOPS 和内存计算。
"""

from typing import Dict, List
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
        
        # MoE特有参数 - 优先从specs中读取，如果没有则从kwargs中获取默认值
        self.num_experts = specs.num_experts if specs.num_experts is not None else kwargs.get('num_experts', 64)
        self.experts_per_token = specs.experts_per_token if specs.experts_per_token is not None else kwargs.get('experts_per_token', 4)
        self.expert_capacity = specs.expert_capacity if specs.expert_capacity is not None else kwargs.get('expert_capacity', None)
        self.moe_layers = specs.moe_layers if specs.moe_layers is not None else kwargs.get('moe_layers', None)
        
        # 计算MoE层数，如果未指定则假设所有FFN层都是MoE
        if self.moe_layers is None:
            self.num_moe_layers = self.specs.layers
        else:
            self.num_moe_layers = len(self.moe_layers)
        
        # 计算激活比例
        self.activation_ratio = self.experts_per_token / self.num_experts
        
        # 路由器开销因子
        self.router_overhead_factor = kwargs.get('router_overhead_factor', 0.1)
    
    def decompose_to_ops(self) -> List['LayerProfile']:
        """分解Qwen3-MoE系列模型为具体操作"""
        from ..estimator.op_level_estimator import LayerProfile, OpProfile, OpType
        
        layer_profiles = []
        
        batch_size = self.config.batch_size
        seq_len = self.config.context_length
        hidden_size = self.specs.hidden_size
        intermediate_size = self.specs.intermediate_size
        num_heads = self.specs.attention_heads
        num_kv_heads = self.specs.num_key_value_heads
        head_dim = self.specs.head_dim
        vocab_size = self.specs.vocab_size
        
        # MoE参数
        num_experts = self.num_experts
        experts_per_token = self.experts_per_token
        
        # 根据推理模式调整序列长度
        # 在decode模式下，只处理一个新token，但需要与cache中的所有token做注意力计算
        is_decode_mode = self.config.inference_mode == "decode" and self.config.use_kv_cache
        current_seq_len = 1 if is_decode_mode else seq_len
        attention_seq_len = seq_len  # 注意力计算时的序列长度（包括cache）
        
        # 为每一层生成操作分解
        for layer_idx in range(self.specs.layers):
            
            # === Attention层操作 (与Qwen3相同) ===
            attention_ops = []
            
            # 1. Q/K/V投影 - 3个GEMM操作 (支持GQA)
            # 在decode模式下，只处理一个新token的投影
            
            # Q投影: [batch*current_seq, hidden] @ [hidden, hidden] = [batch*current_seq, hidden]
            q_proj = OpProfile(
                op_type=OpType.GEMM_QKV_PROJ,
                op_name=f"layer_{layer_idx}_q_proj",
                flops=2 * batch_size * current_seq_len * hidden_size * hidden_size,
                memory_bytes=(batch_size * current_seq_len * hidden_size + hidden_size * hidden_size + 
                             batch_size * current_seq_len * hidden_size) * 2,  # bf16
                is_gemm=True,
                m=batch_size * current_seq_len,
                n=hidden_size,
                k=hidden_size
            )
            attention_ops.append(q_proj)
            
            # K投影: [batch*current_seq, hidden] @ [hidden, kv_hidden] = [batch*current_seq, kv_hidden]
            kv_hidden = num_kv_heads * head_dim
            k_proj = OpProfile(
                op_type=OpType.GEMM_QKV_PROJ,
                op_name=f"layer_{layer_idx}_k_proj",
                flops=2 * batch_size * current_seq_len * hidden_size * kv_hidden,
                memory_bytes=(batch_size * current_seq_len * hidden_size + hidden_size * kv_hidden + 
                             batch_size * current_seq_len * kv_hidden) * 2,
                is_gemm=True,
                m=batch_size * current_seq_len,
                n=kv_hidden,
                k=hidden_size
            )
            attention_ops.append(k_proj)
            
            # V投影: 同K投影
            v_proj = OpProfile(
                op_type=OpType.GEMM_QKV_PROJ,
                op_name=f"layer_{layer_idx}_v_proj",
                flops=2 * batch_size * current_seq_len * hidden_size * kv_hidden,
                memory_bytes=(batch_size * current_seq_len * hidden_size + hidden_size * kv_hidden + 
                             batch_size * current_seq_len * kv_hidden) * 2,
                is_gemm=True,
                m=batch_size * current_seq_len,
                n=kv_hidden,
                k=hidden_size
            )
            attention_ops.append(v_proj)
            
            # 2. RoPE位置编码 - 元素级操作
            # 在decode模式下，只需要为当前token计算RoPE
            rope_op = OpProfile(
                op_type=OpType.ELEMENTWISE_ROPE,
                op_name=f"layer_{layer_idx}_rope",
                flops=6 * batch_size * current_seq_len * hidden_size,  # RoPE rotation computation
                memory_bytes=batch_size * current_seq_len * hidden_size * 2 * 2  # Q和K的RoPE应用
            )
            attention_ops.append(rope_op)
            
            # 3. 注意力计算 - Q@K^T
            # 在decode模式下，Q只有1个token，但要与cache中的所有K做计算
            # 复杂度从O(seq²)降低到O(seq)
            if is_decode_mode:
                # Decode模式: [batch, heads, 1, head_dim] @ [batch, heads, head_dim, seq_len] = [batch, heads, 1, seq_len]
                attention_qk_flops = 2 * batch_size * num_heads * 1 * attention_seq_len * head_dim
                attention_qk_memory = (batch_size * num_heads * 1 * head_dim + 
                                      batch_size * num_heads * head_dim * attention_seq_len + 
                                      batch_size * num_heads * 1 * attention_seq_len) * 2
                qk_m, qk_n, qk_k = 1, attention_seq_len, head_dim
            else:
                # Prefill模式: [batch, heads, seq, head_dim] @ [batch, heads, head_dim, seq] = [batch, heads, seq, seq]
                attention_qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
                attention_qk_memory = (batch_size * num_heads * seq_len * head_dim * 2 + 
                                      batch_size * num_heads * seq_len * seq_len) * 2
                qk_m, qk_n, qk_k = seq_len, seq_len, head_dim
            
            attention_qk = OpProfile(
                op_type=OpType.GEMM_ATTENTION,
                op_name=f"layer_{layer_idx}_attention_qk",
                flops=attention_qk_flops,
                memory_bytes=attention_qk_memory,
                is_gemm=True,
                m=qk_m,
                n=qk_n,
                k=qk_k
            )
            attention_ops.append(attention_qk)
            
            # 4. Softmax - 元素级操作
            # 在decode模式下，只需要对一个query的注意力分数做softmax
            if is_decode_mode:
                softmax_flops = 3 * batch_size * num_heads * 1 * attention_seq_len  # exp + sum + div
                softmax_memory = batch_size * num_heads * 1 * attention_seq_len * 2 * 2  # 读写两次
            else:
                softmax_flops = 3 * batch_size * num_heads * seq_len * seq_len  # exp + sum + div
                softmax_memory = batch_size * num_heads * seq_len * seq_len * 2 * 2  # 读写两次
            
            softmax_op = OpProfile(
                op_type=OpType.ELEMENTWISE_SOFTMAX,
                op_name=f"layer_{layer_idx}_softmax",
                flops=softmax_flops,
                memory_bytes=softmax_memory
            )
            attention_ops.append(softmax_op)
            
            # 5. 注意力输出 - Attention@V  
            # 在decode模式下，[batch, heads, 1, seq_len] @ [batch, heads, seq_len, head_dim] = [batch, heads, 1, head_dim]
            if is_decode_mode:
                attention_v_flops = 2 * batch_size * num_heads * 1 * attention_seq_len * head_dim
                attention_v_memory = (batch_size * num_heads * 1 * attention_seq_len + 
                                     batch_size * num_heads * attention_seq_len * head_dim + 
                                     batch_size * num_heads * 1 * head_dim) * 2
                v_m, v_n, v_k = 1, head_dim, attention_seq_len
            else:
                attention_v_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
                attention_v_memory = (batch_size * num_heads * seq_len * seq_len + 
                                     batch_size * num_heads * seq_len * head_dim + 
                                     batch_size * num_heads * seq_len * head_dim) * 2
                v_m, v_n, v_k = seq_len, head_dim, seq_len
            
            attention_v = OpProfile(
                op_type=OpType.GEMM_ATTENTION,
                op_name=f"layer_{layer_idx}_attention_v",
                flops=attention_v_flops,
                memory_bytes=attention_v_memory,
                is_gemm=True,
                m=v_m,
                n=v_n,
                k=v_k
            )
            attention_ops.append(attention_v)
            
            # 6. 输出投影
            # 在decode模式下，只需要投影一个token的输出
            output_proj = OpProfile(
                op_type=OpType.GEMM_OUTPUT_PROJ,
                op_name=f"layer_{layer_idx}_output_proj",
                flops=2 * batch_size * current_seq_len * hidden_size * hidden_size,
                memory_bytes=(batch_size * current_seq_len * hidden_size + hidden_size * hidden_size + 
                             batch_size * current_seq_len * hidden_size) * 2,
                is_gemm=True,
                m=batch_size * current_seq_len,
                n=hidden_size,
                k=hidden_size
            )
            attention_ops.append(output_proj)
            
            # 7. RMSNorm层归一化 (Qwen3特有)
            # 在decode模式下，只需要归一化一个token
            rms_norm_1 = OpProfile(
                op_type=OpType.ELEMENTWISE_LAYER_NORM,
                op_name=f"layer_{layer_idx}_rms_norm_1",
                flops=2 * batch_size * current_seq_len * hidden_size,  # RMSNorm: sqrt(mean(x^2)) + scale
                memory_bytes=batch_size * current_seq_len * hidden_size * 2 * 2  # 读写两次
            )
            attention_ops.append(rms_norm_1)
            
            attention_layer = LayerProfile(
                layer_idx=layer_idx,
                layer_type="attention",
                ops=attention_ops
            )
            layer_profiles.append(attention_layer)
            
            # === MoE FFN层操作 ===
            moe_ops = []
            
            # 1. 路由器计算 - 选择专家
            # 在decode模式下，只需要为一个token计算路由
            router_op = OpProfile(
                op_type=OpType.GEMM_FFN_GATE,
                op_name=f"layer_{layer_idx}_router",
                flops=2 * batch_size * current_seq_len * hidden_size * num_experts,  # 路由器计算
                memory_bytes=(batch_size * current_seq_len * hidden_size + 
                             batch_size * current_seq_len * num_experts) * 2,  # 只计算激活值内存
                is_gemm=True,
                m=batch_size * current_seq_len,
                n=num_experts,
                k=hidden_size
            )
            moe_ops.append(router_op)
            
            # 1.1 路由器额外开销 - 专家选择和token分发的计算开销
            # 这包括top-k选择、负载均衡、token重排序等
            router_overhead = OpProfile(
                op_type=OpType.ELEMENTWISE_ADD,
                op_name=f"layer_{layer_idx}_router_overhead",
                flops=batch_size * current_seq_len * num_experts * 2,  # 额外的路由开销
                memory_bytes=batch_size * current_seq_len * num_experts * 2 * 2  # 路由表和索引
            )
            moe_ops.append(router_overhead)
            
            # 2. 专家路由Softmax
            # 在decode模式下，只需要为一个token做softmax
            router_softmax = OpProfile(
                op_type=OpType.ELEMENTWISE_SOFTMAX,
                op_name=f"layer_{layer_idx}_router_softmax",
                flops=3 * batch_size * current_seq_len * num_experts,  # exp + sum + div
                memory_bytes=batch_size * current_seq_len * num_experts * 2 * 2  # 读写两次
            )
            moe_ops.append(router_softmax)
            
            # 3-6. 为每个激活的专家生成独立的操作
            # 每个token激活experts_per_token个专家，每个专家都有独立的gate, up, down投影
            for expert_idx in range(experts_per_token):
                # 3. 专家Gate投影
                expert_gate_proj = OpProfile(
                    op_type=OpType.GEMM_FFN_GATE,
                    op_name=f"layer_{layer_idx}_expert_{expert_idx}_gate",
                    flops=2 * batch_size * current_seq_len * hidden_size * intermediate_size,
                    memory_bytes=(batch_size * current_seq_len * hidden_size + 
                                 batch_size * current_seq_len * intermediate_size) * 2,  # 只计算激活值内存
                    is_gemm=True,
                    m=batch_size * current_seq_len,
                    n=intermediate_size,
                    k=hidden_size
                )
                moe_ops.append(expert_gate_proj)
                
                # 4. 专家Up投影
                expert_up_proj = OpProfile(
                    op_type=OpType.GEMM_FFN_UP,
                    op_name=f"layer_{layer_idx}_expert_{expert_idx}_up",
                    flops=2 * batch_size * current_seq_len * hidden_size * intermediate_size,
                    memory_bytes=(batch_size * current_seq_len * hidden_size + 
                                 batch_size * current_seq_len * intermediate_size) * 2,  # 只计算激活值内存
                    is_gemm=True,
                    m=batch_size * current_seq_len,
                    n=intermediate_size,
                    k=hidden_size
                )
                moe_ops.append(expert_up_proj)
                
                # 5. SwiGLU激活（每个专家独立）
                activation_op = OpProfile(
                    op_type=OpType.ELEMENTWISE_ACTIVATION,
                    op_name=f"layer_{layer_idx}_expert_{expert_idx}_swiglu",
                    flops=batch_size * current_seq_len * intermediate_size,
                    memory_bytes=batch_size * current_seq_len * intermediate_size * 2 * 2  # gate * up
                )
                moe_ops.append(activation_op)
                
                # 6. 专家Down投影
                expert_down_proj = OpProfile(
                    op_type=OpType.GEMM_FFN_DOWN,
                    op_name=f"layer_{layer_idx}_expert_{expert_idx}_down",
                    flops=2 * batch_size * current_seq_len * intermediate_size * hidden_size,
                    memory_bytes=(batch_size * current_seq_len * intermediate_size + 
                                 batch_size * current_seq_len * hidden_size) * 2,  # 只计算激活值内存
                    is_gemm=True,
                    m=batch_size * current_seq_len,
                    n=hidden_size,
                    k=intermediate_size
                )
                moe_ops.append(expert_down_proj)
            
            # 7. 专家输出加权合并
            # 在decode模式下，只需要为一个token合并专家输出
            expert_combine = OpProfile(
                op_type=OpType.ELEMENTWISE_ADD,
                op_name=f"layer_{layer_idx}_expert_combine",
                flops=batch_size * current_seq_len * hidden_size * experts_per_token,  # 加权求和
                memory_bytes=batch_size * current_seq_len * hidden_size * experts_per_token * 2  # 读取多个专家输出
            )
            moe_ops.append(expert_combine)
            
            # 7.1 专家输出同步开销 - 多专家并行计算后的结果同步
            # 在实际GPU实现中，不同专家的输出需要同步和重排序
            expert_sync = OpProfile(
                op_type=OpType.MEMORY_COPY,
                op_name=f"layer_{layer_idx}_expert_sync",
                flops=batch_size * current_seq_len * hidden_size * 0.5,  # 同步开销
                memory_bytes=batch_size * current_seq_len * hidden_size * 2  # 临时缓冲区
            )
            moe_ops.append(expert_sync)
            
            # 8. RMSNorm层归一化 (Qwen3特有)
            # 在decode模式下，只需要为一个token做归一化
            rms_norm_2 = OpProfile(
                op_type=OpType.ELEMENTWISE_LAYER_NORM,
                op_name=f"layer_{layer_idx}_rms_norm_2",
                flops=2 * batch_size * current_seq_len * hidden_size,  # RMSNorm计算
                memory_bytes=batch_size * current_seq_len * hidden_size * 2 * 2
            )
            moe_ops.append(rms_norm_2)
            
            moe_layer = LayerProfile(
                layer_idx=layer_idx,
                layer_type="moe_ffn",
                ops=moe_ops
            )
            layer_profiles.append(moe_layer)
        
        # === 添加LM Head ===
        # 在decode模式下，只需要为一个token计算词汇表投影
        lm_head_op = OpProfile(
            op_type=OpType.GEMM_LM_HEAD,
            op_name="lm_head",
            flops=2 * batch_size * current_seq_len * hidden_size * vocab_size,
            memory_bytes=(batch_size * current_seq_len * hidden_size + 
                         hidden_size * vocab_size + 
                         batch_size * current_seq_len * vocab_size) * 2,
            is_gemm=True,
            m=batch_size * current_seq_len,
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
    

    
    def get_performance_analysis(self) -> Dict[str, any]:
        """
        获取Qwen3-MoE模型性能分析报告
        
        Returns:
            包含FLOPS分解和性能指标的字典
        """
        total_flops = self.estimate_flops_per_token()
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