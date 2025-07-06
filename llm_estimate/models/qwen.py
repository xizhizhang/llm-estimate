"""
Qwen系列模型实现

包含Qwen-7B、Qwen-14B等模型的具体实现。
参考 Qwen 官方实现和优化指南修正 FLOPS 和内存计算。
"""

from typing import Dict, List
from .base import BaseModel, ModelSpecs, ModelConfig


class QwenModel(BaseModel):
    """Qwen系列模型实现"""
    
    def decompose_to_ops(self) -> List['LayerProfile']:
        """分解Qwen3系列模型为具体操作"""
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
        
        # 根据推理模式调整序列长度
        # 在decode模式下，只处理一个新token，但需要与cache中的所有token做注意力计算
        is_decode_mode = self.config.inference_mode == "decode" and self.config.use_kv_cache
        current_seq_len = 1 if is_decode_mode else seq_len
        attention_seq_len = seq_len  # 注意力计算时的序列长度（包括cache）
        
        # 为每一层生成操作分解
        for layer_idx in range(self.specs.layers):
            
            # === Attention层操作 ===
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
            
            # === FFN层操作 ===
            ffn_ops = []
            
            # 1. Gate投影
            gate_proj = OpProfile(
                op_type=OpType.GEMM_FFN_GATE,
                op_name=f"layer_{layer_idx}_ffn_gate",
                flops=2 * batch_size * current_seq_len * hidden_size * intermediate_size,
                memory_bytes=(batch_size * current_seq_len * hidden_size + 
                             hidden_size * intermediate_size + 
                             batch_size * current_seq_len * intermediate_size) * 2,
                is_gemm=True,
                m=batch_size * current_seq_len,
                n=intermediate_size,
                k=hidden_size
            )
            ffn_ops.append(gate_proj)
            
            # 2. Up投影
            up_proj = OpProfile(
                op_type=OpType.GEMM_FFN_UP,
                op_name=f"layer_{layer_idx}_ffn_up",
                flops=2 * batch_size * current_seq_len * hidden_size * intermediate_size,
                memory_bytes=(batch_size * current_seq_len * hidden_size + 
                             hidden_size * intermediate_size + 
                             batch_size * current_seq_len * intermediate_size) * 2,
                is_gemm=True,
                m=batch_size * current_seq_len,
                n=intermediate_size,
                k=hidden_size
            )
            ffn_ops.append(up_proj)
            
            # 3. SwiGLU激活
            activation_op = OpProfile(
                op_type=OpType.ELEMENTWISE_ACTIVATION,
                op_name=f"layer_{layer_idx}_swiglu",
                flops=batch_size * current_seq_len * intermediate_size,  # 简化估算
                memory_bytes=batch_size * current_seq_len * intermediate_size * 2 * 2  # gate * up
            )
            ffn_ops.append(activation_op)
            
            # 4. Down投影
            down_proj = OpProfile(
                op_type=OpType.GEMM_FFN_DOWN,
                op_name=f"layer_{layer_idx}_ffn_down",
                flops=2 * batch_size * current_seq_len * intermediate_size * hidden_size,
                memory_bytes=(batch_size * current_seq_len * intermediate_size + 
                             intermediate_size * hidden_size + 
                             batch_size * current_seq_len * hidden_size) * 2,
                is_gemm=True,
                m=batch_size * current_seq_len,
                n=hidden_size,
                k=intermediate_size
            )
            ffn_ops.append(down_proj)
            
            # 5. RMSNorm层归一化 (Qwen3特有)
            rms_norm_2 = OpProfile(
                op_type=OpType.ELEMENTWISE_LAYER_NORM,
                op_name=f"layer_{layer_idx}_rms_norm_2",
                flops=2 * batch_size * current_seq_len * hidden_size,  # RMSNorm计算
                memory_bytes=batch_size * current_seq_len * hidden_size * 2 * 2
            )
            ffn_ops.append(rms_norm_2)
            
            ffn_layer = LayerProfile(
                layer_idx=layer_idx,
                layer_type="ffn",
                ops=ffn_ops
            )
            layer_profiles.append(ffn_layer)
        
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
        计算Qwen模型内存使用量
        
        基于 Qwen 官方实现和 Transformers 库优化指南更新内存计算公式
        
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
        
        # 模型权重内存 (实际参数数量)
        model_memory = self.specs.parameters * 1e9 * bytes_per_param
        
        # KV缓存内存计算 (更精确的公式)
        # Qwen 也使用了类似的 Grouped Query Attention (GQA) 优化
        if self.config.use_kv_cache:
            # KV缓存大小 = 2 (K和V) * num_layers * num_kv_heads * head_dim * context_length * batch_size * precision
            # Qwen 支持更长的上下文，需要更精确的计算
            kv_cache_memory = (
                2 * self.specs.layers * self.specs.num_key_value_heads * 
                self.specs.head_dim * self.config.context_length * 
                self.config.batch_size * bytes_per_param
            )
        else:
            kv_cache_memory = 0
        
        # 激活值内存 (基于 Qwen transformer 架构的更精确估算)
        # 包括注意力分数、中间激活等
        sequence_length = self.config.context_length
        batch_size = self.config.batch_size
        hidden_size = self.specs.hidden_size
        
        # 主要激活值内存组件：
        # 1. 注意力分数矩阵: batch_size * num_heads * seq_len * seq_len
        # Qwen 支持更长上下文，注意力分数内存占用更大
        attention_scores_memory = (
            batch_size * self.specs.attention_heads * 
            sequence_length * sequence_length * bytes_per_param
        )
        
        # 2. 隐藏状态激活: batch_size * seq_len * hidden_size * num_layers
        # Qwen 的隐藏状态激活内存
        hidden_activations_memory = (
            batch_size * sequence_length * hidden_size * 
            self.specs.layers * bytes_per_param
        )
        
        # 3. FFN 中间激活 (使用实际的 intermediate_size)
        # Qwen 使用 SwiGLU 激活函数，需要额外的门控机制
        ffn_activations_memory = (
            batch_size * sequence_length * self.specs.intermediate_size * 
            self.specs.layers * bytes_per_param
        )
        
        # 4. RoPE (旋转位置编码) 相关激活
        # Qwen 使用 RoPE，需要额外的位置编码计算内存
        rope_activations_memory = (
            batch_size * sequence_length * self.specs.attention_heads * 
            self.specs.head_dim * self.specs.layers * bytes_per_param * 0.1  # 估计为10%开销
        )
        
        total_activation_memory = (
            attention_scores_memory + hidden_activations_memory + 
            ffn_activations_memory + rope_activations_memory
        )
        
        # 梯度内存 (训练时需要，与模型权重相同)
        gradient_memory = model_memory
        
        # 优化器状态内存 (训练时需要，Adam优化器需要存储momentum和variance)
        optimizer_memory = model_memory * 2  # Adam需要2份额外状态
        
        # 总内存计算
        total_inference = model_memory + kv_cache_memory + total_activation_memory
        total_training = total_inference + gradient_memory + optimizer_memory
        
        return {
            "model_memory_gb": model_memory / (1024**3),
            "kv_cache_memory_gb": kv_cache_memory / (1024**3),
            "attention_scores_memory_gb": attention_scores_memory / (1024**3),
            "hidden_activations_memory_gb": hidden_activations_memory / (1024**3),
            "ffn_activations_memory_gb": ffn_activations_memory / (1024**3),
            "rope_activations_memory_gb": rope_activations_memory / (1024**3),
            "total_activation_memory_gb": total_activation_memory / (1024**3),
            "gradient_memory_gb": gradient_memory / (1024**3),
            "optimizer_memory_gb": optimizer_memory / (1024**3),
            "total_inference_gb": total_inference / (1024**3),
            "total_training_gb": total_training / (1024**3),
            "precision": precision
        }
    