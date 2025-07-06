"""
DeepSeek-V3 model implementation

DeepSeek-V3 is a strong Mixture-of-Experts (MoE) language model with 671B total parameters
with 37B activated for each token. It adopts Multi-head Latent Attention (MLA) and 
DeepSeekMoE architectures for efficient inference and cost-effective training.

Features:
- 671B total parameters, 37B activated per token
- Hybrid architecture: 3 dense layers + 58 MoE layers
- 256 routed experts + 1 shared expert, 8 experts per token
- 128K context length with YaRN RoPE extension
- Multi-Token Prediction (MTP) support
- FP8 and BF16 precision support
"""

from typing import Dict, List, Optional
from .base import BaseModel, ModelSpecs, ModelConfig


class DeepSeekV3Model(BaseModel):
    """DeepSeek-V3 MoE model implementation"""
    
    def decompose_to_ops(self) -> List['LayerProfile']:
        """Decompose DeepSeek-V3 model into specific operations"""
        from ..estimator.op_level_estimator import LayerProfile, OpProfile, OpType
        
        layer_profiles = []
        
        batch_size = self.config.batch_size
        seq_len = self.config.context_length
        hidden_size = self.specs.hidden_size
        intermediate_size = self.specs.intermediate_size
        moe_intermediate_size = self.specs.moe_intermediate_size
        num_heads = self.specs.attention_heads
        num_kv_heads = self.specs.num_key_value_heads
        head_dim = self.specs.head_dim
        vocab_size = self.specs.vocab_size
        num_experts = self.specs.num_experts
        experts_per_token = self.specs.experts_per_token
        
        # Inference mode adjustment
        is_decode_mode = self.config.inference_mode == "decode" and self.config.use_kv_cache
        current_seq_len = 1 if is_decode_mode else seq_len
        attention_seq_len = seq_len
        
        # Precision bytes
        precision_bytes = {
            "fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5
        }.get(self.config.precision, 2)
        
        # Generate operation decomposition for each layer
        for layer_idx in range(self.specs.layers):
            
            # Determine if this is a dense layer or MoE layer
            # First 3 layers are dense, rest are MoE
            is_dense_layer = layer_idx < 3
            
            # === Attention Layer Operations ===
            attention_ops = []
            
            # 1. RMSNorm (pre-attention)
            rms_norm_attn = OpProfile(
                op_type=OpType.ELEMENTWISE_LAYER_NORM,
                op_name=f"layer_{layer_idx}_attention_rms_norm",
                flops=2 * batch_size * current_seq_len * hidden_size,
                memory_bytes=batch_size * current_seq_len * hidden_size * precision_bytes * 2
            )
            attention_ops.append(rms_norm_attn)
            
            # 2. Q/K/V projections with GQA support
            # Q projection
            q_proj = OpProfile(
                op_type=OpType.GEMM_QKV_PROJ,
                op_name=f"layer_{layer_idx}_q_proj",
                flops=2 * batch_size * current_seq_len * hidden_size * hidden_size,
                memory_bytes=(batch_size * current_seq_len * hidden_size + 
                             hidden_size * hidden_size + 
                             batch_size * current_seq_len * hidden_size) * precision_bytes,
                is_gemm=True,
                m=batch_size * current_seq_len,
                n=hidden_size,
                k=hidden_size
            )
            attention_ops.append(q_proj)
            
            # K and V projections (with GQA)
            kv_hidden = num_kv_heads * head_dim
            for proj_name in ["k_proj", "v_proj"]:
                kv_proj = OpProfile(
                    op_type=OpType.GEMM_QKV_PROJ,
                    op_name=f"layer_{layer_idx}_{proj_name}",
                    flops=2 * batch_size * current_seq_len * hidden_size * kv_hidden,
                    memory_bytes=(batch_size * current_seq_len * hidden_size + 
                                 hidden_size * kv_hidden + 
                                 batch_size * current_seq_len * kv_hidden) * precision_bytes,
                    is_gemm=True,
                    m=batch_size * current_seq_len,
                    n=kv_hidden,
                    k=hidden_size
                )
                attention_ops.append(kv_proj)
            
            # 3. YaRN RoPE (Rotary Position Embedding)
            rope_op = OpProfile(
                op_type=OpType.ELEMENTWISE_ROPE,
                op_name=f"layer_{layer_idx}_yarn_rope",
                flops=8 * batch_size * current_seq_len * hidden_size,  # YaRN has additional complexity
                memory_bytes=batch_size * current_seq_len * hidden_size * precision_bytes * 2
            )
            attention_ops.append(rope_op)
            
            # 4. Attention computation (Q@K^T)
            if is_decode_mode:
                attention_qk_flops = 2 * batch_size * num_heads * 1 * attention_seq_len * head_dim
                attention_qk_memory = (batch_size * num_heads * 1 * head_dim + 
                                      batch_size * num_heads * head_dim * attention_seq_len + 
                                      batch_size * num_heads * 1 * attention_seq_len) * precision_bytes
                qk_m, qk_n, qk_k = 1, attention_seq_len, head_dim
            else:
                attention_qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
                attention_qk_memory = (batch_size * num_heads * seq_len * head_dim * 2 + 
                                      batch_size * num_heads * seq_len * seq_len) * precision_bytes
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
            
            # 5. Softmax
            if is_decode_mode:
                softmax_flops = 3 * batch_size * num_heads * 1 * attention_seq_len
                softmax_memory = batch_size * num_heads * 1 * attention_seq_len * precision_bytes * 2
            else:
                softmax_flops = 3 * batch_size * num_heads * seq_len * seq_len
                softmax_memory = batch_size * num_heads * seq_len * seq_len * precision_bytes * 2
            
            softmax_op = OpProfile(
                op_type=OpType.ELEMENTWISE_SOFTMAX,
                op_name=f"layer_{layer_idx}_softmax",
                flops=softmax_flops,
                memory_bytes=softmax_memory
            )
            attention_ops.append(softmax_op)
            
            # 6. Attention output (Attention@V)
            if is_decode_mode:
                attention_v_flops = 2 * batch_size * num_heads * 1 * attention_seq_len * head_dim
                attention_v_memory = (batch_size * num_heads * 1 * attention_seq_len + 
                                     batch_size * num_heads * attention_seq_len * head_dim + 
                                     batch_size * num_heads * 1 * head_dim) * precision_bytes
                v_m, v_n, v_k = 1, head_dim, attention_seq_len
            else:
                attention_v_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
                attention_v_memory = (batch_size * num_heads * seq_len * seq_len + 
                                     batch_size * num_heads * seq_len * head_dim + 
                                     batch_size * num_heads * seq_len * head_dim) * precision_bytes
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
            
            # 7. Output projection
            output_proj = OpProfile(
                op_type=OpType.GEMM_OUTPUT_PROJ,
                op_name=f"layer_{layer_idx}_output_proj",
                flops=2 * batch_size * current_seq_len * hidden_size * hidden_size,
                memory_bytes=(batch_size * current_seq_len * hidden_size + 
                             hidden_size * hidden_size + 
                             batch_size * current_seq_len * hidden_size) * precision_bytes,
                is_gemm=True,
                m=batch_size * current_seq_len,
                n=hidden_size,
                k=hidden_size
            )
            attention_ops.append(output_proj)
            
            attention_layer = LayerProfile(
                layer_idx=layer_idx,
                layer_type="attention",
                ops=attention_ops
            )
            layer_profiles.append(attention_layer)
            
            # === FFN Layer Operations ===
            ffn_ops = []
            
            # RMSNorm (pre-FFN)
            rms_norm_ffn = OpProfile(
                op_type=OpType.ELEMENTWISE_LAYER_NORM,
                op_name=f"layer_{layer_idx}_ffn_rms_norm",
                flops=2 * batch_size * current_seq_len * hidden_size,
                memory_bytes=batch_size * current_seq_len * hidden_size * precision_bytes * 2
            )
            ffn_ops.append(rms_norm_ffn)
            
            if is_dense_layer:
                # Dense FFN layer
                # Gate projection
                gate_proj = OpProfile(
                    op_type=OpType.GEMM_FFN_GATE,
                    op_name=f"layer_{layer_idx}_gate_proj",
                    flops=2 * batch_size * current_seq_len * hidden_size * intermediate_size,
                    memory_bytes=(batch_size * current_seq_len * hidden_size + 
                                 hidden_size * intermediate_size + 
                                 batch_size * current_seq_len * intermediate_size) * precision_bytes,
                    is_gemm=True,
                    m=batch_size * current_seq_len,
                    n=intermediate_size,
                    k=hidden_size
                )
                ffn_ops.append(gate_proj)
                
                # Up projection
                up_proj = OpProfile(
                    op_type=OpType.GEMM_FFN_UP,
                    op_name=f"layer_{layer_idx}_up_proj",
                    flops=2 * batch_size * current_seq_len * hidden_size * intermediate_size,
                    memory_bytes=(batch_size * current_seq_len * hidden_size + 
                                 hidden_size * intermediate_size + 
                                 batch_size * current_seq_len * intermediate_size) * precision_bytes,
                    is_gemm=True,
                    m=batch_size * current_seq_len,
                    n=intermediate_size,
                    k=hidden_size
                )
                ffn_ops.append(up_proj)
                
                # SiLU activation
                silu_op = OpProfile(
                    op_type=OpType.ELEMENTWISE_ACTIVATION,
                    op_name=f"layer_{layer_idx}_silu",
                    flops=3 * batch_size * current_seq_len * intermediate_size,
                    memory_bytes=batch_size * current_seq_len * intermediate_size * precision_bytes * 2
                )
                ffn_ops.append(silu_op)
                
                # Down projection
                down_proj = OpProfile(
                    op_type=OpType.GEMM_FFN_DOWN,
                    op_name=f"layer_{layer_idx}_down_proj",
                    flops=2 * batch_size * current_seq_len * intermediate_size * hidden_size,
                    memory_bytes=(batch_size * current_seq_len * intermediate_size + 
                                 intermediate_size * hidden_size + 
                                 batch_size * current_seq_len * hidden_size) * precision_bytes,
                    is_gemm=True,
                    m=batch_size * current_seq_len,
                    n=hidden_size,
                    k=intermediate_size
                )
                ffn_ops.append(down_proj)
                
            else:
                # MoE FFN layer
                # Router (gating)
                router_op = OpProfile(
                    op_type=OpType.GEMM_ROUTER,
                    op_name=f"layer_{layer_idx}_moe_router",
                    flops=2 * batch_size * current_seq_len * hidden_size * num_experts,
                    memory_bytes=(batch_size * current_seq_len * hidden_size + 
                                 hidden_size * num_experts + 
                                 batch_size * current_seq_len * num_experts) * precision_bytes,
                    is_gemm=True,
                    m=batch_size * current_seq_len,
                    n=num_experts,
                    k=hidden_size
                )
                ffn_ops.append(router_op)
                
                # Top-k selection
                topk_op = OpProfile(
                    op_type=OpType.ELEMENTWISE_ACTIVATION,
                    op_name=f"layer_{layer_idx}_moe_topk",
                    flops=batch_size * current_seq_len * num_experts * 2,  # top-k selection
                    memory_bytes=batch_size * current_seq_len * num_experts * precision_bytes
                )
                ffn_ops.append(topk_op)
                
                # Shared expert (always activated)
                shared_gate = OpProfile(
                    op_type=OpType.GEMM_FFN_GATE,
                    op_name=f"layer_{layer_idx}_shared_gate",
                    flops=2 * batch_size * current_seq_len * hidden_size * moe_intermediate_size,
                    memory_bytes=(batch_size * current_seq_len * hidden_size + 
                                 hidden_size * moe_intermediate_size + 
                                 batch_size * current_seq_len * moe_intermediate_size) * precision_bytes,
                    is_gemm=True,
                    m=batch_size * current_seq_len,
                    n=moe_intermediate_size,
                    k=hidden_size
                )
                ffn_ops.append(shared_gate)
                
                shared_up = OpProfile(
                    op_type=OpType.GEMM_FFN_UP,
                    op_name=f"layer_{layer_idx}_shared_up",
                    flops=2 * batch_size * current_seq_len * hidden_size * moe_intermediate_size,
                    memory_bytes=(batch_size * current_seq_len * hidden_size + 
                                 hidden_size * moe_intermediate_size + 
                                 batch_size * current_seq_len * moe_intermediate_size) * precision_bytes,
                    is_gemm=True,
                    m=batch_size * current_seq_len,
                    n=moe_intermediate_size,
                    k=hidden_size
                )
                ffn_ops.append(shared_up)
                
                shared_down = OpProfile(
                    op_type=OpType.GEMM_FFN_DOWN,
                    op_name=f"layer_{layer_idx}_shared_down",
                    flops=2 * batch_size * current_seq_len * moe_intermediate_size * hidden_size,
                    memory_bytes=(batch_size * current_seq_len * moe_intermediate_size + 
                                 moe_intermediate_size * hidden_size + 
                                 batch_size * current_seq_len * hidden_size) * precision_bytes,
                    is_gemm=True,
                    m=batch_size * current_seq_len,
                    n=hidden_size,
                    k=moe_intermediate_size
                )
                ffn_ops.append(shared_down)
                
                # Routed experts (only activated experts)
                # Each token activates 8 experts
                expert_tokens = batch_size * current_seq_len * experts_per_token
                expert_gate = OpProfile(
                    op_type=OpType.GEMM_FFN_GATE,
                    op_name=f"layer_{layer_idx}_experts_gate",
                    flops=2 * expert_tokens * hidden_size * moe_intermediate_size,
                    memory_bytes=(expert_tokens * hidden_size + 
                                 experts_per_token * hidden_size * moe_intermediate_size + 
                                 expert_tokens * moe_intermediate_size) * precision_bytes,
                    is_gemm=True,
                    m=expert_tokens,
                    n=moe_intermediate_size,
                    k=hidden_size
                )
                ffn_ops.append(expert_gate)
                
                expert_up = OpProfile(
                    op_type=OpType.GEMM_FFN_UP,
                    op_name=f"layer_{layer_idx}_experts_up",
                    flops=2 * expert_tokens * hidden_size * moe_intermediate_size,
                    memory_bytes=(expert_tokens * hidden_size + 
                                 experts_per_token * hidden_size * moe_intermediate_size + 
                                 expert_tokens * moe_intermediate_size) * precision_bytes,
                    is_gemm=True,
                    m=expert_tokens,
                    n=moe_intermediate_size,
                    k=hidden_size
                )
                ffn_ops.append(expert_up)
                
                expert_down = OpProfile(
                    op_type=OpType.GEMM_FFN_DOWN,
                    op_name=f"layer_{layer_idx}_experts_down",
                    flops=2 * expert_tokens * moe_intermediate_size * hidden_size,
                    memory_bytes=(expert_tokens * moe_intermediate_size + 
                                 experts_per_token * moe_intermediate_size * hidden_size + 
                                 expert_tokens * hidden_size) * precision_bytes,
                    is_gemm=True,
                    m=expert_tokens,
                    n=hidden_size,
                    k=moe_intermediate_size
                )
                ffn_ops.append(expert_down)
                
                # SiLU activations for MoE
                silu_shared = OpProfile(
                    op_type=OpType.ELEMENTWISE_ACTIVATION,
                    op_name=f"layer_{layer_idx}_shared_silu",
                    flops=3 * batch_size * current_seq_len * moe_intermediate_size,
                    memory_bytes=batch_size * current_seq_len * moe_intermediate_size * precision_bytes * 2
                )
                ffn_ops.append(silu_shared)
                
                silu_experts = OpProfile(
                    op_type=OpType.ELEMENTWISE_ACTIVATION,
                    op_name=f"layer_{layer_idx}_experts_silu",
                    flops=3 * expert_tokens * moe_intermediate_size,
                    memory_bytes=expert_tokens * moe_intermediate_size * precision_bytes * 2
                )
                ffn_ops.append(silu_experts)
            
            ffn_layer = LayerProfile(
                layer_idx=layer_idx,
                layer_type="ffn_moe" if not is_dense_layer else "ffn_dense",
                ops=ffn_ops
            )
            layer_profiles.append(ffn_layer)
        
        return layer_profiles
    
    def calculate_memory_usage(self, precision: str = "fp16") -> Dict[str, float]:
        """Calculate memory usage for DeepSeek-V3 model"""
        precision_bytes = {
            "fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5
        }.get(precision, 2)
        
        # Model parameters
        total_params = self.specs.parameters * 1e9  # Convert to actual parameter count
        model_weights = total_params * precision_bytes
        
        # Activation memory
        batch_size = self.config.batch_size
        seq_len = self.config.context_length
        hidden_size = self.specs.hidden_size
        num_layers = self.specs.layers
        
        # Activation memory per layer
        activation_per_layer = (
            batch_size * seq_len * hidden_size * precision_bytes * 4  # Q, K, V, O
            + batch_size * seq_len * self.specs.intermediate_size * precision_bytes * 2  # FFN activations
        )
        
        # Total activation memory
        activation_memory = activation_per_layer * num_layers
        
        # KV Cache memory
        if self.config.use_kv_cache:
            kv_cache_memory = (
                2 * batch_size * seq_len * num_layers * 
                self.specs.num_key_value_heads * self.specs.head_dim * precision_bytes
            )
        else:
            kv_cache_memory = 0
        
        # Optimizer states (for training)
        optimizer_memory = model_weights * 2  # Assuming Adam optimizer
        
        # Gradient memory
        gradient_memory = model_weights
        
        total_memory = model_weights + activation_memory + kv_cache_memory
        
        return {
            "model_weights": model_weights / (1024**3),  # GB
            "activation_memory": activation_memory / (1024**3),  # GB
            "kv_cache_memory": kv_cache_memory / (1024**3),  # GB
            "optimizer_memory": optimizer_memory / (1024**3),  # GB
            "gradient_memory": gradient_memory / (1024**3),  # GB
            "total_inference_memory": total_memory / (1024**3),  # GB
            "total_training_memory": (total_memory + optimizer_memory + gradient_memory) / (1024**3),  # GB
        }

