"""
Llama系列模型实现

包含Llama-2、Code Llama等模型的具体实现。
参考 meta-llama 官方实现修正 FLOPS 和内存计算。
"""

from typing import Dict
from .base import BaseModel, ModelSpecs, ModelConfig


class LlamaModel(BaseModel):
    """Llama系列模型实现"""
    
    def calculate_memory_usage(self, precision: str = "fp16") -> Dict[str, float]:
        """
        计算Llama模型内存使用量
        
        基于 meta-llama 官方实现和 Apple Core ML 优化指南更新内存计算公式
        
        Args:
            precision: 精度类型 ("fp32", "fp16", "int8", "int4")
            
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
        # 参考 Apple Core ML 和 Hugging Face 实现
        if self.config.use_kv_cache:
            # KV缓存大小 = 2 (K和V) * num_layers * num_kv_heads * head_dim * context_length * batch_size * precision
            # Llama 使用 Grouped Query Attention (GQA)
            kv_cache_memory = (
                2 * self.specs.layers * self.specs.num_key_value_heads * 
                self.specs.head_dim * self.config.context_length * 
                self.config.batch_size * bytes_per_param
            )
        else:
            kv_cache_memory = 0
        
        # 激活值内存 (基于 transformer 架构的更精确估算)
        # 包括注意力分数、中间激活等
        sequence_length = self.config.context_length
        batch_size = self.config.batch_size
        hidden_size = self.specs.hidden_size
        
        # 主要激活值内存组件：
        # 1. 注意力分数矩阵: batch_size * num_heads * seq_len * seq_len
        attention_scores_memory = (
            batch_size * self.specs.attention_heads * 
            sequence_length * sequence_length * bytes_per_param
        )
        
        # 2. 隐藏状态激活: batch_size * seq_len * hidden_size * num_layers
        hidden_activations_memory = (
            batch_size * sequence_length * hidden_size * 
            self.specs.layers * bytes_per_param
        )
        
        # 3. FFN 中间激活 (使用实际的 intermediate_size)
        ffn_activations_memory = (
            batch_size * sequence_length * self.specs.intermediate_size * 
            self.specs.layers * bytes_per_param
        )
        
        total_activation_memory = (
            attention_scores_memory + hidden_activations_memory + 
            ffn_activations_memory
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
            "total_activation_memory_gb": total_activation_memory / (1024**3),
            "gradient_memory_gb": gradient_memory / (1024**3),
            "optimizer_memory_gb": optimizer_memory / (1024**3),
            "total_inference_gb": total_inference / (1024**3),
            "total_training_gb": total_training / (1024**3),
            "precision": precision
        }
    
    def estimate_flops_per_token(self) -> float:
        """
        估算Llama模型每个token的计算量（FLOPS）
        
        基于 OpenAI scaling laws 和 meta-llama 官方实现的更精确公式
        参考文献：
        - OpenAI Scaling Laws (Kaplan et al., 2020)
        - Apple Core ML on-device Llama optimization
        - Chinchilla scaling laws (Hoffmann et al., 2022)
        
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
        
        # === 每层的FLOPS计算 ===
        
        # 1. 注意力机制 FLOPS
        # 1.1 QKV投影 (考虑GQA)
        # Q投影: hidden_size * hidden_size
        q_projection_flops = hidden_size * hidden_size
        
        # K,V投影: hidden_size * (num_kv_heads * head_dim) (GQA优化)
        kv_projection_flops = 2 * hidden_size * (num_kv_heads * head_dim)
        
        # 1.2 注意力计算
        # Q @ K^T: num_heads * head_dim * context_length  
        attention_scores_flops = num_heads * head_dim * context_length
        
        # Softmax: 3 * num_heads * context_length (exp + sum + div)
        softmax_flops = 3 * num_heads * context_length
        
        # Attention @ V: num_heads * context_length * head_dim
        attention_output_flops = num_heads * context_length * head_dim
        
        # 1.3 输出投影
        output_projection_flops = hidden_size * hidden_size
        
        # 总注意力FLOPS
        attention_flops = (
            q_projection_flops + kv_projection_flops + 
            attention_scores_flops + softmax_flops + 
            attention_output_flops + output_projection_flops
        )
        
        # 2. Feed-Forward Network (FFN) FLOPS
        # 标准Llama使用SwiGLU激活，包含gate、up、down三个投影
        # Gate投影: hidden_size * intermediate_size
        gate_projection_flops = hidden_size * self.specs.intermediate_size
        
        # Up投影: hidden_size * intermediate_size 
        up_projection_flops = hidden_size * self.specs.intermediate_size
        
        # SwiGLU激活: gate(x) * swish(up(x))，近似为线性复杂度
        activation_flops = self.specs.intermediate_size  # 近似值
        
        # Down投影: intermediate_size * hidden_size
        down_projection_flops = self.specs.intermediate_size * hidden_size
        
        # 总FFN FLOPS
        ffn_flops = (
            gate_projection_flops + up_projection_flops + 
            activation_flops + down_projection_flops
        )
        
        # 3. 层归一化 (RMSNorm) FLOPS
        # 每层有2个RMSNorm，每个需要 ~3 * hidden_size 操作
        layer_norm_flops = 2 * 3 * hidden_size
        
        # 每层总FLOPS
        layer_flops = attention_flops + ffn_flops + layer_norm_flops
        
        # 所有层的FLOPS
        total_layer_flops = layers * layer_flops
        
        # 4. 词嵌入和输出投影 FLOPS
        # 输入嵌入: vocab_size * hidden_size (仅计算当前token)
        embedding_flops = hidden_size  # 单个token的嵌入查找，近似为常数
        
        # 输出投影 (LM head): hidden_size * vocab_size
        output_head_flops = hidden_size * vocab_size
        
        # 总FLOPS (每个token)
        total_flops = total_layer_flops + embedding_flops + output_head_flops
        
        # 乘以2以考虑乘法和加法操作 (multiply-accumulate)
        total_flops *= 2
        
        return total_flops
    
    def estimate_model_flops_per_token_simplified(self) -> float:
        """
        使用简化的 OpenAI scaling laws 公式估算 FLOPS
        
        基于 C_forward ≈ 2N 的公式，其中 N 是非嵌入参数数量
        
        Returns:
            每个token的FLOPS数 (简化估算)
        """
        # 非嵌入参数数量估算
        # 主要包括：attention layers + FFN layers + layer norms
        hidden_size = self.specs.hidden_size
        layers = self.specs.layers
        vocab_size = self.specs.vocab_size
        
        # Attention参数: Q,K,V,O投影
        attention_params = layers * (4 * hidden_size * hidden_size)
        
        # FFN参数: gate, up, down投影 (使用实际的 intermediate_size)
        ffn_params = layers * (3 * hidden_size * self.specs.intermediate_size)
        
        # Layer norm参数 (较少，可忽略)
        layer_norm_params = layers * 2 * hidden_size
        
        # 嵌入参数
        embedding_params = vocab_size * hidden_size
        
        # 非嵌入参数
        non_embedding_params = attention_params + ffn_params + layer_norm_params
        
        # OpenAI公式: forward pass ≈ 2N FLOPs
        forward_flops = 2 * non_embedding_params
        
        return forward_flops
    
    def get_performance_analysis(self) -> Dict[str, any]:
        """
        获取模型性能分析报告
        
        Returns:
            包含FLOPS分解和性能指标的字典
        """
        total_flops = self.estimate_flops_per_token()
        simplified_flops = self.estimate_model_flops_per_token_simplified()
        memory_usage = self.calculate_memory_usage(self.config.precision)
        
        # FLOPS分解分析
        hidden_size = self.specs.hidden_size
        layers = self.specs.layers
        
        # 按组件分解FLOPS
        attention_flops_per_layer = (
            4 * hidden_size * hidden_size +  # QKV + output projections
            self.specs.attention_heads * self.specs.head_dim * self.config.context_length +  # attention computation
            3 * self.specs.attention_heads * self.config.context_length  # softmax
        ) * 2  # multiply-accumulate
        
        ffn_flops_per_layer = (
            3 * hidden_size * self.specs.intermediate_size +  # gate, up, down projections
            self.specs.intermediate_size  # activation
        ) * 2  # multiply-accumulate
        
        total_attention_flops = layers * attention_flops_per_layer
        total_ffn_flops = layers * ffn_flops_per_layer
        
        return {
            "model_info": {
                "name": self.specs.name,
                "parameters": f"{self.specs.parameters}B",
                "layers": layers,
                "hidden_size": hidden_size,
                "attention_heads": self.specs.attention_heads,
                "kv_heads": self.specs.num_key_value_heads,
            },
            "flops_analysis": {
                "total_flops_per_token": total_flops,
                "simplified_flops_per_token": simplified_flops,
                "attention_flops_percentage": (total_attention_flops / total_flops) * 100,
                "ffn_flops_percentage": (total_ffn_flops / total_flops) * 100,
                "attention_flops_per_token": total_attention_flops,
                "ffn_flops_per_token": total_ffn_flops,
                "context_dependency": {
                    "quadratic_terms_flops": layers * self.specs.attention_heads * (self.config.context_length ** 2),
                    "linear_terms_flops": total_flops - layers * self.specs.attention_heads * (self.config.context_length ** 2),
                }
            },
            "memory_analysis": memory_usage,
            "efficiency_metrics": {
                "memory_per_parameter_bytes": (memory_usage["model_memory_gb"] * 1024**3) / (self.specs.parameters * 1e9),
                "flops_per_parameter": total_flops / (self.specs.parameters * 1e9),
                "kv_cache_overhead_percentage": (memory_usage["kv_cache_memory_gb"] / memory_usage["total_inference_gb"]) * 100 if memory_usage["total_inference_gb"] > 0 else 0,
            }
        } 