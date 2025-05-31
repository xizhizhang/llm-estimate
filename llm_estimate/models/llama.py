"""
Llama系列模型实现

包含Llama-2、Code Llama等模型的具体实现。
"""

from typing import Dict
from .base import BaseModel, ModelSpecs, ModelConfig


class LlamaModel(BaseModel):
    """Llama系列模型实现"""
    
    def calculate_memory_usage(self, precision: str = "fp16") -> Dict[str, float]:
        """
        计算Llama模型内存使用量
        
        Args:
            precision: 精度类型 ("fp32", "fp16", "int8", "int4")
            
        Returns:
            包含各种内存使用量的字典
        """
        # 精度对应的字节数
        precision_bytes = {
            "fp32": 4,
            "fp16": 2,
            "int8": 1,
            "int4": 0.5
        }
        
        if precision not in precision_bytes:
            raise ValueError(f"Unsupported precision: {precision}")
        
        bytes_per_param = precision_bytes[precision]
        
        # 模型权重内存
        model_memory = self.specs.parameters * 1e9 * bytes_per_param
        
        # KV缓存内存 (假设使用KV缓存)
        if self.config.use_kv_cache:
            # KV缓存大小 = 2 * layers * hidden_size * context_length * batch_size * precision
            kv_cache_memory = (
                2 * self.specs.layers * self.specs.hidden_size * 
                self.config.context_length * self.config.batch_size * 
                bytes_per_param
            )
        else:
            kv_cache_memory = 0
        
        # 激活值内存 (近似估算)
        activation_memory = (
            self.config.batch_size * self.config.context_length * 
            self.specs.hidden_size * 16  # 经验值
        )
        
        # 梯度内存 (训练时需要)
        gradient_memory = model_memory  # 与模型权重相同
        
        # 优化器状态内存 (训练时需要，Adam优化器)
        optimizer_memory = model_memory * 2  # Adam需要存储momentum和variance
        
        total_inference = model_memory + kv_cache_memory + activation_memory
        total_training = total_inference + gradient_memory + optimizer_memory
        
        return {
            "model_memory_gb": model_memory / (1024**3),
            "kv_cache_memory_gb": kv_cache_memory / (1024**3),
            "activation_memory_gb": activation_memory / (1024**3),
            "gradient_memory_gb": gradient_memory / (1024**3),
            "optimizer_memory_gb": optimizer_memory / (1024**3),
            "total_inference_gb": total_inference / (1024**3),
            "total_training_gb": total_training / (1024**3),
            "precision": precision
        }
    
    def estimate_flops_per_token(self) -> float:
        """
        估算Llama模型每个token的计算量（FLOPS）
        
        Returns:
            每个token的FLOPS数
        """
        # Transformer模型的FLOPS估算公式
        # 主要包括：注意力计算 + FFN计算
        
        hidden_size = self.specs.hidden_size
        layers = self.specs.layers
        vocab_size = self.specs.vocab_size
        context_length = self.config.context_length
        
        # 每层的FLOPS计算
        # 1. 注意力机制：Q, K, V投影 + 注意力计算 + 输出投影
        attention_flops = (
            # QKV投影
            3 * hidden_size * hidden_size +
            # 注意力计算 (Q @ K^T)
            hidden_size * context_length +
            # 注意力 @ V
            hidden_size * context_length +
            # 输出投影
            hidden_size * hidden_size
        )
        
        # 2. FFN (通常是4倍hidden_size)
        ffn_hidden = hidden_size * 4
        ffn_flops = (
            # 第一个线性层
            hidden_size * ffn_hidden +
            # 第二个线性层
            ffn_hidden * hidden_size
        )
        
        # 每层总FLOPS
        layer_flops = attention_flops + ffn_flops
        
        # 所有层的FLOPS
        total_layer_flops = layers * layer_flops
        
        # 词嵌入和输出投影
        embedding_flops = hidden_size * vocab_size
        
        # 总FLOPS
        total_flops = total_layer_flops + embedding_flops
        
        return total_flops 