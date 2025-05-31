# 模型测试

这个目录包含所有模型相关的单元测试。

## 测试文件

### test_llama_corrections.py

测试修正后的 Llama 模型 FLOPS 和内存计算，基于 meta-llama 官方实现进行修正。

**主要测试内容：**

1. **模型创建测试** - 验证所有 Llama 模型变体能正确创建
2. **FLOPS 计算测试** - 验证详细和简化 FLOPS 计算的正确性
3. **内存计算测试** - 验证各种内存组件的计算准确性
4. **GQA 优化测试** - 验证 Grouped Query Attention 的内存节省效果
5. **上下文长度缩放测试** - 验证 KV 缓存随上下文长度的线性增长
6. **精度缩放测试** - 验证不同精度对内存使用的影响
7. **性能分析测试** - 验证完整性能分析报告的正确性

**运行方式：**

```bash
# 直接运行（包含交互式测试和详细输出）
python3 tests/test_models/test_llama_corrections.py

# 使用 pytest 运行（需要先安装 pytest）
pip install pytest
pytest tests/test_models/test_llama_corrections.py -v
```

**测试覆盖的模型：**

- Llama-2 系列: 7B (Multi-Head Attention)
- Llama-3.1 系列: 8B (Grouped Query Attention)

**验证的关键指标：**

- **FLOPS 分布**: 注意力机制 vs FFN 的计算量分布
- **内存优化**: GQA 相对于 MHA 的内存节省（可达 75%）
- **长上下文支持**: Llama-3.1 的 128K 上下文长度处理
- **缩放规律**: 验证 OpenAI scaling laws 的准确性

**测试结果示例：**

```
🔍 测试模型: llama-3.1-8b
📊 模型信息:
   参数量: 8B
   隐藏维度: 4096
   注意力头数: 32
   KV头数: 8 (GQA比率: 4.0)

⚡ FLOPS分析 (每token):
   总FLOPS: 18,795,208,704
   注意力FLOPS: 28.7%
   FFN FLOPS: 68.6%

💾 内存使用:
   推理总内存: 21.40 GB
   KV缓存开销: 2.3%
``` 