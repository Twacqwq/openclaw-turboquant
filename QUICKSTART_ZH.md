# 🚀 TurboQuant 快速开始指南（中文）

## 5分钟入门

### 1️⃣ 安装

```bash
# 克隆项目
git clone https://github.com/openclaw/openclaw-turboquant.git
cd openclaw-turboquant

# 安装依赖
uv sync

# 验证安装
uv run pytest tests/ -v  # 应该看到 35 tests passed ✓
```

### 2️⃣ 基本使用

```python
import numpy as np
from openclaw_turboquant import TurboQuantMSE

# 创建量化器
q = TurboQuantMSE(d=128, bit_width=4, seed=42)

# 生成测试数据
x = np.random.randn(128)
print(f"原始向量大小: {x.nbytes} 字节")

# 量化
compressed = q.quantize(x)
print(f"量化后: {compressed}")

# 反量化
reconstructed = q.dequantize(compressed)
print(f"重建向量大小: {reconstructed.nbytes} 字节")
print(f"重建误差 (MSE): {np.mean((x - reconstructed)**2):.6f}")
```

### 3️⃣ 相似度搜索

```python
from openclaw_turboquant import ContextStore

# 创建存储
store = ContextStore(d=128, bit_width=4)

# 添加文档
docs = [
    ("doc1", np.random.randn(128), "这是第一个文档"),
    ("doc2", np.random.randn(128), "这是第二个文档"),
    ("doc3", np.random.randn(128), "这是第三个文档"),
]

for doc_id, embedding, text in docs:
    store.ingest(doc_id, embedding, text)

# 搜索
query = np.random.randn(128)
results = store.retrieve_top_k(query, k=2)

for entry, score in results:
    print(f"{entry.entry_id}: {score:.4f} - {entry.text}")
```

### 4️⃣ 令牌感知的上下文组装

```python
# 在令牌预算内选择最相关的文档
context = store.assemble_context(query, token_budget=1000)

# context 是一个消息列表，可直接送给AI模型
for msg in context:
    print(f"Role: {msg['role']}")
    print(f"Content: {msg['content'][:50]}...")
    print(f"Relevance: {msg['metadata']['relevance_score']:.4f}\n")
```

### 5️⃣ 命令行使用

```bash
# 运行基准测试
uv run openclaw-turboquant benchmark --dim 128 --bits 4 --n-vectors 100

# 压缩向量文件
uv run openclaw-turboquant compress --input vectors.npy --output vectors.npz --bits 4

# 检索相似向量
uv run openclaw-turboquant retrieve --store vectors.npz --query query.npy --top-k 5
```

---

## 常用参数详解

### TurboQuantMSE / TurboQuantProd

```python
q = TurboQuantMSE(
    d=128,              # 向量维度 [必需]
    bit_width=4,        # 每坐标比特数 [1-8, 默认:4]
    seed=42,            # 随机种子 [可选, 默认:None]
    codebook_max_iter=200  # Lloyd-Max迭代次数 [默认:200]
)
```

| 参数 | 范围 | 默认 | 说明 |
|------|------|------|------|
| `d` | 1-∞ | 必需 | 向量维度 |
| `bit_width` | 1-8 | 4 | 压缩级别（越小越压缩） |
| `seed` | 0-2³² | None | 设置以获得可重复结果 |

### ContextStore

```python
store = ContextStore(
    d=128,           # 嵌入维度 [必需]
    bit_width=4,     # 量化比特数 [默认:4]
    seed=42          # 随机种子 [可选]
)

# 主要方法
store.ingest(
    entry_id="doc1",
    embedding=vec,   # numpy数组
    text="原始文本",
    metadata={"source": "data.md"}
)

store.retrieve_top_k(query_embedding, k=5)
# 返回: [(entry1, score1), (entry2, score2), ...]

store.assemble_context(
    query_embedding,
    token_budget=4096,      # 令牌限制
    avg_chars_per_token=4.0 # 字符估计
)
# 返回: [{"role": "context", "content": "...", "metadata": {...}}, ...]

store.compact(keep_ratio=0.5, query_embedding=query)
# 删除低相关性条目，返回删除数

store.memory_estimate_bytes()
# 返回: 压缩存储的字节数

store.size  # 属性: 存储条目数
```

---

## 🎯 两种量化模式对比

| 特性 | MSE模式 | Product模式 |
|------|--------|-----------|
| **用途** | 向量重建 | 内积估计 |
| **优化目标** | 最小化MSE | 无偏内积估计 |
| **比特使用** | b位完全用于重建 | (b-1)位MSE + 1位QJL |
| **检索方式** | 需要反量化后比较 | 直接内积估计 |
| **速度** | ⚡⚡⚡ | ⚡⚡ |
| **精度** | 中等 | 高（内积） |
| **适用** | 存储压缩 | 相似度搜索 |

### 何时使用MSE模式？
- 需要重建原始向量
- 追求最大压缩率
- 存储为主要约束

### 何时使用Product模式？
- 主要做相似度搜索
- 需要无偏的内积估计
- 内积精度重要（OpenClaw首选）

---

## 📊 性能参考

### 单向量操作 (d=64)

| 操作 | 时间 | 吞吐量 |
|------|------|--------|
| MSE量化 | 4.6 µs | 217K vec/s |
| MSE反量化 | 1.2 µs | 833K vec/s |
| Product内积 | 4.1 µs | 244K vec/s |
| 存储摄取 | 12 µs | 83K ops/s |

### 批量操作 (d=64, n=100)

| 操作 | 时间 |
|------|------|
| 批量量化 | 473 µs |
| 批量反量化 | ~120 µs |
| 存储检索 | 406 µs |

### 压缩效率

```
原始: d × 32位 (float32)
MSE: d × b位 + 64位 (norm)

例如 (d=1024, b=4):
  原始: 32KB
  压缩: 5.1KB
  比率: 6.3倍 ✓
```

---

## 🧪 测试与验证

```bash
# 运行所有测试
uv run pytest

# 只运行特定模块的测试
uv run pytest tests/test_turboquant.py::TestTurboQuantMSE -v

# 运行基准测试
uv run pytest benchmarks/ --benchmark-only

# 生成覆盖率报告
uv run pytest --cov=src --cov-report=html
```

---

## 🔍 调试技巧

### 检查量化质量

```python
from openclaw_turboquant import TurboQuantMSE
import numpy as np

q = TurboQuantMSE(d=128, bit_width=4, seed=42)
x = np.random.randn(128)

# 量化和反量化
cx = q.quantize(x)
rx = q.dequantize(cx)

# 计算重建误差
mse = np.mean((x - rx) ** 2)
print(f"MSE: {mse:.6f}")

# 计算相关性
correlation = np.corrcoef(x, rx)[0, 1]
print(f"相关性: {correlation:.4f}")

# 检查范数保留
original_norm = np.linalg.norm(x)
reconstructed_norm = np.linalg.norm(rx)
print(f"范数比率: {reconstructed_norm / original_norm:.4f}")
```

### 验证内积估计准确性

```python
from openclaw_turboquant import TurboQuantProd
import numpy as np

q = TurboQuantProd(d=128, bit_width=4, seed=42)
x = np.random.randn(128)
y = np.random.randn(128)

# 精确内积
true_ip = x @ y

# 量化后的内积估计
qx = q.quantize(x)
estimated_ip = q.estimate_inner_product(y, qx)

# 计算误差
error = abs(true_ip - estimated_ip)
relative_error = error / abs(true_ip)
print(f"精确内积: {true_ip:.4f}")
print(f"估计内积: {estimated_ip:.4f}")
print(f"相对误差: {relative_error:.4f} ({relative_error*100:.2f}%)")
```

---

## ⚠️ 常见错误

### 错误1: "No module named 'openclaw_turboquant'"

```bash
# ❌ 问题：未正确安装
python -c "from openclaw_turboquant import TurboQuantMSE"

# ✅ 解决方案：使用uv运行
uv run python -c "from openclaw_turboquant import TurboQuantMSE"

# 或者安装到当前环境
uv pip install -e .
```

### 错误2: "got an unexpected keyword argument 'dim'"

```python
# ❌ 错误的参数名
q = TurboQuantMSE(dim=128, bit_width=4)

# ✅ 正确的参数名
q = TurboQuantMSE(d=128, bit_width=4)
```

### 错误3: "bit_width must be between 1 and 8"

```python
# ❌ 不支持的比特宽度
q = TurboQuantMSE(d=128, bit_width=16)

# ✅ 使用有效范围
q = TurboQuantMSE(d=128, bit_width=4)  # 1-8都可以
```

### 错误4: "token_budget exceeded"

```python
# ❌ 令牌预算过小
context = store.assemble_context(query, token_budget=10)

# ✅ 合理设置令牌预算
context = store.assemble_context(query, token_budget=4096)
```

---

## 📚 更多资源

- **完整API文档**: 见 `README.zh-CN.md`
- **论文**: https://arxiv.org/abs/2504.19874
- **GitHub**: https://github.com/openclaw/openclaw-turboquant
- **问题追踪**: https://github.com/openclaw/openclaw-turboquant/issues

---

## 💡 提示

1. **生产环境**：始终设置 `seed` 参数以确保可重复性
2. **性能调优**：使用 `quantize_batch()` 而不是循环 `quantize()`
3. **内存优化**：定期调用 `store.compact()` 来清理低相关性条目
4. **精度要求**：bit_width=6-8 用于高精度需求
5. **极限压缩**：bit_width=1-2 用于超极端压缩场景

---

**最后更新**: 2026年3月28日
