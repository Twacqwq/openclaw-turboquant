# openclaw-turboquant

[English](README.md) | **简体中文**

基于Google Research的**TurboQuant**算法，为[OpenClaw](https://openclaw.dev)上下文压缩提供近最优在线向量量化方案。论文：[arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

## 状态

| 组件 | 状态 | 说明 |
|------|------|------|
| 库 API | ✅ 可用 | 核心量化算法已完整实现 |
| CLI | ✅ 可用 | `benchmark`、`compress`、`retrieve` 命令可用 |
| Agent Skill | ✅ 可用 | CLI 命令可由 Agent 独立调用 |
| Context Engine 插件 | 🚧 开发中 | 接口已定义，核心集成逻辑尚未实现 |

## 概览

TurboQuant 通过简单的两阶段管道实现接近信息论下界2.7倍的近最优失真：

1. **随机旋转** — 应用随机正交矩阵（通过QR分解生成的Haar测度）将信息均匀分散到各坐标
2. **标量量化** — 使用Lloyd-Max编码本对每个旋转后的坐标进行独立量化，编码本针对单位超球面上坐标的Beta分布优化

提供两种量化模式：

| 模式 | 使用场景 | 说明 |
|------|----------|------|
| **MSE** | 向量重建 | 通过*b*位Lloyd-Max标量量化最小化均方误差 |
| **Product** | 内积估计 | 使用*(b−1)*位MSE + 1位QJL（量化Johnson-Lindenstrauss）进行无偏内积估计 |

## 安装

需要 Python ≥ 3.13 和 [uv](https://docs.astral.sh/uv/)

```bash
# 克隆仓库
git clone https://github.com/openclaw/openclaw-turboquant.git
cd openclaw-turboquant

# 使用uv安装
uv sync

# 或者以可编辑模式安装（开发）
uv pip install -e .
```

## 快速开始

### 库 API

```python
import numpy as np
from openclaw_turboquant import TurboQuantMSE, TurboQuantProd

# MSE量化（用于向量重建）
mse_q = TurboQuantMSE(d=128, bit_width=4, seed=42)
x = np.random.randn(128)
compressed = mse_q.quantize(x)
reconstructed = mse_q.dequantize(compressed)

# 内积量化
prod_q = TurboQuantProd(d=128, bit_width=4, seed=42)
x, y = np.random.randn(128), np.random.randn(128)
cx, cy = prod_q.quantize(x), prod_q.quantize(y)
ip_estimate = prod_q.estimate_inner_product(x, cy)
```

### 上下文存储（OpenClaw集成）

```python
from openclaw_turboquant.context_engine import ContextStore

# 创建上下文存储
store = ContextStore(dim=128, bit_width=4, seed=42)

# 添加文档和对应的嵌入向量
store.ingest("key1", embedding, "一些文本内容", metadata={"source": "doc.md"})
store.ingest("key2", embedding2, "更多文本内容", metadata={"source": "doc2.md"})

# 检索最相似的条目
results = store.retrieve_top_k(query_embedding, k=5)
# 返回: [(ContextEntry, score), ...]

# 在令牌预算内组装上下文
context = store.assemble_context(query_embedding, token_budget=4096)
# 返回: OpenClaw兼容的消息列表

# 紧凑化存储（保留最相关的50%条目）
removed_count = store.compact(keep_ratio=0.5, query_embedding=query_embedding)

# 估计压缩后的存储大小（字节）
memory_bytes = store.memory_estimate_bytes()
```

### 命令行工具

```bash
# 运行失真基准测试
openclaw-turboquant benchmark --dim 128 --bits 4 --n-vectors 1000

# 压缩向量文件
openclaw-turboquant compress --input vectors.npy --output compressed.npz --bits 4

# 检索相似向量
openclaw-turboquant retrieve --store compressed.npz --query query.npy --top-k 5
```

## OpenClaw集成

### 上下文引擎插件（开发中）

> **注意：** 插件接口已定义，但核心集成逻辑（Embedding API 调用、Python CLI 桥接）尚未实现。欢迎贡献！

`plugin/` 目录包含在 `ingest → assemble → compact → afterTurn` 生命周期中压缩嵌入向量的Context Engine插件：

- **`plugin/openclaw.plugin.json`** — 插件清单（kind: context-engine）
- **`plugin/index.ts`** — TypeScript入口点，注册 `turboquant-engine`

配置选项（通过插件设置）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bitWidth` | `4` | 每坐标比特数（1–8） |
| `embeddingDim` | `128` | 向量维度 |
| `topK` | `10` | 检索结果数量 |
| `compactKeepRatio` | `0.5` | 紧凑化时保留的比例 |

### AgentSkills

`skills/turboquant/SKILL.md` 为AI代理提供使用TurboQuant CLI和库API的说明。

## 算法详解

### Lloyd-Max编码本

随机旋转后，每个坐标遵循Beta分布：

$$f(x; d) = \frac{\Gamma(d/2)}{\Gamma(1/2)\,\Gamma((d-1)/2)} \cdot (1 - x^2)^{(d-3)/2}, \quad x \in [-1, 1]$$

Lloyd-Max算法迭代优化编码本中心点和决策边界，以最小化此分布下的预期失真。

**预计算编码本**：
- 支持的比特宽度：1–8位
- 编码本大小：2^b 个中心点
- 迭代次数：最多200次（收敛容差：1e-12）

### QJL变换

用于内积估计，TurboQuant使用1位量化Johnson-Lindenstrauss投影：

$$\hat{z} = \text{sign}(S \cdot x)$$

其中$S$是随机高斯投影矩阵。结合MSE残差，得到无偏估计器：$\mathbb{E}[\langle \hat{x}, \hat{y} \rangle] = \langle x, y \rangle$

### 两阶段管道

```
输入向量 x ∈ ℝ^d
     ↓
[阶段1] y = Π · x              (随机旋转)
     ↓
[阶段2] 索引 = 编码本.量化(y)   (标量量化)
     ↓
[反向] x̃ = Π^T · ỹ             (旋转回原空间)
     ↓
输出: MSEQuantized(索引, ||x||)
或 ProdQuantized(mse_idx, qjl_signs, residual_norm, ||x||)
```

## 基准测试

使用 `uv run pytest benchmarks/ --benchmark-only` 运行：

| 操作 | 维度 | 平均时间 | 吞吐量 |
|------|------|----------|--------|
| MSE量化 | 64 | ~4.6 µs | 217K vec/s |
| MSE反量化 | 64 | ~1.2 µs | 833K vec/s |
| MSE批量(100向量) | 64 | ~473 µs | - |
| MSE量化 | 256 | ~9.4 µs | - |
| Product量化 | 64 | ~11 µs | - |
| Product反量化 | 64 | ~3.6 µs | - |
| Product内积计算 | 64 | ~4.1 µs | - |
| QJL量化 | 64 | ~2.4 µs | - |
| QJL反量化 | 64 | ~1.2 µs | - |
| 上下文存储摄取 | 64 | ~12 µs | - |
| 上下文存储检索(100条) | 64 | ~406 µs | - |

### 压缩效率

对于 d=128, bit_width=4：
- 原始大小：512字节（float32 × 128）
- 压缩后：80字节（128×4位 + 64位范数）
- **压缩比：6.4倍**

## 架构

项目采用分层架构：

```
┌─────────────────────────────────────────────┐
│ 应用层                                       │
│ ├─ context_engine.py (高层API)              │
│ └─ cli.py (命令行工具)                       │
└────────────┬────────────────────────────────┘
             │
┌────────────┴────────────────────────────────┐
│ 算法层                                       │
│ └─ quantizer.py (TurboQuantMSE/Prod)       │
└────────────┬────────────────────────────────┘
             │
┌────────────┴─────────────────────────────────┐
│ 核心层（独立构建块）                         │
│ ├─ rotation.py (随机旋转)                    │
│ ├─ codebook.py (Lloyd-Max编码本)            │
│ └─ qjl.py (QJL变换)                         │
└─────────────────────────────────────────────┘
```

### 模块说明

| 模块 | 行数 | 职责 | 依赖 |
|------|------|------|------|
| `rotation.py` | 40 | 生成Haar随机旋转矩阵 | numpy |
| `codebook.py` | 127 | Lloyd-Max标量量化器 | scipy |
| `qjl.py` | 98 | 1位QJL变换 | numpy |
| `quantizer.py` | 344 | 两阶段管道编排 | 上述三个模块 |
| `context_engine.py` | 227 | 上下文压缩/检索API | quantizer |
| `cli.py` | 164 | 命令行接口 | quantizer, context_engine |

## 开发

```bash
# 运行所有测试
uv run pytest

# 运行基准测试
uv run pytest benchmarks/ --benchmark-only -v

# 代码检查与格式化
uv run ruff check src/ tests/ benchmarks/
uv run ruff format src/ tests/ benchmarks/

# 类型检查
uv run mypy src/
```

## 测试

项目包含35个全面的测试用例：

```bash
uv run pytest -v
```

测试覆盖：
- ✓ Beta分布PDF正确性
- ✓ Lloyd-Max编码本属性
- ✓ 随机旋转正交性和Haar测度
- ✓ QJL变换与无偏估计
- ✓ TurboQuantMSE/Prod量化与反量化
- ✓ 批量操作
- ✓ 上下文存储（摄取、检索、紧凑化）
- ✓ 令牌预算管理

## 使用场景

### 1. 大规模向量存储压缩

```python
from openclaw_turboquant import TurboQuantMSE

# 初始化量化器
q = TurboQuantMSE(d=1024, bit_width=4, seed=42)

# 压缩大量向量
vectors = np.random.randn(10000, 1024)
compressed = q.quantize_batch(vectors)

# 存储压缩后的向量而不是原始向量
# 节省6.4倍存储空间
```

### 2. 快速相似度搜索

```python
from openclaw_turboquant import ContextStore

store = ContextStore(d=1024, bit_width=4)

# 批量添加文档
for doc_id, embedding, text in documents:
    store.ingest(doc_id, embedding, text)

# 检索最相似的文档
query = get_query_embedding()
similar_docs = store.retrieve_top_k(query, k=10)
```

### 3. OpenClaw AI代理的上下文管理

```python
# 自动选择最相关且符合令牌预算的文档
context_messages = store.assemble_context(
    query_embedding,
    token_budget=8192,  # 上下文窗口
    avg_chars_per_token=4.0
)

# 发送给AI模型
response = model.generate(context_messages)

# 满足内存限制的存储紧凑化
store.compact(keep_ratio=0.7, query_embedding=query)
```

## API参考

### TurboQuantMSE

```python
class TurboQuantMSE:
    def __init__(
        self,
        d: int,
        bit_width: int,
        *,
        seed: int | None = None,
        codebook_max_iter: int = 200,
    ):
        """初始化MSE量化器
        
        参数:
            d: 向量维度
            bit_width: 每坐标比特数（1-8）
            seed: 随机种子
            codebook_max_iter: Lloyd-Max最大迭代次数
        """
    
    def quantize(self, x: NDArray) -> MSEQuantized:
        """量化单个向量"""
    
    def dequantize(self, compressed: MSEQuantized) -> NDArray:
        """反量化单个向量"""
    
    def quantize_batch(self, X: NDArray) -> list[MSEQuantized]:
        """批量量化向量"""
    
    def dequantize_batch(self, quantized: list[MSEQuantized]) -> NDArray:
        """批量反量化向量"""
```

### TurboQuantProd

```python
class TurboQuantProd:
    def __init__(
        self,
        d: int,
        bit_width: int,
        *,
        seed: int | None = None,
        codebook_max_iter: int = 200,
    ):
        """初始化内积量化器"""
    
    def quantize(self, x: NDArray) -> ProdQuantized:
        """量化单个向量（包含残差处理）"""
    
    def estimate_inner_product(
        self,
        y: NDArray,
        quantized_x: ProdQuantized,
    ) -> float:
        """估计 ⟨y, x⟩ 的无偏值"""
    
    # 其他方法同TurboQuantMSE
```

### ContextStore

```python
class ContextStore:
    def __init__(
        self,
        d: int,
        bit_width: int = 4,
        *,
        seed: int | None = None,
    ):
        """初始化上下文存储
        
        参数:
            d: 嵌入向量维度
            bit_width: 量化比特数
            seed: 随机种子
        """
    
    def ingest(
        self,
        entry_id: str,
        embedding: NDArray,
        text: str,
        metadata: dict | None = None,
    ) -> None:
        """添加或更新上下文条目"""
    
    def retrieve_top_k(
        self,
        query_embedding: NDArray,
        k: int = 5,
    ) -> list[tuple[ContextEntry, float]]:
        """检索最相似的k个条目（返回条目和相似度得分）"""
    
    def assemble_context(
        self,
        query_embedding: NDArray,
        token_budget: int,
        *,
        avg_chars_per_token: float = 4.0,
    ) -> list[dict]:
        """在令牌预算内组装上下文消息"""
    
    def compact(
        self,
        keep_ratio: float = 0.5,
        query_embedding: NDArray | None = None,
    ) -> int:
        """紧凑化存储，返回删除的条目数"""
    
    def memory_estimate_bytes(self) -> int:
        """估计压缩后的内存使用（字节）"""
    
    def remove(self, entry_id: str) -> bool:
        """删除条目"""
    
    @property
    def size(self) -> int:
        """返回存储的条目数"""
```

## 论文与参考

- **论文**: [TurboQuant: Near-Optimal Quantization for Efficient Scaled Dot-Product Attention](https://arxiv.org/abs/2504.19874)
- **作者**: Google Research
- **算法**: Two-stage pipeline (random rotation + scalar quantization)

## 许可证

MIT

## 常见问题

### Q: bit_width应该设置多少？
A: 对于大多数应用，4-6位是一个很好的平衡点，能在压缩率和精度之间取得平衡。1位用于极端压缩，8位用于高精度需求。

### Q: 如何选择合适的维度？
A: 通常与嵌入模型的输出维度相同。常见值：128（小模型）、768（BERT）、1024（较大模型）。

### Q: ContextStore中的令牌预算如何计算？
A: 通常设置为你的模型上下文窗口的50-75%，为模型生成的响应留出空间。例如，对于4K上下文模型，设置`token_budget=2048`。

### Q: 如何启用可重复性？
A: 在初始化时指定`seed`参数：
```python
q = TurboQuantMSE(d=128, bit_width=4, seed=42)
```

### Q: 压缩后精度损失有多大？
A: 这取决于bit_width和数据分布。4位通常保持90%以上的相似度精度。详见基准测试部分。

## 贡献

欢迎提交问题报告和拉取请求！

---

**更新日期**: 2026年3月28日  
**版本**: 0.1.0
