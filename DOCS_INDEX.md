# 文档导航 (Documentation Index)

## 📚 中文文档 (Chinese Documentation)

### 🚀 快速开始
- **[QUICKSTART_ZH.md](./QUICKSTART_ZH.md)** - 5分钟快速入门指南
  - 安装步骤
  - 基本使用示例
  - 常用参数详解
  - 性能参考
  - 常见错误解决

### 📖 完整文档
- **[README.zh-CN.md](./README.zh-CN.md)** - 完整的中文README
  - 项目概览和核心算法
  - 详细的API参考
  - OpenClaw集成说明
  - 架构设计
  - 使用场景和最佳实践
  - FAQ（常见问题）

---

## 📚 English Documentation

### 🚀 Quick Start
- **[README.md](./README.md)** - Official English documentation
  - Project overview
  - Installation instructions
  - Quick start examples
  - CLI usage
  - Benchmarks
  - Algorithm details

---

## 🔍 其他资源 (Other Resources)

### 项目结构
```
openclaw-turboquant/
├── src/openclaw_turboquant/     # 核心代码
│   ├── rotation.py              # 随机旋转
│   ├── codebook.py              # Lloyd-Max编码本
│   ├── qjl.py                   # QJL变换
│   ├── quantizer.py             # 主算法
│   ├── context_engine.py        # 上下文API
│   └── cli.py                   # 命令行工具
├── tests/                       # 测试（35个用例）
├── benchmarks/                  # 性能基准
├── plugin/                      # OpenClaw插件
├── skills/                      # AI Agent技能定义
├── README.md                    # English README
├── README.zh-CN.md              # 中文README
├── QUICKSTART_ZH.md             # 中文快速开始
└── pyproject.toml               # 项目配置
```

### 关键模块说明
| 文件 | 行数 | 职责 |
|------|------|------|
| rotation.py | 40 | 生成Haar随机旋转矩阵 |
| codebook.py | 127 | Lloyd-Max标量量化器 |
| qjl.py | 98 | 1位QJL变换 |
| quantizer.py | 344 | TurboQuantMSE/Prod编排 |
| context_engine.py | 227 | 上下文压缩/检索API |
| cli.py | 164 | 命令行工具 |

---

## 🛠️ 常用命令

```bash
# 安装依赖
uv sync

# 运行测试
uv run pytest -v

# 运行基准
uv run pytest benchmarks/ --benchmark-only

# CLI使用
uv run openclaw-turboquant benchmark --dim 128 --bits 4
uv run openclaw-turboquant compress --input vectors.npy --output compressed.npz
uv run openclaw-turboquant retrieve --store compressed.npz --query query.npy

# 代码检查
uv run ruff check src/
uv run mypy src/
```

---

## 📊 快速参考

### TurboQuantMSE（向量重建）
```python
from openclaw_turboquant import TurboQuantMSE
import numpy as np

q = TurboQuantMSE(d=128, bit_width=4, seed=42)
x = np.random.randn(128)
compressed = q.quantize(x)
reconstructed = q.dequantize(compressed)
```

### TurboQuantProd（内积估计）
```python
from openclaw_turboquant import TurboQuantProd

q = TurboQuantProd(d=128, bit_width=4, seed=42)
x, y = np.random.randn(128), np.random.randn(128)
qx = q.quantize(x)
ip_estimate = q.estimate_inner_product(y, qx)
```

### ContextStore（上下文管理）
```python
from openclaw_turboquant.context_engine import ContextStore

store = ContextStore(d=128, bit_width=4)
store.ingest("doc1", embedding, text)
results = store.retrieve_top_k(query, k=5)
context = store.assemble_context(query, token_budget=4096)
```

---

## 🎯 选择正确的文档

### 如果你想...
- 🚀 **快速上手** → 阅读 [QUICKSTART_ZH.md](./QUICKSTART_ZH.md)
- 📖 **全面了解** → 阅读 [README.zh-CN.md](./README.zh-CN.md)
- 🔍 **查看API** → 参考 [README.zh-CN.md](./README.zh-CN.md) 中的API参考部分
- 💻 **查看代码示例** → [QUICKSTART_ZH.md](./QUICKSTART_ZH.md) 中有丰富示例
- 🐛 **解决问题** → 查看 [QUICKSTART_ZH.md](./QUICKSTART_ZH.md) 中的常见错误部分
- 📚 **深入学习算法** → [README.zh-CN.md](./README.zh-CN.md) 中的"算法详解"
- ⚡ **性能优化** → [QUICKSTART_ZH.md](./QUICKSTART_ZH.md) 中的性能参考和提示部分

---

## 🌍 语言选择

- 中文文档: [README.zh-CN.md](./README.zh-CN.md) 和 [QUICKSTART_ZH.md](./QUICKSTART_ZH.md)
- English: [README.md](./README.md)

---

**上次更新**: 2026年3月28日
