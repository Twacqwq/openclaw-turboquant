# openclaw-turboquant

**English** | [简体中文](README.zh-CN.md)

Near-optimal online vector quantization for [OpenClaw](https://openclaw.dev) context compression, based on the **TurboQuant** algorithm from [Google Research](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)).

## Status

| Component | Status | Description |
|-----------|--------|-------------|
| Library API | ✅ Ready | Core quantization algorithms fully implemented |
| CLI | ✅ Ready | `benchmark`, `compress`, `retrieve` commands available |
| Agent Skill | ✅ Ready | CLI commands can be used independently by agents |
| Context Engine Plugin | 🚧 WIP | Interface defined, core integration logic not yet implemented |

## Overview

TurboQuant achieves near-optimal distortion (within ~2.7× of the information-theoretic lower bound) using a simple two-stage pipeline:

1. **Random Rotation** — Apply a random orthogonal matrix (Haar measure via QR decomposition) to spread information uniformly across coordinates.
2. **Scalar Quantization** — Quantize each rotated coordinate independently using a Lloyd-Max codebook optimized for the Beta distribution of coordinates on the unit hypersphere.

Two quantization modes are provided:

| Mode | Use Case | Description |
|------|----------|-------------|
| **MSE** | Reconstruction | Minimizes mean squared error via Lloyd-Max scalar quantization at *b* bits per coordinate |
| **Product** | Inner-product estimation | Uses MSE at *(b−1)* bits + 1-bit QJL (Quantized Johnson-Lindenstrauss) on the residual for unbiased inner-product estimation |

## Installation

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
# Clone the repository
git clone https://github.com/openclaw/openclaw-turboquant.git
cd openclaw-turboquant

# Install with uv
uv sync
```

## Quick Start

### Library API

```python
import numpy as np
from openclaw_turboquant import TurboQuantMSE, TurboQuantProd

# MSE quantization (for reconstruction)
mse_q = TurboQuantMSE(dim=128, bit_width=4, seed=42)
x = np.random.randn(128)
compressed = mse_q.quantize(x)
reconstructed = mse_q.dequantize(compressed)

# Inner-product quantization
prod_q = TurboQuantProd(dim=128, bit_width=4, seed=42)
x, y = np.random.randn(128), np.random.randn(128)
cx, cy = prod_q.quantize(x), prod_q.quantize(y)
ip_estimate = prod_q.estimate_inner_product(cx, cy)
```

### Context Store (OpenClaw Integration)

```python
from openclaw_turboquant.context_engine import ContextStore

store = ContextStore(dim=128, bit_width=4, seed=42)
store.ingest("key1", embedding, "Some text content", metadata={"source": "doc.md"})

# Retrieve top-k similar entries
results = store.retrieve_top_k(query_embedding, k=5)

# Assemble context within token budget
context = store.assemble_context(query_embedding, max_tokens=4096)

# Compact the store (keep 50% most relevant entries)
store.compact(keep_ratio=0.5, query_embedding=query_embedding)
```

### CLI

```bash
# Run benchmarks
openclaw-turboquant benchmark --dim 128 --bits 4

# Compress vectors from a .npy file
openclaw-turboquant compress --input vectors.npy --output compressed.npz --bits 4

# Retrieve similar vectors
openclaw-turboquant retrieve --store compressed.npz --query query.npy --top-k 5
```

## OpenClaw Integration

### Context Engine Plugin (WIP)

> **Note:** The plugin interface is defined but the core integration logic (embedding API calls, Python CLI bridge) is not yet implemented. Contributions welcome!

The `plugin/` directory contains a Context Engine plugin that compresses embeddings during the `ingest → assemble → compact → afterTurn` lifecycle:

- **`plugin/openclaw.plugin.json`** — Plugin manifest (`kind: context-engine`)
- **`plugin/index.ts`** — TypeScript entry point registering the `turboquant-engine`

Configuration options (via plugin settings):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bitWidth` | `4` | Bits per coordinate (1–8) |
| `embeddingDim` | `128` | Vector dimension |
| `topK` | `10` | Number of results for retrieval |
| `compactKeepRatio` | `0.5` | Fraction of entries kept during compaction |

### AgentSkills

The `skills/turboquant/SKILL.md` skill provides AI agents with instructions for using the TurboQuant CLI and library API.

## Algorithm Details

### Lloyd-Max Codebook

After random rotation, each coordinate follows a Beta distribution:

$$f(x; d) = \frac{\Gamma(d/2)}{\Gamma(1/2)\,\Gamma((d-1)/2)} \cdot (1 - x^2)^{(d-3)/2}, \quad x \in [-1, 1]$$

The Lloyd-Max algorithm iteratively optimizes codebook centroids and decision boundaries to minimize expected distortion under this distribution.

### QJL Transform

For inner-product estimation, TurboQuant uses a 1-bit Quantized Johnson-Lindenstrauss projection:

$$\hat{z} = \text{sign}(S \cdot x)$$

where $S$ is a random Gaussian projection matrix. Combined with the MSE residual, this yields an unbiased estimator: $\mathbb{E}[\langle \hat{x}, \hat{y} \rangle] = \langle x, y \rangle$.

## Benchmarks

Run with `uv run pytest benchmarks/ --benchmark-only`:

| Operation | Dimension | Mean |
|-----------|-----------|------|
| MSE quantize | 64 | ~4.6 µs |
| MSE dequantize | 64 | ~1.2 µs |
| MSE batch (100 vectors) | 64 | ~473 µs |
| MSE quantize | 256 | ~9.4 µs |
| Product quantize | 64 | ~11 µs |
| Product dequantize | 64 | ~3.6 µs |
| Product inner product | 64 | ~4.1 µs |
| QJL quantize | 64 | ~2.4 µs |
| QJL dequantize | 64 | ~1.2 µs |
| Context Store ingest | 64 | ~12 µs |
| Context Store retrieve (100 entries) | 64 | ~406 µs |

## Development

```bash
# Run tests
uv run pytest

# Run benchmarks
uv run pytest benchmarks/ --benchmark-only -v

# Lint & format
uv run ruff check src/ tests/ benchmarks/
uv run ruff format src/ tests/ benchmarks/

# Type check
uv run mypy src/
```

## License

MIT