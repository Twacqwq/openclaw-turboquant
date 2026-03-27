---
name: turboquant
version: 0.1.0
description: Compress and retrieve context vectors using TurboQuant near-optimal quantization for efficient context management.
metadata:
  {
    "openclaw":
      {
        "requires": { "bins": ["uv"] },
        "emoji": "⚡",
        "homepage": "https://github.com/openclaw/openclaw-turboquant"
      }
  }
---

# TurboQuant Context Compression Skill

Use the `exec` tool to invoke the TurboQuant CLI for context vector compression and retrieval.

## When to use

- When the context window is filling up and older messages need compression
- When you need to find the most relevant historical context from a large session
- When `/compact` is triggered and you want vector-aware compaction

## Setup

Ensure the project is installed before using any commands:

```bash
cd <project-root> && uv sync
```

## Commands

### Compress context embeddings

```bash
uv run openclaw-turboquant compress --input embeddings.npy --output compressed.npz --bit-width 4
```

### Retrieve top-k similar entries

```bash
uv run openclaw-turboquant retrieve --query query.npy --index compressed.npz --top-k 5
```

### Benchmark compression quality

```bash
uv run openclaw-turboquant benchmark --dim 128 --bit-width 4 --n-vectors 1000
```

## Notes

- TurboQuant achieves near-optimal distortion at 3-4 bits per coordinate
- The `prod` mode provides unbiased inner product estimation (best for retrieval)
- The `mse` mode minimizes reconstruction error (best for storage)
- Compression ratio: 6-8x compared to float32, with negligible quality loss
- Requires `uv` to be installed and `uv sync` run in the project directory
