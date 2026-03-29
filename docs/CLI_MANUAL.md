# openclaw-turboquant — CLI Command Manual

> **Tool:** `openclaw-turboquant`  
> **Purpose:** TurboQuant near-optimal vector quantization for OpenClaw context compression

---

## Quick Reference

| Command | Category | One-line description |
|---------|----------|----------------------|
| [`ingest`](#ingest) | Context store | Add a message embedding to the persistent context store |
| [`assemble`](#assemble) | Context store | Retrieve the most relevant context entries within a token budget |
| [`compact`](#compact) | Context store | Remove least-relevant entries from the store to free memory |
| [`store-info`](#store-info) | Context store | Display statistics about the current context store |
| [`compress`](#compress) | Batch file | Batch-compress a `.npy` file of embeddings into a compressed index |
| [`retrieve`](#retrieve) | Batch file | Retrieve top-k similar vectors from a compressed index |
| [`benchmark`](#benchmark) | Utility | Run a distortion benchmark to measure quantization quality and speed |
| [`help`](#help) | Utility | Show detailed help for any command |

---

## Global Usage

```
openclaw-turboquant <command> [options]
openclaw-turboquant help [<command>]
```

---

## Context Store Commands

These commands are designed for the **OpenClaw memory integration** lifecycle:
`ingest` (Phase A) → `assemble` (Phase B) → `compact` (Phase C).

The store path defaults to `$OPENCLAW_MEMORY_DIR/turboquant-$OPENCLAW_SESSION_ID`
(falls back to `~/.openclaw/memory/turboquant-<session>` when env vars are absent).

---

### `ingest`

**Add a message embedding to the persistent context store.**

Quantizes an embedding vector using TurboQuant and saves it, along with the
original text and optional metadata, to the context store on disk.

#### Syntax

```
openclaw-turboquant ingest --id ID --text TEXT --embedding FILE [options]
```

#### Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--id` | ✅ | — | Unique identifier for this entry (e.g. `turn_001`) |
| `--text` | ✅ | — | Raw text content to store alongside the vector |
| `--embedding` | ✅ | — | Path to a `.npy` file containing the embedding vector (shape: `[d]`) |
| `--store` | — | auto | Store directory path |
| `--dim` | — | inferred | Embedding dimension; required only when creating a **new** store |
| `--bit-width` | — | `4` | Bits per coordinate for quantization (1–8) |
| `--seed` | — | `42` | Random seed for reproducibility |
| `--metadata` | — | `{}` | JSON string of extra metadata, e.g. `'{"role":"user"}'` |

#### Output (JSON)

```json
{"action":"ingest","entry_id":"turn_001","store_size":5,"store_path":"/path/to/store","ok":true}
```

#### Example

```bash
uv run openclaw-turboquant ingest \
  --id turn_001 \
  --text 'What is TurboQuant?' \
  --embedding /tmp/turn.npy \
  --metadata '{"role":"user"}'
```

---

### `assemble`

**Retrieve the most relevant context entries within a token budget.**

Searches the context store for entries most similar to a query vector and
returns them ranked by relevance score, stopping when the cumulative token
count exceeds the budget.

#### Syntax

```
openclaw-turboquant assemble --query FILE [options]
```

#### Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--query` | ✅ | — | Path to a `.npy` query embedding vector (shape: `[d]`) |
| `--store` | — | auto | Store directory path |
| `--token-budget` | — | `4096` | Maximum total tokens for the assembled context |

#### Output (JSON Lines, one object per returned entry)

```json
{"role":"context","content":"What is TurboQuant?","entry_id":"turn_001","score":0.912}
{"role":"context","content":"Tell me more.","entry_id":"turn_003","score":0.743}
```

#### Example

```bash
uv run openclaw-turboquant assemble \
  --query /tmp/query.npy \
  --token-budget 2048
```

---

### `compact`

**Remove least-relevant entries from the store to free memory.**

Keeps only the top `keep-ratio` fraction of entries. If `--query` is supplied,
relevance is measured by similarity to that vector; otherwise the newest entries
are kept.

#### Syntax

```
openclaw-turboquant compact [options]
```

#### Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--store` | — | auto | Store directory path |
| `--query` | — | — | Path to a `.npy` query vector for relevance-based compaction |
| `--keep-ratio` | — | `0.5` | Fraction of entries to keep (0.0–1.0) |

#### Output (JSON)

```json
{"action":"compact","before":20,"after":10,"removed":10,"store_path":"/path/to/store","ok":true}
```

#### Example

```bash
# Keep the top 50% most relevant entries for the current turn
uv run openclaw-turboquant compact \
  --query /tmp/current_turn.npy \
  --keep-ratio 0.5
```

---

### `store-info`

**Display statistics about the current context store.**

Prints a JSON summary of the store: size, dimension, bit-width, and memory
footprint.

#### Syntax

```
openclaw-turboquant store-info [--store PATH]
```

#### Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--store` | — | auto | Store directory path |

#### Output (JSON)

```json
{
  "path": "/Users/you/.openclaw/memory/turboquant-session123",
  "size": 15,
  "dim": 1536,
  "bit_width": 4,
  "memory_bytes": 14400,
  "memory_kb": 14.06
}
```

#### Example

```bash
uv run openclaw-turboquant store-info
```

---

## Batch File Commands

These commands operate directly on `.npy` / `.npz` files and do **not** use the
context store. They are suitable for offline batch processing.

---

### `compress`

**Batch-compress a `.npy` file of embeddings into a compressed index.**

Reads `N` vectors from a `.npy` file, applies TurboQuant Product quantization,
and saves the compressed index to a `.npz` file for later retrieval.

#### Syntax

```
openclaw-turboquant compress --input FILE --output FILE [options]
```

#### Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--input` | ✅ | — | Path to `.npy` input file (shape: `[N, d]`) |
| `--output` | ✅ | — | Output path for the compressed `.npz` index |
| `--bit-width` | — | `4` | Bits per coordinate (1–8) |
| `--seed` | — | `42` | Random seed |

#### Output

```
Compressed 100 vectors to compressed.npz
```

#### Example

```bash
uv run openclaw-turboquant compress \
  --input embeddings.npy \
  --output compressed.npz \
  --bit-width 4
```

---

### `retrieve`

**Retrieve top-k similar vectors from a compressed index.**

Loads a `.npz` index produced by `compress`, computes estimated inner-product
scores between the query and all stored vectors, and returns the top-k results.

> **Note:** `--seed` must match the seed used in the corresponding `compress`
> call, because TurboQuant uses the same random rotation matrix.

#### Syntax

```
openclaw-turboquant retrieve --query FILE --index FILE [options]
```

#### Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--query` | ✅ | — | Path to `.npy` query vector (shape: `[d]`) |
| `--index` | ✅ | — | Path to `.npz` index file produced by `compress` |
| `--top-k` | — | `5` | Number of results to return |
| `--seed` | — | `42` | Must match the seed used in `compress` |

#### Output (JSON Lines)

```json
{"index": 8, "score": 11.166}
{"index": 3, "score": 4.332}
```

#### Example

```bash
uv run openclaw-turboquant retrieve \
  --query query.npy \
  --index compressed.npz \
  --top-k 5
```

---

## Utility Commands

---

### `benchmark`

**Run a distortion benchmark to measure quantization quality and speed.**

Generates random unit vectors and measures MSE distortion, inner-product
distortion, quantization throughput, and compression ratio.

#### Syntax

```
openclaw-turboquant benchmark [options]
```

#### Arguments

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--dim` | — | `128` | Vector dimension |
| `--bit-width` | — | `4` | Bits per coordinate |
| `--n-vectors` | — | `100` | Number of test vectors |
| `--seed` | — | `42` | Random seed |

#### Output

```
TurboQuant Benchmark: d=128, b=4, n=100
============================================================
MSE setup: 0.002s
MSE quantize: 0.001s (82000 vec/s)
MSE distortion: 0.006123

Prod setup: 0.003s
Prod quantize: 0.001s (75000 vec/s)
IP distortion: 0.000412

Compression ratio: 6.3x
Bits per vector: 576 (from 4096)
```

#### Example

```bash
uv run openclaw-turboquant benchmark --dim 1536 --bit-width 4 --n-vectors 500
```

---

### `help`

**Show this help, or detailed help for a specific command.**

#### Syntax

```
openclaw-turboquant help [COMMAND]
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `COMMAND` | — | Command name (omit to list all commands) |

#### Examples

```bash
# List all commands
uv run openclaw-turboquant help

# Show detailed help for the ingest command
uv run openclaw-turboquant help ingest
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENCLAW_SESSION_ID` | Session identifier injected by the OpenClaw exec tool; used to auto-derive the store path |
| `OPENCLAW_MEMORY_DIR` | Override the base memory directory (default: `~/.openclaw/memory/`) |

---

## Error Codes

| Exit Code | Meaning |
|-----------|---------|
| `0` | Success |
| `1` | User error (bad arguments, file not found, unknown command) |
| `2` | Internal / unexpected error |

---

## OpenClaw Skill Lifecycle

```
Turn start  →  ingest (embed each message)
Query time  →  assemble (retrieve relevant context)
Turn end    →  compact (prune store when it grows large)
Diagnostics →  store-info
```

For full skill integration instructions, see [`skills/turboquant/SKILL.md`](../skills/turboquant/SKILL.md).
