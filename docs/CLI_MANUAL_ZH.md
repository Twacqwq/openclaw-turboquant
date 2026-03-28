# openclaw-turboquant — CLI 指令集操作手册

> **工具名称：** `openclaw-turboquant`  
> **用途：** 基于 TurboQuant 近最优向量量化算法的 OpenClaw 上下文压缩工具

---

## 快速索引

| 指令 | 类别 | 功能说明 |
|------|------|----------|
| [`ingest`](#ingest) | 上下文存储 | 将消息 embedding 向量添加到持久化上下文存储 |
| [`assemble`](#assemble) | 上下文存储 | 在 Token 预算内检索最相关的上下文条目 |
| [`compact`](#compact) | 上下文存储 | 删除最不相关的条目以释放存储空间 |
| [`store-info`](#store-info) | 上下文存储 | 显示当前上下文存储的统计信息 |
| [`compress`](#compress) | 批量文件 | 将 `.npy` embedding 文件批量压缩为索引文件 |
| [`retrieve`](#retrieve) | 批量文件 | 从压缩索引文件中检索 Top-k 相似向量 |
| [`benchmark`](#benchmark) | 工具 | 运行失真度基准测试以评估量化质量和速度 |
| [`help`](#help) | 工具 | 查看任意指令的详细帮助信息 |

---

## 全局用法

```
openclaw-turboquant <指令> [选项]
openclaw-turboquant help [<指令>]
```

---

## 上下文存储类指令

这些指令专为 **OpenClaw 记忆集成** 生命周期设计：  
`ingest`（阶段 A：入库）→ `assemble`（阶段 B：组装）→ `compact`（阶段 C：压缩）

存储路径默认为 `$OPENCLAW_MEMORY_DIR/turboquant-$OPENCLAW_SESSION_ID`  
（未设置环境变量时回退为 `~/.openclaw/memory/turboquant-<session>`）

---

### `ingest`

**将消息 embedding 向量添加到持久化上下文存储。**

使用 TurboQuant 对 embedding 向量进行量化，并将其与原始文本及可选元数据一起保存到磁盘上的上下文存储中。

#### 语法

```
openclaw-turboquant ingest --id ID --text TEXT --embedding FILE [选项]
```

#### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--id` | ✅ | — | 当前条目的唯一标识符（如 `turn_001`） |
| `--text` | ✅ | — | 与向量一起存储的原始文本内容 |
| `--embedding` | ✅ | — | `.npy` embedding 向量文件路径（形状：`[d]`） |
| `--store` | — | 自动推断 | 上下文存储目录路径 |
| `--dim` | — | 自动推断 | embedding 维度；仅在**新建**存储时需要指定 |
| `--bit-width` | — | `4` | 每个坐标的量化位宽（1–8） |
| `--seed` | — | `42` | 随机种子，用于可复现性 |
| `--metadata` | — | `{}` | JSON 格式的额外元数据，如 `'{"role":"user"}'` |

#### 输出（JSON）

```json
{"action":"ingest","entry_id":"turn_001","store_size":5,"store_path":"/path/to/store","ok":true}
```

#### 示例

```bash
uv run openclaw-turboquant ingest \
  --id turn_001 \
  --text '什么是 TurboQuant？' \
  --embedding /tmp/turn.npy \
  --metadata '{"role":"user"}'
```

---

### `assemble`

**在 Token 预算内检索最相关的上下文条目。**

在上下文存储中搜索与查询向量最相似的条目，按相关性分数排序后返回，当累计 Token 数超出预算时停止。

#### 语法

```
openclaw-turboquant assemble --query FILE [选项]
```

#### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--query` | ✅ | — | `.npy` 查询向量文件路径（形状：`[d]`） |
| `--store` | — | 自动推断 | 上下文存储目录路径 |
| `--token-budget` | — | `4096` | 组装上下文的最大 Token 数 |

#### 输出（JSON Lines，每个返回条目一行）

```json
{"role":"context","content":"什么是 TurboQuant？","entry_id":"turn_001","score":0.912}
{"role":"context","content":"请继续介绍。","entry_id":"turn_003","score":0.743}
```

#### 示例

```bash
uv run openclaw-turboquant assemble \
  --query /tmp/query.npy \
  --token-budget 2048
```

---

### `compact`

**删除最不相关的条目以释放存储空间。**

仅保留 `keep-ratio` 比例的条目。若指定了 `--query`，则按与该向量的相似度排序；否则保留最新的条目。

#### 语法

```
openclaw-turboquant compact [选项]
```

#### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--store` | — | 自动推断 | 上下文存储目录路径 |
| `--query` | — | — | 用于相关性排序的查询向量 `.npy` 文件路径 |
| `--keep-ratio` | — | `0.5` | 保留条目的比例（0.0–1.0） |

#### 输出（JSON）

```json
{"action":"compact","before":20,"after":10,"removed":10,"store_path":"/path/to/store","ok":true}
```

#### 示例

```bash
# 保留与当前轮次最相关的 50% 条目
uv run openclaw-turboquant compact \
  --query /tmp/current_turn.npy \
  --keep-ratio 0.5
```

---

### `store-info`

**显示当前上下文存储的统计信息。**

输出存储的 JSON 摘要，包括条目数量、向量维度、量化位宽和内存占用。

#### 语法

```
openclaw-turboquant store-info [--store PATH]
```

#### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--store` | — | 自动推断 | 上下文存储目录路径 |

#### 输出（JSON）

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

#### 示例

```bash
uv run openclaw-turboquant store-info
```

---

## 批量文件类指令

这些指令直接操作 `.npy` / `.npz` 文件，**不使用**上下文存储，适用于离线批量处理。

---

### `compress`

**将 `.npy` embedding 文件批量压缩为压缩索引文件。**

从 `.npy` 文件中读取 N 个向量，应用 TurboQuant Product 量化，并将压缩后的索引保存为 `.npz` 文件供后续检索使用。

#### 语法

```
openclaw-turboquant compress --input FILE --output FILE [选项]
```

#### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | ✅ | — | 输入 `.npy` 文件路径（形状：`[N, d]`） |
| `--output` | ✅ | — | 输出 `.npz` 压缩索引文件路径 |
| `--bit-width` | — | `4` | 每个坐标的量化位宽（1–8） |
| `--seed` | — | `42` | 随机种子 |

#### 输出

```
Compressed 100 vectors to compressed.npz
```

#### 示例

```bash
uv run openclaw-turboquant compress \
  --input embeddings.npy \
  --output compressed.npz \
  --bit-width 4
```

---

### `retrieve`

**从压缩索引文件中检索 Top-k 相似向量。**

加载 `compress` 生成的 `.npz` 索引，计算查询向量与所有存储向量之间的估计内积分数，并返回 Top-k 结果。

> **注意：** `--seed` 必须与对应 `compress` 操作中使用的种子一致，因为 TurboQuant 使用相同的随机旋转矩阵。

#### 语法

```
openclaw-turboquant retrieve --query FILE --index FILE [选项]
```

#### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--query` | ✅ | — | `.npy` 查询向量文件路径（形状：`[d]`） |
| `--index` | ✅ | — | 由 `compress` 生成的 `.npz` 索引文件路径 |
| `--top-k` | — | `5` | 返回结果数量 |
| `--seed` | — | `42` | 必须与 `compress` 时使用的种子一致 |

#### 输出（JSON Lines）

```json
{"index": 8, "score": 11.166}
{"index": 3, "score": 4.332}
```

#### 示例

```bash
uv run openclaw-turboquant retrieve \
  --query query.npy \
  --index compressed.npz \
  --top-k 5
```

---

## 工具类指令

---

### `benchmark`

**运行失真度基准测试以评估量化质量和速度。**

生成随机单位向量，测量 MSE 失真度、内积失真度、量化吞吐量和压缩比。

#### 语法

```
openclaw-turboquant benchmark [选项]
```

#### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--dim` | — | `128` | 向量维度 |
| `--bit-width` | — | `4` | 每个坐标的量化位宽 |
| `--n-vectors` | — | `100` | 测试向量数量 |
| `--seed` | — | `42` | 随机种子 |

#### 输出

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

#### 示例

```bash
uv run openclaw-turboquant benchmark --dim 1536 --bit-width 4 --n-vectors 500
```

---

### `help`

**查看所有指令列表，或查看指定指令的详细帮助信息。**

#### 语法

```
openclaw-turboquant help [指令名]
```

#### 参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `指令名` | — | 要查询的指令名称（省略则列出所有指令） |

#### 示例

```bash
# 列出所有指令
uv run openclaw-turboquant help

# 查看 ingest 指令的详细说明
uv run openclaw-turboquant help ingest

# 查看 assemble 指令的详细说明
uv run openclaw-turboquant help assemble
```

---

## 环境变量

| 变量 | 说明 |
|------|------|
| `OPENCLAW_SESSION_ID` | 由 OpenClaw exec 工具注入的会话标识符；用于自动推断存储路径 |
| `OPENCLAW_MEMORY_DIR` | 覆盖基础记忆目录（默认：`~/.openclaw/memory/`） |

---

## 退出码

| 退出码 | 含义 |
|--------|------|
| `0` | 成功 |
| `1` | 用户错误（参数错误、文件未找到、未知指令等） |
| `2` | 内部/意外错误 |

---

## OpenClaw Skill 生命周期

```
对话开始  →  ingest（对每条消息进行 embedding 入库）
查询阶段  →  assemble（检索相关上下文）
对话结束  →  compact（存储增长时进行剪枝）
诊断/调试 →  store-info（查看存储状态）
```

完整的 Skill 集成说明，请参阅 [`skills/turboquant/SKILL.md`](../skills/turboquant/SKILL.md)。

---

## 典型使用场景

### 场景一：OpenClaw 集成（推荐）

```bash
# 1. 每轮对话入库
uv run openclaw-turboquant ingest \
  --id "turn_$(date +%s)" \
  --text "$USER_MESSAGE" \
  --embedding /tmp/embedding.npy

# 2. 组装相关上下文（传入下一轮 LLM）
uv run openclaw-turboquant assemble \
  --query /tmp/query.npy \
  --token-budget 4096

# 3. 对话结束后压缩（可选，存储过大时使用）
uv run openclaw-turboquant compact \
  --query /tmp/latest.npy \
  --keep-ratio 0.7
```

### 场景二：离线批量处理

```bash
# 压缩 embedding 文件
uv run openclaw-turboquant compress \
  --input my_embeddings.npy \
  --output index.npz

# 检索最相似的 10 条
uv run openclaw-turboquant retrieve \
  --query query.npy \
  --index index.npz \
  --top-k 10
```

### 场景三：评估量化质量

```bash
# 使用与生产环境相同的参数进行基准测试
uv run openclaw-turboquant benchmark \
  --dim 1536 \
  --bit-width 4 \
  --n-vectors 1000
```
