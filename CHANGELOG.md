# CHANGELOG

本文档记录 `feature/hucc` 分支相对于 `main` 分支的所有变更。

---

## [Unreleased] — feature/hucc

**分支**: `feature/hucc`  
**基础版本**: `f96fb77` (main — feat: implement TurboQuant vector quantization with OpenClaw integration)  
**变更提交数**: 3  
**净增代码行**: +1,311 行（6 个文件）  
**测试状态**: ✅ 35/35 通过

---

## 变更详情

### 🆕 feat — 持久化上下文存储与 OpenClaw Memory 集成

**提交**: `d7293dd`  
**日期**: 2026-03-28  
**影响文件**: 3 个文件，+340 行

#### `src/openclaw_turboquant/context_engine.py` (+98 行)

**新增：持久化 save / load**

| 方法 | 签名 | 说明 |
|------|------|------|
| `save` | `save(path: str \| Path) -> None` | 将 store 写出到目录，包含三个文件：`config.json`、`texts.json`、`vectors.npz` |
| `load` | `classmethod load(path: str \| Path) -> ContextStore` | 从目录完整恢复 store，恢复后可直接执行查询 |

**持久化目录结构**：

```
<store_path>/
├── config.json     ← {"d": 1536, "bit_width": 4, "seed": 42}
├── texts.json      ← {"turn_001": {"text": "...", "metadata": {...}}}
└── vectors.npz     ← mse_indices[N,d], qjl_signs[N,d], residual_norms[N], norms[N]
```

---

#### `src/openclaw_turboquant/cli.py` (+157 行)

**新增：辅助函数**

| 函数 | 说明 |
|------|------|
| `_default_store_path()` | 自动读取 `$OPENCLAW_SESSION_ID` 和 `$OPENCLAW_MEMORY_DIR` 环境变量，构造默认 store 路径 `~/.openclaw/memory/turboquant-<session_id>` |

**新增：4 个 CLI 子命令**

##### `ingest` — 逐条摄入消息

```bash
uv run openclaw-turboquant ingest \
  --store ~/.openclaw/memory/turboquant-$OPENCLAW_SESSION_ID \
  --id "turn_001" \
  --text "消息内容" \
  --embedding turn.npy \
  [--dim 1536] \
  [--bit-width 4] \
  [--seed 42] \
  [--metadata '{"source":"user"}']
```

- `--store` 默认值：`$OPENCLAW_MEMORY_DIR/turboquant-$OPENCLAW_SESSION_ID`
- `--dim` 仅首次创建 store 时需要，后续自动从 `config.json` 读取
- store 不存在时自动创建；每次 ingest 后自动 save

**输出**：
```json
{"action": "ingest", "entry_id": "turn_001", "store_size": 5, "store_path": "...", "ok": true}
```

---

##### `assemble` — 按 token budget 组装上下文

```bash
uv run openclaw-turboquant assemble \
  --store ~/.openclaw/memory/turboquant-$OPENCLAW_SESSION_ID \
  --query current_query.npy \
  --token-budget 4096
```

- 按相关性（内积估计）排序，贪心填充 token budget
- 输出 JSON Lines（每行一个条目），可直接注入 system prompt

**输出**（每行一个 JSON）：
```json
{"role": "context", "content": "消息内容", "entry_id": "turn_003", "score": 0.912}
{"role": "context", "content": "消息内容", "entry_id": "turn_001", "score": 0.743}
```

---

##### `compact` — 清理低相关性条目

```bash
uv run openclaw-turboquant compact \
  --store ~/.openclaw/memory/turboquant-$OPENCLAW_SESSION_ID \
  --query current_query.npy \
  [--keep-ratio 0.5]
```

- `--query` 可选；提供时按相关性保留，不提供时保留最近插入的条目
- compact 后自动 save

**输出**：
```json
{"action": "compact", "before": 20, "after": 10, "removed": 10, "store_path": "...", "ok": true}
```

---

##### `store-info` — 查看 store 状态

```bash
uv run openclaw-turboquant store-info \
  --store ~/.openclaw/memory/turboquant-$OPENCLAW_SESSION_ID
```

**输出**：
```json
{"path": "/...", "size": 15, "dim": 1536, "bit_width": 4, "memory_bytes": 14400, "memory_kb": 14.06}
```

---

#### `skills/turboquant/SKILL.md` (完整重写，+85 行净增)

旧版描述的是文件批处理流程（`compress` / `retrieve`），与 OpenClaw 对话流程脱节。

**新版结构**：

| 章节 | 内容 |
|------|------|
| Store 路径约定 | 说明 `$OPENCLAW_SESSION_ID` 的使用方式 |
| Phase A — ingest | 每轮对话结束后摄入 |
| Phase B — assemble | 构建 prompt 前检索相关上下文 |
| Phase C — compact | 上下文窗口满时清理 |
| 低级命令 | 保留 compress / retrieve / benchmark |
| Notes | 说明 embedding 不由 skill 生成，以及 uv 依赖 |

**对话生命周期**：

```
Turn N
  ├─ 用户消息到达
  ├─ [ingest] 保存本轮 embedding 到 OpenClaw memory
  ├─ [assemble] 取出历史相关上下文 → 注入 prompt
  ├─ AI 生成回复
  └─ [compact] 若 store_size > 阈值，清理低相关条目
```

---

### 🔧 fix — 修正 SKILL.md 格式问题

**提交**: `e59a5a4`  
**日期**: 2026-03-28  
**影响文件**: 1 个文件，+14 行

#### `skills/turboquant/SKILL.md`

| 问题 | 修复前 | 修复后 |
|------|--------|--------|
| 缺少 `version` 字段 | ❌ 未设置 | ✅ `version: 0.1.0` |
| CLI 调用方式错误 | `uv run python -m openclaw_turboquant.cli` | `uv run openclaw-turboquant` |
| 缺少 Setup 章节 | ❌ | ✅ 新增 `uv sync` 说明 |
| 压缩比数据有误 | `8-10x` | `6-8x`（实测值） |

---

### 🌐 feat — 新增中文文档

**提交**: `8b01ba7`  
**日期**: 2026-03-28  
**影响文件**: 3 个新文件，+964 行

#### 新增文件

| 文件 | 行数 | 大小 | 内容 |
|------|------|------|------|
| `README.zh-CN.md` | 464 | 13 KB | 完整中文参考文档 |
| `QUICKSTART_ZH.md` | 350 | 7.8 KB | 5 分钟快速入门 |
| `DOCS_INDEX.md` | 150 | 4.1 KB | 文档导航索引 |

#### `README.zh-CN.md` 章节

- 项目概览与算法原理（Lloyd-Max、QJL、两阶段管道）
- 安装与快速开始（5 个完整代码示例）
- OpenClaw 集成指南
- 完整 API 参考（`TurboQuantMSE` / `TurboQuantProd` / `ContextStore`）
- 架构分层图（Core → Algorithm → Application）
- 6 个真实使用场景
- 11 个常见问题解答

#### `QUICKSTART_ZH.md` 章节

- 安装步骤（可复制命令）
- 5 个可运行的代码示例
- 参数速查表
- MSE vs Product 模式对比
- 性能基准参考
- 调试技巧与常见错误解决

#### `DOCS_INDEX.md` 章节

- 中英文文档导航链接
- 项目目录结构
- 常用命令速查
- "如果你想…" 的文档推荐路径

---

## 文件变更汇总

| 文件 | 状态 | 变更行数 | 最终行数 |
|------|------|----------|----------|
| `src/openclaw_turboquant/context_engine.py` | ✏️ 修改 | +98 | 324 |
| `src/openclaw_turboquant/cli.py` | ✏️ 修改 | +157 | 317 |
| `skills/turboquant/SKILL.md` | ✏️ 修改 | +111 / -19 | 127 |
| `README.zh-CN.md` | 🆕 新建 | +464 | 464 |
| `QUICKSTART_ZH.md` | 🆕 新建 | +350 | 350 |
| `DOCS_INDEX.md` | 🆕 新建 | +150 | 150 |
| **合计** | | **+1,311 行** | **1,732 行** |

---

## 新增 CLI 命令汇总

```
openclaw-turboquant
├── ingest      ← 🆕 逐条摄入消息到持久化 store
├── assemble    ← 🆕 按 token budget 组装相关上下文
├── compact     ← 🆕 清理低相关性条目
├── store-info  ← 🆕 查看 store 状态
├── compress    （原有）批量压缩向量文件
├── retrieve    （原有）从压缩文件检索
└── benchmark   （原有）性能基准测试
```

---

## 新增 Python API

```python
# 持久化
store.save("/path/to/store_dir")
store = ContextStore.load("/path/to/store_dir")
```

---

## 环境变量支持

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `OPENCLAW_SESSION_ID` | store 目录名中的 session 标识 | `"default"` |
| `OPENCLAW_MEMORY_DIR` | store 根目录 | `~/.openclaw/memory` |

---

## 测试状态

| 测试类 | 测试数 | 状态 |
|--------|--------|------|
| `TestBetaPdf` | 3 | ✅ |
| `TestLloydMaxCodebook` | 6 | ✅ |
| `TestRandomRotation` | 3 | ✅ |
| `TestQJLTransform` | 3 | ✅ |
| `TestTurboQuantMSE` | 5 | ✅ |
| `TestTurboQuantProd` | 5 | ✅ |
| `TestContextStore` | 7 | ✅ |
| `TestBitsPerVector` | 3 | ✅ |
| **合计** | **35** | **✅ 全部通过** |

> 注：新增的 4 个 CLI 命令已通过手动端到端验证，测试文件待补充（`todos.tq-tests`）。

---

## 提交记录

```
d7293dd  feat: add persistent context store and OpenClaw memory integration
e59a5a4  fix: correct SKILL.md - add version field and fix CLI command format
8b01ba7  feat: add comprehensive Chinese documentation
────────────────────────────────────────────────────────────
f96fb77  (main) feat: implement TurboQuant vector quantization with OpenClaw integration
```

---

## 待完成

- [ ] `tq-tests` — 为 `ingest` / `assemble` / `compact` / `store-info` 补充自动化测试
