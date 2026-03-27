"""CLI entry point for TurboQuant operations."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from openclaw_turboquant.context_engine import ContextStore
from openclaw_turboquant.quantizer import TurboQuantMSE, TurboQuantProd


def _default_store_path() -> str:
    """Return the default store path using $OPENCLAW_SESSION_ID if set."""
    session_id = os.environ.get("OPENCLAW_SESSION_ID", "default")
    base = Path(os.environ.get("OPENCLAW_MEMORY_DIR", Path.home() / ".openclaw" / "memory"))
    return str(base / f"turboquant-{session_id}")


def cmd_ingest(args: argparse.Namespace) -> None:
    """Add a single message embedding to the persistent context store."""
    store_path = Path(args.store)
    embedding = np.load(args.embedding).astype(np.float64)
    if embedding.ndim > 1:
        embedding = embedding[0]

    # Load existing store or create a new one
    if store_path.is_dir() and (store_path / "config.json").exists():
        store = ContextStore.load(store_path)
    else:
        if args.dim is None:
            dim = int(embedding.shape[0])
        else:
            dim = args.dim
        store = ContextStore(d=dim, bit_width=args.bit_width, seed=args.seed)

    metadata: dict[str, object] = {}
    if args.metadata:
        metadata = json.loads(args.metadata)

    store.ingest(args.id, embedding, args.text, metadata=metadata)
    store.save(store_path)

    print(json.dumps({
        "action": "ingest",
        "entry_id": args.id,
        "store_size": store.size,
        "store_path": str(store_path),
        "ok": True,
    }))


def cmd_assemble(args: argparse.Namespace) -> None:
    """Retrieve budget-aware context from the persistent store."""
    store_path = Path(args.store)
    if not store_path.is_dir():
        print(json.dumps({"error": f"Store not found: {store_path}"}))
        raise SystemExit(1)

    query = np.load(args.query).astype(np.float64)
    if query.ndim > 1:
        query = query[0]

    store = ContextStore.load(store_path)
    messages = store.assemble_context(query, token_budget=args.token_budget)

    for msg in messages:
        score = msg["metadata"].pop("relevance_score", None)
        entry_id = msg["metadata"].pop("entry_id", None)
        print(json.dumps({
            "role": msg["role"],
            "content": msg["content"],
            "entry_id": entry_id,
            "score": round(score, 6) if score is not None else None,
            **msg["metadata"],
        }, ensure_ascii=False))


def cmd_compact(args: argparse.Namespace) -> None:
    """Remove least-relevant entries from the persistent store."""
    store_path = Path(args.store)
    if not store_path.is_dir():
        print(json.dumps({"error": f"Store not found: {store_path}"}))
        raise SystemExit(1)

    store = ContextStore.load(store_path)
    before = store.size

    query: np.ndarray | None = None
    if args.query:
        query = np.load(args.query).astype(np.float64)
        if query.ndim > 1:
            query = query[0]

    removed = store.compact(keep_ratio=args.keep_ratio, query_embedding=query)
    store.save(store_path)

    print(json.dumps({
        "action": "compact",
        "before": before,
        "after": store.size,
        "removed": removed,
        "store_path": str(store_path),
        "ok": True,
    }))


def cmd_store_info(args: argparse.Namespace) -> None:
    """Display statistics about a persistent context store."""
    store_path = Path(args.store)
    if not store_path.is_dir():
        print(json.dumps({"error": f"Store not found: {store_path}"}))
        raise SystemExit(1)

    store = ContextStore.load(store_path)
    mem = store.memory_estimate_bytes()
    print(json.dumps({
        "path": str(store_path.resolve()),
        "size": store.size,
        "dim": store.d,
        "bit_width": store.bit_width,
        "memory_bytes": mem,
        "memory_kb": round(mem / 1024, 2),
    }))


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run a quick distortion benchmark."""
    d = args.dim
    bit_width = args.bit_width
    n = args.n_vectors
    seed = args.seed

    rng = np.random.default_rng(seed)

    print(f"TurboQuant Benchmark: d={d}, b={bit_width}, n={n}")
    print("=" * 60)

    # Generate random unit vectors
    vectors = rng.standard_normal((n, d))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # MSE quantizer
    t0 = time.perf_counter()
    mse_q = TurboQuantMSE(d, bit_width, seed=seed)
    setup_time = time.perf_counter() - t0
    print(f"MSE setup: {setup_time:.3f}s")

    t0 = time.perf_counter()
    quantized = mse_q.quantize_batch(vectors)
    quant_time = time.perf_counter() - t0

    reconstructed = mse_q.dequantize_batch(quantized)
    mse_errors = np.mean(np.sum((vectors - reconstructed) ** 2, axis=1))
    print(f"MSE quantize: {quant_time:.3f}s ({n / quant_time:.0f} vec/s)")
    print(f"MSE distortion: {mse_errors:.6f}")

    # Inner product quantizer
    if bit_width >= 2:
        t0 = time.perf_counter()
        prod_q = TurboQuantProd(d, bit_width, seed=seed)
        setup_time = time.perf_counter() - t0
        print(f"\nProd setup: {setup_time:.3f}s")

        t0 = time.perf_counter()
        prod_quantized = prod_q.quantize_batch(vectors)
        quant_time = time.perf_counter() - t0

        # Measure inner product distortion
        n_pairs = min(n, 100)
        ip_errors = []
        for i in range(n_pairs):
            j = (i + 1) % n
            true_ip = float(vectors[i] @ vectors[j])
            est_ip = prod_q.estimate_inner_product(vectors[i], prod_quantized[j])
            ip_errors.append((true_ip - est_ip) ** 2)

        print(f"Prod quantize: {quant_time:.3f}s ({n / quant_time:.0f} vec/s)")
        print(f"IP distortion: {np.mean(ip_errors):.6f}")

    # Compression ratio
    original_bits = d * 32
    compressed_bits = d * bit_width + 64  # + norm storage
    ratio = original_bits / compressed_bits
    print(f"\nCompression ratio: {ratio:.1f}x")
    print(f"Bits per vector: {compressed_bits} (from {original_bits})")


def cmd_compress(args: argparse.Namespace) -> None:
    """Compress embeddings from a .npy file."""
    vectors = np.load(args.input)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    d = vectors.shape[1]
    quantizer = TurboQuantProd(d, args.bit_width, seed=args.seed)

    results = quantizer.quantize_batch(vectors)

    # Save compressed
    np.savez_compressed(
        args.output,
        mse_indices=np.array([r.mse_indices for r in results]),
        qjl_signs=np.array([r.qjl_signs for r in results]),
        residual_norms=np.array([r.residual_norm for r in results]),
        norms=np.array([r.norm for r in results]),
        bit_width=args.bit_width,
        d=d,
    )
    print(f"Compressed {len(results)} vectors to {args.output}")


def cmd_retrieve(args: argparse.Namespace) -> None:
    """Retrieve top-k similar entries from a compressed index."""
    query = np.load(args.query)
    if query.ndim > 1:
        query = query[0]

    data = np.load(args.index)
    d = int(data["d"])
    bit_width = int(data["bit_width"])
    quantizer = TurboQuantProd(d, bit_width, seed=args.seed)

    results = []
    for i in range(len(data["norms"])):
        q = type(quantizer.quantize(np.zeros(d)))(
            mse_indices=data["mse_indices"][i],
            qjl_signs=data["qjl_signs"][i],
            residual_norm=float(data["residual_norms"][i]),
            norm=float(data["norms"][i]),
        )
        score = quantizer.estimate_inner_product(query, q)
        results.append({"index": i, "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    for r in results[: args.top_k]:
        print(json.dumps(r))


def main() -> None:
    parser = argparse.ArgumentParser(prog="openclaw-turboquant")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── ingest ────────────────────────────────────────────────────────
    p_ing = sub.add_parser("ingest", help="Add a message embedding to the context store")
    p_ing.add_argument("--store", default=_default_store_path(),
                       help="Path to the context store directory (default: ~/.openclaw/memory/turboquant-$OPENCLAW_SESSION_ID)")
    p_ing.add_argument("--id", required=True, dest="id", help="Unique entry identifier")
    p_ing.add_argument("--text", required=True, help="Raw text content for this entry")
    p_ing.add_argument("--embedding", required=True, help="Path to .npy embedding vector")
    p_ing.add_argument("--dim", type=int, default=None, help="Embedding dimension (required for new stores)")
    p_ing.add_argument("--bit-width", type=int, default=4)
    p_ing.add_argument("--seed", type=int, default=42)
    p_ing.add_argument("--metadata", default=None, help="Optional JSON string of extra metadata")

    # ── assemble ──────────────────────────────────────────────────────
    p_asm = sub.add_parser("assemble", help="Assemble context within a token budget")
    p_asm.add_argument("--store", default=_default_store_path())
    p_asm.add_argument("--query", required=True, help="Path to .npy query embedding")
    p_asm.add_argument("--token-budget", type=int, default=4096)

    # ── compact ───────────────────────────────────────────────────────
    p_cmp = sub.add_parser("compact", help="Remove least-relevant entries from the store")
    p_cmp.add_argument("--store", default=_default_store_path())
    p_cmp.add_argument("--query", default=None, help="Path to .npy query embedding for relevance-based compaction")
    p_cmp.add_argument("--keep-ratio", type=float, default=0.5)

    # ── store-info ────────────────────────────────────────────────────
    p_info = sub.add_parser("store-info", help="Show statistics for a context store")
    p_info.add_argument("--store", default=_default_store_path())

    # ── benchmark ─────────────────────────────────────────────────────
    p_bench = sub.add_parser("benchmark", help="Run distortion benchmark")
    p_bench.add_argument("--dim", type=int, default=128)
    p_bench.add_argument("--bit-width", type=int, default=4)
    p_bench.add_argument("--n-vectors", type=int, default=100)
    p_bench.add_argument("--seed", type=int, default=42)

    # compress
    p_comp = sub.add_parser("compress", help="Compress embeddings")
    p_comp.add_argument("--input", required=True)
    p_comp.add_argument("--output", required=True)
    p_comp.add_argument("--bit-width", type=int, default=4)
    p_comp.add_argument("--seed", type=int, default=42)

    # retrieve
    p_ret = sub.add_parser("retrieve", help="Retrieve similar vectors")
    p_ret.add_argument("--query", required=True)
    p_ret.add_argument("--index", required=True)
    p_ret.add_argument("--top-k", type=int, default=5)
    p_ret.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "assemble":
        cmd_assemble(args)
    elif args.command == "compact":
        cmd_compact(args)
    elif args.command == "store-info":
        cmd_store_info(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "compress":
        cmd_compress(args)
    elif args.command == "retrieve":
        cmd_retrieve(args)


if __name__ == "__main__":
    main()
