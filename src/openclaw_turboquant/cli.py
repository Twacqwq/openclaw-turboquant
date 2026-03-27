"""CLI entry point for TurboQuant operations."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from openclaw_turboquant.quantizer import TurboQuantMSE, TurboQuantProd


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

    # benchmark
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

    if args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "compress":
        cmd_compress(args)
    elif args.command == "retrieve":
        cmd_retrieve(args)


if __name__ == "__main__":
    main()
