"""Pytest-benchmark tests for TurboQuant performance."""

from __future__ import annotations

import numpy as np
import pytest

from openclaw_turboquant.codebook import LloydMaxCodebook
from openclaw_turboquant.context_engine import ContextStore
from openclaw_turboquant.qjl import QJLTransform
from openclaw_turboquant.quantizer import TurboQuantMSE, TurboQuantProd

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def unit_vectors_64(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal((100, 64))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def unit_vectors_256(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal((100, 256))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ======================================================================
# Codebook construction
# ======================================================================


def test_bench_codebook_construction_d64_b2(benchmark) -> None:  # type: ignore[no-untyped-def]
    benchmark(LloydMaxCodebook, 64, 2)


def test_bench_codebook_construction_d128_b4(benchmark) -> None:  # type: ignore[no-untyped-def]
    benchmark(LloydMaxCodebook, 128, 4)


# ======================================================================
# MSE quantize
# ======================================================================


@pytest.fixture(scope="module")
def mse_q64() -> TurboQuantMSE:
    return TurboQuantMSE(64, bit_width=4, seed=42)


@pytest.fixture(scope="module")
def mse_q256() -> TurboQuantMSE:
    return TurboQuantMSE(256, bit_width=4, seed=42)


def test_bench_mse_quantize_d64(
    benchmark,
    mse_q64: TurboQuantMSE,
    unit_vectors_64: np.ndarray,  # type: ignore[no-untyped-def]
) -> None:
    benchmark(mse_q64.quantize, unit_vectors_64[0])


def test_bench_mse_dequantize_d64(
    benchmark,
    mse_q64: TurboQuantMSE,
    unit_vectors_64: np.ndarray,  # type: ignore[no-untyped-def]
) -> None:
    q = mse_q64.quantize(unit_vectors_64[0])
    benchmark(mse_q64.dequantize, q)


def test_bench_mse_quantize_batch_d64(
    benchmark,
    mse_q64: TurboQuantMSE,
    unit_vectors_64: np.ndarray,  # type: ignore[no-untyped-def]
) -> None:
    benchmark(mse_q64.quantize_batch, unit_vectors_64)


def test_bench_mse_quantize_d256(
    benchmark,
    mse_q256: TurboQuantMSE,
    unit_vectors_256: np.ndarray,  # type: ignore[no-untyped-def]
) -> None:
    benchmark(mse_q256.quantize, unit_vectors_256[0])


# ======================================================================
# Prod quantize
# ======================================================================


@pytest.fixture(scope="module")
def prod_q64() -> TurboQuantProd:
    return TurboQuantProd(64, bit_width=4, seed=42)


def test_bench_prod_quantize_d64(
    benchmark,
    prod_q64: TurboQuantProd,
    unit_vectors_64: np.ndarray,  # type: ignore[no-untyped-def]
) -> None:
    benchmark(prod_q64.quantize, unit_vectors_64[0])


def test_bench_prod_dequantize_d64(
    benchmark,
    prod_q64: TurboQuantProd,
    unit_vectors_64: np.ndarray,  # type: ignore[no-untyped-def]
) -> None:
    q = prod_q64.quantize(unit_vectors_64[0])
    benchmark(prod_q64.dequantize, q)


def test_bench_prod_inner_product_d64(
    benchmark,
    prod_q64: TurboQuantProd,
    unit_vectors_64: np.ndarray,  # type: ignore[no-untyped-def]
) -> None:
    q = prod_q64.quantize(unit_vectors_64[0])
    benchmark(prod_q64.estimate_inner_product, unit_vectors_64[1], q)


# ======================================================================
# QJL
# ======================================================================


@pytest.fixture(scope="module")
def qjl64() -> QJLTransform:
    return QJLTransform(64, seed=42)


def test_bench_qjl_quantize_d64(
    benchmark,
    qjl64: QJLTransform,
    unit_vectors_64: np.ndarray,  # type: ignore[no-untyped-def]
) -> None:
    benchmark(qjl64.quantize, unit_vectors_64[0])


def test_bench_qjl_dequantize_d64(
    benchmark,
    qjl64: QJLTransform,
    unit_vectors_64: np.ndarray,  # type: ignore[no-untyped-def]
) -> None:
    q = qjl64.quantize(unit_vectors_64[0])
    benchmark(qjl64.dequantize, q)


# ======================================================================
# Context Store
# ======================================================================


def test_bench_context_store_ingest(benchmark, rng: np.random.Generator) -> None:  # type: ignore[no-untyped-def]
    store = ContextStore(d=64, bit_width=3, seed=42)
    emb = rng.standard_normal(64)

    def do_ingest() -> None:
        store.ingest(f"msg-{store.size}", emb, "Test message")

    benchmark(do_ingest)


def test_bench_context_store_retrieve(benchmark, rng: np.random.Generator) -> None:  # type: ignore[no-untyped-def]
    store = ContextStore(d=64, bit_width=3, seed=42)
    for i in range(100):
        store.ingest(f"msg-{i}", rng.standard_normal(64), f"Message {i}")

    query = rng.standard_normal(64)
    benchmark(store.retrieve_top_k, query, 10)
