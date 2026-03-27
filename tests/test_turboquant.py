"""Unit tests for TurboQuant core components."""

from __future__ import annotations

import numpy as np
import pytest

from openclaw_turboquant.codebook import LloydMaxCodebook, beta_pdf
from openclaw_turboquant.context_engine import ContextStore
from openclaw_turboquant.qjl import QJLTransform
from openclaw_turboquant.quantizer import TurboQuantMSE, TurboQuantProd, bits_per_vector
from openclaw_turboquant.rotation import random_rotation_matrix

# ======================================================================
# Beta PDF
# ======================================================================


class TestBetaPdf:
    def test_nonnegative(self) -> None:
        for d in [16, 64, 256]:
            for x in np.linspace(-0.99, 0.99, 50):
                assert beta_pdf(float(x), d) >= 0.0

    def test_zero_outside_interval(self) -> None:
        assert beta_pdf(1.0, 64) == 0.0
        assert beta_pdf(-1.0, 64) == 0.0
        assert beta_pdf(1.5, 64) == 0.0

    def test_integrates_to_one(self) -> None:
        from scipy.integrate import quad

        for d in [16, 64, 128]:
            val, _ = quad(lambda x, _d=d: beta_pdf(x, _d), -1.0, 1.0)
            assert abs(val - 1.0) < 1e-6, f"d={d}: integral={val}"


# ======================================================================
# Codebook
# ======================================================================


class TestLloydMaxCodebook:
    @pytest.fixture
    def cb(self) -> LloydMaxCodebook:
        return LloydMaxCodebook(128, bit_width=2)

    def test_centroid_count(self, cb: LloydMaxCodebook) -> None:
        assert len(cb.centroids) == 4  # 2^2

    def test_centroids_sorted(self, cb: LloydMaxCodebook) -> None:
        assert np.all(np.diff(cb.centroids) > 0)

    def test_centroids_in_range(self, cb: LloydMaxCodebook) -> None:
        assert np.all(cb.centroids >= -1.0)
        assert np.all(cb.centroids <= 1.0)

    def test_boundaries_count(self, cb: LloydMaxCodebook) -> None:
        assert len(cb.boundaries) == 5  # n_levels + 1

    def test_quantize_dequantize_roundtrip(self, cb: LloydMaxCodebook) -> None:
        values = np.array([-0.5, -0.1, 0.0, 0.1, 0.5])
        indices = cb.quantize_scalar(values)
        recovered = cb.dequantize_scalar(indices)
        assert recovered.shape == values.shape
        # Each recovered value should be a centroid
        for v in recovered:
            assert v in cb.centroids

    def test_1bit_codebook_symmetric(self) -> None:
        cb = LloydMaxCodebook(128, bit_width=1)
        assert len(cb.centroids) == 2
        assert abs(cb.centroids[0] + cb.centroids[1]) < 0.01


# ======================================================================
# Random Rotation
# ======================================================================


class TestRandomRotation:
    def test_orthogonality(self) -> None:
        d = 64
        q = random_rotation_matrix(d, rng=np.random.default_rng(42))
        identity = q @ q.T
        np.testing.assert_allclose(identity, np.eye(d), atol=1e-10)

    def test_determinant_one(self) -> None:
        q = random_rotation_matrix(32, rng=np.random.default_rng(0))
        assert abs(abs(np.linalg.det(q)) - 1.0) < 1e-10

    def test_reproducibility(self) -> None:
        q1 = random_rotation_matrix(16, rng=np.random.default_rng(123))
        q2 = random_rotation_matrix(16, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(q1, q2)


# ======================================================================
# QJL Transform
# ======================================================================


class TestQJLTransform:
    @pytest.fixture
    def qjl(self) -> QJLTransform:
        return QJLTransform(64, seed=42)

    def test_quantize_produces_signs(self, qjl: QJLTransform) -> None:
        x = np.random.default_rng(0).standard_normal(64)
        x = x / np.linalg.norm(x)
        result = qjl.quantize(x)
        assert set(np.unique(result.signs)).issubset({-1, 1})

    def test_dequantize_shape(self, qjl: QJLTransform) -> None:
        x = np.random.default_rng(0).standard_normal(64)
        x = x / np.linalg.norm(x)
        result = qjl.quantize(x)
        recon = qjl.dequantize(result)
        assert recon.shape == (64,)

    def test_unbiased_inner_product(self) -> None:
        """QJL should give unbiased inner product estimates on average."""
        d = 256
        rng = np.random.default_rng(42)
        x = rng.standard_normal(d)
        x = x / np.linalg.norm(x)
        y = rng.standard_normal(d)

        true_ip = float(x @ y)

        # Average over many random QJL instances
        estimates = []
        for seed in range(200):
            qjl = QJLTransform(d, seed=seed)
            result = qjl.quantize(x)
            est = qjl.estimate_inner_product(y, result)
            estimates.append(est)

        mean_est = np.mean(estimates)
        # Should be close to true inner product (unbiased)
        assert abs(mean_est - true_ip) < 0.5, (
            f"Bias too large: true={true_ip:.4f}, mean_est={mean_est:.4f}"
        )


# ======================================================================
# TurboQuant MSE
# ======================================================================


class TestTurboQuantMSE:
    @pytest.fixture
    def quantizer(self) -> TurboQuantMSE:
        return TurboQuantMSE(64, bit_width=4, seed=42)

    def test_quantize_dequantize(self, quantizer: TurboQuantMSE) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(64)
        q = quantizer.quantize(x)
        x_hat = quantizer.dequantize(q)
        assert x_hat.shape == x.shape

    def test_mse_decreases_with_bits(self) -> None:
        d = 64
        rng = np.random.default_rng(42)
        x = rng.standard_normal(d)
        x = x / np.linalg.norm(x)

        mse_values = []
        for b in [1, 2, 3, 4]:
            q = TurboQuantMSE(d, b, seed=0)
            qx = q.quantize(x)
            xh = q.dequantize(qx)
            mse = float(np.sum((x - xh) ** 2))
            mse_values.append(mse)

        # MSE should monotonically decrease (or stay same) as bits increase
        for i in range(len(mse_values) - 1):
            assert mse_values[i] >= mse_values[i + 1] - 1e-6

    def test_zero_vector(self, quantizer: TurboQuantMSE) -> None:
        x = np.zeros(64)
        q = quantizer.quantize(x)
        x_hat = quantizer.dequantize(q)
        np.testing.assert_allclose(x_hat, x, atol=1e-10)

    def test_batch(self, quantizer: TurboQuantMSE) -> None:
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        quantized = quantizer.quantize_batch(vectors)
        assert len(quantized) == 10
        recon = quantizer.dequantize_batch(quantized)
        assert recon.shape == (10, 64)

    def test_norm_preservation(self, quantizer: TurboQuantMSE) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(64) * 5.0  # non-unit norm
        q = quantizer.quantize(x)
        x_hat = quantizer.dequantize(q)
        # Norm should be approximately preserved
        assert abs(np.linalg.norm(x_hat) - np.linalg.norm(x)) / np.linalg.norm(x) < 0.5


# ======================================================================
# TurboQuant Prod
# ======================================================================


class TestTurboQuantProd:
    @pytest.fixture
    def quantizer(self) -> TurboQuantProd:
        return TurboQuantProd(64, bit_width=4, seed=42)

    def test_quantize_dequantize(self, quantizer: TurboQuantProd) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(64)
        q = quantizer.quantize(x)
        x_hat = quantizer.dequantize(q)
        assert x_hat.shape == x.shape

    def test_bit_width_minimum(self) -> None:
        with pytest.raises(ValueError, match="bit_width must be >= 2"):
            TurboQuantProd(64, bit_width=1)

    def test_inner_product_estimation(self, quantizer: TurboQuantProd) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(64)
        x = x / np.linalg.norm(x)
        y = rng.standard_normal(64)

        q = quantizer.quantize(x)
        est = quantizer.estimate_inner_product(y, q)
        true_ip = float(x @ y)

        # Should be in the right ballpark
        assert abs(est - true_ip) < 5.0

    def test_zero_vector(self, quantizer: TurboQuantProd) -> None:
        x = np.zeros(64)
        q = quantizer.quantize(x)
        x_hat = quantizer.dequantize(q)
        np.testing.assert_allclose(x_hat, x, atol=1e-10)

    def test_batch(self, quantizer: TurboQuantProd) -> None:
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((5, 64))
        quantized = quantizer.quantize_batch(vectors)
        assert len(quantized) == 5
        recon = quantizer.dequantize_batch(quantized)
        assert recon.shape == (5, 64)


# ======================================================================
# Context Store
# ======================================================================


class TestContextStore:
    @pytest.fixture
    def store(self) -> ContextStore:
        return ContextStore(d=32, bit_width=3, seed=42)

    def test_ingest_and_size(self, store: ContextStore) -> None:
        rng = np.random.default_rng(0)
        store.ingest("msg-1", rng.standard_normal(32), "Hello world")
        store.ingest("msg-2", rng.standard_normal(32), "Foo bar")
        assert store.size == 2

    def test_remove(self, store: ContextStore) -> None:
        rng = np.random.default_rng(0)
        store.ingest("msg-1", rng.standard_normal(32), "Hello")
        assert store.remove("msg-1") is True
        assert store.remove("nonexistent") is False
        assert store.size == 0

    def test_retrieve_top_k(self, store: ContextStore) -> None:
        rng = np.random.default_rng(0)
        for i in range(10):
            emb = rng.standard_normal(32)
            store.ingest(f"msg-{i}", emb, f"Message {i}")

        query = rng.standard_normal(32)
        results = store.retrieve_top_k(query, k=3)
        assert len(results) == 3
        # Scores should be descending
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_assemble_context(self, store: ContextStore) -> None:
        rng = np.random.default_rng(0)
        for i in range(5):
            emb = rng.standard_normal(32)
            store.ingest(f"msg-{i}", emb, f"Short msg {i}")

        query = rng.standard_normal(32)
        messages = store.assemble_context(query, token_budget=100)
        assert len(messages) > 0
        assert all(m["role"] == "context" for m in messages)

    def test_compact(self, store: ContextStore) -> None:
        rng = np.random.default_rng(0)
        for i in range(10):
            store.ingest(f"msg-{i}", rng.standard_normal(32), f"Msg {i}")

        removed = store.compact(keep_ratio=0.3)
        assert removed == 7
        assert store.size == 3

    def test_compact_with_query(self, store: ContextStore) -> None:
        rng = np.random.default_rng(0)
        for i in range(10):
            store.ingest(f"msg-{i}", rng.standard_normal(32), f"Msg {i}")

        query = rng.standard_normal(32)
        removed = store.compact(keep_ratio=0.5, query_embedding=query)
        assert removed == 5
        assert store.size == 5

    def test_memory_estimate(self, store: ContextStore) -> None:
        assert store.memory_estimate_bytes() == 0
        rng = np.random.default_rng(0)
        store.ingest("msg-1", rng.standard_normal(32), "Hello")
        assert store.memory_estimate_bytes() > 0


# ======================================================================
# Utility
# ======================================================================


class TestBitsPerVector:
    def test_mse(self) -> None:
        assert bits_per_vector(128, 4, mode="mse") == 512

    def test_prod(self) -> None:
        assert bits_per_vector(128, 4, mode="prod") == 512

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError):
            bits_per_vector(128, 4, mode="invalid")
