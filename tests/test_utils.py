"""Tests for spectrum utilities, indexing, and metrics."""

import numpy as np
import pandas as pd
import pytest

from src.utils.spectrum import SpectralSimilarity
from src.utils.indexing import PrecursorIndex, EmbeddingIndex
from src.utils.metrics import (
    top_k_accuracy,
    mean_reciprocal_rank,
    per_class_breakdown,
)


def _spec(mz, ints, precursor_mz=500.0, mode="positive"):
    return {
        "mz_list": np.array(mz),
        "intensity_list": np.array(ints),
        "precursor_mz": precursor_mz,
        "mode": mode,
    }


# ── spectrum.py ──

class TestSpectralSimilarity:
    s3 = _spec([100.0, 200.0, 300.0], [50.0, 100.0, 25.0])
    s2a = _spec([100.0, 200.0], [100.0, 50.0])
    s2b = _spec([100.0, 200.0], [50.0, 100.0])
    s_far = _spec([500.0], [100.0])

    @pytest.mark.parametrize("method", ["cosine", "cosine_hungarian", "modified_cosine", "entropy"])
    def test_identical_spectra(self, method):
        scorer = SpectralSimilarity(method)
        score, n_matches = scorer.score(self.s3, self.s3)
        assert score == pytest.approx(1.0, abs=1e-5)
        assert n_matches == 3

    @pytest.mark.parametrize("method", ["cosine", "cosine_hungarian"])
    def test_no_overlap(self, method):
        scorer = SpectralSimilarity(method)
        a = _spec([100.0], [100.0])
        score, n_matches = scorer.score(a, self.s_far)
        assert score == 0.0
        assert n_matches == 0

    @pytest.mark.parametrize("method", ["cosine", "cosine_hungarian"])
    def test_partial_score(self, method):
        scorer = SpectralSimilarity(method)
        score, n_matches = scorer.score(self.s2a, self.s2b)
        assert 0.0 < score < 1.0
        assert n_matches == 2

    def test_neutral_loss_cosine_identical(self):
        scorer = SpectralSimilarity("neutral_loss_cosine")
        score, n_matches = scorer.score(self.s2a, self.s2a)
        assert score == pytest.approx(1.0, abs=1e-6)
        assert n_matches == 2

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            SpectralSimilarity("nonexistent")


# ── indexing.py ──

class TestPrecursorIndex:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "precursor_mz": [100.0, 200.0, 300.0, 150.0, 250.0],
            "mode": ["positive", "positive", "negative", "positive", "negative"],
            "name": list("abcde"),
        })

    def test_finds_within_tolerance(self, sample_df):
        idx = PrecursorIndex(sample_df)
        assert 1 in idx.query(200.0, "positive", tol_da=0.5)

    def test_mode_filtering(self, sample_df):
        idx = PrecursorIndex(sample_df)
        assert len(idx.query(300.0, "positive", tol_da=0.5)) == 0

    def test_tolerance_range(self, sample_df):
        idx = PrecursorIndex(sample_df)
        assert len(idx.query(200.0, "positive", tol_da=100.0)) == 3

    def test_missing_mode(self, sample_df):
        idx = PrecursorIndex(sample_df)
        assert len(idx.query(100.0, "nonexistent", tol_da=1.0)) == 0


class TestEmbeddingIndex:
    def test_build_and_query(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((100, 64)).astype(np.float32)

        idx = EmbeddingIndex()
        idx.build(embeddings, np.arange(100))
        assert idx.ntotal == 100

        results = idx.query(embeddings[0], top_k=5)
        assert len(results) == 5
        assert results[0][0] == 0
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_batch_query(self):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((50, 32)).astype(np.float32)

        idx = EmbeddingIndex()
        idx.build(embeddings)

        results = idx.batch_query(embeddings[:3], top_k=5)
        assert len(results) == 3
        for i, row in enumerate(results):
            assert row[0][0] == i

    def test_save_load(self, tmp_path):
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((20, 16)).astype(np.float32)
        ids = np.arange(100, 120)

        idx = EmbeddingIndex()
        idx.build(embeddings, ids)
        idx.save(tmp_path / "test_index")

        idx2 = EmbeddingIndex()
        idx2.load(tmp_path / "test_index")
        assert idx2.ntotal == 20
        assert idx2.query(embeddings[0], top_k=1)[0][0] == 100


# ── metrics.py ──

class TestMetrics:
    def test_top_k_perfect(self):
        preds = [["a", "b"], ["c", "d"], ["e", "f"]]
        gts = ["a", "c", "e"]
        assert top_k_accuracy(preds, gts, 1) == pytest.approx(1.0)

    def test_top_k_partial(self):
        preds = [["a", "b"], ["x", "c"], ["y", "z"]]
        gts = ["a", "c", "e"]
        assert top_k_accuracy(preds, gts, 1) == pytest.approx(1 / 3)
        assert top_k_accuracy(preds, gts, 2) == pytest.approx(2 / 3)

    def test_top_k_miss(self):
        assert top_k_accuracy([["x"]], ["a"], 1) == 0.0

    def test_mrr_perfect(self):
        assert mean_reciprocal_rank([["a"], ["b"], ["c"]], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_mrr_second_rank(self):
        assert mean_reciprocal_rank([["x", "a"]], ["a"]) == pytest.approx(0.5)

    def test_mrr_miss(self):
        assert mean_reciprocal_rank([["x", "y"]], ["a"]) == 0.0

    def test_per_class_breakdown(self):
        preds = [["a"], ["b"], ["x"], ["d"]]
        gts = ["a", "b", "c", "d"]
        cls = ["PC", "PC", "PE", "PE"]
        df = per_class_breakdown(preds, gts, cls, k=1)
        assert len(df) == 2
        assert df[df["lipid_class"] == "PC"].iloc[0]["top_1_accuracy"] == pytest.approx(1.0)
        assert df[df["lipid_class"] == "PE"].iloc[0]["top_1_accuracy"] == pytest.approx(0.5)

    def test_empty(self):
        assert top_k_accuracy([], [], 1) == 0.0
        assert mean_reciprocal_rank([], []) == 0.0
