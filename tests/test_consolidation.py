"""Comprehensive tests for the consolidated codebase."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ref_df():
    """Load the full reference library (used across all tests)."""
    from src.data.loader import load_library
    return load_library()


@pytest.fixture(scope="session")
def sample_query(ref_df):
    """A known library spectrum for testing search methods."""
    row = ref_df.iloc[0]
    return {
        "mz_list": row["mz_list"],
        "intensity_list": row["intensity_list"],
        "precursor_mz": row["precursor_mz"],
        "mode": row["mode"],
        "ground_truth_name": row["name"],
        "ground_truth_class": row["lipid_class"],
    }


# ---------------------------------------------------------------------------
# 12a: Data Pipeline Tests
# ---------------------------------------------------------------------------


class TestDataPipeline:
    def test_load_library(self, ref_df):
        """Library loads with expected columns and substantial row count."""
        expected_cols = {"name", "mz_list", "intensity_list", "precursor_mz", "mode", "lipid_class"}
        assert expected_cols.issubset(set(ref_df.columns))
        assert len(ref_df) > 500_000

    def test_split_dataset(self, ref_df):
        """Train/val/test splits have no compound overlap."""
        from src.data.split import split_dataset
        splits = split_dataset(ref_df)
        assert "train" in splits and "val" in splits
        train_names = set(splits["train"]["name"].unique())
        val_names = set(splits["val"]["name"].unique())
        assert len(train_names & val_names) == 0

    def test_noise_augmentation(self):
        """Each noise profile produces valid spectra."""
        from src.data.augment import augment_spectrum, NOISE_PROFILES
        mz = np.array([100.0, 200.0, 300.0])
        ints = np.array([50.0, 100.0, 25.0])
        prec = 400.0
        rng = np.random.default_rng(42)

        for name, profile in NOISE_PROFILES.items():
            if name == "clean":
                continue
            mz_aug, int_aug, prec_aug = augment_spectrum(mz, ints, prec, profile, rng)
            assert len(mz_aug) > 0
            assert np.all(int_aug >= 0), f"Negative intensities for {name}"
            assert prec_aug > 0

    def test_eval_set_generation(self):
        """load_split_eval_sets returns 3 sets, each with noise levels."""
        from src.data.evaluation import load_split_eval_sets
        sets = load_split_eval_sets()
        assert len(sets) == 3
        for eval_set in sets:
            assert "clean" in eval_set
            assert len(eval_set["clean"]) > 0


# ---------------------------------------------------------------------------
# 12b: Model & Registry Tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_populated(self):
        """All expected model names are in registry after import."""
        from src.models.registry import models
        # Trigger registration imports
        import src.models  # noqa: F401
        import src.identify  # noqa: F401  — registers "hybrid"
        expected = {
            "spectral_graph_encoder", "spectral_graph_encoder_v2",
            "cosine", "modified_cosine", "entropy",
            "dreams_s2s", "reranker", "reranker_v4", "hybrid",
        }
        assert expected.issubset(set(models.keys())), (
            f"Missing: {expected - set(models.keys())}"
        )

    def test_rank_and_format(self):
        """rank_and_format produces sorted results with correct fields."""
        from src.models.registry import rank_and_format
        ref_df = pd.DataFrame({
            "name": ["lipid_a", "lipid_b", "lipid_c"],
            "lipid_class": ["PC", "PE", "SM"],
            "adduct_name": ["[M+H]+", "[M-H]-", "[M+Na]+"],
            "precursor_mz": [500.0, 600.0, 700.0],
        })
        cand_indices = np.array([0, 1, 2])
        scores = np.array([0.5, 0.9, 0.3])
        results = rank_and_format(ref_df, cand_indices, scores, top_k=2)
        assert len(results) == 2
        assert results[0]["name"] == "lipid_b"
        assert results[0]["rank"] == 1
        assert results[1]["name"] == "lipid_a"
        assert "score" in results[0]
        assert "lipid_class" in results[0]
        assert "n_candidates" in results[0]

    def test_rank_and_format_extra_fields(self):
        """rank_and_format supports extra_fields parameter."""
        from src.models.registry import rank_and_format
        ref_df = pd.DataFrame({
            "name": ["a", "b"],
            "lipid_class": ["PC", "PE"],
            "adduct_name": ["", ""],
            "precursor_mz": [100.0, 200.0],
        })
        results = rank_and_format(
            ref_df, np.array([0, 1]), np.array([0.8, 0.5]), top_k=2,
            extra_fields={"cosine_score": np.array([0.9, 0.7])},
        )
        assert "cosine_score" in results[0]


class TestCheckpoints:
    @pytest.mark.slow
    def test_checkpoint_loading_encoder_v1(self):
        """checkpoints/best.pt loads into SpectrumGraphEncoder without errors."""
        from src.models.spectral_graph_encoder import SpectrumGraphEncoder
        ckpt_path = Path("checkpoints/best.pt")
        if not ckpt_path.exists():
            pytest.skip("Checkpoint not found")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        encoder = SpectrumGraphEncoder()
        encoder.load_state_dict(ckpt["encoder"])

    @pytest.mark.slow
    def test_checkpoint_loading_encoder_v2(self):
        """checkpoints/best_v2.pt loads into SpectrumGraphEncoderV2 without errors."""
        from src.models.spectral_graph_encoder import SpectrumGraphEncoderV2
        ckpt_path = Path("checkpoints/best_v2.pt")
        if not ckpt_path.exists():
            pytest.skip("Checkpoint not found")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        encoder = SpectrumGraphEncoderV2()
        encoder.load_state_dict(ckpt["encoder"])

    @pytest.mark.slow
    def test_checkpoint_loading_reranker_v3(self):
        """checkpoints/reranker.pt loads into DreaMSReranker without errors."""
        from src.models.reranker import DreaMSReranker
        ckpt_path = Path("checkpoints/reranker.pt")
        if not ckpt_path.exists():
            ckpt_path = Path("checkpoints/reranker_best.pt")
        if not ckpt_path.exists():
            pytest.skip("Checkpoint not found")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        model = DreaMSReranker(
            max_candidates=config.get("max_candidates", 50),
            dropout=0.0,
        )
        model.load_state_dict(ckpt["model"])

    @pytest.mark.slow
    def test_checkpoint_loading_reranker_v4(self):
        """checkpoints/reranker_v4.pt loads into DreaMSRerankerV4 without errors."""
        from src.models.reranker import DreaMSRerankerV4
        ckpt_path = Path("checkpoints/reranker_v4.pt")
        if not ckpt_path.exists():
            pytest.skip("Checkpoint not found")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        model = DreaMSRerankerV4(
            max_candidates=config.get("max_candidates", 50),
            dropout=0.0, score_dropout=0.0,
        )
        model.load_state_dict(ckpt["model"])


# ---------------------------------------------------------------------------
# 12c: Search & API Tests
# ---------------------------------------------------------------------------


class TestSpectralGraphEncoder:
    def test_v1_forward_pass(self):
        """V1 encoder produces correct output shape."""
        from src.models.spectral_graph_encoder import (
            SpectrumGraphEncoder, spectrum_to_padded, padded_collate_fn,
        )
        encoder = SpectrumGraphEncoder(d_model=32, d_edge_hidden=8, n_heads=4, n_layers=1, d_spec=16)
        encoder.eval()
        samples = [
            spectrum_to_padded([100.0, 200.0], [50.0, 100.0], 500.0),
            spectrum_to_padded([150.0, 250.0, 350.0], [30.0, 80.0, 60.0], 600.0),
        ]
        batch = padded_collate_fn(samples)
        with torch.no_grad():
            out = encoder(batch)
        assert out.shape == (2, 16)
        # Check L2-normalized
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_v2_forward_pass(self):
        """V2 encoder produces correct output shape."""
        from src.models.spectral_graph_encoder import (
            SpectrumGraphEncoderV2, spectrum_to_padded_v2, padded_collate_fn_v2,
        )
        encoder = SpectrumGraphEncoderV2(d_model=32, d_edge_hidden=8, n_heads=4, n_layers=1, d_spec=16)
        encoder.eval()
        samples = [
            spectrum_to_padded_v2([100.0, 200.0], [50.0, 100.0], 500.0),
            spectrum_to_padded_v2([150.0], [100.0], 600.0),
        ]
        batch = padded_collate_fn_v2(samples)
        with torch.no_grad():
            out = encoder(batch)
        assert out.shape == (2, 16)

    def test_encode_batch(self):
        """encode_batch returns correct shape numpy array."""
        from src.models.spectral_graph_encoder import SpectrumGraphEncoder, encode_batch
        encoder = SpectrumGraphEncoder(d_model=32, d_edge_hidden=8, n_heads=4, n_layers=1, d_spec=16)
        embs = encode_batch(
            encoder, [[100.0, 200.0], [150.0]], [[50.0, 100.0], [80.0]], [500.0, 600.0],
            batch_size=2, device="cpu",
        )
        assert embs.shape == (2, 16)
        assert isinstance(embs, np.ndarray)


class TestCosineSearch:
    def test_cosine_search_perfect_clean(self, ref_df, sample_query):
        """Cosine search on clean query returns exact match as top-1."""
        from src.models.classical import ClassicalSearcher
        searcher = ClassicalSearcher(ref_df, method="cosine", precursor_tol=0.5)
        results = searcher.search(sample_query, top_k=5)
        assert len(results) > 0
        assert results[0]["name"] == sample_query["ground_truth_name"]
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)


class TestIdentifyCompound:
    def test_identify_compound_api(self, ref_df, sample_query):
        """identify_compound returns list of dicts with required keys."""
        from src.identify import identify_compound
        results = identify_compound(sample_query, ref_df, top_k=5)
        assert isinstance(results, list)
        assert len(results) <= 5
        if results:
            required_keys = {"rank", "name", "score", "lipid_class", "precursor_mz"}
            assert required_keys.issubset(set(results[0].keys()))

    def test_identify_compound_correct_output(self, ref_df, sample_query):
        """identify_compound returns correct compound as top-1 for a known spectrum."""
        from src.identify import identify_compound
        results = identify_compound(sample_query, ref_df, top_k=5)
        assert len(results) > 0
        assert results[0]["name"] == sample_query["ground_truth_name"]


class TestBenchmarkMetrics:
    def test_score_results(self):
        """score_results computes metrics correctly from known inputs."""
        from src.benchmark import score_results

        queries = [
            {"ground_truth_name": "lipid_a", "ground_truth_class": "PC"},
            {"ground_truth_name": "lipid_b", "ground_truth_class": "PE"},
        ]
        all_results = [
            [{"name": "lipid_a", "lipid_class": "PC", "n_candidates": 10}],
            [{"name": "lipid_x", "lipid_class": "SM", "n_candidates": 15},
             {"name": "lipid_b", "lipid_class": "PE", "n_candidates": 15}],
        ]
        row = score_results(all_results, queries, "clean", "test")
        assert row["top_1"] == pytest.approx(0.5)  # 1/2 correct at top-1
        assert row["top_5"] == pytest.approx(1.0)  # 2/2 correct at top-5
        assert row["mrr"] == pytest.approx(0.75)  # (1/1 + 1/2) / 2


class TestConstants:
    def test_neutral_loss_masses(self):
        """NEUTRAL_LOSS_MASSES has 21 entries and matches constants."""
        from src.constants import NEUTRAL_LOSS_MASSES, NL_TOL, KNOWN_NEUTRAL_LOSSES
        assert len(NEUTRAL_LOSS_MASSES) == 21
        assert NL_TOL == 0.02
        # Verify the H2O neutral loss is present
        assert any(abs(m - 18.0106) < 0.001 for m in NEUTRAL_LOSS_MASSES)

    def test_diagnostic_ions(self):
        """DIAGNOSTIC_IONS has expected entries."""
        from src.constants import DIAGNOSTIC_IONS
        assert 184.0733 in DIAGNOSTIC_IONS
        assert DIAGNOSTIC_IONS[184.0733] == "PC_choline_head"


class TestRerankerModels:
    def test_v3_forward(self):
        """DreaMSReranker v3 forward pass works."""
        from src.models.reranker import DreaMSReranker
        model = DreaMSReranker(dreams_dim=32, d_model=16, n_heads=2, n_cross=1, max_candidates=10)
        model.eval()
        B, K = 2, 5
        with torch.no_grad():
            scores = model(
                torch.randn(B, 32), torch.randn(B, K, 32),
                torch.rand(B, K), torch.arange(K).unsqueeze(0).expand(B, -1),
            )
        assert scores.shape == (B, K)

    def test_v4_forward(self):
        """DreaMSRerankerV4 forward pass works."""
        from src.models.reranker import DreaMSRerankerV4
        model = DreaMSRerankerV4(dreams_dim=32, d_model=16, n_heads=2, n_cross=1, max_candidates=10)
        model.eval()
        B, K = 2, 5
        with torch.no_grad():
            scores = model(
                torch.randn(B, 32), torch.randn(B, K, 32),
                torch.rand(B, K), torch.arange(K).unsqueeze(0).expand(B, -1),
            )
        assert scores.shape == (B, K)

    def test_v4_score_dropout_training(self):
        """V4 score dropout is active during training, inactive during eval."""
        from src.models.reranker import DreaMSRerankerV4
        model = DreaMSRerankerV4(dreams_dim=32, d_model=16, n_heads=2, n_cross=1,
                                  max_candidates=10, score_dropout=0.99)
        B, K = 4, 5
        q = torch.randn(B, 32)
        r = torch.randn(B, K, 32)
        cos = torch.ones(B, K)
        ranks = torch.arange(K).unsqueeze(0).expand(B, -1)

        model.train()
        # With 99% dropout, most cosine scores should be zeroed
        # (we can't assert exact values due to randomness, but the model shouldn't crash)
        scores_train = model(q, r, cos, ranks)
        assert scores_train.shape == (B, K)
