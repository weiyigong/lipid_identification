"""Public API for lipid identification from MS2 spectra."""

from pathlib import Path

import numpy as np
import torch

from src.models.classical import ClassicalSearcher
from src.models.registry import register_model, rank_and_format
from src.utils.indexing import PrecursorIndex

_RERANKER_CKPT = Path("checkpoints/reranker.pt")
_RERANKER_BEST_CKPT = Path("checkpoints/reranker_best.pt")


def _load_reranker(device=None):
    """Load DreaMS reranker model + dependencies from checkpoint if available."""
    from src.models.reranker import DreaMSReranker

    ckpt_path = _RERANKER_BEST_CKPT if _RERANKER_BEST_CKPT.exists() else _RERANKER_CKPT
    if not ckpt_path.exists():
        return None, None, None

    dreams_ref_path = Path("cache/dreams_ref_embeddings.npy")
    if not dreams_ref_path.exists():
        return None, None, None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = DreaMSReranker(
        dreams_dim=config.get("dreams_dim", 1024),
        d_model=config.get("d_model", 256),
        n_heads=config.get("n_heads", 4),
        n_cross=config.get("n_cross", 3),
        max_candidates=config.get("max_candidates", 50),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    from dreams.api import PreTrainedModel
    from src.models.dreams_pipeline import _normalize
    dreams_model = PreTrainedModel.from_name("DreaMS_embedding")
    dreams_ref_embs = _normalize(np.load(dreams_ref_path))

    return model, dreams_model, dreams_ref_embs


def identify_compound(query, reference_library, top_k=10):
    """Assessment interface. Uses DreaMS reranker if available, falls back to cosine."""
    try:
        model, dreams_model, dreams_ref_embs = _load_reranker()
        if model is not None:
            from src.models.reranker import DreaMSRerankerSearcher
            device = next(model.parameters()).device
            searcher = DreaMSRerankerSearcher(
                reference_library, model, dreams_ref_embs, dreams_model,
                cosine_top_k=50, precursor_tol=0.5, device=device,
            )
            return searcher.search(query, top_k=top_k)
    except Exception:
        pass

    searcher = ClassicalSearcher(reference_library, method="cosine", precursor_tol=0.5)
    return searcher.search(query, top_k=top_k)


@register_model("hybrid")
class HybridSearcher:
    """Two-stage re-ranking: cosine top-N candidates, then blend with encoder scores."""

    def __init__(self, ref_df, encoder, lam=0.5, precursor_tol=0.5,
                 cosine_top_n=50, device=None):
        self.ref_df = ref_df
        self.encoder = encoder
        self.lam = lam
        self.precursor_tol = precursor_tol
        self.cosine_top_n = cosine_top_n
        self.device = device or next(encoder.parameters()).device
        self.precursor_index = PrecursorIndex(ref_df)
        self.cosine_scorer = ClassicalSearcher(
            ref_df, method="cosine", precursor_tol=precursor_tol,
        )
        self._ref_embs = None

    def precompute_embeddings(self, batch_size=256, cache_path=None,
                              encode_batch_fn=None):
        """Precompute encoder embeddings for all references."""
        if self._ref_embs is not None:
            return
        if cache_path is not None and cache_path.exists():
            self._ref_embs = np.load(cache_path)
            return

        if encode_batch_fn is None:
            from src.models.spectral_graph_encoder import encode_batch
            encode_batch_fn = encode_batch

        self._ref_embs = encode_batch_fn(
            self.encoder,
            self.ref_df["mz_list"].tolist(),
            self.ref_df["intensity_list"].tolist(),
            self.ref_df["precursor_mz"].tolist(),
            batch_size=batch_size, device=self.device,
        )
        if cache_path is not None:
            np.save(cache_path, self._ref_embs)

    def search(self, query, top_k=10):
        """Single-query hybrid search."""
        cosine_results = self.cosine_scorer.search(query, top_k=self.cosine_top_n)
        if not cosine_results:
            return []

        cand_names = [r["name"] for r in cosine_results]
        cosine_scores = {r["name"]: r["score"] for r in cosine_results}

        cand_indices = self.precursor_index.query(
            query["precursor_mz"], query["mode"], self.precursor_tol,
        )
        if len(cand_indices) == 0:
            return cosine_results[:top_k]

        cosine_name_set = set(cand_names)
        matched_indices = []
        matched_cosine_scores = []
        for idx in cand_indices:
            name = self.ref_df.iloc[idx]["name"]
            if name in cosine_name_set:
                matched_indices.append(idx)
                matched_cosine_scores.append(cosine_scores.get(name, 0.0))

        if not matched_indices:
            return cosine_results[:top_k]

        matched_indices = np.array(matched_indices)
        cosine_arr = np.array(matched_cosine_scores)

        from src.models.spectral_graph_encoder import encode_batch
        q_emb = encode_batch(
            self.encoder,
            [np.asarray(query["mz_list"])],
            [np.asarray(query["intensity_list"])],
            [float(query["precursor_mz"])],
            batch_size=1, device=self.device,
        )[0]

        encoder_scores = self._ref_embs[matched_indices] @ q_emb

        cos_min, cos_max = cosine_arr.min(), cosine_arr.max()
        if cos_max > cos_min:
            cosine_norm = (cosine_arr - cos_min) / (cos_max - cos_min)
        else:
            cosine_norm = np.ones_like(cosine_arr)

        enc_min, enc_max = encoder_scores.min(), encoder_scores.max()
        if enc_max > enc_min:
            encoder_norm = (encoder_scores - enc_min) / (enc_max - enc_min)
        else:
            encoder_norm = np.ones_like(encoder_scores)

        blended = self.lam * cosine_norm + (1 - self.lam) * encoder_norm

        return rank_and_format(
            self.ref_df, matched_indices, blended, top_k,
            extra_fields={
                "cosine_score": cosine_arr,
                "encoder_score": encoder_scores,
            },
        )

    def batch_search(self, queries, top_k=10, batch_size=128):
        return [self.search(q, top_k=top_k) for q in queries]

    def batch_search_multi_noise(self, eval_sets, top_k=10, batch_size=128):
        results = {}
        for noise_name, queries in eval_sets.items():
            results[noise_name] = self.batch_search(queries, top_k=top_k)
        return results
