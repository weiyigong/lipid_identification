"""DreaMS cross-attention rerankers (v3 + v4).

v3: cosine prior via direct addition of rank embedding + score projection.
v4: gated cosine prior (sigmoid gate) + cosine score dropout during training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.classical import ClassicalSearcher
from src.models.registry import register_model, rank_and_format
from src.utils.indexing import PrecursorIndex


class CrossAttnBlock(nn.Module):
    """Bidirectional cross-attention: query attends to ref, ref attends to query."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_r = nn.LayerNorm(d_model)
        self.cross_q2r = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_r2q = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_q2 = nn.LayerNorm(d_model)
        self.norm_r2 = nn.LayerNorm(d_model)
        self.ffn_q = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_r = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q, r):
        hq = self.norm_q(q)
        hr = self.norm_r(r)
        q_update, _ = self.cross_q2r(hq, hr, hr)
        r_update, _ = self.cross_r2q(hr, hq, hq)

        q = q + q_update
        r = r + r_update
        q = q + self.ffn_q(self.norm_q2(q))
        r = r + self.ffn_r(self.norm_r2(r))
        return q, r


# ---------------------------------------------------------------------------
# v3: direct cosine prior
# ---------------------------------------------------------------------------


class DreaMSReranker(nn.Module):
    """Cross-attention reranker with cosine prior (v3).

    Per-candidate: ref_proj(dreams_emb) + rank_embedding(rank) + score_proj(score).
    Cross-attention enables listwise reasoning across candidates.
    """

    def __init__(self, dreams_dim=1024, d_model=256, n_heads=4, n_cross=3,
                 max_candidates=50, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.query_proj = nn.Sequential(
            nn.Linear(dreams_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.ref_proj = nn.Sequential(
            nn.Linear(dreams_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.rank_embedding = nn.Embedding(max_candidates, d_model)
        self.score_proj = nn.Linear(1, d_model)

        self.cross_layers = nn.ModuleList([
            CrossAttnBlock(d_model, n_heads, dropout) for _ in range(n_cross)
        ])

        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, query_emb, ref_embs, cosine_scores, cosine_ranks):
        q = self.query_proj(query_emb).unsqueeze(1)
        r = self.ref_proj(ref_embs)

        r = r + self.rank_embedding(cosine_ranks)
        r = r + self.score_proj(cosine_scores.unsqueeze(-1))

        for layer in self.cross_layers:
            q, r = layer(q, r)

        return self.score_head(r).squeeze(-1)


# ---------------------------------------------------------------------------
# v4: gated cosine prior + score dropout
# ---------------------------------------------------------------------------


class DreaMSRerankerV4(nn.Module):
    """Cross-attention reranker with gated cosine prior and score dropout (v4).

    Sigmoid gate controls how much cosine info flows into candidate representations.
    Score dropout zeros cosine scores for random samples during training.
    """

    def __init__(self, dreams_dim=1024, d_model=256, n_heads=4, n_cross=3,
                 max_candidates=50, dropout=0.1, score_dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.score_dropout_p = score_dropout

        self.query_proj = nn.Sequential(
            nn.Linear(dreams_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.ref_proj = nn.Sequential(
            nn.Linear(dreams_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.rank_embedding = nn.Embedding(max_candidates, d_model)
        self.score_proj = nn.Linear(1, d_model)

        # Learned gate initialized with bias=2.0 so sigmoid(2)~0.88 — starts trusting cosine
        self.cosine_gate = nn.Linear(d_model, d_model)
        nn.init.constant_(self.cosine_gate.bias, 2.0)

        self.cross_layers = nn.ModuleList([
            CrossAttnBlock(d_model, n_heads, dropout) for _ in range(n_cross)
        ])

        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, query_emb, ref_embs, cosine_scores, cosine_ranks):
        q = self.query_proj(query_emb).unsqueeze(1)
        r = self.ref_proj(ref_embs)

        if self.training and self.score_dropout_p > 0:
            mask = (torch.rand(cosine_scores.shape[0], 1, device=cosine_scores.device)
                    > self.score_dropout_p).float()
            cosine_scores = cosine_scores * mask

        cosine_prior = (self.rank_embedding(cosine_ranks)
                        + self.score_proj(cosine_scores.unsqueeze(-1)))

        gate = torch.sigmoid(self.cosine_gate(r))
        r = r + gate * cosine_prior

        for layer in self.cross_layers:
            q, r = layer(q, r)

        return self.score_head(r).squeeze(-1)


# ---------------------------------------------------------------------------
# Searcher classes (shared search logic, different model types)
# ---------------------------------------------------------------------------


def _reranker_search(searcher, query, top_k):
    """Shared two-stage search logic for both reranker versions."""
    cosine_results = searcher.cosine_searcher.search(query, top_k=searcher.cosine_top_k)
    if not cosine_results:
        return []

    cand_indices = searcher.precursor_index.query(
        query["precursor_mz"], query["mode"], searcher.precursor_tol,
    )
    cosine_score_map = {r["name"]: r["score"] for r in cosine_results}

    matched = []
    for idx in cand_indices:
        name = searcher.ref_df.iloc[idx]["name"]
        if name in cosine_score_map:
            matched.append((idx, cosine_score_map[name]))

    if not matched:
        return cosine_results[:top_k]

    matched.sort(key=lambda x: -x[1])
    matched_indices = [m[0] for m in matched]
    matched_cosine = np.array([m[1] for m in matched], dtype=np.float32)

    q_emb = searcher._embed_query(query)
    ref_embs = searcher.dreams_ref_embs[matched_indices]

    n_total = len(matched_indices)
    q_t = torch.from_numpy(q_emb).float().unsqueeze(0).to(searcher.device)
    r_t = torch.from_numpy(ref_embs).float().unsqueeze(0).to(searcher.device)
    cos_t = torch.from_numpy(matched_cosine).float().unsqueeze(0).to(searcher.device)
    rank_t = torch.arange(n_total, dtype=torch.long).unsqueeze(0).to(searcher.device)
    rank_t = rank_t.clamp(max=searcher.model.rank_embedding.num_embeddings - 1)

    searcher.model.eval()
    with torch.no_grad():
        scores = searcher.model(q_t, r_t, cos_t, rank_t).squeeze(0).cpu().numpy()

    cand_indices_arr = np.array(matched_indices)
    return rank_and_format(
        searcher.ref_df, cand_indices_arr, scores, top_k,
        extra_fields={"cosine_score": matched_cosine},
    )


class _BaseRerankerSearcher:
    """Shared base for reranker searchers."""

    def __init__(self, ref_df, model, dreams_ref_embs, dreams_model,
                 cosine_top_k=50, precursor_tol=0.5, device=None):
        self.ref_df = ref_df
        self.model = model
        self.dreams_ref_embs = dreams_ref_embs
        self.dreams_model = dreams_model
        self.cosine_top_k = cosine_top_k
        self.precursor_tol = precursor_tol
        self.device = device or torch.device("cpu")
        self.cosine_searcher = ClassicalSearcher(
            ref_df, method="cosine", precursor_tol=precursor_tol,
        )
        self.precursor_index = PrecursorIndex(ref_df)

    def _embed_query(self, query):
        from src.models.dreams_pipeline import _embed_spectra, _normalize
        embs = _embed_spectra(
            self.dreams_model,
            [np.asarray(query["mz_list"])],
            [np.asarray(query["intensity_list"])],
            [float(query["precursor_mz"])],
        )
        return _normalize(embs)[0]

    def search(self, query, top_k=10):
        return _reranker_search(self, query, top_k)

    def batch_search(self, queries, top_k=10):
        return [self.search(q, top_k=top_k) for q in queries]

    def batch_search_multi_noise(self, eval_sets, top_k=10):
        results = {}
        for noise_name, queries in eval_sets.items():
            results[noise_name] = self.batch_search(queries, top_k=top_k)
        return results


@register_model("reranker")
class DreaMSRerankerSearcher(_BaseRerankerSearcher):
    """Two-stage search: cosine top-K -> DreaMS cross-attention reranker v3."""
    pass


@register_model("reranker_v4")
class DreaMSRerankerSearcherV4(_BaseRerankerSearcher):
    """Two-stage search: cosine top-K -> DreaMS cross-attention reranker v4."""
    pass
