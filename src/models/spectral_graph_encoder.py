"""Dense attention spectral graph encoders (V1 + V2).

V1: 6-dim nodes, 25-dim edges, d_model=256, 6 layers (~5.1M params).
    Checkpoint: checkpoints/best.pt
V2: 13-dim nodes (quantized intensity), 4-dim edges, d_model=128, 2 layers (~0.5M params).
    Checkpoint: checkpoints/best_v2.pt
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import NEUTRAL_LOSS_MASSES, NL_TOL
from src.models.registry import register_model, rank_and_format
from src.utils.indexing import PrecursorIndex

# ---------------------------------------------------------------------------
# V2 feature constants
# ---------------------------------------------------------------------------

# 7 canonical LipidBlast intensity tiers (86.3% of all peaks)
INTENSITY_BINS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00])
N_BINS = len(INTENSITY_BINS)
NODE_DIM_V2 = 13  # log_mz + 7-bin one-hot + rank + mz_ratio + nl_to_precursor + nl_count + is_virtual
EDGE_DIM_V2 = 4   # log_mass_diff + mass_diff_normalized + mass_ratio + is_to_virtual

# ---------------------------------------------------------------------------
# Shared building blocks (identical architecture in V1 and V2)
# ---------------------------------------------------------------------------


class DenseTransformerConv(nn.Module):
    """Dense edge-conditioned multi-head attention matching PyG TransformerConv formula."""

    def __init__(self, d_model, n_heads, d_edge, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        d_out = n_heads * self.d_head

        self.lin_query = nn.Linear(d_model, d_out)
        self.lin_key = nn.Linear(d_model, d_out)
        self.lin_value = nn.Linear(d_model, d_out)
        self.lin_edge = nn.Linear(d_edge, d_out, bias=False)
        self.lin_skip = nn.Linear(d_model, d_out)
        self.dropout = nn.Dropout(dropout)
        self._scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x, edge_attr, attn_mask):
        B, N, _ = x.shape
        H, C = self.n_heads, self.d_head

        Q = self.lin_query(x).view(B, N, H, C)
        K = self.lin_key(x).view(B, N, H, C)
        V = self.lin_value(x).view(B, N, H, C)
        E = self.lin_edge(edge_attr).view(B, N, N, H, C)

        K_plus_E = K[:, None, :, :, :] + E
        alpha = (Q[:, :, None, :, :] * K_plus_E).sum(-1) * self._scale
        alpha = alpha.masked_fill(~attn_mask.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(alpha, dim=2)
        alpha = alpha.nan_to_num(0.0)
        alpha = self.dropout(alpha)

        V_plus_E = V[:, None, :, :, :] + E
        out = (alpha.unsqueeze(-1) * V_plus_E).sum(dim=2)
        out = out.reshape(B, N, H * C)
        return out + self.lin_skip(x)


class TransformerBlock(nn.Module):
    """Pre-norm dense transformer block with residual and FFN."""

    def __init__(self, d_model, n_heads, d_edge, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = DenseTransformerConv(d_model, n_heads, d_edge, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, edge_attr, attn_mask):
        h = self.norm1(x)
        h = self.conv(h, edge_attr, attn_mask)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


def _dual_readout(x, node_mask, is_virtual, attn_gate, output_proj):
    """Shared dual readout: attention-pooled peaks + virtual node -> L2-normalized output."""
    x = x.masked_fill(~node_mask.unsqueeze(-1), 0.0)

    virtual_states = (x * is_virtual.unsqueeze(-1)).sum(dim=1)

    peak_mask = node_mask & ~is_virtual
    attn_logits = attn_gate(x).squeeze(-1)
    attn_logits = attn_logits.masked_fill(~peak_mask, float("-inf"))
    attn_weights = torch.softmax(attn_logits, dim=1)
    pooled_peaks = (attn_weights.unsqueeze(-1) * x).sum(dim=1)

    combined = torch.cat([pooled_peaks, virtual_states], dim=-1)
    out = output_proj(combined)
    return F.normalize(out, p=2, dim=-1)


# ---------------------------------------------------------------------------
# V1: 6-dim node features, 25-dim edge features
# ---------------------------------------------------------------------------


def spectrum_to_padded(mz_list, intensity_list, precursor_mz):
    """Convert a spectrum to dense numpy arrays (V1: 6-dim nodes, 25-dim edges)."""
    mz = np.asarray(mz_list, dtype=np.float64)
    ints = np.asarray(intensity_list, dtype=np.float64)
    prec = float(precursor_mz)

    max_int = ints.max() if len(ints) > 0 and ints.max() > 0 else 1.0
    rel = ints / max_int
    n_peaks = len(mz)
    n_nodes = n_peaks + 1

    x = np.zeros((n_nodes, 6), dtype=np.float32)
    if n_peaks > 0:
        x[:n_peaks, 0] = np.log10(mz + 1)
        x[:n_peaks, 1] = rel
        x[:n_peaks, 2] = np.sqrt(rel)
        x[:n_peaks, 3] = mz / prec if prec > 0 else 0.0
        x[:n_peaks, 4] = np.log10(np.maximum(prec - mz, 0) + 1)

    x[n_peaks] = [np.log10(prec + 1), 1.0, 1.0, 1.0, 0.0, 1.0]

    all_mz = np.append(mz, prec)
    diff = all_mz[None, :] - all_mz[:, None]
    abs_diff = np.abs(diff)

    edge_attr = np.zeros((n_nodes, n_nodes, 25), dtype=np.float32)
    edge_attr[:, :, 0] = np.log10(abs_diff + 1)
    edge_attr[:, :, 1] = np.sign(diff)
    max_mz = np.maximum(all_mz[:, None], all_mz[None, :])
    min_mz = np.minimum(all_mz[:, None], all_mz[None, :])
    edge_attr[:, :, 2] = np.where(max_mz > 0, min_mz / (max_mz + 1e-10), 0.0)
    edge_attr[:, :, 3:24] = (np.abs(abs_diff[:, :, None] - NEUTRAL_LOSS_MASSES[None, None, :]) < NL_TOL).astype(np.float32)
    edge_attr[:, n_peaks, 24] = 1.0

    np.fill_diagonal(edge_attr[:, :, 0], 0.0)
    np.fill_diagonal(edge_attr[:, :, 1], 0.0)
    np.fill_diagonal(edge_attr[:, :, 2], 0.0)

    node_mask = np.ones(n_nodes, dtype=np.bool_)
    is_virtual = np.zeros(n_nodes, dtype=np.bool_)
    is_virtual[n_peaks] = True

    return {
        "x": x,
        "edge_attr": edge_attr,
        "node_mask": node_mask,
        "is_virtual": is_virtual,
        "n_nodes": n_nodes,
    }


def padded_collate_fn(samples):
    """Collate list of V1 spectrum dicts into a batched dict, padding to max N."""
    max_n = max(s["n_nodes"] for s in samples)
    B = len(samples)

    x = np.zeros((B, max_n, 6), dtype=np.float32)
    edge_attr = np.zeros((B, max_n, max_n, 25), dtype=np.float32)
    node_mask = np.zeros((B, max_n), dtype=np.bool_)
    is_virtual = np.zeros((B, max_n), dtype=np.bool_)

    for i, s in enumerate(samples):
        n = s["n_nodes"]
        x[i, :n] = s["x"]
        edge_attr[i, :n, :n] = s["edge_attr"]
        node_mask[i, :n] = s["node_mask"]
        is_virtual[i, :n] = s["is_virtual"]

    return {
        "x": torch.from_numpy(x),
        "edge_attr": torch.from_numpy(edge_attr),
        "node_mask": torch.from_numpy(node_mask),
        "is_virtual": torch.from_numpy(is_virtual),
    }


class SpectrumGraphEncoder(nn.Module):
    """V1 edge-conditioned dense transformer for MS2 spectra (~5.1M params)."""

    def __init__(self, d_model=256, d_edge_hidden=64, n_heads=8,
                 n_layers=6, d_spec=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_spec = d_spec

        self.node_proj = nn.Sequential(
            nn.Linear(6, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(25, d_edge_hidden),
            nn.LayerNorm(d_edge_hidden),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_edge_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.attn_gate = nn.Linear(d_model, 1)
        self.output_proj = nn.Linear(2 * d_model, d_spec)

    def forward(self, batch):
        """Returns L2-normalized (B, d_spec)."""
        x_in = batch["x"]
        ea_in = batch["edge_attr"]
        node_mask = batch["node_mask"]
        is_virtual = batch["is_virtual"]

        attn_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        diag = torch.eye(x_in.shape[1], dtype=torch.bool, device=x_in.device)
        attn_mask = attn_mask & ~diag.unsqueeze(0)

        x = self.node_proj(x_in)
        edge_attr = self.edge_proj(ea_in)

        for layer in self.layers:
            x = layer(x, edge_attr, attn_mask)
        x = self.final_norm(x)

        return _dual_readout(x, node_mask, is_virtual, self.attn_gate, self.output_proj)


def encode_batch(encoder, mz_lists, intensity_lists, precursor_mzs,
                 batch_size=128, device=None):
    """Encode spectra with V1 SpectrumGraphEncoder. Returns (N, d_spec) numpy array."""
    if device is None:
        device = next(encoder.parameters()).device

    encoder.eval()
    n = len(mz_lists)
    all_embs = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            samples = [
                spectrum_to_padded(mz_lists[i], intensity_lists[i], precursor_mzs[i])
                for i in range(start, end)
            ]
            batch = padded_collate_fn(samples)
            batch = {k: v.to(device) for k, v in batch.items()}
            embs = encoder(batch).cpu().numpy()
            all_embs.append(embs)

    return np.concatenate(all_embs, axis=0)


@register_model("spectral_graph_encoder")
class SpectralGraphEncoderSearcher:
    """Precursor filter + V1 graph encoder cosine ranking."""

    def __init__(self, ref_df, encoder, precursor_tol=0.5, device=None):
        self.ref_df = ref_df
        self.encoder = encoder
        self.precursor_tol = precursor_tol
        self.device = device or next(encoder.parameters()).device
        self.precursor_index = PrecursorIndex(ref_df)
        self._ref_embs = None

    def precompute_embeddings(self, batch_size=256, cache_path=None):
        if self._ref_embs is not None:
            return
        if cache_path is not None and cache_path.exists():
            self._ref_embs = np.load(cache_path)
            print(f"Loaded cached embeddings: {self._ref_embs.shape}")
        else:
            self._ref_embs = encode_batch(
                self.encoder,
                self.ref_df["mz_list"].tolist(),
                self.ref_df["intensity_list"].tolist(),
                self.ref_df["precursor_mz"].tolist(),
                batch_size=batch_size, device=self.device,
            )
            if cache_path is not None:
                np.save(cache_path, self._ref_embs)
                print(f"Saved embeddings to {cache_path}")

    def _encode_queries(self, mz_lists, int_lists, prec_mzs, batch_size=128):
        return encode_batch(
            self.encoder, mz_lists, int_lists, prec_mzs,
            batch_size=batch_size, device=self.device,
        )

    def search(self, query, top_k=10):
        cand_indices = self.precursor_index.query(
            query["precursor_mz"], query["mode"], self.precursor_tol,
        )
        if len(cand_indices) == 0:
            return []

        q_emb = self._encode_queries(
            [np.asarray(query["mz_list"])],
            [np.asarray(query["intensity_list"])],
            [float(query["precursor_mz"])],
        )[0]

        scores = self._ref_embs[cand_indices] @ q_emb
        return rank_and_format(self.ref_df, cand_indices, scores, top_k)

    def batch_search(self, queries, top_k=10, batch_size=128):
        query_cands = [
            self.precursor_index.query(q["precursor_mz"], q["mode"], self.precursor_tol)
            for q in queries
        ]
        mz_lists = [np.asarray(q["mz_list"]) for q in queries]
        int_lists = [np.asarray(q["intensity_list"]) for q in queries]
        prec_mzs = [float(q["precursor_mz"]) for q in queries]
        query_embs = self._encode_queries(mz_lists, int_lists, prec_mzs, batch_size)
        return self._rank_queries(query_embs, query_cands, top_k)

    def batch_search_multi_noise(self, eval_sets, top_k=10, batch_size=128):
        all_query_cands = {}
        for noise_name, queries in eval_sets.items():
            all_query_cands[noise_name] = [
                self.precursor_index.query(q["precursor_mz"], q["mode"], self.precursor_tol)
                for q in queries
            ]

        results = {}
        for noise_name, queries in eval_sets.items():
            mz_lists = [np.asarray(q["mz_list"]) for q in queries]
            int_lists = [np.asarray(q["intensity_list"]) for q in queries]
            prec_mzs = [float(q["precursor_mz"]) for q in queries]
            query_embs = self._encode_queries(mz_lists, int_lists, prec_mzs, batch_size)
            results[noise_name] = self._rank_queries(
                query_embs, all_query_cands[noise_name], top_k,
            )
        return results

    def _rank_queries(self, query_embs, query_cands, top_k):
        all_results = []
        for i, cands in enumerate(query_cands):
            if len(cands) == 0:
                all_results.append([])
                continue
            scores = self._ref_embs[cands] @ query_embs[i]
            all_results.append(rank_and_format(self.ref_df, cands, scores, top_k))
        return all_results


# ---------------------------------------------------------------------------
# V2: 13-dim node features (quantized intensity), 4-dim edge features
# ---------------------------------------------------------------------------


def _quantize_intensity(rel_int):
    """Assign each intensity to the nearest canonical bin. Returns bin indices (0-6)."""
    dists = np.abs(rel_int[:, None] - INTENSITY_BINS[None, :])
    return np.argmin(dists, axis=1)


def spectrum_to_padded_v2(mz_list, intensity_list, precursor_mz):
    """Convert a spectrum to dense numpy arrays (V2: 13-dim nodes, 4-dim edges)."""
    mz = np.asarray(mz_list, dtype=np.float64)
    ints = np.asarray(intensity_list, dtype=np.float64)
    prec = float(precursor_mz)

    max_int = ints.max() if len(ints) > 0 and ints.max() > 0 else 1.0
    rel = ints / max_int
    n_peaks = len(mz)
    n_nodes = n_peaks + 1

    x = np.zeros((n_nodes, NODE_DIM_V2), dtype=np.float32)
    if n_peaks > 0:
        x[:n_peaks, 0] = np.log10(mz + 1)

        bin_idx = _quantize_intensity(rel)
        x[np.arange(n_peaks), 1 + bin_idx] = 1.0

        ranks = np.argsort(np.argsort(rel)).astype(np.float32)
        if n_peaks > 1:
            ranks /= (n_peaks - 1)
        x[:n_peaks, 8] = ranks

        x[:n_peaks, 9] = mz / prec if prec > 0 else 0.0
        x[:n_peaks, 10] = np.log10(np.maximum(prec - mz, 0) + 1)

        if n_peaks > 1:
            all_mz_tmp = np.append(mz, prec)
            diff_matrix = np.abs(all_mz_tmp[None, :] - all_mz_tmp[:, None])
            nl_hits = np.any(
                np.abs(diff_matrix[:n_peaks, :n_peaks, None] - NEUTRAL_LOSS_MASSES[None, None, :]) < NL_TOL,
                axis=2,
            )
            np.fill_diagonal(nl_hits, False)
            x[:n_peaks, 11] = nl_hits.sum(axis=1).astype(np.float32)
            max_nl = max(nl_hits.sum(axis=1).max(), 1)
            x[:n_peaks, 11] /= max_nl

    # Virtual node
    x[n_peaks] = 0.0
    x[n_peaks, 0] = np.log10(prec + 1)
    x[n_peaks, 1 + (N_BINS - 1)] = 1.0
    x[n_peaks, 8] = 1.0
    x[n_peaks, 9] = 1.0
    x[n_peaks, 12] = 1.0

    all_mz = np.append(mz, prec)
    diff = all_mz[None, :] - all_mz[:, None]
    abs_diff = np.abs(diff)

    edge_attr = np.zeros((n_nodes, n_nodes, EDGE_DIM_V2), dtype=np.float32)
    edge_attr[:, :, 0] = np.log10(abs_diff + 1)
    edge_attr[:, :, 1] = diff / (prec + 1e-10)
    max_mz = np.maximum(all_mz[:, None], all_mz[None, :])
    min_mz = np.minimum(all_mz[:, None], all_mz[None, :])
    edge_attr[:, :, 2] = np.where(max_mz > 0, min_mz / (max_mz + 1e-10), 0.0)
    edge_attr[:, n_peaks, 3] = 1.0

    for d in range(EDGE_DIM_V2):
        np.fill_diagonal(edge_attr[:, :, d], 0.0)

    node_mask = np.ones(n_nodes, dtype=np.bool_)
    is_virtual = np.zeros(n_nodes, dtype=np.bool_)
    is_virtual[n_peaks] = True

    return {
        "x": x,
        "edge_attr": edge_attr,
        "node_mask": node_mask,
        "is_virtual": is_virtual,
        "n_nodes": n_nodes,
    }


def padded_collate_fn_v2(samples):
    """Collate list of V2 spectrum dicts into a batched dict, padding to max N."""
    max_n = max(s["n_nodes"] for s in samples)
    B = len(samples)

    x = np.zeros((B, max_n, NODE_DIM_V2), dtype=np.float32)
    edge_attr = np.zeros((B, max_n, max_n, EDGE_DIM_V2), dtype=np.float32)
    node_mask = np.zeros((B, max_n), dtype=np.bool_)
    is_virtual = np.zeros((B, max_n), dtype=np.bool_)

    for i, s in enumerate(samples):
        n = s["n_nodes"]
        x[i, :n] = s["x"]
        edge_attr[i, :n, :n] = s["edge_attr"]
        node_mask[i, :n] = s["node_mask"]
        is_virtual[i, :n] = s["is_virtual"]

    return {
        "x": torch.from_numpy(x),
        "edge_attr": torch.from_numpy(edge_attr),
        "node_mask": torch.from_numpy(node_mask),
        "is_virtual": torch.from_numpy(is_virtual),
    }


class SpectrumGraphEncoderV2(nn.Module):
    """V2 edge-conditioned dense transformer for MS2 spectra (~0.5M params)."""

    def __init__(self, d_model=128, d_edge_hidden=32, n_heads=4,
                 n_layers=2, d_spec=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_spec = d_spec

        self.node_proj = nn.Sequential(
            nn.Linear(NODE_DIM_V2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(EDGE_DIM_V2, d_edge_hidden),
            nn.LayerNorm(d_edge_hidden),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_edge_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.attn_gate = nn.Linear(d_model, 1)
        self.output_proj = nn.Linear(2 * d_model, d_spec)

    def forward(self, batch):
        """Returns L2-normalized (B, d_spec)."""
        x_in = batch["x"]
        ea_in = batch["edge_attr"]
        node_mask = batch["node_mask"]
        is_virtual = batch["is_virtual"]

        attn_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        diag = torch.eye(x_in.shape[1], dtype=torch.bool, device=x_in.device)
        attn_mask = attn_mask & ~diag.unsqueeze(0)

        x = self.node_proj(x_in)
        edge_attr = self.edge_proj(ea_in)

        for layer in self.layers:
            x = layer(x, edge_attr, attn_mask)
        x = self.final_norm(x)

        return _dual_readout(x, node_mask, is_virtual, self.attn_gate, self.output_proj)


def encode_batch_v2(encoder, mz_lists, intensity_lists, precursor_mzs,
                    batch_size=128, device=None):
    """Encode spectra with V2 SpectrumGraphEncoderV2. Returns (N, d_spec) numpy array."""
    if device is None:
        device = next(encoder.parameters()).device

    encoder.eval()
    n = len(mz_lists)
    all_embs = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            samples = [
                spectrum_to_padded_v2(mz_lists[i], intensity_lists[i], precursor_mzs[i])
                for i in range(start, end)
            ]
            batch = padded_collate_fn_v2(samples)
            batch = {k: v.to(device) for k, v in batch.items()}
            embs = encoder(batch).cpu().numpy()
            all_embs.append(embs)

    return np.concatenate(all_embs, axis=0)


@register_model("spectral_graph_encoder_v2")
class SpectralGraphEncoderSearcherV2:
    """Precursor filter + V2 graph encoder cosine ranking."""

    def __init__(self, ref_df, encoder, precursor_tol=0.5, device=None):
        self.ref_df = ref_df
        self.encoder = encoder
        self.precursor_tol = precursor_tol
        self.device = device or next(encoder.parameters()).device
        self.precursor_index = PrecursorIndex(ref_df)
        self._ref_embs = None

    def precompute_embeddings(self, batch_size=256, cache_path=None):
        if self._ref_embs is not None:
            return
        if cache_path is not None and cache_path.exists():
            self._ref_embs = np.load(cache_path)
            print(f"Loaded cached V2 embeddings: {self._ref_embs.shape}")
        else:
            self._ref_embs = encode_batch_v2(
                self.encoder,
                self.ref_df["mz_list"].tolist(),
                self.ref_df["intensity_list"].tolist(),
                self.ref_df["precursor_mz"].tolist(),
                batch_size=batch_size, device=self.device,
            )
            if cache_path is not None:
                np.save(cache_path, self._ref_embs)
                print(f"Saved V2 embeddings to {cache_path}")

    def _encode_queries(self, mz_lists, int_lists, prec_mzs, batch_size=128):
        return encode_batch_v2(
            self.encoder, mz_lists, int_lists, prec_mzs,
            batch_size=batch_size, device=self.device,
        )

    def search(self, query, top_k=10):
        cand_indices = self.precursor_index.query(
            query["precursor_mz"], query["mode"], self.precursor_tol,
        )
        if len(cand_indices) == 0:
            return []

        q_emb = self._encode_queries(
            [np.asarray(query["mz_list"])],
            [np.asarray(query["intensity_list"])],
            [float(query["precursor_mz"])],
        )[0]

        scores = self._ref_embs[cand_indices] @ q_emb
        return rank_and_format(self.ref_df, cand_indices, scores, top_k)

    def batch_search(self, queries, top_k=10, batch_size=128):
        query_cands = [
            self.precursor_index.query(q["precursor_mz"], q["mode"], self.precursor_tol)
            for q in queries
        ]
        mz_lists = [np.asarray(q["mz_list"]) for q in queries]
        int_lists = [np.asarray(q["intensity_list"]) for q in queries]
        prec_mzs = [float(q["precursor_mz"]) for q in queries]
        query_embs = self._encode_queries(mz_lists, int_lists, prec_mzs, batch_size)
        return self._rank_queries(query_embs, query_cands, top_k)

    def batch_search_multi_noise(self, eval_sets, top_k=10, batch_size=128):
        all_query_cands = {}
        for noise_name, queries in eval_sets.items():
            all_query_cands[noise_name] = [
                self.precursor_index.query(q["precursor_mz"], q["mode"], self.precursor_tol)
                for q in queries
            ]

        results = {}
        for noise_name, queries in eval_sets.items():
            mz_lists = [np.asarray(q["mz_list"]) for q in queries]
            int_lists = [np.asarray(q["intensity_list"]) for q in queries]
            prec_mzs = [float(q["precursor_mz"]) for q in queries]
            query_embs = self._encode_queries(mz_lists, int_lists, prec_mzs, batch_size)
            results[noise_name] = self._rank_queries(
                query_embs, all_query_cands[noise_name], top_k,
            )
        return results

    def _rank_queries(self, query_embs, query_cands, top_k):
        all_results = []
        for i, cands in enumerate(query_cands):
            if len(cands) == 0:
                all_results.append([])
                continue
            scores = self._ref_embs[cands] @ query_embs[i]
            all_results.append(rank_and_format(self.ref_df, cands, scores, top_k))
        return all_results


# Backward-compat aliases used by existing imports
CustomEncoderSearcher = SpectralGraphEncoderSearcher
CustomEncoderSearcherV2 = SpectralGraphEncoderSearcherV2
