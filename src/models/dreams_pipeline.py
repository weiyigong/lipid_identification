"""Approach A: DreaMS spectrum-to-spectrum retrieval with precursor filtering."""

import numpy as np
import torch

# DreaMS checkpoints contain arbitrary pickled classes (pathlib.PosixPath,
# msml.*, argparse.Namespace, etc.) and pytorch-lightning explicitly passes
# weights_only=True. Force weights_only=False so the checkpoint loads.
# We must patch both torch.load AND lightning_fabric's cached reference.
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

try:
    import lightning_fabric.utilities.cloud_io as _cloud_io
    _cloud_io.torch.load = _patched_torch_load
except ImportError:
    pass

from dreams.api import PreTrainedModel

from src.models.registry import register_model, rank_and_format
from src.utils.indexing import PrecursorIndex

# DreaMS DataFormatA constants
_MAX_MZ = 1000.0
_N_PEAKS = 100
_PREC_INTENS = 1.1


def _preprocess_spectrum(mz, ints, prec_mz):
    """Preprocess one spectrum to DreaMS input format. Returns (101, 2) float32.

    Replicates DreaMS SpectrumPreprocessor: trim to 100 highest peaks,
    pad to 100, relative intensity, prepend precursor, normalize m/z by 1000.
    """
    mz = np.asarray(mz, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)

    # Trim to top-100 by intensity, keeping m/z order
    if len(mz) > _N_PEAKS:
        top_idx = np.argsort(ints)[-_N_PEAKS:]
        top_idx = np.sort(top_idx)
        mz, ints = mz[top_idx], ints[top_idx]

    # Relative intensity
    max_int = ints.max() if len(ints) > 0 else 1.0
    if max_int > 0:
        ints = ints / max_int

    # Pad to 100 peaks
    n = len(mz)
    if n < _N_PEAKS:
        mz = np.pad(mz, (0, _N_PEAKS - n))
        ints = np.pad(ints, (0, _N_PEAKS - n))

    # Prepend precursor peak, then normalize m/z
    out = np.empty((_N_PEAKS + 1, 2), dtype=np.float32)
    out[0, 0] = prec_mz / _MAX_MZ
    out[0, 1] = _PREC_INTENS
    out[1:, 0] = mz / _MAX_MZ
    out[1:, 1] = ints
    return out


def _preprocess_batch(mz_lists, intensity_lists, precursor_mzs):
    """Vectorized preprocessing of N spectra. Returns (N, 101, 2) float32."""
    n = len(mz_lists)
    batch = np.empty((n, _N_PEAKS + 1, 2), dtype=np.float32)
    for i in range(n):
        batch[i] = _preprocess_spectrum(mz_lists[i], intensity_lists[i], precursor_mzs[i])
    return batch


def _embed_spectra(model, mz_lists, intensity_lists, precursor_mzs, batch_size=256,
                   progress_bar=False):
    """Encode spectra with DreaMS. Returns (N, 1024) array.

    Bypasses DreaMS's HDF5/MSData pipeline by preprocessing in-memory
    and calling the model directly.
    """
    from tqdm import tqdm

    device = model.model.device
    dtype = model.model.dtype
    preprocessed = _preprocess_batch(mz_lists, intensity_lists, precursor_mzs)
    n = len(preprocessed)
    all_embs = []

    model.model.eval()
    with torch.inference_mode():
        for start in tqdm(range(0, n, batch_size), desc="DreaMS embedding",
                          total=(n + batch_size - 1) // batch_size,
                          disable=not progress_bar):
            batch = torch.from_numpy(
                preprocessed[start:start + batch_size]
            ).to(device=device, dtype=dtype)
            embs = model.model(batch).cpu().numpy()
            all_embs.append(embs)

    return np.concatenate(all_embs, axis=0)


def _normalize(embs):
    """L2-normalize rows in place, returns normalized copy."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


@register_model("dreams_s2s")
class DreaMSSearcher:
    """Precursor filter → DreaMS cosine ranking. Embeds candidates on the fly."""

    def __init__(self, ref_df, precursor_tol=0.5):
        self.ref_df = ref_df
        self.precursor_tol = precursor_tol
        self.precursor_index = PrecursorIndex(ref_df)
        self._model = PreTrainedModel.from_name("DreaMS_embedding")

    def search(self, query, top_k=10):
        """Single-query search. Compatible with run_benchmark."""
        cand_indices = self.precursor_index.query(
            query["precursor_mz"], query["mode"], self.precursor_tol,
        )
        if len(cand_indices) == 0:
            return []

        # Collect query + candidates into one batch
        mz_lists = [np.asarray(query["mz_list"])]
        int_lists = [np.asarray(query["intensity_list"])]
        prec_mzs = [float(query["precursor_mz"])]

        for idx in cand_indices:
            row = self.ref_df.iloc[idx]
            mz_lists.append(np.asarray(row["mz_list"]))
            int_lists.append(np.asarray(row["intensity_list"]))
            prec_mzs.append(float(row["precursor_mz"]))

        embs = _normalize(_embed_spectra(self._model, mz_lists, int_lists, prec_mzs))
        q_emb = embs[0]
        cand_embs = embs[1:]

        scores = cand_embs @ q_emb
        return rank_and_format(self.ref_df, cand_indices, scores, top_k)

    def batch_search(self, queries, top_k=10, batch_size=128):
        """Batch search: collect all unique candidates, embed once, rank per query."""
        query_cands, unique_cands, cand_to_pos, cand_embs = self._gather_and_embed_candidates(
            queries, batch_size
        )

        # Embed queries
        mz_lists = [np.asarray(q["mz_list"]) for q in queries]
        int_lists = [np.asarray(q["intensity_list"]) for q in queries]
        prec_mzs = [float(q["precursor_mz"]) for q in queries]
        query_embs = _normalize(
            _embed_spectra(self._model, mz_lists, int_lists, prec_mzs, batch_size=batch_size)
        )

        return self._rank_queries(queries, query_embs, query_cands, cand_to_pos, cand_embs, top_k)

    def batch_search_multi_noise(self, eval_sets, top_k=10, batch_size=128):
        """Batch search across multiple noise levels, embedding candidates only once.

        eval_sets: {noise_name: [query_dicts]}  — all queries should share the same
        base precursor m/z values (same sample, different augmentations).
        Returns {noise_name: list[list[dict]]}.
        """
        # Collect candidate sets from ALL noise levels
        all_query_cands = {}
        global_cand_set = set()
        for noise_name, queries in eval_sets.items():
            query_cands = []
            for q in queries:
                cands = self.precursor_index.query(
                    q["precursor_mz"], q["mode"], self.precursor_tol,
                )
                query_cands.append(cands)
                global_cand_set.update(cands.tolist())
            all_query_cands[noise_name] = query_cands

        # Embed all unique candidates once
        unique_cands = sorted(global_cand_set)
        cand_to_pos = {idx: i for i, idx in enumerate(unique_cands)}
        print(f"  Embedding {len(unique_cands)} unique candidates...")

        mz_lists, int_lists, prec_mzs = [], [], []
        for idx in unique_cands:
            row = self.ref_df.iloc[idx]
            mz_lists.append(np.asarray(row["mz_list"]))
            int_lists.append(np.asarray(row["intensity_list"]))
            prec_mzs.append(float(row["precursor_mz"]))
        cand_embs = _normalize(
            _embed_spectra(self._model, mz_lists, int_lists, prec_mzs, batch_size=batch_size)
        )

        # Per noise level: embed only the queries, then rank
        results = {}
        for noise_name, queries in eval_sets.items():
            q_mz = [np.asarray(q["mz_list"]) for q in queries]
            q_int = [np.asarray(q["intensity_list"]) for q in queries]
            q_prec = [float(q["precursor_mz"]) for q in queries]
            print(f"  Embedding {len(queries)} queries for {noise_name}...")
            query_embs = _normalize(
                _embed_spectra(self._model, q_mz, q_int, q_prec, batch_size=batch_size)
            )
            results[noise_name] = self._rank_queries(
                queries, query_embs, all_query_cands[noise_name], cand_to_pos, cand_embs, top_k
            )
        return results

    def _gather_and_embed_candidates(self, queries, batch_size):
        """Precursor-filter and embed all unique candidates for a query set."""
        query_cands = []
        all_cand_set = set()
        for q in queries:
            cands = self.precursor_index.query(
                q["precursor_mz"], q["mode"], self.precursor_tol,
            )
            query_cands.append(cands)
            all_cand_set.update(cands.tolist())

        unique_cands = sorted(all_cand_set)
        cand_to_pos = {idx: i for i, idx in enumerate(unique_cands)}

        mz_lists, int_lists, prec_mzs = [], [], []
        for idx in unique_cands:
            row = self.ref_df.iloc[idx]
            mz_lists.append(np.asarray(row["mz_list"]))
            int_lists.append(np.asarray(row["intensity_list"]))
            prec_mzs.append(float(row["precursor_mz"]))

        cand_embs = _normalize(
            _embed_spectra(self._model, mz_lists, int_lists, prec_mzs, batch_size=batch_size)
        )
        return query_cands, unique_cands, cand_to_pos, cand_embs

    def _rank_queries(self, queries, query_embs, query_cands, cand_to_pos, cand_embs, top_k):
        """Rank precomputed query embeddings against precomputed candidate embeddings."""
        all_results = []
        for i, (q, cands) in enumerate(zip(queries, query_cands)):
            if len(cands) == 0:
                all_results.append([])
                continue
            positions = np.array([cand_to_pos[c] for c in cands])
            scores = cand_embs[positions] @ query_embs[i]
            all_results.append(
                rank_and_format(self.ref_df, cands, scores, top_k)
            )
        return all_results
