"""Approach C: classical spectral similarity baselines."""

import numpy as np

from src.models.registry import register_model
from src.utils.indexing import PrecursorIndex
from src.utils.spectrum import SpectralSimilarity


def _format_results(ref_df, scored_indices, top_k):
    """Sort by (-score, -n_matches) descending, return top-k result dicts."""
    if len(scored_indices) == 0:
        return []
    indices, scores, n_matches_list = zip(*scored_indices)
    sort_keys = [(-s, -m) for s, m in zip(scores, n_matches_list)]
    order = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])[:top_k]
    n_cand = len(scored_indices)
    results = []
    for rank, i in enumerate(order, 1):
        row = ref_df.iloc[indices[i]]
        results.append({
            "rank": rank,
            "name": row["name"],
            "score": float(scores[i]),
            "n_matches": int(n_matches_list[i]),
            "lipid_class": row["lipid_class"],
            "adduct": row.get("adduct_name", ""),
            "precursor_mz": float(row["precursor_mz"]),
            "n_candidates": n_cand,
        })
    return results


class ClassicalSearcher:
    """Pre-built precursor index + scorer for fast classical search.

    Instantiate once with the reference library, then call .search() per query.
    """

    def __init__(self, ref_df, method="cosine", precursor_tol=0.5,
                 precursor_filter=True, **scorer_kwargs):
        self.ref_df = ref_df
        self.precursor_tol = precursor_tol
        self.precursor_filter = precursor_filter
        self.precursor_index = PrecursorIndex(ref_df) if precursor_filter else None
        self.scorer = SpectralSimilarity(method, **scorer_kwargs)

    def search(self, query, top_k=10):
        """Search reference library for query spectrum."""
        if self.precursor_filter:
            cand_indices = self.precursor_index.query(
                query["precursor_mz"], query["mode"], self.precursor_tol,
            )
        else:
            cand_indices = np.arange(len(self.ref_df))

        if len(cand_indices) == 0:
            return []

        scored = []
        for idx in cand_indices:
            ref = self.ref_df.iloc[idx]
            ref_dict = {
                "mz_list": ref["mz_list"],
                "intensity_list": ref["intensity_list"],
                "precursor_mz": ref["precursor_mz"],
            }
            score, n_matches = self.scorer.score(query, ref_dict)
            scored.append((idx, score, n_matches))

        return _format_results(self.ref_df, scored, top_k)


def cosine_similarity_search(query, ref_df, precursor_index=None,
                             precursor_tol=0.5, top_k=10):
    """Precursor filter → CosineGreedy similarity."""
    if precursor_index is None:
        precursor_index = PrecursorIndex(ref_df)

    cand_indices = precursor_index.query(
        query["precursor_mz"], query["mode"], precursor_tol,
    )
    if len(cand_indices) == 0:
        return []

    scorer = SpectralSimilarity("cosine")
    scored = []
    for idx in cand_indices:
        ref = ref_df.iloc[idx]
        ref_dict = {
            "mz_list": ref["mz_list"],
            "intensity_list": ref["intensity_list"],
            "precursor_mz": ref["precursor_mz"],
        }
        score, n_matches = scorer.score(query, ref_dict)
        scored.append((idx, score, n_matches))

    return _format_results(ref_df, scored, top_k)


def modified_cosine_search(query, ref_df, precursor_index=None,
                           precursor_tol=0.5, top_k=10):
    """Precursor filter → matchms ModifiedCosine."""
    if precursor_index is None:
        precursor_index = PrecursorIndex(ref_df)

    cand_indices = precursor_index.query(
        query["precursor_mz"], query["mode"], precursor_tol,
    )
    if len(cand_indices) == 0:
        return []

    scorer = SpectralSimilarity("modified_cosine")
    scored = []
    for idx in cand_indices:
        ref = ref_df.iloc[idx]
        ref_dict = {
            "mz_list": ref["mz_list"],
            "intensity_list": ref["intensity_list"],
            "precursor_mz": ref["precursor_mz"],
        }
        score, n_matches = scorer.score(query, ref_dict)
        scored.append((idx, score, n_matches))

    return _format_results(ref_df, scored, top_k)


def entropy_similarity_search(query, ref_df, precursor_index=None,
                              precursor_tol=0.5, top_k=10):
    """Precursor filter → entropy-based spectral similarity."""
    if precursor_index is None:
        precursor_index = PrecursorIndex(ref_df)

    cand_indices = precursor_index.query(
        query["precursor_mz"], query["mode"], precursor_tol,
    )
    if len(cand_indices) == 0:
        return []

    scorer = SpectralSimilarity("entropy")
    scored = []
    for idx in cand_indices:
        ref = ref_df.iloc[idx]
        ref_dict = {
            "mz_list": ref["mz_list"],
            "intensity_list": ref["intensity_list"],
            "precursor_mz": ref["precursor_mz"],
        }
        score, n_matches = scorer.score(query, ref_dict)
        scored.append((idx, score, n_matches))

    return _format_results(ref_df, scored, top_k)


# ── Random Forest pairwise features ──

def compute_pairwise_features(query, candidate):
    """10-dim feature vector for a (query, candidate) spectrum pair."""
    qmz = np.asarray(query["mz_list"], dtype=float)
    qint = np.asarray(query["intensity_list"], dtype=float)
    cmz = np.asarray(candidate["mz_list"], dtype=float)
    cint = np.asarray(candidate["intensity_list"], dtype=float)
    qprec = float(query["precursor_mz"])
    cprec = float(candidate["precursor_mz"])

    # 1. Precursor m/z difference
    prec_diff = abs(qprec - cprec)

    # Peak matching within tolerance
    tol = 0.02
    if len(qmz) == 0 or len(cmz) == 0:
        return np.array([prec_diff, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

    diffs = np.abs(qmz[:, None] - cmz[None, :])
    matched_q = np.any(diffs <= tol, axis=1)
    matched_c = np.any(diffs <= tol, axis=0)

    n_shared = int(matched_q.sum())
    frac_q = n_shared / len(qmz)
    frac_c = n_shared / len(cmz)

    # Jaccard on peak bins
    q_bins = set(np.round(qmz / tol).astype(int))
    c_bins = set(np.round(cmz / tol).astype(int))
    union = len(q_bins | c_bins)
    jaccard = len(q_bins & c_bins) / union if union > 0 else 0.0

    # Intensity-weighted matched fraction
    int_weighted = float(qint[matched_q].sum() / qint.sum()) if qint.sum() > 0 else 0.0

    # Max matched intensity
    max_matched = float(qint[matched_q].max()) if matched_q.any() else 0.0
    if qint.max() > 0:
        max_matched /= qint.max()

    # Cosine similarity (quick dot product on matched peaks)
    cosine = 0.0
    if matched_q.any():
        best_match = np.argmin(diffs, axis=1)
        qi = np.sqrt(qint[matched_q])
        ci = np.sqrt(cint[best_match[matched_q]])
        dot = np.dot(qi, ci)
        nq = np.linalg.norm(np.sqrt(qint))
        nc = np.linalg.norm(np.sqrt(cint))
        if nq > 0 and nc > 0:
            cosine = dot / (nq * nc)

    # Neutral loss cosine
    q_nl = np.sort(qprec - qmz)
    c_nl = np.sort(cprec - cmz)
    nl_diffs = np.abs(q_nl[:, None] - c_nl[None, :])
    nl_matched_q = np.any(nl_diffs <= tol, axis=1)
    nl_cosine = 0.0
    if nl_matched_q.any():
        nl_best = np.argmin(nl_diffs, axis=1)
        qi_nl = np.sqrt(qint[np.argsort(qprec - qmz)][nl_matched_q])
        ci_nl = np.sqrt(cint[np.argsort(cprec - cmz)][nl_best[nl_matched_q]])
        dot_nl = np.dot(qi_nl, ci_nl)
        nq_nl = np.linalg.norm(np.sqrt(qint))
        nc_nl = np.linalg.norm(np.sqrt(cint))
        if nq_nl > 0 and nc_nl > 0:
            nl_cosine = dot_nl / (nq_nl * nc_nl)

    # Spectral entropy difference
    from scipy.stats import entropy as _entropy
    qe = _entropy(qint / qint.sum()) if qint.sum() > 0 else 0.0
    ce = _entropy(cint / cint.sum()) if cint.sum() > 0 else 0.0
    entropy_diff = abs(qe - ce)

    return np.array([
        prec_diff, n_shared, frac_q, frac_c,
        cosine, nl_cosine, jaccard,
        int_weighted, max_matched, entropy_diff,
    ], dtype=float)


FEATURE_NAMES = [
    "precursor_mz_diff", "n_shared_peaks", "frac_query_matched", "frac_ref_matched",
    "cosine_sim", "neutral_loss_cosine", "jaccard",
    "intensity_weighted_matches", "max_matched_intensity", "spectral_entropy_diff",
]


class RandomForestSearcher:
    """Train a random forest on pairwise features, then use it to re-rank candidates."""

    def __init__(self, ref_df, precursor_tol=0.5, n_estimators=200, n_jobs=-1, seed=42):
        from sklearn.ensemble import RandomForestClassifier
        self.ref_df = ref_df
        self.precursor_tol = precursor_tol
        self.precursor_index = PrecursorIndex(ref_df)
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=n_jobs, random_state=seed,
        )
        self._fitted = False

    def build_training_data(self, train_queries, n_negatives=5, rng=None):
        """Build (features, labels) from train queries vs reference library."""
        if rng is None:
            rng = np.random.default_rng(42)

        X, y = [], []
        for q in train_queries:
            gt_name = q["ground_truth_name"]
            cand_indices = self.precursor_index.query(
                q["precursor_mz"], q["mode"], self.precursor_tol,
            )
            if len(cand_indices) == 0:
                continue

            # Find positive(s)
            positives = [i for i in cand_indices if self.ref_df.iloc[i]["name"] == gt_name]
            negatives = [i for i in cand_indices if self.ref_df.iloc[i]["name"] != gt_name]

            for idx in positives:
                ref = self.ref_df.iloc[idx]
                feats = compute_pairwise_features(q, {
                    "mz_list": ref["mz_list"],
                    "intensity_list": ref["intensity_list"],
                    "precursor_mz": ref["precursor_mz"],
                })
                X.append(feats)
                y.append(1)

            # Sample negatives
            if len(negatives) > n_negatives:
                neg_sample = rng.choice(negatives, n_negatives, replace=False)
            else:
                neg_sample = negatives
            for idx in neg_sample:
                ref = self.ref_df.iloc[idx]
                feats = compute_pairwise_features(q, {
                    "mz_list": ref["mz_list"],
                    "intensity_list": ref["intensity_list"],
                    "precursor_mz": ref["precursor_mz"],
                })
                X.append(feats)
                y.append(0)

        return np.array(X), np.array(y)

    def fit(self, X, y):
        """Fit the random forest classifier."""
        self.clf.fit(X, y)
        self._fitted = True

    def search(self, query, top_k=10):
        """Score candidates by RF probability and return top-k."""
        assert self._fitted, "Must call fit() before search()"

        cand_indices = self.precursor_index.query(
            query["precursor_mz"], query["mode"], self.precursor_tol,
        )
        if len(cand_indices) == 0:
            return []

        features = []
        for idx in cand_indices:
            ref = self.ref_df.iloc[idx]
            feats = compute_pairwise_features(query, {
                "mz_list": ref["mz_list"],
                "intensity_list": ref["intensity_list"],
                "precursor_mz": ref["precursor_mz"],
            })
            features.append(feats)

        X = np.array(features)
        probs = self.clf.predict_proba(X)[:, 1]

        scored = [(idx, prob, 0) for idx, prob in zip(cand_indices, probs)]
        return _format_results(self.ref_df, scored, top_k)


# ---------------------------------------------------------------------------
# Registry entries for classical methods
# ---------------------------------------------------------------------------


@register_model("cosine")
class CosineSearcher(ClassicalSearcher):
    def __init__(self, ref_df, **kwargs):
        super().__init__(ref_df, method="cosine", **kwargs)


@register_model("modified_cosine")
class ModifiedCosineSearcher(ClassicalSearcher):
    def __init__(self, ref_df, **kwargs):
        super().__init__(ref_df, method="modified_cosine", **kwargs)


@register_model("entropy")
class EntropySearcher(ClassicalSearcher):
    def __init__(self, ref_df, **kwargs):
        super().__init__(ref_df, method="entropy", **kwargs)
