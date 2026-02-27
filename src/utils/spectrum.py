"""Spectrum similarity scoring. Wraps matchms scorers + custom entropy similarity."""

import numpy as np
from matchms import Spectrum
from matchms.similarity import (
    CosineGreedy,
    CosineHungarian,
    ModifiedCosine,
    NeutralLossesCosine,
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import entropy


def to_spectrum(data):
    """Convert a data dict to a matchms Spectrum."""
    metadata = {}
    if "precursor_mz" in data:
        metadata["precursor_mz"] = data["precursor_mz"]
    if "mode" in data:
        metadata["ionmode"] = data["mode"]
    return Spectrum(
        mz=np.asarray(data["mz_list"], dtype=float),
        intensities=np.asarray(data["intensity_list"], dtype=float),
        metadata=metadata,
    )


_MATCHMS_SCORERS = {
    "cosine": CosineGreedy,
    "cosine_hungarian": CosineHungarian,
    "modified_cosine": ModifiedCosine,
    "neutral_loss_cosine": NeutralLossesCosine,
}

_DEFAULT_PARAMS = dict(tolerance=0.02, mz_power=0.0, intensity_power=0.5)


class SpectralSimilarity:
    """Unified spectral similarity scorer.

    Methods: cosine, cosine_hungarian, modified_cosine, neutral_loss_cosine, entropy.
    """

    def __init__(self, method="cosine", **kwargs):
        self.method = method
        if method in _MATCHMS_SCORERS:
            params = {**_DEFAULT_PARAMS, **kwargs}
            self._scorer = _MATCHMS_SCORERS[method](**params)
        elif method == "entropy":
            self._scorer = None
            self._tol = kwargs.get("tolerance", _DEFAULT_PARAMS["tolerance"])
        else:
            raise ValueError(f"Unknown method: {method}. "
                             f"Choose from {list(_MATCHMS_SCORERS) + ['entropy']}")

    def score(self, query, reference):
        """Compute pairwise similarity. Returns (score, n_matches)."""
        if self._scorer is not None:
            result = self._scorer.pair(to_spectrum(query), to_spectrum(reference))
            return float(result["score"]), int(result["matches"])
        return self._entropy_score(query, reference)

    def _entropy_score(self, query, reference):
        """Entropy-based spectral similarity (Li & Bohman, 2021)."""
        mz_a = np.asarray(query["mz_list"], dtype=float)
        mz_b = np.asarray(reference["mz_list"], dtype=float)
        if len(mz_a) == 0 or len(mz_b) == 0:
            return 0.0, 0

        cost = cdist(mz_a.reshape(-1, 1), mz_b.reshape(-1, 1))
        row_ind, col_ind = linear_sum_assignment(cost)
        mask = cost[row_ind, col_ind] <= self._tol
        n_matches = int(mask.sum())
        if not mask.any():
            return 0.0, 0

        int_a = np.asarray(query["intensity_list"], dtype=float)
        int_b = np.asarray(reference["intensity_list"], dtype=float)

        ha = entropy(int_a)
        hb = entropy(int_b)
        if ha + hb == 0:
            return 0.0, n_matches

        merged = np.concatenate([int_a, int_b])
        for ia, ib in zip(row_ind[mask], col_ind[mask]):
            merged[ia] += int_b[ib]
            merged[len(int_a) + ib] = 0
        merged = merged[merged > 0]

        hab = entropy(merged)
        return float(1.0 - (2.0 * hab - ha - hb) / np.log(4)), n_matches
