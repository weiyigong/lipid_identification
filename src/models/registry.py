"""Model registry and shared result formatting."""

import numpy as np

models = {}


def register_model(name):
    """Decorator that registers a Searcher class by name."""
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def get_model(name, **kwargs):
    """Instantiate a registered model by name."""
    if name not in models:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(models.keys())}")
    return models[name](**kwargs)


def rank_and_format(ref_df, cand_indices, scores, top_k, extra_fields=None):
    """Sort by score descending, return top-k result dicts.

    Replaces the 4 private _rank_and_format copies across model files.
    extra_fields: optional dict mapping field_name -> array of per-candidate values.
    """
    n_cand = len(cand_indices)
    if n_cand == 0:
        return []
    k = min(top_k, n_cand)
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    results = []
    for rank, i in enumerate(top_idx, 1):
        row = ref_df.iloc[cand_indices[i]]
        d = {
            "rank": rank,
            "name": row["name"],
            "score": float(scores[i]),
            "n_matches": 0,
            "lipid_class": row["lipid_class"],
            "adduct": row.get("adduct_name", ""),
            "precursor_mz": float(row["precursor_mz"]),
            "n_candidates": n_cand,
        }
        if extra_fields:
            for field_name, arr in extra_fields.items():
                d[field_name] = float(arr[i])
        results.append(d)
    return results
