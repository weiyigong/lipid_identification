"""Run all methods across all noise levels and collect results."""

import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.metrics import top_k_accuracy, mean_reciprocal_rank, class_level_accuracy


def run_benchmark(methods, eval_sets, top_k=10, show_progress=True):
    """Benchmark all methods on all noise levels.

    Each method callable takes a query dict and returns a list of result dicts,
    each with at least a "name" key, ordered by rank.

    Returns tidy DataFrame with columns:
        method, noise, top_1, top_5, top_10, mrr, n_queries, mean_query_ms,
        mean_candidates, median_candidates
    """
    rows = []

    for method_name, method_fn in methods.items():
        for noise_name, queries in eval_sets.items():
            ground_truths = [q["ground_truth_name"] for q in queries]
            gt_classes = [q["ground_truth_class"] for q in queries]
            all_preds = []
            all_pred_classes = []
            all_n_cands = []

            t0 = time.perf_counter()
            iterator = tqdm(
                queries,
                desc=f"{method_name}/{noise_name}",
                leave=False,
                disable=not show_progress,
            )
            for q in iterator:
                results = method_fn(q)
                all_preds.append([r["name"] for r in results])
                all_pred_classes.append([r.get("lipid_class", "") for r in results])
                if results and "n_candidates" in results[0]:
                    all_n_cands.append(results[0]["n_candidates"])
            elapsed = time.perf_counter() - t0

            rows.append({
                "method": method_name,
                "noise": noise_name,
                "top_1": top_k_accuracy(all_preds, ground_truths, 1),
                "top_5": top_k_accuracy(all_preds, ground_truths, 5),
                "top_10": top_k_accuracy(all_preds, ground_truths, 10),
                "mrr": mean_reciprocal_rank(all_preds, ground_truths),
                "class_top_1": class_level_accuracy(all_pred_classes, gt_classes, 1),
                "class_top_5": class_level_accuracy(all_pred_classes, gt_classes, 5),
                "n_queries": len(queries),
                "mean_query_ms": elapsed / len(queries) * 1000 if queries else 0,
                "mean_candidates": float(np.mean(all_n_cands)) if all_n_cands else 0,
                "median_candidates": float(np.median(all_n_cands)) if all_n_cands else 0,
            })

            if show_progress:
                r = rows[-1]
                print(
                    f"  {method_name:20s} | {noise_name:10s} | "
                    f"top1={r['top_1']:.3f}  cls1={r['class_top_1']:.3f}  "
                    f"mrr={r['mrr']:.3f}  "
                    f"{r['mean_query_ms']:.1f} ms/query  "
                    f"cands={r['mean_candidates']:.0f}"
                )

    return pd.DataFrame(rows)


def run_benchmark_batch(batch_methods, eval_sets, top_k=10, show_progress=True):
    """Benchmark methods that support batch search.

    batch_methods: {name: callable(queries, top_k) -> list[list[dict]]}
    """
    rows = []

    for method_name, method_fn in batch_methods.items():
        for noise_name, queries in eval_sets.items():
            ground_truths = [q["ground_truth_name"] for q in queries]
            gt_classes = [q["ground_truth_class"] for q in queries]

            t0 = time.perf_counter()
            all_results = method_fn(queries, top_k=top_k)
            elapsed = time.perf_counter() - t0

            all_preds = [[r["name"] for r in res] for res in all_results]
            all_pred_classes = [[r.get("lipid_class", "") for r in res] for res in all_results]
            all_n_cands = [
                res[0]["n_candidates"] for res in all_results
                if res and "n_candidates" in res[0]
            ]

            rows.append({
                "method": method_name,
                "noise": noise_name,
                "top_1": top_k_accuracy(all_preds, ground_truths, 1),
                "top_5": top_k_accuracy(all_preds, ground_truths, 5),
                "top_10": top_k_accuracy(all_preds, ground_truths, 10),
                "mrr": mean_reciprocal_rank(all_preds, ground_truths),
                "class_top_1": class_level_accuracy(all_pred_classes, gt_classes, 1),
                "class_top_5": class_level_accuracy(all_pred_classes, gt_classes, 5),
                "n_queries": len(queries),
                "mean_query_ms": elapsed / len(queries) * 1000 if queries else 0,
                "mean_candidates": float(np.mean(all_n_cands)) if all_n_cands else 0,
                "median_candidates": float(np.median(all_n_cands)) if all_n_cands else 0,
            })

            if show_progress:
                r = rows[-1]
                print(
                    f"  {method_name:20s} | {noise_name:10s} | "
                    f"top1={r['top_1']:.3f}  cls1={r['class_top_1']:.3f}  "
                    f"mrr={r['mrr']:.3f}  "
                    f"{r['mean_query_ms']:.1f} ms/query  "
                    f"cands={r['mean_candidates']:.0f}"
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared benchmark helpers (extracted from individual benchmark scripts)
# ---------------------------------------------------------------------------


def score_results(all_results, queries, noise_name, method_name=""):
    """Compute benchmark metrics from search results."""
    ground_truths = [q["ground_truth_name"] for q in queries]
    gt_classes = [q["ground_truth_class"] for q in queries]
    all_preds = [[r["name"] for r in res] for res in all_results]
    all_pred_classes = [[r.get("lipid_class", "") for r in res] for res in all_results]
    all_n_cands = [
        res[0]["n_candidates"] for res in all_results if res and "n_candidates" in res[0]
    ]
    return {
        "method": method_name,
        "noise": noise_name,
        "top_1": top_k_accuracy(all_preds, ground_truths, 1),
        "top_5": top_k_accuracy(all_preds, ground_truths, 5),
        "top_10": top_k_accuracy(all_preds, ground_truths, 10),
        "mrr": mean_reciprocal_rank(all_preds, ground_truths),
        "class_top_1": class_level_accuracy(all_pred_classes, gt_classes, 1),
        "class_top_5": class_level_accuracy(all_pred_classes, gt_classes, 5),
        "n_queries": len(queries),
        "mean_candidates": float(np.mean(all_n_cands)) if all_n_cands else 0,
        "median_candidates": float(np.median(all_n_cands)) if all_n_cands else 0,
    }


def load_dreams_model():
    """Load DreaMS pretrained model."""
    from dreams.api import PreTrainedModel
    return PreTrainedModel.from_name("DreaMS_embedding")


def load_dreams_ref_embs():
    """Load and normalize cached reference DreaMS embeddings."""
    from pathlib import Path
    from src.models.dreams_pipeline import _normalize
    path = Path("cache/dreams_ref_embeddings.npy")
    embs = np.load(path)
    return _normalize(embs)
