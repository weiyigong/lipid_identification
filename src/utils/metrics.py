"""Evaluation metrics for retrieval-based identification."""

import timeit

import numpy as np
import pandas as pd


def top_k_accuracy(predictions, ground_truths, k):
    """Fraction of queries where the ground truth appears in the top-k predictions."""
    if len(predictions) == 0:
        return 0.0
    return float(np.mean([gt in preds[:k] for preds, gt in zip(predictions, ground_truths)]))


def mean_reciprocal_rank(predictions, ground_truths):
    """Mean of 1/rank where the ground truth first appears (0 if not found)."""
    if len(predictions) == 0:
        return 0.0

    def _rr(preds, gt):
        try:
            return 1.0 / (preds.index(gt) + 1)
        except ValueError:
            return 0.0

    return float(np.mean([_rr(p, gt) for p, gt in zip(predictions, ground_truths)]))


def class_level_accuracy(pred_classes, gt_classes, k=1):
    """Fraction of queries where any of the top-k predicted classes matches the GT class."""
    if len(pred_classes) == 0:
        return 0.0
    return float(np.mean([gt in preds[:k] for preds, gt in zip(pred_classes, gt_classes)]))


def per_class_breakdown(predictions, ground_truths, gt_classes, k=1):
    """Top-k accuracy broken down by lipid class."""
    df = pd.DataFrame({
        "lipid_class": gt_classes,
        "hit": [gt in preds[:k] for preds, gt in zip(predictions, ground_truths)],
    })
    result = df.groupby("lipid_class").agg(
        n_queries=("hit", "size"),
        **{f"top_{k}_accuracy": ("hit", "mean")},
    ).sort_values("n_queries", ascending=False).reset_index()
    return result


def query_time_benchmark(method, queries, n_runs=3, max_queries=100):
    """Time method over queries. Returns {mean_ms, std_ms, total_queries}."""
    subset = queries[:max_queries]
    timer = timeit.Timer(lambda: [method(q) for q in subset])
    times = [t / len(subset) * 1000 for t in timer.repeat(n_runs, 1)]
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "total_queries": len(subset),
    }
