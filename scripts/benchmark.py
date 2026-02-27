"""Unified benchmark script: train + evaluate any method.

Usage:
    uv run python scripts/benchmark.py --method spectral_graph_encoder
    uv run python scripts/benchmark.py --method spectral_graph_encoder_v2 --fresh
    uv run python scripts/benchmark.py --method reranker --skip-train
    uv run python scripts/benchmark.py --method reranker_v4 --skip-train
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.loader import load_library
from src.data.split import split_dataset, print_split_stats
from src.data.evaluation import load_split_eval_sets
from src.benchmark import score_results, load_dreams_model, load_dreams_ref_embs
from src.utils.logging import TrainingLogger

CACHE_DIR = Path("cache")
CHECKPOINT_DIR = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cached_embeddings(ref_df, splits):
    """Load precomputed ChemBERTa + optional DreaMS embeddings, sliced by split."""
    dreams_path = CACHE_DIR / "dreams_ref_embeddings.npy"
    chemberta_path = CACHE_DIR / "chemberta_ref_embeddings.npy"

    if not chemberta_path.exists():
        raise FileNotFoundError(
            "ChemBERTa embeddings not found. Run:\n"
            "  uv run python scripts/precompute_embeddings.py"
        )

    mol_all = np.load(chemberta_path)
    assert len(ref_df) == mol_all.shape[0]

    has_dreams = dreams_path.exists()
    if has_dreams:
        dreams_all = np.load(dreams_path)
        assert len(ref_df) == dreams_all.shape[0]
        print(f"Loaded DreaMS: {dreams_all.shape}, ChemBERTa: {mol_all.shape}")
    else:
        dreams_all = None
        print(f"Loaded ChemBERTa: {mol_all.shape} (no DreaMS — distillation disabled)")

    ref_names = ref_df["name"].values
    name_to_idx = {n: i for i, n in enumerate(ref_names)}

    result = {}
    for split_name, split_df in splits.items():
        indices = np.array([name_to_idx[n] for n in split_df["name"].values])
        result[split_name] = {
            "dreams": dreams_all[indices] if has_dreams else None,
            "mol": mol_all[indices],
        }
    return result, mol_all


def run_encoder_benchmark(method, args, ref_df, splits, eval_sets_named, logger):
    """Train and benchmark a spectral graph encoder (V1 or V2)."""
    is_v2 = method == "spectral_graph_encoder_v2"
    train_df, val_df = splits["train"], splits["val"]
    emb_splits, _ = load_cached_embeddings(ref_df, splits)

    if is_v2:
        from src.models.spectral_graph_encoder import SpectrumGraphEncoderV2 as EncoderCls
        from src.models.spectral_graph_encoder import SpectralGraphEncoderSearcherV2 as SearcherCls
        from src.train.train_encoder import train_joint_v2 as train_fn, evaluate_retrieval_v2 as eval_fn
        encoder = EncoderCls(d_model=128, d_edge_hidden=32, n_heads=4, n_layers=2, d_spec=512)
        config = {
            "d_spec": 512, "lr": 3e-4, "epochs": 30, "batch_size": 2048,
            "alpha": 0.5, "beta_0": 0.3, "gamma": 0.1, "temperature": 0.05,
            "patience": 5, "warmup_frac": 0.05, "beta_decay_frac": 0.6,
            "curriculum_warmup": 5, "seed": 42, "num_workers": 8,
        }
    else:
        from src.models.spectral_graph_encoder import SpectrumGraphEncoder as EncoderCls
        from src.models.spectral_graph_encoder import SpectralGraphEncoderSearcher as SearcherCls
        from src.train.train_encoder import train_joint as train_fn, evaluate_retrieval as eval_fn
        encoder = EncoderCls(d_model=256, d_edge_hidden=64, n_heads=8, n_layers=6, d_spec=512)
        config = {
            "d_spec": 512, "lr": 3e-4, "epochs": 30, "batch_size": 2048,
            "alpha": 0.2, "beta_0": 0.3, "gamma": 0.1, "temperature": 0.07,
            "patience": 5, "warmup_frac": 0.05, "beta_decay_frac": 0.6,
            "curriculum_warmup": 5, "seed": 42, "num_workers": 8,
        }

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {n_params:,}")

    def eval_callback(enc, mproj, epoch):
        searcher = SearcherCls(ref_df, enc, precursor_tol=0.5, device=DEVICE)
        searcher.precompute_embeddings(batch_size=512)
        metrics = {"epoch": epoch + 1}
        for label, ev in eval_sets_named.items():
            noise_results = searcher.batch_search_multi_noise(ev, top_k=10)
            for noise_name, results_list in noise_results.items():
                row = score_results(results_list, ev[noise_name], noise_name, method)
                prefix = f"eval/{label}/{noise_name}"
                metrics[f"{prefix}/top1"] = row["top_1"]
                metrics[f"{prefix}/top5"] = row["top_5"]
                metrics[f"{prefix}/mrr"] = row["mrr"]
                metrics[f"{prefix}/cls1"] = row["class_top_1"]
        return metrics

    if not args.skip_train:
        logger.print(f"\n{'=' * 60}")
        logger.print(f"  Joint Multi-Teacher Training ({method})")
        logger.print("=" * 60)
        encoder, mol_proj = train_fn(
            encoder, train_df, val_df,
            mol_embs_train=emb_splits["train"]["mol"],
            mol_embs_val=emb_splits["val"]["mol"],
            dreams_embs_train=emb_splits["train"]["dreams"],
            dreams_embs_val=emb_splits["val"]["dreams"],
            config=config, device=DEVICE, checkpoint_dir=CHECKPOINT_DIR,
            logger=logger, eval_callback=eval_callback,
        )

        suffix = "_v2" if is_v2 else ""
        ckpt_path = CHECKPOINT_DIR / f"custom_encoder{suffix}.pt"
        torch.save({"encoder": encoder.state_dict(), "mol_proj": mol_proj.state_dict(), "config": config}, ckpt_path)
        logger.print(f"\nSaved checkpoint to {ckpt_path}")
        eval_fn(encoder, val_df, DEVICE)
    else:
        suffix = "_v2" if is_v2 else ""
        ckpt_path = CHECKPOINT_DIR / f"best{suffix}.pt"
        if not ckpt_path.exists():
            ckpt_path = CHECKPOINT_DIR / f"custom_encoder{suffix}.pt"
        logger.print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        encoder = encoder.to(DEVICE)

    # Final benchmark
    searcher = SearcherCls(ref_df, encoder, precursor_tol=0.5, device=DEVICE)
    searcher.precompute_embeddings(batch_size=512)
    return searcher


def run_reranker_benchmark(method, args, ref_df, splits, eval_sets_named, logger):
    """Train and benchmark a DreaMS reranker (v3 or v4)."""
    is_v4 = method == "reranker_v4"
    train_df, val_df = splits["train"], splits["val"]

    rng = np.random.default_rng(42)
    n_train = min(args.train_queries, len(train_df))
    n_val = min(args.val_queries, len(val_df))
    train_sample = train_df.iloc[rng.choice(len(train_df), n_train, replace=False)].reset_index(drop=True)
    val_sample = val_df.iloc[rng.choice(len(val_df), n_val, replace=False)].reset_index(drop=True)

    print(f"Loading DreaMS model and reference embeddings...")
    dreams_model = load_dreams_model()
    dreams_ref_embs = load_dreams_ref_embs()

    # Load or precompute candidate lists
    from src.train.train_reranker import precompute_noisy_candidate_lists, train_reranker
    train_cache = CACHE_DIR / "reranker_v3_train_candidates.pkl"
    val_cache = CACHE_DIR / "reranker_v3_val_candidates.pkl"

    if not args.fresh and train_cache.exists() and val_cache.exists():
        print("Loading cached candidate lists...")
        with open(train_cache, "rb") as f:
            train_candidates = pickle.load(f)
        with open(val_cache, "rb") as f:
            val_candidates = pickle.load(f)
    else:
        print("Precomputing noisy candidate lists...")
        train_candidates = precompute_noisy_candidate_lists(
            train_sample, ref_df, dreams_model, dreams_ref_embs, k=args.cosine_top_k, seed=42,
        )
        val_candidates = precompute_noisy_candidate_lists(
            val_sample, ref_df, dreams_model, dreams_ref_embs, k=args.cosine_top_k, seed=123,
        )
        with open(train_cache, "wb") as f:
            pickle.dump(train_candidates, f)
        with open(val_cache, "wb") as f:
            pickle.dump(val_candidates, f)

    # Build model
    if is_v4:
        from src.models.reranker import DreaMSRerankerV4, DreaMSRerankerSearcherV4 as SearcherCls
        model = DreaMSRerankerV4(
            dreams_dim=1024, d_model=256, n_heads=4, n_cross=3,
            max_candidates=args.cosine_top_k, dropout=0.1,
            score_dropout=args.score_dropout,
        )
    else:
        from src.models.reranker import DreaMSReranker, DreaMSRerankerSearcher as SearcherCls
        model = DreaMSReranker(
            dreams_dim=1024, d_model=256, n_heads=4, n_cross=3,
            max_candidates=args.cosine_top_k, dropout=0.1,
        )

    config = {
        "lr": 3e-4, "epochs": 30, "batch_size": 128,
        "patience": 12 if is_v4 else 7,
        "warmup_frac": 0.05, "seed": 42, "num_workers": 4,
        "cosine_top_k": args.cosine_top_k,
        "d_model": 256, "n_heads": 4, "n_cross": 3,
        "max_candidates": args.cosine_top_k,
    }
    if is_v4:
        config["score_dropout"] = args.score_dropout

    def eval_callback(mdl, epoch):
        searcher = SearcherCls(
            ref_df, mdl, dreams_ref_embs, dreams_model,
            cosine_top_k=args.cosine_top_k, precursor_tol=0.5, device=DEVICE,
        )
        metrics = {"epoch": epoch + 1}
        ev = eval_sets_named["0% isomer"]
        for noise_name in ("clean", "moderate", "severe"):
            queries = ev[noise_name]
            results = searcher.batch_search(queries, top_k=10)
            row = score_results(results, queries, noise_name, method)
            prefix = f"eval/0% isomer/{noise_name}"
            metrics[f"{prefix}/top1"] = row["top_1"]
            metrics[f"{prefix}/mrr"] = row["mrr"]
        composite = (
            metrics.get("eval/0% isomer/clean/top1", 0)
            + metrics.get("eval/0% isomer/moderate/mrr", 0)
            + metrics.get("eval/0% isomer/severe/top1", 0)
        ) / 3
        metrics["eval/composite"] = composite
        return metrics

    ckpt_name = "reranker_v4.pt" if is_v4 else "reranker.pt"
    if not args.skip_train:
        logger.print(f"\n{'=' * 60}")
        logger.print(f"  DreaMS Reranker Training ({method})")
        logger.print("=" * 60)
        model = train_reranker(
            model, train_candidates, val_candidates,
            config=config, device=DEVICE, checkpoint_dir=CHECKPOINT_DIR,
            logger=logger, eval_callback=eval_callback,
        )
        torch.save({"model": model.state_dict(), "config": config}, CHECKPOINT_DIR / ckpt_name)
    else:
        ckpt_path = CHECKPOINT_DIR / "reranker_best.pt"
        if not ckpt_path.exists():
            ckpt_path = CHECKPOINT_DIR / ckpt_name
        logger.print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model = model.to(DEVICE)

    searcher = SearcherCls(
        ref_df, model, dreams_ref_embs, dreams_model,
        cosine_top_k=args.cosine_top_k, precursor_tol=0.5, device=DEVICE,
    )
    return searcher


def final_benchmark(searcher, method, eval_sets_named, logger):
    """Run final benchmark across all eval sets and noise levels."""
    logger.print(f"\n{'=' * 60}")
    logger.print(f"  Final Benchmark ({method})")
    logger.print("=" * 60)

    all_rows = []
    for label, ev in eval_sets_named.items():
        logger.print(f"\n  Eval set: {label} ({len(ev['clean'])} queries)")

        t0 = time.perf_counter()
        noise_results = searcher.batch_search_multi_noise(ev, top_k=10)
        elapsed = time.perf_counter() - t0

        n_noise = len(ev)
        n_queries = len(ev["clean"])
        per_query_ms = elapsed / (n_noise * n_queries) * 1000

        for noise_name, results_list in noise_results.items():
            row = score_results(results_list, ev[noise_name], noise_name, method)
            row["mean_query_ms"] = per_query_ms
            row["eval_set"] = label
            all_rows.append(row)
            logger.print(
                f"    {method:25s} | {noise_name:10s} | "
                f"top1={row['top_1']:.3f}  cls1={row['class_top_1']:.3f}  "
                f"mrr={row['mrr']:.3f}"
            )

    combined = pd.DataFrame(all_rows)
    out_path = Path(f"data/benchmark_{method}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    logger.print(f"\nSaved {len(combined)} rows to {out_path}")
    return combined


def parse_args():
    parser = argparse.ArgumentParser(description="Unified benchmark for all methods")
    parser.add_argument(
        "--method", required=True,
        choices=["spectral_graph_encoder", "spectral_graph_encoder_v2", "reranker", "reranker_v4"],
    )
    parser.add_argument("--log-backend", choices=["wandb", "file"], default="file")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Ignore cached data")
    parser.add_argument("--cosine-top-k", type=int, default=50)
    parser.add_argument("--train-queries", type=int, default=5000)
    parser.add_argument("--val-queries", type=int, default=500)
    parser.add_argument("--score-dropout", type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()
    method = args.method
    print(f"Device: {DEVICE}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ref_df = load_library()
    splits = split_dataset(ref_df)
    print_split_stats(splits)

    eval_no_iso, eval_mixed, eval_with_iso = load_split_eval_sets()
    eval_sets_named = {
        "0% isomer": eval_no_iso,
        "50% isomer": eval_mixed,
        "100% isomer": eval_with_iso,
    }

    logger = TrainingLogger(
        backend=args.log_backend,
        project="lipid-identification",
        name=method,
        config={"method": method, "device": str(DEVICE)},
        log_dir=args.log_dir,
        resume=True,
    )

    if method in ("spectral_graph_encoder", "spectral_graph_encoder_v2"):
        searcher = run_encoder_benchmark(method, args, ref_df, splits, eval_sets_named, logger)
    else:
        searcher = run_reranker_benchmark(method, args, ref_df, splits, eval_sets_named, logger)

    combined = final_benchmark(searcher, method, eval_sets_named, logger)
    logger.log_table(f"benchmark_{method}", combined)
    logger.finish()


if __name__ == "__main__":
    main()
