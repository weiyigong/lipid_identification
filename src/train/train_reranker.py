"""Training pipeline for the DreaMS cross-attention reranker (v3).

Key changes from v2: candidates sorted by cosine score, cosine scores + ranks
passed to model as prior information.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.data.augment import augment_spectrum, NOISE_PROFILES
from src.models.classical import ClassicalSearcher
from src.utils.indexing import PrecursorIndex


def precompute_noisy_candidate_lists(df, ref_df, dreams_model, dreams_ref_embs,
                                     k=50, precursor_tol=0.5,
                                     noise_profiles=None, seed=42):
    """Augment-first candidate precomputation with DreaMS embeddings.

    Candidates are sorted by cosine score (descending) and cosine scores are stored
    alongside DreaMS embeddings for use as model input features.
    """
    from src.models.dreams_pipeline import _embed_spectra, _normalize

    if noise_profiles is None:
        noise_profiles = [NOISE_PROFILES[k] for k in ("mild", "moderate", "severe", "extreme")]

    searcher = ClassicalSearcher(ref_df, method="cosine", precursor_tol=precursor_tol)
    precursor_index = PrecursorIndex(ref_df)
    ref_names = ref_df["name"].values
    rng = np.random.default_rng(seed)

    all_items = []
    all_mz, all_int, all_prec = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building noisy candidate lists"):
        mz = np.asarray(row["mz_list"], dtype=np.float64)
        ints = np.asarray(row["intensity_list"], dtype=np.float64)
        prec = float(row["precursor_mz"])
        gt_name = row["name"]

        for profile in noise_profiles:
            aug_mz, aug_int, aug_prec = augment_spectrum(mz, ints, prec, profile, rng)

            query = {
                "mz_list": aug_mz,
                "intensity_list": aug_int,
                "precursor_mz": aug_prec,
                "mode": row["mode"],
                "adduct_name": row["adduct_name"],
            }
            results = searcher.search(query, top_k=k)
            if not results:
                continue

            # Map results to ref_df indices with cosine scores
            cand_indices = precursor_index.query(aug_prec, row["mode"], precursor_tol)
            cosine_score_map = {r["name"]: r["score"] for r in results}

            matched = []
            for idx in cand_indices:
                name = ref_names[idx]
                if name in cosine_score_map:
                    matched.append((idx, cosine_score_map[name]))
            if not matched:
                continue

            # Sort by cosine score descending (rank 0 = best)
            matched.sort(key=lambda x: -x[1])
            matched = matched[:k]

            matched_idx = [m[0] for m in matched]
            matched_cos = np.array([m[1] for m in matched], dtype=np.float32)
            labels = np.array(
                [1.0 if ref_names[i] == gt_name else 0.0 for i in matched_idx],
                dtype=np.float32,
            )

            if labels.sum() == 0:
                continue

            # Pad to k candidates (padded entries get cosine_score=0, label=0)
            n_matched = len(matched_idx)
            if n_matched < k:
                matched_idx += [matched_idx[0]] * (k - n_matched)
                matched_cos = np.pad(matched_cos, (0, k - n_matched), constant_values=0.0)
                labels = np.pad(labels, (0, k - n_matched), constant_values=0.0)

            all_items.append({
                "ref_indices": np.array(matched_idx, dtype=np.int64),
                "cosine_scores": matched_cos,
                "labels": labels,
            })
            all_mz.append(aug_mz)
            all_int.append(aug_int)
            all_prec.append(aug_prec)

    if not all_items:
        return []

    # Batch-embed all noisy queries with DreaMS
    print(f"  Embedding {len(all_mz)} noisy queries with DreaMS...")
    query_embs = _normalize(_embed_spectra(
        dreams_model, all_mz, all_int, all_prec,
        batch_size=256, progress_bar=True,
    ))

    candidate_lists = []
    for i, item in enumerate(all_items):
        ref_indices = item["ref_indices"]
        candidate_lists.append({
            "query_dreams_emb": query_embs[i].astype(np.float32),
            "ref_dreams_embs": dreams_ref_embs[ref_indices].astype(np.float32),
            "cosine_scores": item["cosine_scores"],
            "labels": item["labels"],
        })

    return candidate_lists


class RerankerDataset(Dataset):
    """Serves precomputed DreaMS embedding pairs with cosine prior features."""

    def __init__(self, candidate_lists):
        self.candidate_lists = candidate_lists

    def __len__(self):
        return len(self.candidate_lists)

    def __getitem__(self, idx):
        item = self.candidate_lists[idx]
        return (
            torch.from_numpy(item["query_dreams_emb"]),
            torch.from_numpy(item["ref_dreams_embs"]),
            torch.from_numpy(item["cosine_scores"]),
            torch.from_numpy(item["labels"]),
        )


def reranker_collate_fn(batch):
    """Stack precomputed tensors and generate rank indices."""
    query_embs, ref_embs, cosine_scores, labels = zip(*batch)
    K = ref_embs[0].shape[0]
    B = len(query_embs)
    # Ranks are implicit from sorted order: 0, 1, 2, ..., K-1
    cosine_ranks = torch.arange(K, dtype=torch.long).unsqueeze(0).repeat(B, 1)
    return {
        "query_embs": torch.stack(query_embs),
        "ref_embs": torch.stack(ref_embs),
        "cosine_scores": torch.stack(cosine_scores),
        "cosine_ranks": cosine_ranks,
        "labels": torch.stack(labels),
    }


def candidate_softmax_loss(scores, labels):
    """Softmax CE over candidate list — directly optimizes top-1 accuracy."""
    targets = labels.argmax(dim=1)
    return F.cross_entropy(scores, targets)


def _save_checkpoint(path, model, optimizer, scheduler, epoch, best_score, wait, config):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_score": best_score,
        "wait": wait,
        "config": config,
    }, path)


def train_reranker(model, train_candidates, val_candidates,
                   config=None, device=None, checkpoint_dir=None, logger=None,
                   eval_callback=None):
    """Train the DreaMS cross-attention reranker with softmax CE."""
    config = config or {}
    lr = config.get("lr", 3e-4)
    epochs = config.get("epochs", 30)
    batch_size = config.get("batch_size", 128)
    patience = config.get("patience", 7)
    warmup_frac = config.get("warmup_frac", 0.05)
    num_workers = config.get("num_workers", 0)

    def _log(msg):
        if logger is not None:
            logger.print(msg)
        else:
            print(msg)

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    use_amp = device.type == "cuda"
    if use_amp:
        model = torch.compile(model)

    train_ds = RerankerDataset(train_candidates)
    val_ds = RerankerDataset(val_candidates)
    pin = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=reranker_collate_fn, num_workers=num_workers,
        drop_last=True, pin_memory=pin, persistent_workers=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=reranker_collate_fn, num_workers=num_workers,
        pin_memory=pin, persistent_workers=pin,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_frac)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_score = -float("inf")
    best_state = None
    wait = 0
    start_epoch = 0

    # Resume from checkpoint
    if checkpoint_dir is not None:
        latest_path = checkpoint_dir / "reranker_latest.pt"
        if latest_path.exists():
            _log(f"  Resuming from {latest_path}")
            ckpt = torch.load(latest_path, map_location=device, weights_only=False)
            try:
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                start_epoch = ckpt["epoch"] + 1
                best_val_score = ckpt["best_score"]
                wait = ckpt["wait"]
                best_path = checkpoint_dir / "reranker_best.pt"
                if best_path.exists():
                    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                    best_state = {"model": best_ckpt["model"]}
                _log(f"  Resumed at epoch {start_epoch}, best_score={best_val_score:.4f}")
            except (RuntimeError, KeyError):
                _log("  Checkpoint incompatible with current model, starting fresh")
                start_epoch = 0
                best_val_score = -float("inf")

    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1:2d}/{epochs}", leave=False)
        for batch in pbar:
            q = batch["query_embs"].to(device)
            r = batch["ref_embs"].to(device)
            cs = batch["cosine_scores"].to(device)
            cr = batch["cosine_ranks"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                scores = model(q, r, cs, cr)
                loss = candidate_softmax_loss(scores, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())

            if logger is not None and global_step % 10 == 0:
                logger.log_batch({
                    "batch/loss": loss.item(),
                    "batch/lr": scheduler.get_last_lr()[0],
                    "batch/epoch": epoch + 1,
                }, step=global_step)

            global_step += 1

        # Validate
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                q = batch["query_embs"].to(device)
                r = batch["ref_embs"].to(device)
                cs = batch["cosine_scores"].to(device)
                cr = batch["cosine_ranks"].to(device)
                labels = batch["labels"].to(device)

                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    scores = model(q, r, cs, cr)
                    loss = candidate_softmax_loss(scores, labels)

                val_losses.append(loss.item())

                top_idx = scores.argmax(dim=1)
                top_labels = labels.gather(1, top_idx.unsqueeze(1)).squeeze(1)
                val_correct += (top_labels > 0.5).sum().item()
                val_total += q.shape[0]

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = val_correct / max(val_total, 1)

        epoch_msg = (
            f"  Epoch {epoch+1:3d}/{epochs} | "
            f"loss={train_loss:.4f} val_loss={val_loss:.4f} val_top1={val_acc:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )
        _log(epoch_msg)

        if logger is not None:
            logger.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/top1_acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            })

        # Eval callback for full benchmark metrics
        current_score = val_acc
        if eval_callback is not None:
            eval_metrics = eval_callback(model, epoch)
            if eval_metrics:
                if logger is not None:
                    logger.log(eval_metrics, step=global_step)
                composite = eval_metrics.get("eval/composite", val_acc)
                if composite > 0:
                    current_score = composite

        improved = current_score > best_val_score
        if improved:
            best_val_score = current_score
            best_state = {
                "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            }
            wait = 0
        else:
            wait += 1

        if checkpoint_dir is not None:
            _save_checkpoint(
                checkpoint_dir / "reranker_latest.pt",
                model, optimizer, scheduler, epoch, best_val_score, wait, config,
            )
            if improved:
                _save_checkpoint(
                    checkpoint_dir / "reranker_best.pt",
                    model, optimizer, scheduler, epoch, best_val_score, wait, config,
                )

        if wait >= patience:
            _log(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
    return model
