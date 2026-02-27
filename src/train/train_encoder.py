"""Joint multi-teacher training for spectral graph encoders (V1 + V2).

Shared: asymmetric_nce_loss, LinearProj, _save_checkpoint, _log_eval_summary.
V1-specific: JointTrainingDataset, joint_collate_fn, train_joint, evaluate_retrieval.
V2-specific: JointTrainingDatasetV2, joint_collate_fn_v2, train_joint_v2, evaluate_retrieval_v2.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.data.augment import augment_spectrum, NOISE_PROFILES

# ---------------------------------------------------------------------------
# Shared components
# ---------------------------------------------------------------------------


def asymmetric_nce_loss(z_anchor, z_positive, temperature=0.07):
    """One-directional InfoNCE: each positive retrieves its anchor."""
    z_anchor = F.normalize(z_anchor, dim=-1)
    z_positive = F.normalize(z_positive, dim=-1)
    sim = z_positive @ z_anchor.T / temperature
    labels = torch.arange(z_positive.shape[0], device=z_positive.device)
    return F.cross_entropy(sim, labels)


class LinearProj(nn.Linear):
    """Single linear projection with Procrustes-style orthogonal init."""

    def reset_parameters(self):
        W = torch.randn(self.out_features, self.in_features)
        U, _, Vt = torch.linalg.svd(W, full_matrices=False)
        self.weight.data = U @ Vt
        if self.bias is not None:
            nn.init.zeros_(self.bias)


def _save_checkpoint(path, encoder, mol_proj, optimizer, scheduler, epoch,
                     best_val_metric, wait, config, spec_proj=None,
                     metric_key="best_val_loss"):
    """Save a training checkpoint to disk."""
    state = {
        "epoch": epoch,
        metric_key: best_val_metric,
        "wait": wait,
        "config": config,
        "encoder": encoder.state_dict(),
        "mol_proj": mol_proj.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if spec_proj is not None:
        state["spec_proj"] = spec_proj.state_dict()
    torch.save(state, path)


def _log_eval_summary(eval_metrics, _log):
    """Print a compact per-set summary of eval metrics."""
    sets_seen = {}
    for k, v in eval_metrics.items():
        if not isinstance(v, float):
            continue
        parts = k.split("/")
        if len(parts) == 4 and parts[0] == "eval":
            set_name, noise, metric = parts[1], parts[2], parts[3]
            sets_seen.setdefault(set_name, {}).setdefault(noise, {})[metric] = v
    for sname, noises in sets_seen.items():
        summaries = []
        for noise in ("clean", "mild", "moderate", "severe", "extreme"):
            if noise in noises:
                m = noises[noise]
                summaries.append(
                    f"{noise}: t1={m.get('top1', 0):.3f} mrr={m.get('mrr', 0):.3f}"
                )
        _log(f"    eval [{sname}] {' | '.join(summaries)}")


def _joint_training_loop(encoder, train_loader, val_loader, train_ds, val_ds,
                         mol_proj, spec_proj, config, device, checkpoint_dir,
                         logger, eval_callback, use_distill, ckpt_prefix="",
                         checkpoint_selection="val_align"):
    """Core joint training loop shared by V1 and V2.

    checkpoint_selection: "val_align" (V1) or "composite" (V2).
    """
    d_spec = config.get("d_spec", 512)
    alpha = config.get("alpha", 0.2)
    beta_0 = config.get("beta_0", 0.3) if use_distill else 0.0
    gamma = config.get("gamma", 0.1)
    temperature = config.get("temperature", 0.07)
    patience = config.get("patience", 5)
    warmup_frac = config.get("warmup_frac", 0.05)
    beta_decay_frac = config.get("beta_decay_frac", 0.6)
    epochs = config.get("epochs", 30)
    use_amp = device.type == "cuda"

    def _log(msg):
        if logger is not None:
            logger.print(msg)
        else:
            print(msg)

    if not use_distill:
        _log("  (DreaMS embeddings not provided — distillation disabled)")

    all_params = list(encoder.parameters()) + list(mol_proj.parameters())
    if spec_proj is not None:
        all_params += list(spec_proj.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=config.get("lr", 3e-4), weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_frac)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_composite = checkpoint_selection == "composite"
    if use_composite:
        best_val_metric = -float("inf")
    else:
        best_val_metric = float("inf")
    best_state = None
    wait = 0
    start_epoch = 0

    metric_key = "best_val_score" if use_composite else "best_val_loss"
    best_ckpt_name = f"best{ckpt_prefix}.pt"
    latest_ckpt_name = f"latest{ckpt_prefix}.pt"

    # Resume from latest checkpoint
    if checkpoint_dir is not None:
        latest_path = checkpoint_dir / latest_ckpt_name
        if latest_path.exists():
            _log(f"  Resuming from {latest_path}")
            ckpt = torch.load(latest_path, map_location=device, weights_only=False)
            encoder.load_state_dict(ckpt["encoder"])
            mol_proj.load_state_dict(ckpt["mol_proj"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if spec_proj is not None and "spec_proj" in ckpt:
                spec_proj.load_state_dict(ckpt["spec_proj"])
            start_epoch = ckpt["epoch"] + 1
            best_val_metric = ckpt.get(metric_key, best_val_metric)
            wait = ckpt["wait"]
            best_path = checkpoint_dir / best_ckpt_name
            if best_path.exists():
                best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                best_state = {
                    "encoder": best_ckpt["encoder"],
                    "mol_proj": best_ckpt["mol_proj"],
                }
            _log(f"  Resumed at epoch {start_epoch}, {metric_key}={best_val_metric:.6f}, wait={wait}")

    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, epochs):
        train_ds.set_epoch(epoch)
        val_ds.set_epoch(epoch)

        frac = epoch / max(epochs - 1, 1)
        beta = max(0.0, beta_0 * (1 - frac / beta_decay_frac))

        encoder.train()
        mol_proj.train()
        if spec_proj is not None:
            spec_proj.train()
        train_losses = []
        train_align_losses = []
        train_contrastive_losses = []
        train_distill_losses = []
        train_orth_losses = []

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1:2d}/{epochs}", leave=False)
        for batch1, batch2, mol_target, dreams_target in pbar:
            batch1 = {k: v.to(device) for k, v in batch1.items()}
            batch2 = {k: v.to(device) for k, v in batch2.items()}
            mol_target = mol_target.to(device)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                z1 = encoder(batch1)
                z2 = encoder(batch2)

                loss_align = 0.5 * (F.mse_loss(mol_proj(z1), mol_target)
                                    + F.mse_loss(mol_proj(z2), mol_target))
                loss_contrastive = asymmetric_nce_loss(z1, z2, temperature)

                if use_distill and beta > 0:
                    dreams_target = dreams_target.to(device)
                    loss_distill = F.mse_loss(spec_proj(z1), dreams_target)
                else:
                    loss_distill = torch.zeros(1, device=device)

                W = mol_proj.weight
                WTW = W.T @ W
                loss_orth = torch.norm(WTW - torch.eye(W.shape[1], device=device))

                loss = loss_align + alpha * loss_contrastive + beta * loss_distill + gamma * loss_orth

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(loss.item())
            train_align_losses.append(loss_align.item())
            train_contrastive_losses.append(loss_contrastive.item())
            train_distill_losses.append(loss_distill.item())
            train_orth_losses.append(loss_orth.item())
            pbar.set_postfix(loss=loss.item(), align=loss_align.item())

            if logger is not None and global_step % 10 == 0:
                logger.log_batch({
                    "batch/loss": loss.item(),
                    "batch/align": loss_align.item(),
                    "batch/contrastive": loss_contrastive.item(),
                    "batch/distill": loss_distill.item(),
                    "batch/orth": loss_orth.item(),
                    "batch/lr": scheduler.get_last_lr()[0],
                    "batch/epoch": epoch + 1,
                }, step=global_step)

            global_step += 1

        # Validate
        encoder.eval()
        mol_proj.eval()
        val_align_losses = []
        with torch.no_grad():
            for batch1, batch2, mol_target, dreams_target in val_loader:
                batch1 = {k: v.to(device) for k, v in batch1.items()}
                mol_target = mol_target.to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    z1 = encoder(batch1)
                    mapped_mol = mol_proj(z1)
                val_align_losses.append(F.mse_loss(mapped_mol, mol_target).item())

        train_loss = np.mean(train_losses)
        train_align = np.mean(train_align_losses)
        val_align = np.mean(val_align_losses)
        current_lr = scheduler.get_last_lr()[0]

        epoch_msg = (
            f"  Epoch {epoch+1:3d}/{epochs} | "
            f"loss={train_loss:.4f} align={train_align:.4f} val_align={val_align:.4f} | "
            f"beta={beta:.3f} lr={current_lr:.2e}"
        )
        _log(epoch_msg)
        if logger is not None:
            logger.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/align": train_align,
                "train/contrastive": np.mean(train_contrastive_losses),
                "train/distill": np.mean(train_distill_losses),
                "train/orth": np.mean(train_orth_losses),
                "val/align": val_align,
                "beta": beta,
                "lr": current_lr,
            })

        # Eval callback
        composite_score = 0.0
        if eval_callback is not None:
            eval_metrics = eval_callback(encoder, mol_proj, epoch)
            if eval_metrics:
                if logger is not None:
                    logger.log(eval_metrics, step=global_step)
                _log_eval_summary(eval_metrics, _log)
                if use_composite:
                    composite_score = _compute_composite_score(eval_metrics)
                    _log(f"    composite_score={composite_score:.4f} (best={best_val_metric:.4f})")

        # Checkpoint selection
        if use_composite and eval_callback is not None and composite_score > 0:
            current_metric = composite_score
            improved = current_metric > best_val_metric
        else:
            current_metric = val_align
            improved = current_metric < best_val_metric

        if improved:
            best_val_metric = current_metric
            best_state = {
                "encoder": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                "mol_proj": {k: v.cpu().clone() for k, v in mol_proj.state_dict().items()},
            }
            wait = 0
        else:
            wait += 1

        if checkpoint_dir is not None:
            _save_checkpoint(
                checkpoint_dir / latest_ckpt_name,
                encoder, mol_proj, optimizer, scheduler, epoch,
                best_val_metric, wait, config, spec_proj,
                metric_key=metric_key,
            )
            if improved:
                _save_checkpoint(
                    checkpoint_dir / best_ckpt_name,
                    encoder, mol_proj, optimizer, scheduler, epoch,
                    best_val_metric, wait, config, spec_proj,
                    metric_key=metric_key,
                )

        if wait >= patience:
            _log(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        mol_proj.load_state_dict(best_state["mol_proj"])
    return encoder, mol_proj


def _compute_composite_score(eval_metrics):
    """Composite retrieval metric for V2 checkpoint selection."""
    keys = [
        "eval/0% isomer/clean/top1",
        "eval/0% isomer/moderate/mrr",
        "eval/50% isomer/clean/top1",
    ]
    vals = [eval_metrics.get(k) for k in keys]
    if any(v is None for v in vals):
        return 0.0
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# V1: JointTrainingDataset + train_joint
# ---------------------------------------------------------------------------


class JointTrainingDataset(Dataset):
    """Yields clean + noisy graph views with V1 features + teacher targets."""

    def __init__(self, df, mol_embs, dreams_embs=None, seed=42, warmup_epochs=5):
        from src.models.spectral_graph_encoder import spectrum_to_padded
        self._spectrum_to_padded = spectrum_to_padded
        self.mz_lists = df["mz_list"].tolist()
        self.int_lists = df["intensity_list"].tolist()
        self.prec_mzs = df["precursor_mz"].values
        self.mol_embs = mol_embs
        self.dreams_embs = dreams_embs
        self.has_dreams = dreams_embs is not None
        self._warmup_profiles = [NOISE_PROFILES["mild"], NOISE_PROFILES["moderate"]]
        self._all_profiles = [NOISE_PROFILES[k] for k in ("mild", "moderate", "severe", "extreme")]
        self._epoch = 0
        self._warmup_epochs = warmup_epochs
        self.rng = np.random.default_rng(seed)

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return len(self.mz_lists)

    def __getitem__(self, idx):
        mz = np.asarray(self.mz_lists[idx])
        ints = np.asarray(self.int_lists[idx])
        prec = float(self.prec_mzs[idx])

        g1 = self._spectrum_to_padded(mz, ints, prec)

        pool = self._warmup_profiles if self._epoch < self._warmup_epochs else self._all_profiles
        p = pool[self.rng.integers(len(pool))]
        mz2, int2, prec2 = augment_spectrum(mz, ints, prec, p, self.rng)
        g2 = self._spectrum_to_padded(mz2, int2, prec2)

        mol_emb = torch.from_numpy(self.mol_embs[idx]).float()
        dreams_emb = (torch.from_numpy(self.dreams_embs[idx]).float()
                      if self.has_dreams else torch.zeros(1))
        return g1, g2, mol_emb, dreams_emb


def joint_collate_fn(batch):
    """Collate V1 joint training samples."""
    from src.models.spectral_graph_encoder import padded_collate_fn
    g1s, g2s, mol_embs, dreams_embs = zip(*batch)
    return (
        padded_collate_fn(list(g1s)),
        padded_collate_fn(list(g2s)),
        torch.stack(mol_embs),
        torch.stack(dreams_embs),
    )


def train_joint(encoder, train_df, val_df,
                mol_embs_train, mol_embs_val,
                dreams_embs_train=None, dreams_embs_val=None,
                config=None, device=None, checkpoint_dir=None, logger=None,
                eval_callback=None):
    """Joint multi-teacher training for V1 encoder. Early stops on val_align."""
    config = config or {}
    d_spec = config.get("d_spec", 512)
    d_mol = mol_embs_train.shape[1]
    use_distill = dreams_embs_train is not None
    d_dreams = dreams_embs_train.shape[1] if use_distill else d_spec
    batch_size = config.get("batch_size", 512)
    num_workers = config.get("num_workers", 0)

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mol_proj = LinearProj(d_spec, d_mol).to(device)
    encoder = encoder.to(device)
    use_amp = device.type == "cuda"
    if use_amp:
        encoder = torch.compile(encoder)

    curriculum_warmup = config.get("curriculum_warmup", 5)
    train_ds = JointTrainingDataset(
        train_df, mol_embs_train, dreams_embs_train,
        seed=config.get("seed", 42), warmup_epochs=curriculum_warmup,
    )
    val_ds = JointTrainingDataset(
        val_df, mol_embs_val, dreams_embs_val,
        seed=config.get("seed", 42) + 1, warmup_epochs=curriculum_warmup,
    )
    pin = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=joint_collate_fn, num_workers=num_workers,
        drop_last=True, pin_memory=pin, persistent_workers=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=joint_collate_fn, num_workers=num_workers,
        pin_memory=pin, persistent_workers=pin,
    )

    spec_proj = LinearProj(d_spec, d_dreams).to(device) if use_distill else None

    return _joint_training_loop(
        encoder, train_loader, val_loader, train_ds, val_ds,
        mol_proj, spec_proj, config, device, checkpoint_dir,
        logger, eval_callback, use_distill,
        ckpt_prefix="", checkpoint_selection="val_align",
    )


def evaluate_retrieval(encoder, val_df, device, batch_size=256, **_kwargs):
    """Evaluate class-level nearest-neighbor accuracy on validation set."""
    from src.models.spectral_graph_encoder import encode_batch

    embs = encode_batch(
        encoder,
        val_df["mz_list"].tolist(), val_df["intensity_list"].tolist(),
        val_df["precursor_mz"].tolist(),
        batch_size=batch_size, device=device,
    )

    sim = embs @ embs.T
    np.fill_diagonal(sim, -1)

    classes = val_df["lipid_class"].values
    nn_indices = sim.argmax(axis=1)
    nn_classes = classes[nn_indices]
    accuracy = (nn_classes == classes).mean()
    print(f"  Val NN class accuracy: {accuracy:.4f}")
    return accuracy


# ---------------------------------------------------------------------------
# V2: JointTrainingDatasetV2 + train_joint_v2
# ---------------------------------------------------------------------------


class JointTrainingDatasetV2(Dataset):
    """Yields clean + noisy graph views with V2 features + teacher targets."""

    def __init__(self, df, mol_embs, dreams_embs=None, seed=42, warmup_epochs=5):
        from src.models.spectral_graph_encoder import spectrum_to_padded_v2
        self._spectrum_to_padded = spectrum_to_padded_v2
        self.mz_lists = df["mz_list"].tolist()
        self.int_lists = df["intensity_list"].tolist()
        self.prec_mzs = df["precursor_mz"].values
        self.mol_embs = mol_embs
        self.dreams_embs = dreams_embs
        self.has_dreams = dreams_embs is not None
        self._warmup_profiles = [NOISE_PROFILES["mild"], NOISE_PROFILES["moderate"]]
        self._all_profiles = [NOISE_PROFILES[k] for k in ("mild", "moderate", "severe", "extreme")]
        self._epoch = 0
        self._warmup_epochs = warmup_epochs
        self.rng = np.random.default_rng(seed)

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return len(self.mz_lists)

    def __getitem__(self, idx):
        mz = np.asarray(self.mz_lists[idx])
        ints = np.asarray(self.int_lists[idx])
        prec = float(self.prec_mzs[idx])

        g1 = self._spectrum_to_padded(mz, ints, prec)

        pool = self._warmup_profiles if self._epoch < self._warmup_epochs else self._all_profiles
        p = pool[self.rng.integers(len(pool))]
        mz2, int2, prec2 = augment_spectrum(mz, ints, prec, p, self.rng)
        g2 = self._spectrum_to_padded(mz2, int2, prec2)

        mol_emb = torch.from_numpy(self.mol_embs[idx]).float()
        dreams_emb = (torch.from_numpy(self.dreams_embs[idx]).float()
                      if self.has_dreams else torch.zeros(1))
        return g1, g2, mol_emb, dreams_emb


def joint_collate_fn_v2(batch):
    """Collate V2 joint training samples."""
    from src.models.spectral_graph_encoder import padded_collate_fn_v2
    g1s, g2s, mol_embs, dreams_embs = zip(*batch)
    return (
        padded_collate_fn_v2(list(g1s)),
        padded_collate_fn_v2(list(g2s)),
        torch.stack(mol_embs),
        torch.stack(dreams_embs),
    )


def train_joint_v2(encoder, train_df, val_df,
                   mol_embs_train, mol_embs_val,
                   dreams_embs_train=None, dreams_embs_val=None,
                   config=None, device=None, checkpoint_dir=None, logger=None,
                   eval_callback=None):
    """Joint multi-teacher training for V2 encoder. Early stops on composite retrieval metric."""
    config = config or {}
    d_spec = config.get("d_spec", 512)
    d_mol = mol_embs_train.shape[1]
    use_distill = dreams_embs_train is not None
    d_dreams = dreams_embs_train.shape[1] if use_distill else d_spec
    batch_size = config.get("batch_size", 512)
    num_workers = config.get("num_workers", 0)

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mol_proj = LinearProj(d_spec, d_mol).to(device)
    encoder = encoder.to(device)
    use_amp = device.type == "cuda"
    if use_amp:
        encoder = torch.compile(encoder)

    curriculum_warmup = config.get("curriculum_warmup", 5)
    train_ds = JointTrainingDatasetV2(
        train_df, mol_embs_train, dreams_embs_train,
        seed=config.get("seed", 42), warmup_epochs=curriculum_warmup,
    )
    val_ds = JointTrainingDatasetV2(
        val_df, mol_embs_val, dreams_embs_val,
        seed=config.get("seed", 42) + 1, warmup_epochs=curriculum_warmup,
    )
    pin = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=joint_collate_fn_v2, num_workers=num_workers,
        drop_last=True, pin_memory=pin, persistent_workers=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=joint_collate_fn_v2, num_workers=num_workers,
        pin_memory=pin, persistent_workers=pin,
    )

    spec_proj = LinearProj(d_spec, d_dreams).to(device) if use_distill else None

    return _joint_training_loop(
        encoder, train_loader, val_loader, train_ds, val_ds,
        mol_proj, spec_proj, config, device, checkpoint_dir,
        logger, eval_callback, use_distill,
        ckpt_prefix="_v2", checkpoint_selection="composite",
    )


def evaluate_retrieval_v2(encoder, val_df, device, batch_size=256):
    """Evaluate class-level nearest-neighbor accuracy on validation set (V2)."""
    from src.models.spectral_graph_encoder import encode_batch_v2

    embs = encode_batch_v2(
        encoder,
        val_df["mz_list"].tolist(), val_df["intensity_list"].tolist(),
        val_df["precursor_mz"].tolist(),
        batch_size=batch_size, device=device,
    )
    sim = embs @ embs.T
    np.fill_diagonal(sim, -1)
    classes = val_df["lipid_class"].values
    nn_indices = sim.argmax(axis=1)
    nn_classes = classes[nn_indices]
    accuracy = (nn_classes == classes).mean()
    print(f"  Val NN class accuracy: {accuracy:.4f}")
    return accuracy
