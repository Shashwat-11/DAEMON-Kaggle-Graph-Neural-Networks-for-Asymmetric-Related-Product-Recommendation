"""
DAEMON — Training Module

Training loop, checkpointing, and logging for the Directed Asymmetric
Graph Neural Network for product recommendation on Kaggle.

Key design decisions:
  - Mixed-precision (AMP) with GradScaler for T4 GPU efficiency
  - Gradient accumulation to simulate larger effective batch sizes
  - Early stopping with patience-based validation AUC tracking
  - Full-graph embedding generation for validation metrics
  - Checkpointing supports multi-session training (Kaggle 9hr limit)

References:
  Virinchi et al. (2022). Recommending Related Products Using Graph
  Neural Networks in Directed Graphs. ECML-PKDD.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def memory_summary() -> Dict[str, float]:
    """Return a dictionary of current GPU memory statistics.

    Returns:
        Dict with keys:
          - allocated_gb:  current allocated VRAM in GB
          - reserved_gb:   current reserved VRAM in GB
          - peak_gb:       peak allocated VRAM since last reset
          - free_gb:       estimated free VRAM in GB
    """
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "peak_gb": 0.0, "free_gb": 0.0}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_mem / 1e9
    free_gb = total - allocated

    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "peak_gb": round(peak, 2),
        "free_gb": round(free_gb, 2),
    }


def print_memory_summary() -> None:
    """Print formatted GPU memory statistics to console."""
    stats = memory_summary()
    print(
        f"GPU Memory  |  allocated={stats['allocated_gb']:.1f}GB  "
        f"reserved={stats['reserved_gb']:.1f}GB  "
        f"peak={stats['peak_gb']:.1f}GB  "
        f"free={stats['free_gb']:.1f}GB"
    )


# ---------------------------------------------------------------------------
# Epoch-level training
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    cfg: Any,
    epoch_idx: int = 0,
) -> float:
    """Run one training epoch with AMP, gradient accumulation, and clipping.

    The loader yields DGL mini-batches (``input_nodes``, ``output_nodes``,
    ``blocks``) from a neighbor sampler.  Each batch runs:

        1. ``autocast`` forward → loss
        2. ``scaler.scale(loss).backward()``
        3. Every ``grad_accum_steps``: unscale → clip → step → update

    Args:
        model:      DAEMON model (produces ``(src_emb, tgt_emb)``)
        loader:     DGL DataLoader yielding neighbor-sampled subgraphs
        criterion:  Loss function expecting signature
                    ``forward(src_emb, tgt_emb, batch_data: dict)``
        optimizer:  PyTorch optimizer (e.g. Adam)
        scaler:     ``GradScaler`` for AMP
        cfg:        Config object with ``grad_accum_steps``, ``use_amp``,
                    ``grad_clip_norm``
        epoch_idx:  Current epoch number (for progress bar label)

    Returns:
        Average loss over all batches in this epoch (float).

    Note:
        Batch-local negative sampling is used as an intentional adaptation
        for mini-batch training. This trades a minor accuracy reduction for
        significantly lower memory overhead compared to global negative
        sampling across the full graph.
    """
    model.train()
    total_loss: float = 0.0
    num_batches: int = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx:3d}", leave=False)
    # Use model device dynamically instead of hardcoded "cuda"
    train_device = next(model.parameters()).device
    for step, batch in enumerate(pbar):
        # ---- DGL DataLoader yields (input_nodes, output_nodes, blocks) ----
        input_nodes, output_nodes, blocks = batch
        blocks = [b.to(train_device, non_blocking=True) for b in blocks]
        batch_inputs = blocks[0].srcdata["feat"]

        # ---- Forward with optional AMP ----
        with autocast(enabled=getattr(cfg, "use_amp", True)):
            src_emb, tgt_emb = model(blocks, batch_inputs)

            # The criterion expects the batch data dict with pairs for
            # co-purchase, one-way, and co-view edges.
            # loader provides batch_data as an attribute on the last block.
            batch_data = blocks[-1].dstdata.get("batch_data", None)
            if batch_data is None:
                # Fallback: build minimal pairs from the subgraph edges
                batch_data = _build_default_batch_data(blocks[-1])

            loss, _ = criterion(src_emb, tgt_emb, batch_data)

            # Scale loss by accumulation steps so that the effective loss
            # is averaged across the accumulation window.
            accum_steps = getattr(cfg, "grad_accum_steps", 1)
            loss = loss / accum_steps

        # ---- Backward with AMP scaling ----
        scaler.scale(loss).backward()

        # ---- Gradient accumulation step ----
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            # Unscale before clipping (so clip sees true gradient norms)
            scaler.unscale_(optimizer)
            clip_norm = getattr(cfg, "grad_clip_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # ---- Track loss (undo accumulation scaling for logging) ----
        batch_loss = loss.item() * accum_steps
        total_loss += batch_loss
        num_batches += 1

        # ---- Progress bar ----
        pbar.set_postfix(
            loss=f"{batch_loss:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            vram=f"{torch.cuda.memory_allocated() / 1e9:.1f}G",
        )

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def _build_default_batch_data(block) -> Dict[str, torch.Tensor]:
    """Build a minimal batch_data dict from a DGL block when the loader
    does not attach structured pairs.

    Separates edges by type (co-purchase=0, co-view=1) so that the
    AsymmetricLoss receives correctly typed edges for each component.
    One-way pairs default to empty tensors (asymmetry detection requires
    the full graph, not just a block).
    """
    src, dst = block.edges()
    device = src.device

    # Separate edges by type if available
    if "type" in block.edata:
        cp_mask = block.edata["type"] == 0
        cv_mask = block.edata["type"] == 1
        cp_src = src[cp_mask]
        cp_dst = dst[cp_mask]
        cv_src = src[cv_mask]
        cv_dst = dst[cv_mask]
    else:
        # No type info — default all to co-purchase
        cp_src, cp_dst = src, dst
        cv_src = cv_dst = torch.empty(0, dtype=torch.long, device=device)

    # Use total graph nodes for negative sampling range;
    # fall back to block destination nodes if full count unavailable.
    num_nodes = block.num_dst_nodes()

    return {
        "cp_u": cp_src,
        "cp_v": cp_dst,
        "ow_u": torch.empty(0, dtype=torch.long, device=device),
        "ow_v": torch.empty(0, dtype=torch.long, device=device),
        "cv_u": cv_src,
        "cv_v": cv_dst,
        "num_nodes": num_nodes,
    }


# ---------------------------------------------------------------------------
# Embedding generation (full graph)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_embeddings(
    model: nn.Module,
    g,
    cfg: Any,
    batch_size: int = 4096,
    desc: str = "Embedding",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run full-graph inference to produce source and target embeddings.

    Uses the same neighbor-sampler architecture as training (fanouts from
    ``cfg.num_neighbors``) but processes the graph in batches so the full
    embedding matrix never needs to live on GPU at once.

    Args:
        model:      DAEMON model
        g:          Full DGL graph (CPU-resident)
        cfg:        Config with ``num_neighbors``, ``num_layers``, etc.
        batch_size: Number of seed nodes per batch (default 4096)
        desc:       Label for tqdm progress bar

    Returns:
        Tuple of ``(source_embeddings, target_embeddings)``, each of shape
        ``[num_nodes, out_dim]``, on CPU.
    """
    model.eval()

    try:
        from dgl.dataloading import DataLoader as DGLDataLoader
        from dgl.dataloading import MultiLayerNeighborSampler
    except ImportError:
        raise ImportError("DGL is required for embedding generation.")

    fanouts = getattr(cfg, "num_neighbors", [20, 10, 10])
    num_layers = getattr(cfg, "num_layers", len(fanouts))
    if len(fanouts) < num_layers:
        # Pad fanouts with -1 (all neighbors) for remaining layers
        fanouts = list(fanouts) + [-1] * (num_layers - len(fanouts))

    sampler = MultiLayerNeighborSampler(fanouts)

    device = next(model.parameters()).device
    num_nodes = g.num_nodes()

    # Use all nodes as seeds for full-graph inference
    all_nodes = torch.arange(num_nodes)
    loader = DGLDataLoader(
        g,
        all_nodes,
        sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        device=device,
    )

    src_list: List[torch.Tensor] = []
    tgt_list: List[torch.Tensor] = []

    for input_nodes, output_nodes, blocks in tqdm(loader, desc=desc):
        blocks = [b.to(device) for b in blocks]
        batch_inputs = blocks[0].srcdata["feat"]

        with autocast(enabled=getattr(cfg, "use_amp", True)):
            src_batch, tgt_batch = model(blocks, batch_inputs)

        src_list.append(src_batch.cpu())
        tgt_list.append(tgt_batch.cpu())

    source_embeddings = torch.cat(src_list, dim=0)
    target_embeddings = torch.cat(tgt_list, dim=0)

    return source_embeddings, target_embeddings


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    g,
    masks: Dict[str, torch.Tensor],
    cfg: Any,
) -> Dict[str, float]:
    """Evaluate the model on validation edges.

    Generates full-graph embeddings, then computes:

    - **AUC**: area under the ROC curve for link prediction (positive =
      real validation edges vs negative = random non-edges)
    - **HitRate@10**: fraction of queries whose ground-truth co-purchase
      partner appears in the top-10 by ``rel(q, v) = θ_q^s · θ_v^t``

    Args:
        model:  DAEMON model
        g:      Full DGL graph
        masks:  Dict with ``"val"`` mask (bool tensor of shape ``[E,]``)
        cfg:    Config object

    Returns:
        Dict with keys ``"auc"``, ``"hitrate_10"``, and optionally
        ``"hitrate_5"``, ``"hitrate_20"``.
    """
    model.eval()

    # ---- Generate full-graph embeddings ----
    src_emb, tgt_emb = generate_embeddings(
        model, g, cfg, desc="Validation"
    )
    # Move embeddings to model device (use model device, not hardcoded cuda)
    model_device = next(model.parameters()).device
    src_emb = src_emb.to(model_device)
    tgt_emb = tgt_emb.to(model_device)

    # ---- Validation edges ----
    val_mask = masks["val"]
    if val_mask.sum() == 0:
        return {"auc": 0.0, "hitrate_10": 0.0}

    all_src, all_dst = g.edges()
    val_src = all_src[val_mask].to(model_device)
    val_dst = all_dst[val_mask].to(model_device)
    num_val = val_src.size(0)

    # ---- Negative sampling (uniform, same count as positives) ----
    num_nodes = g.num_nodes()
    neg_src = torch.randint(0, num_nodes, (num_val,), device=model_device)
    neg_dst = torch.randint(0, num_nodes, (num_val,), device=model_device)

    # ---- AUC (link prediction) ----
    pos_scores = (src_emb[val_src] * tgt_emb[val_dst]).sum(dim=1)
    neg_scores = (src_emb[neg_src] * tgt_emb[neg_dst]).sum(dim=1)

    all_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    all_labels = np.concatenate([np.ones(num_val), np.zeros(num_val)])
    auc = float(roc_auc_score(all_labels, all_scores))

    # ---- HitRate@k ----
    # For each unique query node, check if its true candidate appears in top-k.
    # Group by query node to handle multiple ground-truth candidates per query.
    hitrate_10 = _compute_hitrate(src_emb, tgt_emb, val_src, val_dst, k=10)

    # ---- Collect results ----
    results: Dict[str, float] = {
        "auc": round(auc, 6),
        "hitrate_10": round(hitrate_10, 6),
    }

    # Optionally compute additional k values
    for k in (1, 5, 20):
        if k == 10:
            continue
        hr = _compute_hitrate(src_emb, tgt_emb, val_src, val_dst, k=k)
        results[f"hitrate_{k}"] = round(hr, 6)

    # Clean up GPU memory
    del src_emb, tgt_emb, pos_scores, neg_scores
    torch.cuda.empty_cache()

    return results


@torch.no_grad()
def _compute_hitrate(
    src_emb: torch.Tensor,
    tgt_emb: torch.Tensor,
    query_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    k: int = 10,
) -> float:
    """Compute HitRate@k for link prediction.

    For each unique query, we check if *any* of its ground-truth candidates
    appear in the top-k results ranked by relevance score.

    Args:
        src_emb:        Source embeddings ``[N, d]``
        tgt_emb:        Target embeddings ``[N, d]``
        query_ids:      Query node indices ``[E,]``
        candidate_ids:  Ground-truth candidate indices ``[E,]``
        k:              Cutoff for top-k (default 10)

    Returns:
        HitRate@k as a float in [0, 1].
    """
    if query_ids.numel() == 0:
        return 0.0

    # Group candidates by query node (support multiple gt per query)
    unique_queries, inverse = torch.unique(query_ids, return_inverse=True)
    num_queries = unique_queries.size(0)
    hits = 0

    for i in range(num_queries):
        q = unique_queries[i]
        # All ground-truth candidates for this query
        gt_mask = inverse == i
        gt_candidates = candidate_ids[gt_mask]

        # Compute scores: θ_q^s · θ_v^t for all v
        q_src = src_emb[q]  # [d]
        scores = q_src @ tgt_emb.T  # [num_nodes]
        _, topk_idx = torch.topk(scores, min(k, scores.size(0)))

        # Check if any gt candidate is in top-k
        if (gt_candidates.unsqueeze(1) == topk_idx.unsqueeze(0)).any():
            hits += 1

    return hits / max(num_queries, 1)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    is_best: bool = False,
) -> str:
    """Save training checkpoint to disk.

    Saves model state dict, optimizer state, scaler state, scheduler state,
    epoch number, and metrics.  If ``is_best=True``, also saves a copy to
    ``<path>.best``.

    Args:
        model:     DAEMON model
        optimizer: PyTorch optimizer
        scaler:    GradScaler for AMP
        scheduler: LR scheduler
        epoch:     Current epoch (0-indexed)
        metrics:   Dict of validation metrics (e.g. ``{"auc": 0.85}``)
        path:      Destination path (e.g. ``"/kaggle/working/checkpoint.pt"``)
        is_best:   If True, also save as ``<path>.best``

    Returns:
        The path the checkpoint was saved to.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved → {path}  (epoch {epoch})")

    if is_best:
        best_path = path + ".best"
        torch.save(checkpoint, best_path)
        print(f"Best model saved → {best_path}")

    return path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    scheduler: Any = None,
    map_location: Optional[str] = None,
) -> Tuple[int, Dict[str, float]]:
    """Load a training checkpoint and restore all states.

    Args:
        path:         Checkpoint file path
        model:        Model instance (state is loaded in-place)
        optimizer:    Optional optimizer (state restored if provided)
        scaler:       Optional GradScaler (state restored if provided)
        scheduler:    Optional LR scheduler (state restored if provided)
        map_location: Device map string (e.g. ``"cuda:0"`` or ``"cpu"``).
                      Defaults to ``"cuda"`` if available else ``"cpu"``.

    Returns:
        Tuple of ``(starting_epoch, metrics)`` where ``starting_epoch``
        is the epoch to resume from (``checkpoint["epoch"]``) and
        ``metrics`` is the saved metrics dict.
    """
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)

    # Restore model
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore optimizer
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore scaler
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Restore scheduler
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        sd = checkpoint["scheduler_state_dict"]
        if sd is not None:
            scheduler.load_state_dict(sd)

    start_epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    best_val_auc = metrics.get("auc", 0.0)

    print(
        f"Checkpoint loaded ← {path}  "
        f"(resume epoch {start_epoch}, best AUC={best_val_auc:.4f})"
    )

    return start_epoch + 1, metrics


# ---------------------------------------------------------------------------
# Full training orchestration
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_data: Tuple,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    cfg: Any,
    resume_epoch: int = 0,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """Run the full training loop with early stopping, validation, and
    best-model tracking.

    Args:
        model:        DAEMON model
        train_loader: DGL DataLoader for training subgraphs
        val_data:     Tuple of ``(g, masks)`` where ``g`` is the full DGL
                      graph and ``masks`` is a dict with a ``"val"`` key.
        criterion:    AsymmetricLoss from ``daemon_model.py``
        optimizer:    PyTorch optimizer
        scheduler:    LR scheduler (called after each validation)
        scaler:       GradScaler for AMP
        cfg:          Config object with ``epochs``, ``patience``,
                      ``checkpoint_path``, ``val_every``, etc.
        resume_epoch: Epoch to start from (for checkpoint resume)

    Returns:
        Tuple of ``(history, best_metrics)``:

        - ``history``: dict with ``"train_loss"`` (list of per-epoch losses)
          and ``"val_auc"`` (list of per-validation AUC values).
        - ``best_metrics``: dict with the best validation metrics and
          best epoch number.
    """
    g, masks = val_data

    num_epochs = getattr(cfg, "epochs", 30)
    patience = getattr(cfg, "patience", 5)
    val_every = getattr(cfg, "val_every", 1)
    checkpoint_path = getattr(cfg, "checkpoint_path", "/tmp/daemon_checkpoint.pt")
    cleanup_every = getattr(cfg, "cleanup_every_n_epochs", 5)

    # Early stopping state
    best_val_auc: float = 0.0
    best_metrics: Dict[str, float] = {"auc": 0.0, "epoch": -1}
    patience_counter: int = 0

    # History tracking
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_auc": [],
        "val_hitrate_10": [],
        "lr": [],
    }

    print(f"{'=' * 60}")
    print(f"  DAEMON Training — {num_epochs} max epochs, patience={patience}")
    print(f"{'=' * 60}")
    print_memory_summary()

    for epoch in range(resume_epoch, num_epochs):
        epoch_start = time.time()

        # ---- Training ----
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, cfg,
            epoch_idx=epoch,
        )
        history["train_loss"].append(train_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        # ---- Validation (every val_every epochs) ----
        val_results: Optional[Dict[str, float]] = None
        if (epoch + 1) % val_every == 0:
            val_results = validate(model, g, masks, cfg)
            val_auc = val_results.get("auc", 0.0)
            val_hr10 = val_results.get("hitrate_10", 0.0)

            history["val_auc"].append(val_auc)
            history["val_hitrate_10"].append(val_hr10)

            # LR scheduler step (ReduceLROnPlateau uses val_auc)
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_auc)
                else:
                    scheduler.step()

            # ---- Best model tracking ----
            is_best = val_auc > best_val_auc
            if is_best:
                best_val_auc = val_auc
                best_metrics = {
                    "auc": val_auc,
                    "hitrate_10": val_hr10,
                    "epoch": epoch,
                }
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint (always save latest; flag best separately)
            save_checkpoint(
                model, optimizer, scaler, scheduler,
                epoch=epoch,
                metrics=val_results,
                path=checkpoint_path,
                is_best=is_best,
            )
        else:
            # No validation this epoch
            history.setdefault("val_auc", []).append(
                history["val_auc"][-1] if history["val_auc"] else 0.0
            )
            history.setdefault("val_hitrate_10", []).append(
                history["val_hitrate_10"][-1] if history["val_hitrate_10"] else 0.0
            )

        # ---- Epoch summary ----
        epoch_time = time.time() - epoch_start
        log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_results=val_results,
            lr=current_lr,
            history=history,
            epoch_time=epoch_time,
            best_val_auc=best_val_auc,
        )

        # ---- Memory cleanup ----
        if (epoch + 1) % cleanup_every == 0:
            gc.collect()
            torch.cuda.empty_cache()

        # ---- Early stopping ----
        if patience_counter >= patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch} "
                f"(no improvement for {patience} validations).  "
                f"Best AUC: {best_val_auc:.4f} at epoch {best_metrics['epoch']}."
            )
            break

    # ---- Final summary ----
    print(f"\n{'=' * 60}")
    print(f"  Training Complete — {len(history['train_loss'])} epochs")
    print(f"  Best Val AUC:    {best_metrics.get('auc', 0):.4f}  "
          f"(epoch {best_metrics.get('epoch', -1)})")
    print(f"  Best HitRate@10: {best_metrics.get('hitrate_10', 0):.4f}")
    print(f"{'=' * 60}")

    return history, best_metrics


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_training(
    cfg: Any,
    model_class: type,
    criterion_class: type,
) -> Dict[str, Any]:
    """Initialize all training components from a config object.

    Creates and moves to GPU:

    - Model instance (``model_class(cfg)``)
    - Adam optimizer with configurable LR and weight decay
    - ``AsymmetricLoss`` criterion (``criterion_class(cfg)``)
    - LR scheduler (``CosineAnnealingWarmRestarts`` or fallback to
      ``ReduceLROnPlateau``)
    - ``GradScaler`` for AMP

    Args:
        cfg:             Config object (dataclass) with fields:
                         ``in_feats``, ``hidden_dim``, ``out_dim``,
                         ``num_layers``, ``dropout``, ``lr``,
                         ``weight_decay``, ``use_amp``, ``epochs``.
        model_class:     DAEMON model class (e.g. ``daemon_model.DAEMONModel``)
        criterion_class: Loss class (e.g. ``daemon_model.AsymmetricLoss``)

    Returns:
        Dict with keys:
          - ``"model"``:      model on GPU
          - ``"optimizer"``:  Adam optimizer
          - ``"criterion"``:  loss function on GPU
          - ``"scheduler"``:  LR scheduler
          - ``"scaler"``:     ``GradScaler``
          - ``"start_epoch"``: 0 (override if loading checkpoint)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Model ----
    model = model_class(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_class.__name__}  |  "
          f"Params: {total_params:,}  |  "
          f"Trainable: {trainable_params:,}")

    # ---- Optimizer ----
    lr = getattr(cfg, "lr", 1e-4)
    weight_decay = getattr(cfg, "weight_decay", 1e-5)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    # ---- Criterion ----
    criterion = criterion_class(cfg).to(device)

    # ---- LR Scheduler ----
    # Prefer CosineAnnealingWarmRestarts; fall back to ReduceLROnPlateau.
    scheduler: Any
    if hasattr(cfg, "scheduler") and cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6,
        )
    else:
        T_0 = getattr(cfg, "scheduler_t0", 10)
        T_mult = getattr(cfg, "scheduler_tmult", 2)
        eta_min = getattr(cfg, "scheduler_eta_min", 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min,
        )

    # ---- GradScaler (AMP) ----
    use_amp = getattr(cfg, "use_amp", True)
    scaler = GradScaler(enabled=use_amp)

    print(f"Optimizer: Adam (lr={lr:.2e}, wd={weight_decay:.1e})")
    print(f"Scheduler: {type(scheduler).__name__}")
    print(f"AMP: {'enabled' if use_amp else 'disabled'}  |  "
          f"Device: {device}")

    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
        "scaler": scaler,
        "start_epoch": 0,
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_epoch(
    epoch: int,
    train_loss: float,
    val_results: Optional[Dict[str, float]],
    lr: float,
    history: Dict[str, List[float]],
    epoch_time: Optional[float] = None,
    best_val_auc: Optional[float] = None,
) -> None:
    """Print a formatted per-epoch summary to console.

    Args:
        epoch:        Current epoch number (0-indexed)
        train_loss:   Average training loss for this epoch
        val_results:  Validation metrics dict (or None if not validated)
        lr:           Current learning rate
        history:      Training history dict (for displaying trailing avg)
        epoch_time:   Wall time for this epoch in seconds
        best_val_auc: Current best validation AUC
    """
    mem = memory_summary()
    line_parts = [
        f"Epoch {epoch:3d}",
        f"Loss {train_loss:.4f}",
    ]

    if val_results is not None:
        auc = val_results.get("auc", 0.0)
        hr10 = val_results.get("hitrate_10", 0.0)
        line_parts.append(f"AUC {auc:.4f}")
        line_parts.append(f"HR@10 {hr10:.4f}")

    line_parts.append(f"LR {lr:.2e}")

    if epoch_time is not None:
        line_parts.append(f"Time {epoch_time:.1f}s")

    if best_val_auc is not None:
        line_parts.append(f"Best {best_val_auc:.4f}")

    line_parts.append(f"VRAM {mem['allocated_gb']:.1f}G")

    print("  │  ".join(line_parts))
