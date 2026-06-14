#!/usr/bin/env python3
"""
build_notebook.py — Generate daemon_kaggle.ipynb using nbformat.

Creates a complete, self-contained Kaggle notebook for the DAEMON model
(Related Product Recommendation via GNNs on Directed Graphs).

Notebook structure:
  Cell Group 1:  Environment Setup (pip, imports, GPU check)
  Cell Group 2:  Configuration (DAEMONConfig with overrides)
  Cell Group 3:  Data Loading with Synthetic Fallback
  Cell Group 4:  Graph Splitting & DataLoaders
  Cell Group 5:  Model Definition & Smoke Test
  Cell Group 6:  Training (setup_training + train_model with early stopping)
  Cell Group 7:  Evaluation (evaluate_full — EQ1, EQ2, EQ3)
  Cell Group 8:  FAISS Indexing & Latency Benchmark
  Cell Group 9:  Recommendation Demo (5 sample queries)
  Cell Group 10: Cold-Start Demo
  Cell Group 11: Ablation Studies
  Cell Group 12: Export Results & Summary Dashboard

Usage:
  python build_notebook.py
  → produces /home/shashwat-11/ML/daemon_kaggle.ipynb
"""

import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell


def make_notebook() -> nbf.NotebookNode:
    """Build the complete Kaggle notebook and return the NBNode object."""
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    }

    cells = []

    # ========================================================================
    # Cell Group 1: Environment Setup
    # ========================================================================
    cells.append(
        new_markdown_cell(
            "## 📦 Cell Group 1: Environment Setup\n\n"
            "Install dependencies, import all modules, verify GPU, set seeds."
        )
    )

    cells.append(
        new_code_cell(
            r"""# ════════════════════════════════════════════════════════════════
# Cell 1a — Install DGL and FAISS (Kaggle / Colab compatible)
# ════════════════════════════════════════════════════════════════
# ⚠️ REQUIRED: Enable Internet in Kaggle notebook settings
# (Right sidebar → Settings → Internet → ON)
# Then run this cell to install dependencies.

# ── Detect PyTorch / CUDA version for correct DGL wheel ─────────────
import torch
tv = torch.__version__.split("+")[0]                # e.g. "2.6.0"
cv = "cu" + torch.version.cuda.replace(".", "")     # e.g. "cu124"
dgl_url = f"https://data.dgl.ai/wheels/torch-{tv}/{cv}/repo.html"
print(f"Detected PyTorch {tv} + CUDA {cv}")
print(f"DGL wheel URL: {dgl_url}")

# ── Install DGL with matching CUDA support ─────────────────────────
!pip install dgl -f {dgl_url} -q

# ── Install FAISS for GPU-accelerated similarity search ────────────
!pip install faiss-gpu -q

# ── Core data-science packages ─────────────────────────────────────
!pip install pandas matplotlib tqdm scikit-learn -q

# Optional: install if using real product text features (not needed for synthetic)
# !pip install sentence-transformers -q

print("✅ Dependencies installed")
"""
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 1b — Imports from src/ modules and all dependencies
# ════════════════════════════════════════════════════════════════
import sys
import os
import json
import math
import time
import gc
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Add src/ to Python path so we can import project modules directly
sys.path.append("src")

# ── DAEMON model components ──────────────────────────────────────────────
from daemon_model import (
    DAEMONConfig,
    DAEMONModel,
    AsymmetricLoss,
    count_parameters,
)

# ── Data pipeline ─────────────────────────────────────────────────────────
from data_pipeline import (
    generate_synthetic_graph,
    build_product_graph,
    split_edges_by_type,
    find_one_way_edges,
    print_graph_stats,
    NegativeSampler,
    validate_graph,
)

# ── Training ──────────────────────────────────────────────────────────────
from training import (
    setup_training,
    train_model,
    load_checkpoint,
    memory_summary,
    print_memory_summary,
)

# ── Evaluation ────────────────────────────────────────────────────────────
from evaluation import (
    generate_all_embeddings,
    build_faiss_index,
    evaluate_full,
    recommend_related,
    cold_start_recommend,
    compute_hit_rate_at_k,
    compute_mrr_at_k,
)

# ── DGL (Deep Graph Library) ──────────────────────────────────────────────
import dgl
from dgl.dataloading import DataLoader as DGLDataLoader
from dgl.dataloading import MultiLayerNeighborSampler

print("✅ All imports successful")
print(f"   DGL version: {dgl.__version__}")
print(f"   PyTorch version: {torch.__version__}")
"""
        )
    )

    cells.append(
        new_code_cell(
            '''# ════════════════════════════════════════════════════════════════
# Cell 1c — GPU verification, seed setting, memory check
# ════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# ── Device detection ─────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"CUDA:   {torch.version.cuda}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print_memory_summary()
else:
    print("WARNING: Running on CPU — training will be very slow.")
    print("         Enable a GPU accelerator on Kaggle (Accelerator → GPU T4 x2).")
'''
        )
    )

    # ========================================================================
    # Cell Group 2: Configuration
    # ========================================================================
    cells.append(
        new_markdown_cell(
            "## ⚙️ Cell Group 2: DAEMON Configuration\n\n"
            "All hyperparameters are defined in the `DAEMONConfig` dataclass. "
            "Override defaults for the Kaggle environment."
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 2 — DAEMONConfig with Kaggle-optimised overrides
# ════════════════════════════════════════════════════════════════
cfg = DAEMONConfig(
    # ── Model architecture ──────────────────────────────────────────────
    in_feats=384,       # Input feature dimension (Sentence-BERT / synthetic)
    hidden_dim=128,     # Hidden dimension for GNN layers
    out_dim=64,         # Output embedding dimension (final)
    num_layers=3,       # Number of DAEMON layers
    dropout=0.1,        # Dropout rate

    # ── Training hyperparameters ────────────────────────────────────────
    epochs=30,          # Max training epochs
    batch_size=1024,    # Batch size (neighbourhoods per step)
    num_neighbors=(20, 10, 10),  # Fan-out per layer
    lr=1e-4,            # Peak learning rate
    weight_decay=1e-5,  # L2 regularisation
    grad_accum_steps=1, # Gradient accumulation (set >1 for larger effective BS)
    use_amp=True,       # Mixed-precision (FP16) for T4 GPU efficiency
    patience=5,         # Early-stopping patience
    num_neg=5,          # Negative samples per positive pair

    # ── Evaluation ──────────────────────────────────────────────────────
    hitrate_k=(5, 10, 20),  # Recall cutoffs
    val_every=1,            # Validate every N epochs

    # ── Paths (Kaggle convention) ───────────────────────────────────────
    data_dir="/kaggle/input/daemon-data",
    output_dir="/kaggle/working",
    checkpoint_path="/kaggle/working/daemon_best.pt",

    # ── Memory management ───────────────────────────────────────────────
    cleanup_every_n_epochs=4,
)

# Override output dir for local testing (Kaggle working dir not available)
if not os.path.isdir("/kaggle/working"):
    cfg.output_dir = "./output"
    cfg.checkpoint_path = "./output/daemon_best.pt"

os.makedirs(cfg.output_dir, exist_ok=True)

# Print config
print("=" * 62)
print("  DAEMON Configuration")
print("=" * 62)
for field in cfg.__dataclass_fields__:
    val = getattr(cfg, field)
    print(f"  {field:30s} = {val}")
print("=" * 62)
"""
        )
    )

    # ========================================================================
    # Cell Group 3: Data Loading with Synthetic Fallback
    # ========================================================================
    cells.append(
        new_markdown_cell(
            "## 📊 Cell Group 3: Data Loading (with Synthetic Fallback)\n\n"
            "Attempt to load the Kaggle competition dataset first, then "
            "fall back to synthetic data if unavailable. The `RUN_SYNTHETIC` "
            "flag allows quick switching."
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 3a — Load or generate product graph data
# ════════════════════════════════════════════════════════════════
RUN_SYNTHETIC = True  # ← Set to False to use real Kaggle data

loaded_from_kaggle = False

if not RUN_SYNTHETIC and os.path.isdir("/kaggle/input/daemon-data"):
    try:
        print("Loading Kaggle dataset from /kaggle/input/daemon-data/ …")
        edges_cp = np.load("/kaggle/input/daemon-data/cp_edges.npy")
        edges_cv = np.load("/kaggle/input/daemon-data/cv_edges.npy")
        features_np = np.load("/kaggle/input/daemon-data/features.npy")
        loaded_from_kaggle = True
        print(f"  CP edges:    {edges_cp.shape[1]:>12,}")
        print(f"  CV edges:    {edges_cv.shape[1]:>12,}")
        print(f"  Features:    {features_np.shape}")
        print(f"  Feature dim: {features_np.shape[1]}")
    except Exception as e:
        print(f"Could not load Kaggle data: {e}")
        print("Falling back to synthetic data …")

if not loaded_from_kaggle:
    # Try local preprocessed data
    local_data_dir = Path("./data")
    if local_data_dir.is_dir():
        try:
            print("Loading local preprocessed data from ./data/ …")
            edges_cp = np.load(local_data_dir / "cp_edges.npy")
            edges_cv = np.load(local_data_dir / "cv_edges.npy")
            features_np = np.load(local_data_dir / "features.npy")
            loaded_from_kaggle = True
            print(f"  CP edges:    {edges_cp.shape[1]:>12,}")
            print(f"  CV edges:    {edges_cv.shape[1]:>12,}")
            print(f"  Features:    {features_np.shape}")
        except Exception as e:
            print(f"Could not load local data: {e}")

if not loaded_from_kaggle:
    NUM_SYNTHETIC_NODES = 50000
    print("=" * 52)
    print(f"  Generating synthetic graph ({NUM_SYNTHETIC_NODES:,} nodes)")
    print("=" * 52)
    edges_cp, edges_cv, features_np, categories = generate_synthetic_graph(
        num_products=NUM_SYNTHETIC_NODES,
        feature_dim=cfg.in_feats,
        avg_cp_degree=5,
        avg_cv_degree=8,
        asymmetry_ratio=0.75,
        seed=42,
    )

print(f"\\nFinal data shapes:")
print(f"  CP edges:    {edges_cp.shape[1]:>12,}")
print(f"  CV edges:    {edges_cv.shape[1]:>12,}")
print(f"  Features:    {features_np.shape}")
print(f"  Feature dim: {features_np.shape[1]}")
"""
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 3b — Build the DGL graph
# ════════════════════════════════════════════════════════════════
print("Building DGL product graph …")

# Minimal product DataFrame (required by build_product_graph API)
product_df = pd.DataFrame({
    "title": [f"product_{i}" for i in range(features_np.shape[0])],
})
product_df["description"] = product_df["title"]
product_df["category"] = "default"

g, _ = build_product_graph(
    product_df=product_df,
    cp_edges=edges_cp,
    cv_edges=edges_cv,
    feature_dim=cfg.in_feats,
    features=features_np,
)

# Update config with actual graph dimensions
cfg.num_nodes = g.num_nodes()
cfg.num_edges = g.num_edges()

print(f"\\n  Nodes: {g.num_nodes():,}  |  Edges: {g.num_edges():,}")
print(f"  Feature dim: {g.ndata['feat'].shape[1]}  |  Edge types: {g.edata['type'].unique().tolist()}")

# Validate graph integrity
print("\\nValidating graph …")
validate_graph(g)
print("✅ Graph construction and validation passed!")
"""
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 3c — Detailed graph statistics
# ════════════════════════════════════════════════════════════════
print_graph_stats(g)

# Additional diagnostics
src_degrees = g.out_degrees().float()
dst_degrees = g.in_degrees().float()

print(f"\\n  Degree correlation (src vs dst): {torch.corrcoef(torch.stack([src_degrees, dst_degrees]))[0,1]:.4f}")
print(f"  Sparsity: {1 - g.num_edges() / (g.num_nodes() ** 2):.8f}")
"""
        )
    )

    # ========================================================================
    # Cell Group 4: Graph Splitting & DataLoaders
    # ========================================================================
    cells.append(
        new_markdown_cell(
            "## ✂️ Cell Group 4: Graph Splitting & DataLoaders\n\n"
            "Stratified edge split (75/5/20 train/val/test), one-way edge "
            "detection, negative sampler, and DGL DataLoader creation."
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 4a — Edge split by type (stratified)
# ════════════════════════════════════════════════════════════════
split = split_edges_by_type(g, train_ratio=0.75, val_ratio=0.05)

# Assemble edge-ID sets
train_eids = torch.cat([split["train_cp"], split["train_cv"]])
val_eids   = torch.cat([split["val_cp"],   split["val_cv"]])
test_eids  = torch.cat([split["test_cp"],  split["test_cv"]])

# Build boolean masks over ALL edges in the full graph
val_mask  = torch.zeros(g.num_edges(), dtype=torch.bool)
test_mask = torch.zeros(g.num_edges(), dtype=torch.bool)
val_mask[val_eids] = True
test_mask[test_eids] = True

# ── Print summary table ──────────────────────────────────────────────────
n_total = g.num_edges()
print("Edge split summary:")
print(f"  {'Split':<8} {'Edges':>12} {'%':>8}")
print(f"  {'─'*8} {'─'*12} {'─'*8}")
print(f"  {'Train':<8} {len(train_eids):>12,} {100*len(train_eids)/n_total:>7.1f}%")
print(f"  {'Val':<8}   {len(val_eids):>12,} {100*len(val_eids)/n_total:>7.1f}%")
print(f"  {'Test':<8}  {len(test_eids):>12,} {100*len(test_eids)/n_total:>7.1f}%")
print(f"  {'Total':<8} {n_total:>12,} {'100.0%':>8}")
"""
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 4b — One-way edges & negative sampler
# ════════════════════════════════════════════════════════════════
# Find one-way co-purchase edges (asymmetry component of loss)
one_way_u, one_way_v = find_one_way_edges(g)
print(f"One-way co-purchase edges: {len(one_way_u):,}  "
      f"({100*len(one_way_u)/max(g.num_edges(),1):.1f}% of all edges)")

# Negative sampler for link-precision evaluation
neg_sampler = NegativeSampler(
    num_nodes=g.num_nodes(),
    num_neg=cfg.num_neg,
    device=device,
)
print(f"Negative sampler: {cfg.num_neg} negatives / positive pair")
print(f"  (Uniform random across {g.num_nodes():,} nodes)")

# Additional evaluation negatives cache
from data_pipeline import generate_eval_negatives
eval_negatives = generate_eval_negatives(
    g, torch.stack(g.edges())[:, test_mask], num_neg_ratio=1,
)
print(f"Evaluation negatives: {eval_negatives.shape[1]:,}")
"""
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 4c — DGL neighbour-sampler DataLoader for training
# ════════════════════════════════════════════════════════════════
# Subgraph containing only training edges (val/test edges are hidden)
train_g = dgl.edge_subgraph(g, train_eids, relabel_nodes=False)
print(f"Training subgraph:  {train_g.num_nodes():,} nodes,  "
      f"{train_g.num_edges():,} edges")

# Neighbour sampler with decreasing fan-out per layer
sampler = MultiLayerNeighborSampler(list(cfg.num_neighbors))

# Node-based DataLoader — iterates over all training nodes as seeds
train_loader = DGLDataLoader(
    train_g,
    torch.arange(train_g.num_nodes()),
    sampler,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    device=device,
    use_uva=False,
)

n_batches = math.ceil(train_g.num_nodes() / cfg.batch_size)
print(f"Train batches per epoch: ~{n_batches:,}  "
      f"(batch size = {cfg.batch_size})")
print(f"Neighbour fan-outs:      {cfg.num_neighbors}")

# ── Validation data tuple for train_model() ──────────────────────────────
masks = {"val": val_mask}
val_data = (g, masks)

print("✅ DataLoaders ready")
"""
        )
    )

    # ========================================================================
    # Cell Group 5: Model Definition
    # ========================================================================
    cells.append(
        new_markdown_cell(
            "## 🧠 Cell Group 5: Model Definition & Smoke Test\n\n"
            "Instantiate the DAEMON model and AsymmetricLoss, count parameters, "
            "and run a single-batch forward pass to verify correctness."
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 5a — Instantiate DAEMON model and loss
# ════════════════════════════════════════════════════════════════
model = DAEMONModel(cfg).to(device)
criterion = AsymmetricLoss(cfg).to(device)

# Count trainable parameters
n_params = count_parameters(model)
print(f"DAEMON Model — {n_params:,} trainable parameters")
print(f"  Layers:        {cfg.num_layers}")
print(f"  Hidden dim:    {cfg.hidden_dim}")
print(f"  Output dim:    {cfg.out_dim}")
print(f"  Dropout:       {cfg.dropout}")
print(f"  Input proj:    {'Yes' if model.input_proj is not None else 'No (in_feats=hidden_dim)'}")

# Per-layer breakdown
print(f"\\n  Layer breakdown:")
total = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"    {name:40s}  {list(param.shape)!r:20s}  {param.numel():>8,}")
        total += param.numel()
print(f"    {'─' * 70}")
print(f"    {'Total':40s}  {'':20s}  {total:>8,}")
"""
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 5b — Single-batch smoke test (forward + loss)
# ════════════════════════════════════════════════════════════════
print("Running smoke test (single batch) …")
model.eval()

with torch.no_grad():
    # Grab one batch from the loader
    for batch_idx, batch in enumerate(train_loader):
        input_nodes, output_nodes, blocks = batch
        blocks = [b.to(device) for b in blocks]
        batch_inputs = blocks[0].srcdata["feat"]

        src_emb, tgt_emb = model(blocks, batch_inputs)

        # Verify shapes
        print(f"  Input features:  {list(batch_inputs.shape)}  "
              f"(src_nodes={batch_inputs.size(0)}, feat_dim={batch_inputs.size(1)})")
        print(f"  Source emb:      {list(src_emb.shape)}  "
              f"(dst_nodes={src_emb.size(0)}, out_dim={src_emb.size(1)})")
        print(f"  Target emb:      {list(tgt_emb.shape)}")
        print(f"  L2 norm (src):   {src_emb.norm(dim=1).mean().item():.4f}")
        print(f"  L2 norm (tgt):   {tgt_emb.norm(dim=1).mean().item():.4f}")
        print(f"  Device:          {src_emb.device}")

        # Compute loss on this batch
        blk = blocks[-1]
        batch_src, batch_dst = blk.edges()
        etype = blk.edata["type"]
        cp_mask = etype == 0
        cv_mask = etype == 1

        loss, components = criterion(
            src_emb, tgt_emb,
            batch_src[cp_mask], batch_dst[cp_mask],   # co-purchase
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),  # one-way (simplified)
            batch_src[cv_mask], batch_dst[cv_mask],   # co-view
            tgt_emb.size(0),  # use batch output nodes, not full graph
        )

        print(f"\\n  Loss components:")
        for k, v in components.items():
            print(f"    {k}: {v.item():.4f}")
        print(f"  Total loss:      {loss.item():.4f}")

        if torch.isfinite(loss):
            print(f"\\n  ✅ Smoke test PASSED — loss is finite, shapes correct")
        else:
            print(f"\\n  ❌ Smoke test FAILED — loss is NaN/Inf")
        break  # single batch
"""
        )
    )

    # ========================================================================
    # Cell Group 6: Training
    # ========================================================================
    cells.append(
        new_markdown_cell(
            "## 🏋️ Cell Group 6: Full Training with Early Stopping\n\n"
            "Use `setup_training()` to initialise all components, wrap the "
            "loss function for compatibility, and run the full training loop "
            "with validation, checkpointing, and early stopping."
        )
    )

    cells.append(
        new_code_cell(
            '''# ════════════════════════════════════════════════════════════════
# Cell 6a — Training setup with loss wrapper
# ════════════════════════════════════════════════════════════════

class WrappedAsymmetricLoss(nn.Module):
    """Adapter: unpacks batch_data dict for AsymmetricLoss.forward().

    The ``train_epoch`` function in ``training.py`` passes batch data as a
    dict via ``criterion(src_emb, tgt_emb, batch_data)``, but
    ``AsymmetricLoss.forward`` expects individual tensor arguments.
    This wrapper bridges the two APIs.
    """

    def __init__(self, cfg: DAEMONConfig) -> None:
        super().__init__()
        self.inner = AsymmetricLoss(cfg)

    def forward(
        self,
        src_emb: torch.Tensor,
        tgt_emb: torch.Tensor,
        batch_data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.inner(
            src_emb,
            tgt_emb,
            batch_data["cp_u"],
            batch_data["cp_v"],
            batch_data["ow_u"],
            batch_data["ow_v"],
            batch_data["cv_u"],
            batch_data["cv_v"],
            batch_data["num_nodes"],
        )


# Setup all training components via setup_training
train_setup = setup_training(cfg, DAEMONModel, WrappedAsymmetricLoss)

model = train_setup["model"]
optimizer = train_setup["optimizer"]
criterion_wrapped = train_setup["criterion"]
scheduler = train_setup["scheduler"]
scaler = train_setup["scaler"]
start_epoch = train_setup["start_epoch"]

print_memory_summary()
print("✅ Training setup complete — ready to train!")
'''
        )
    )

    # ========================================================================
    # Cell 6b: Checkpoint Resume (survive Kaggle 9-hour session limit)
    # ========================================================================
    cells.append(
        new_code_cell(
            '''# ════════════════════════════════════════════════════════════════
# Cell 6b — Resume from checkpoint if available (Kaggle session recovery)
# ════════════════════════════════════════════════════════════════
resume_path = cfg.checkpoint_path

if os.path.isfile(resume_path):
    print(f"🔍 Found checkpoint: {resume_path}")
    resume_epoch, saved_metrics = load_checkpoint(
        path=resume_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location=device,
    )
    start_epoch = resume_epoch
    print(f"   Resumed from epoch {resume_epoch}")
    print(f"   Saved metrics: AUC={saved_metrics.get('auc', 0):.4f}, "
          f"HitRate@10={saved_metrics.get('hitrate_10', 0):.4f}")
    print("✅ Continuing training from checkpoint")
else:
    print("ℹ️  No checkpoint found — starting training from scratch")
    print(f"   (Checkpoint path: {resume_path})")
'''
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 6c — Run full training loop with early stopping
# ════════════════════════════════════════════════════════════════
print("=" * 62)
print(f"  Starting DAEMON Training — max {cfg.epochs} epochs")
print(f"  Device: {device}  |  AMP: {cfg.use_amp}  |  Patience: {cfg.patience}")
print("=" * 62)
print_memory_summary()

train_history, best_metrics = train_model(
    model=model,
    train_loader=train_loader,
    val_data=(g, masks),
    criterion=criterion_wrapped,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    cfg=cfg,
    resume_epoch=start_epoch,
)

print(f"\\n{'=' * 62}")
print(f"  TRAINING COMPLETE")
print(f"  Best Val AUC:      {best_metrics.get('auc', 0):.4f}")
print(f"  Best HitRate@10:    {best_metrics.get('hitrate_10', 0):.4f}")
print(f"  Best Epoch:         {best_metrics.get('epoch', -1)}")
print(f"{'=' * 62}")

# Clean up to free GPU memory before evaluation
gc.collect()
torch.cuda.empty_cache()
"""
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 6d — Training history visualisation
# ════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("DAEMON Training History", fontsize=14, fontweight="bold")

epochs_range = range(1, len(train_history["train_loss"]) + 1)

# ── Loss curve ───────────────────────────────────────────────────────────
axes[0].plot(epochs_range, train_history["train_loss"], "b-", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Training Loss")
axes[0].set_title("Training Loss")
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(1, len(epochs_range))

# ── Validation AUC ───────────────────────────────────────────────────────
if train_history.get("val_auc"):
    axes[1].plot(epochs_range, train_history["val_auc"], "g-", linewidth=2)
    best_auc = best_metrics.get("auc", 0)
    best_ep = best_metrics.get("epoch", -1) + 1  # 0-indexed → 1-indexed
    axes[1].axhline(y=best_auc, color="r", linestyle="--", alpha=0.7,
                    label=f"Best AUC = {best_auc:.4f} (ep {best_ep})")
    axes[1].legend()
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation AUC")
axes[1].set_title("Validation AUC")
axes[1].grid(True, alpha=0.3)

# ── Learning rate ────────────────────────────────────────────────────────
if train_history.get("lr"):
    axes[2].plot(epochs_range, train_history["lr"], "m-", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(cfg.output_dir, "training_history.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Training history plot saved → {plot_path}")
"""
        )
    )

    # ========================================================================
    # Cell Group 7: Evaluation
    # ========================================================================
    cells.append(
        new_markdown_cell(
            "## 📈 Cell Group 7: Full Evaluation (EQ1–EQ3)\n\n"
            "Load the best checkpoint, run the complete evaluation suite, "
            "and display results for node recommendation, link prediction, "
            "and direction prediction."
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 7a — Load best model checkpoint
# ════════════════════════════════════════════════════════════════
print("Loading best model checkpoint …")

best_path = cfg.checkpoint_path + ".best"
latest_path = cfg.checkpoint_path

eval_model = DAEMONModel(cfg).to(device)

if os.path.isfile(best_path):
    resume_epoch, saved_metrics = load_checkpoint(
        path=best_path,
        model=eval_model,
        map_location=device,
    )
    print(f"  Loaded BEST model (epoch {resume_epoch - 1}, "
          f"AUC={saved_metrics.get('auc', 0):.4f})")
elif os.path.isfile(latest_path):
    resume_epoch, saved_metrics = load_checkpoint(
        path=latest_path,
        model=eval_model,
        map_location=device,
    )
    print(f"  Loaded LATEST model (epoch {resume_epoch - 1})")
else:
    eval_model = model  # fall back to in-memory model
    print("  Using in-memory trained model (no checkpoint found)")

print_memory_summary()
"""
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 7b — Run full evaluation (EQ1: HitRate/MRR, EQ2: Link AUC, EQ3: Direction AUC)
# ════════════════════════════════════════════════════════════════
print("Running full evaluation suite (EQ1–EQ3) …")
print("=" * 62)

eval_masks = {
    "test": test_mask,
}

eval_results = evaluate_full(
    model=eval_model,
    g=g,
    masks=eval_masks,
    cfg=cfg,
    device=device,
    batch_size=4096,
)

# ── Display results ──────────────────────────────────────────────────────
print(f"\\n{'=' * 62}")
print(f"  EVALUATION RESULTS")
print(f"{'=' * 62}")

# EQ1: Node recommendation
if "hitrate" in eval_results:
    print(f"\\n  📌 EQ1 — Node Recommendation")
    print(f"  {'─' * 42}")
    hr = eval_results["hitrate"]
    mrr = eval_results.get("mrr", {})
    print(f"  {'k':<8} {'HitRate@k':<18} {'MRR@k':<18}")
    print(f"  {'─'*8} {'─'*18} {'─'*18}")
    for k in sorted(hr.keys()):
        print(f"  {k:<8} {hr[k]:<18.4f} {mrr.get(k, 0):<18.4f}")

# EQ2: Link prediction
if "auc_link" in eval_results:
    print(f"\\n  📌 EQ2 — Existential Link Prediction")
    print(f"  AUC:  {eval_results['auc_link']:.4f}")

# EQ3: Direction prediction
if "auc_direction" in eval_results:
    print(f"\\n  📌 EQ3 — Direction Link Prediction")
    print(f"  AUC:       {eval_results['auc_direction']:.4f}")
    if "direction_accuracy" in eval_results:
        print(f"  Accuracy:  {eval_results['direction_accuracy']:.4f}  "
              f"(forward > reverse)")

print(f"\\n{'=' * 62}")
"""
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 7c — Formatted results table (pandas DataFrame)
# ════════════════════════════════════════════════════════════════
from IPython.display import display

# Build rows
rows = []
if "hitrate" in eval_results:
    for k in sorted(eval_results["hitrate"].keys()):
        rows.append({"Metric": f"HitRate@{k}", "Type": "EQ1", "Value": eval_results["hitrate"][k]})
if "mrr" in eval_results:
    for k in sorted(eval_results["mrr"].keys()):
        rows.append({"Metric": f"MRR@{k}", "Type": "EQ1", "Value": eval_results["mrr"][k]})
if "auc_link" in eval_results:
    rows.append({"Metric": "Link Prediction AUC", "Type": "EQ2", "Value": eval_results["auc_link"]})
if "auc_direction" in eval_results:
    rows.append({"Metric": "Direction Prediction AUC", "Type": "EQ3", "Value": eval_results["auc_direction"]})
if "direction_accuracy" in eval_results:
    rows.append({"Metric": "Direction Accuracy", "Type": "EQ3", "Value": eval_results["direction_accuracy"]})

if rows:
    results_df = pd.DataFrame(rows)
    print("\\n📊 Evaluation Metrics Summary:")
    display(results_df.style.hide(axis="index").format({"Value": "{:.4f}"}))
else:
    print("No evaluation results available.")
"""
        )
    )

    # ========================================================================
    # Cell Group 8: FAISS Indexing
    # ========================================================================
    cells.append(
        new_markdown_cell(
            "## ⚡ Cell Group 8: FAISS Indexing & Latency Benchmark\n\n"
            "Generate full-graph embeddings, build a GPU-accelerated FAISS "
            "index (IVF with inner-product search), and benchmark query latency."
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 8a — Generate full-graph embeddings
# ════════════════════════════════════════════════════════════════
print("Generating full-graph embeddings …")
t_start = time.time()

embeddings = generate_all_embeddings(
    model=eval_model,
    g=g,
    batch_size=4096,
    device=device,
)

t_elapsed = time.time() - t_start

if isinstance(embeddings, tuple):
    src_emb, tgt_emb = embeddings
    print(f"  Source embeddings: {list(src_emb.shape)}  "
          f"[{src_emb.device}]")
    print(f"  Target embeddings: {list(tgt_emb.shape)}  "
          f"[{tgt_emb.device}]")
    # Use source embeddings for queries, target for candidates
    query_emb = src_emb
    candidate_emb = tgt_emb
else:
    query_emb = embeddings
    candidate_emb = embeddings
    print(f"  Unified embeddings: {list(embeddings.shape)}")

print(f"  Generation time: {t_elapsed:.2f}s  "
      f"({g.num_nodes() / t_elapsed:.0f} nodes/sec)")
print_memory_summary()
"""
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 8b — Build FAISS (IVF) index + latency benchmark
# ════════════════════════════════════════════════════════════════
print("Building FAISS IVF index …")
faiss_index = None

try:
    t_start = time.time()

    faiss_index = build_faiss_index(
        embeds=candidate_emb,
        use_gpu=torch.cuda.is_available(),
        nprobe=32,
    )

    t_elapsed = time.time() - t_start
    print(f"  Index built in {t_elapsed:.2f}s")
    print(f"  Type:     {type(faiss_index).__name__}")
    print(f"  Vectors:  {candidate_emb.shape[0]:,}")
    print(f"  Dim:      {candidate_emb.shape[1]}")
    print(f"  nprobe:   {getattr(faiss_index, 'nprobe', 'N/A')}")

    # ── Latency benchmark ─────────────────────────────────────────────────
    print(f"\\n  Running latency benchmark (10 random queries) …")
    rng = np.random.default_rng(42)
    test_queries = rng.integers(0, g.num_nodes(), size=10)

    latencies = []
    for qidx in test_queries:
        t0 = time.time()
        rec_indices, rec_dist = recommend_related(
            embeds=query_emb,
            index=faiss_index,
            product_idx=int(qidx),
            k=10,
        )
        latencies.append((time.time() - t0) * 1000)  # ms

    print(f"  Mean latency:  {np.mean(latencies):.1f} ms")
    print(f"  Median:        {np.median(latencies):.1f} ms")
    print(f"  P99:           {np.percentile(latencies, 99):.1f} ms")
    print(f"  Min / Max:     {np.min(latencies):.1f} / {np.max(latencies):.1f} ms")
    print(f"  Throughput:    {1000 / np.mean(latencies):.0f} queries/sec")

except ImportError as e:
    print(f"  FAISS not available: {e}")
    print("  Skipping FAISS index — will use brute-force fallback.")
except Exception as e:
    print(f"  FAISS build failed: {e}")
    print("  Skipping FAISS index.")
"""
        )
    )

    # ========================================================================
    # Cell Group 9: Recommendation Demo
    # ========================================================================
    cells.append(
        new_markdown_cell(
            "## 🎯 Cell Group 9: Recommendation Demo\n\n"
            "Pick 5 sample products and show their top-10 related recommendations."
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 9 — Recommend related products for 5 sample queries
# ════════════════════════════════════════════════════════════════
print("Top-10 Recommendations for 5 Sample Products")
print("=" * 70)

rng = np.random.default_rng(123)
sample_products = rng.integers(0, min(g.num_nodes(), 50000), size=5)

if faiss_index is not None:
    # ── FAISS-powered recommendations ────────────────────────────────────────
    for i, prod_idx in enumerate(sample_products):
        rec_indices, rec_dist = recommend_related(
            embeds=query_emb,
            index=faiss_index,
            product_idx=int(prod_idx),
            k=10,
        )

        print(f"\\n📌 Query Product #{int(prod_idx):<8} (Demo {i + 1})")
        print(f"  {'─' * 60}")
        print(f"  {'Rank':<6} {'Product ID':<14} {'Score':<10}")
        print(f"  {'─' * 6} {'─' * 14} {'─' * 10}")
        for rank, (ridx, dist) in enumerate(zip(rec_indices, rec_dist), 1):
            print(f"  {rank:<6} {int(ridx):<14} {float(dist):.4f}")
else:
    # ── Brute-force fallback ─────────────────────────────────────────────
    print("(FAISS unavailable — using brute-force torch search)")
    query_np = query_emb.cpu().numpy()
    candidate_np = candidate_emb.cpu().numpy()

    for i, prod_idx in enumerate(sample_products):
        q_vec = query_np[int(prod_idx)]
        scores = candidate_np @ q_vec
        top_k = np.argsort(scores)[::-1]
        top_k = top_k[top_k != int(prod_idx)][:10]

        print(f"\\n📌 Query Product #{int(prod_idx):<8} (Demo {i + 1})")
        print(f"  {'─' * 60}")
        print(f"  {'Rank':<6} {'Product ID':<14} {'Score':<10}")
        print(f"  {'─' * 6} {'─' * 14} {'─' * 10}")
        for rank, ridx in enumerate(top_k, 1):
            print(f"  {rank:<6} {int(ridx):<14} {float(scores[ridx]):.4f}")

print(f"\\n{'=' * 70}")
print("Recommendation demo complete!")
"""
        )
    )

    # ========================================================================
    # Cell Group 10: Cold-Start Demo
    # ========================================================================
    cells.append(
        new_markdown_cell(
            "## 🆕 Cell Group 10: Cold-Start Recommendation Demo\n\n"
            "Simulate a new product with no purchase/view history — recommend "
            "based solely on its feature vector."
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 10 — Cold-start product recommendation (k-NN + graph augmentation)
# ════════════════════════════════════════════════════════════════
if faiss_index is None:
    print("⚠️  FAISS index unavailable — cold-start demo requires it.")
    print("   Falling back to brute-force feature similarity.")
    # Show simple feature-space neighbours as a proxy
    rng = np.random.default_rng(42)
    new_feat = rng.standard_normal(cfg.in_feats).astype(np.float32)
    feat_np = features_np.astype(np.float32)
    sims = feat_np @ new_feat
    top_k = np.argsort(sims)[::-1][:10]
    print(f"\\nTop-10 feature-similar products (cosine):")
    for rank, ridx in enumerate(top_k, 1):
        print(f"  {rank:<4} Product #{int(ridx):<8}  sim={float(sims[ridx]):.4f}")
else:
    print("Cold-Start Recommendation Demo")
    print("=" * 60)
    print("  Simulating a new product with feature metadata only")
    print("  (no edges in the graph — pure cold start)")
    print("=" * 60)

    # Generate random feature vector (would come from product catalog in prod)
    rng = np.random.default_rng(42)
    new_features = torch.tensor(
        rng.standard_normal(cfg.in_feats), dtype=torch.float32
    )

    # Get existing feature matrix from the graph
    existing_features = g.ndata["feat"].cpu()

    # Run cold-start recommendation via k-NN + graph augmentation + GNN forward
    # (cold_start_recommend handles projection internally using the trained model)
    rec_indices, rec_dist = cold_start_recommend(
        model=eval_model,
        index=faiss_index,
        new_features=new_features,
        g=g,
        existing_features=existing_features,
        k_nn=5,
        k=10,
        device=device,
    )

    print(f"\\nTop-10 Recommendations for Cold-Start Product:")
    print(f"  {'─' * 50}")
    print(f"  {'Rank':<6} {'Product ID':<14} {'Score':<10}")
    print(f"  {'─' * 6} {'─' * 14} {'─' * 10}")
    for rank, (ridx, dist) in enumerate(zip(rec_indices, rec_dist), 1):
        print(f"  {rank:<6} {int(ridx):<14} {float(dist):.4f}")

    print(f"\\n✅ Cold-start recommendations generated!")
    print("Note: Cold-start quality depends on feature-representation alignment.")
"""
        )
    )

    # ========================================================================
    # Cell Group 11: Ablation Studies
    # ========================================================================
    cells.append(
        new_markdown_cell(
            "## 🔬 Cell Group 11: Ablation Studies (Optional)\n\n"
            "Quick comparison of loss-component variants. Train for 5 epochs "
            "each with: full loss, no asymmetry term, and no co-view term."
        )
    )

    cells.append(
        new_code_cell(
            '''# ════════════════════════════════════════════════════════════════
# Cell 11a — Ablation: loss-component comparison
# ════════════════════════════════════════════════════════════════
print("Ablation Study — Loss Component Comparison")
print("=" * 60)
print("Training 3 variants for 5 epochs each (quick comparison) …")
print("=" * 60)

ABLATION_EPOCHS = 5
ablation_results: Dict[str, Dict[str, float]] = {}

# Smaller subgraph for faster ablation (500K edges max)
ablation_eids = train_eids[:min(len(train_eids), 500_000)]
abl_g = dgl.edge_subgraph(g, ablation_eids, relabel_nodes=False)

abl_loader = DGLDataLoader(
    abl_g,
    torch.arange(abl_g.num_nodes()),
    MultiLayerNeighborSampler(list(cfg.num_neighbors)),
    batch_size=512,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    device=device,
)


class AblationLoss(nn.Module):
    """AsymmetricLoss with configurable component toggles."""

    def __init__(self, cfg: DAEMONConfig, use_ow: bool, use_cv: bool):
        super().__init__()
        self.inner = AsymmetricLoss(cfg)
        self.use_ow = use_ow
        self.use_cv = use_cv

    def forward(self, src_emb, tgt_emb, cp_u, cp_v, ow_u, ow_v, cv_u, cv_v, num_nodes):
        if not self.use_ow:
            ow_u = ow_u.new_empty(0)
            ow_v = ow_v.new_empty(0)
        if not self.use_cv:
            cv_u = cv_u.new_empty(0)
            cv_v = cv_v.new_empty(0)
        return self.inner(src_emb, tgt_emb, cp_u, cp_v,
                          ow_u, ow_v, cv_u, cv_v, num_nodes)


for variant_name, use_ow, use_cv in [
    ("Full (CP + OW + CV)", True, True),
    ("No Asymmetry (CP + CV)", False, True),
    ("No Co-view (CP + OW)", True, False),
]:
    print(f"\\n{'─' * 55}")
    print(f"  Variant: {variant_name}")
    print(f"{'─' * 55}")

    # Fresh model and optimiser per variant
    abl_model = DAEMONModel(cfg).to(device)
    abl_opt = torch.optim.Adam(
        abl_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    abl_loss_fn = AblationLoss(cfg, use_ow, use_cv).to(device)
    abl_scaler = GradScaler(enabled=cfg.use_amp)

    losses = []
    for epoch in range(ABLATION_EPOCHS):
        abl_model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in abl_loader:
            _in, _out, blocks = batch
            blocks = [b.to(device) for b in blocks]
            h = blocks[0].srcdata["feat"]

            with autocast(enabled=cfg.use_amp):
                s_emb, t_emb = abl_model(blocks, h)
                blk = blocks[-1]
                src, dst = blk.edges()
                etype = blk.edata["type"]
                cp_mask = etype == 0
                cv_mask = etype == 1

                loss, _ = abl_loss_fn(
                    s_emb, t_emb,
                    src[cp_mask], dst[cp_mask],
                    src[cp_mask].new_empty(0), dst[cp_mask].new_empty(0),
                    src[cv_mask], dst[cv_mask],
                    g.num_nodes(),
                )

            abl_scaler.scale(loss).backward()
            abl_scaler.step(abl_opt)
            abl_scaler.update()
            abl_opt.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        losses.append(avg_loss)
        print(f"  Epoch {epoch + 1:2d}:  Loss = {avg_loss:.4f}")

    # Quick evaluation
    print(f"  Evaluating …")
    abl_model.eval()
    try:
        with torch.no_grad():
            s_emb, t_emb = generate_all_embeddings(
                abl_model, g, batch_size=4096, device=device
            )
            s_emb, t_emb = s_emb.to(device), t_emb.to(device)
            test_edges_2col = torch.stack(g.edges(), dim=1)[test_mask]
            q, t = test_edges_2col[:, 0], test_edges_2col[:, 1]
            hr10 = compute_hit_rate_at_k(s_emb, t_emb, q, t, k=10)
            mrr10 = compute_mrr_at_k(s_emb, t_emb, q, t, k=10)
            ablation_results[variant_name] = {
                "hitrate_10": float(f"{hr10:.4f}"),
                "mrr_10": float(f"{mrr10:.4f}"),
                "loss_5ep": float(f"{losses[-1]:.4f}"),
            }
            print(f"  → HitRate@10: {hr10:.4f}  |  MRR@10: {mrr10:.4f}")
    except Exception as exc:
        print(f"  → Eval skipped: {exc}")

    # Clean up
    del abl_model, abl_opt, abl_loss_fn, abl_scaler, s_emb, t_emb
    torch.cuda.empty_cache()
    gc.collect()

# ── Ablation summary table ────────────────────────────────────────────────
print(f"\\n{'=' * 60}")
print(f"  ABLATION SUMMARY")
print(f"{'=' * 60}")
ab_df = pd.DataFrame([
    {"Variant": k, "HitRate@10": v["hitrate_10"],
     "MRR@10": v["mrr_10"], "Loss (ep5)": v["loss_5ep"]}
    for k, v in ablation_results.items()
])
display(ab_df.style.hide(axis="index"))
print(f"\\nNote: Each variant trained for only {ABLATION_EPOCHS} epochs")
print("      (full results require longer training)")
'''
        )
    )

    # ========================================================================
    # Cell Group 12: Export Results
    # ========================================================================
    cells.append(
        new_markdown_cell(
            "## 📦 Cell Group 12: Export Results & Summary Dashboard\n\n"
            "Save all outputs (model, embeddings, FAISS index, metrics), "
            "display a final summary, and show download links."
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 12a — Save model, embeddings, FAISS index, metrics
# ════════════════════════════════════════════════════════════════
out = Path(cfg.output_dir)
out.mkdir(parents=True, exist_ok=True)

# ── 1. Model checkpoint ──────────────────────────────────────────────────
torch.save(eval_model.state_dict(), out / "daemon_model.pt")
print(f"✅ Model:          {out / 'daemon_model.pt'}  "
      f"({os.path.getsize(out / 'daemon_model.pt') / 1e6:.1f} MB)")

# ── 2. Embeddings ────────────────────────────────────────────────────────
if isinstance(embeddings, tuple):
    torch.save(embeddings[0], out / "source_embeddings.pt")
    torch.save(embeddings[1], out / "target_embeddings.pt")
    print(f"✅ Source emb:     {out / 'source_embeddings.pt'}  "
          f"{list(embeddings[0].shape)}")
    print(f"✅ Target emb:     {out / 'target_embeddings.pt'}  "
          f"{list(embeddings[1].shape)}")
else:
    torch.save(embeddings, out / "embeddings.pt")
    print(f"✅ Embeddings:     {out / 'embeddings.pt'}  "
          f"{list(embeddings.shape)}")

# ── 3. FAISS index ───────────────────────────────────────────────────────
if faiss_index is not None:
    try:
        import faiss
        idx_to_save = faiss_index
        if hasattr(faiss_index, "index") and faiss_index.index is not None:
            # GPU → CPU conversion for serialisation
            idx_to_save = faiss.index_gpu_to_cpu(faiss_index)
        faiss.write_index(idx_to_save, str(out / "faiss_index.index"))
        print(f"✅ FAISS index:    {out / 'faiss_index.index'}  "
              f"({os.path.getsize(out / 'faiss_index.index') / 1e6:.1f} MB)")
    except Exception as e:
        print(f"⚠️  FAISS save skipped: {e}")

# ── 4. Metrics JSON ──────────────────────────────────────────────────────
serialisable = {}
for k, v in eval_results.items():
    if isinstance(v, dict):
        serialisable[k] = {str(kk): round(float(vv), 6) for kk, vv in v.items()}
    elif isinstance(v, (np.floating, np.integer)):
        serialisable[k] = float(v)
    elif isinstance(v, torch.Tensor):
        serialisable[k] = float(v.item())
    else:
        serialisable[k] = v

serialisable["best_training_metrics"] = {
    "best_val_auc": float(best_metrics.get("auc", 0)),
    "best_hitrate_10": float(best_metrics.get("hitrate_10", 0)),
    "best_epoch": int(best_metrics.get("epoch", -1)),
    "total_epochs": len(train_history.get("train_loss", [])),
}

with open(out / "metrics.json", "w") as f:
    json.dump(serialisable, f, indent=2)
print(f"✅ Metrics JSON:   {out / 'metrics.json'}  "
      f"({os.path.getsize(out / 'metrics.json') / 1e3:.1f} KB)")

# ── 5. Copy training plot if exists ──────────────────────────────────────
plot_src = Path(cfg.output_dir) / "training_history.png"
if plot_src.exists():
    print(f"✅ Training plot:  {plot_src}")

print(f"\\n📂 All outputs saved to: {out.resolve()}")
"""
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 12b — Final summary dashboard
# ════════════════════════════════════════════════════════════════
print("=" * 62)
print("  🏆  DAEMON KAGGLE NOTEBOOK — FINAL SUMMARY")
print("=" * 62)

print(f"\\n  📐 Configuration")
print(f"  {'─' * 58}")
print(f"    {'Nodes':<20} {g.num_nodes():>12,}")
print(f"    {'Edges':<20} {g.num_edges():>12,}")
print(f"    {'Feature dim':<20} {cfg.in_feats:>12}")
print(f"    {'Hidden dim':<20} {cfg.hidden_dim:>12}")
print(f"    {'Output dim':<20} {cfg.out_dim:>12}")
print(f"    {'Trainable params':<20} {count_parameters(eval_model):>12,}")

print(f"\\n  🏋️  Training")
print(f"  {'─' * 58}")
print(f"    {'Epochs completed':<20} {len(train_history['train_loss']):>12}")
print(f"    {'Best Val AUC':<20} {best_metrics.get('auc', 0):>12.4f}")
print(f"    {'Best HitRate@10':<20} {best_metrics.get('hitrate_10', 0):>12.4f}")

print(f"\\n  📈 Evaluation")
print(f"  {'─' * 58}")
if "hitrate" in eval_results:
    for k in sorted(eval_results["hitrate"].keys()):
        print(f"    {'HitRate@' + str(k):<20} {eval_results['hitrate'][k]:>12.4f}")
if "auc_link" in eval_results:
    print(f"    {'Link AUC (EQ2)':<20} {eval_results['auc_link']:>12.4f}")
if "auc_direction" in eval_results:
    print(f"    {'Direction AUC (EQ3)':<20} {eval_results['auc_direction']:>12.4f}")

print(f"\\n  ⚡ Indexing")
print(f"  {'─' * 58}")
print(f"    {'Embedding dim':<20} {candidate_emb.shape[1]:>12}")
print(f"    {'Index type':<20} {'FAISS IVF' if faiss_index is not None else 'N/A':>12}")

print(f"\\n  📦 Outputs ({cfg.output_dir})")
print(f"  {'─' * 58}")
for fname in sorted(os.listdir(cfg.output_dir)):
    fpath = os.path.join(cfg.output_dir, fname)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        unit = "MB" if size > 1e6 else "KB"
        sz = size / 1e6 if size > 1e6 else size / 1e3
        print(f"    {fname:<30} {sz:>8.1f} {unit}")

print(f"\\n{'=' * 62}")
print(f"  ✅ NOTEBOOK COMPLETE")
print(f"{'=' * 62}")
"""
        )
    )

    cells.append(
        new_code_cell(
            """# ════════════════════════════════════════════════════════════════
# Cell 12c — Download instructions & file links
# ════════════════════════════════════════════════════════════════
from IPython.display import FileLink, display

print("📥 Download Output Files")
print("─" * 40)

out = Path(cfg.output_dir)
files_found = 0

# List each output file as a clickable link
for fname in ["daemon_model.pt", "metrics.json", "faiss_index.index",
              "source_embeddings.pt", "target_embeddings.pt",
              "embeddings.pt", "training_history.png"]:
    fpath = out / fname
    if fpath.exists():
        display(FileLink(str(fpath), result_html_prefix=f"📎 "))
        files_found += 1

if files_found == 0:
    print("(No output files found — run earlier cells first)")

print(f"\\n{files_found} file(s) available for download\\n")

# Helper: create ZIP archive
print("To download everything as a single ZIP:")
print(f"  !zip -r {cfg.output_dir}/daemon_results.zip {cfg.output_dir}/")
print(f"  FileLink(r'{cfg.output_dir}/daemon_results.zip')")
print()
print("Or list all output files:")
print(f"  !ls -lh {cfg.output_dir}/")
"""
        )
    )

    # Assemble notebook
    nb.cells = cells
    return nb


if __name__ == "__main__":
    nb = make_notebook()
    output_path = "/home/shashwat-11/ML/daemon_kaggle.ipynb"
    with open(output_path, "w") as f:
        nbf.write(nb, f)
    print(f"✅ Notebook generated: {output_path}")
    print(f"   Total cells: {len(nb.cells)}")
    print(f"   Ready to run on Kaggle (or locally with synthetic data).")
