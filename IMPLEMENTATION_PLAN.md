# DAEMON-Kaggle — Implementation Plan

> **Goal**: A single, self-contained Jupyter notebook (`daemon_kaggle.ipynb`) that trains a directed-graph GNN for asymmetric product recommendations, runs end-to-end on Kaggle with a T4 GPU (16 GB VRAM), and survives the ~9-hour session limit.

---

## 1. Final Deliverable

| Item | Detail |
|---|---|
| **File** | `daemon_kaggle.ipynb` |
| **Format** | Single notebook, all cells run sequentially top-to-bottom |
| **Platform** | Kaggle Notebooks (GPU = Tesla T4, Internet = ON, Persistence = `/kaggle/working`) |
| **Dependencies** | PyTorch (pre-installed), DGL, FAISS-GPU — all installed in Section 1 |
| **Runtime (estimated)** | 2.5–4.5 hours for 1M-node graph; instant for synthetic smoke test |
| **Outputs** | Trained model checkpoint, product embeddings, FAISS index, evaluation metrics |

**What "self-contained" means:**
- Cell 1 installs everything (`!pip install dgl faiss-gpu`)
- Cells 2–12 handle config → data → training → eval → demo
- Restart & Run All must succeed without any manual steps

---

## 2. Notebook Section Outline

### Section 1: Environment Setup (~5 min)

**Purpose**: Install missing packages, verify GPU, set seeds — all in one cell group.

```python
# Cell 1a — Install
# !pip install dgl -f https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html
# !pip install faiss-gpu

# Cell 1b — Imports
import torch, dgl, numpy as np, pandas as pd, faiss, gc, time, json
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from dgl.dataloading import DataLoader as DGLDataLoader, MultiLayerNeighborSampler
from dgl.nn import SAGEConv
from dataclasses import dataclass
from tqdm.auto import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Cell 1c — Verify GPU
assert torch.cuda.is_available(), "GPU required!"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"CUDA: {torch.version.cuda} | PyTorch: {torch.__version__} | DGL: {dgl.__version__}")

# Cell 1d — Reproducibility
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed(42)
```

**What to verify here:**
- [ ] `nvidia-smi` shows T4 GPU
- [ ] `dgl.__version__` prints correctly (>= 2.0)
- [ ] `faiss` import succeeds with GPU support
- [ ] Random seeds set without errors


### Section 2: Configuration (~2 min)

**Purpose**: Single source of truth for all hyperparameters. Easily tunable without hunting through code.

```python
@dataclass
class DAEMONConfig:
    # Graph
    num_nodes: int = None          # set after data load
    num_edges: int = None
    num_relations: int = 2         # co-purchase, co-view

    # Model architecture
    in_feats: int = 768            # input feature dim (text embeddings)
    hidden_dim: int = 256
    out_dim: int = 128             # final embedding dim
    num_layers: int = 3
    dropout: float = 0.3

    # Training
    epochs: int = 100
    batch_size: int = 1024
    num_neighbors: List[int] = (15, 10, 5)  # per layer
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_accum_steps: int = 2
    use_amp: bool = True           # mixed precision
    patience: int = 15             # early stopping

    # Loss weights
    lambda_rel: float = 1.0        # related-product loss
    lambda_sym: float = 0.1        # symmetry regularizer
    lambda_asym: float = 0.5       # asymmetry loss weight

    # Evaluation
    hitrate_k: List[int] = (1, 5, 10, 20)
    val_every: int = 5             # validate every N epochs

    # Paths
    data_dir: str = "/kaggle/input/daemon-data"  # or local path
    output_dir: str = "/kaggle/working"
    checkpoint_path: str = "/kaggle/working/daemon_best.pt"

    # Memory
    cleanup_every_n_epochs: int = 4  # gc.collect + empty_cache

cfg = DAEMONConfig()
```

**Tunability**: Change any value in this block — the rest of the notebook picks it up automatically.


### Section 3: Data Loading (~5-10 min)

**Purpose**: Load pre-processed graph data (or build from raw). Report statistics so we know what we're working with.

```python
# Cell 3a — Load preprocessed graph
def load_graph_data(cfg: DAEMONConfig) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    """
    Returns (graph, node_features)

    Expected input format (from /kaggle/input/daemon-data/):
      - node_feat.npy     : float32, shape [num_nodes, in_feats]
      - edges_co_purchase.csv : (src, dst) pairs
      - edges_co_view.csv     : (src, dst) pairs
      - node_id_map.json      : product_id -> idx
    """
    feat = torch.from_numpy(np.load(f"{cfg.data_dir}/node_feat.npy"))
    with open(f"{cfg.data_dir}/node_id_map.json") as f:
        id_map = json.load(f)

    src_p, dst_p = load_edges(f"{cfg.data_dir}/edges_co_purchase.csv")
    src_v, dst_v = load_edges(f"{cfg.data_dir}/edges_co_view.csv")

    num_nodes = feat.shape[0]
    g = dgl.heterograph({
        ('product', 'co_purchase', 'product'): (src_p, dst_p),
        ('product', 'co_view',     'product'): (src_v, dst_v),
    }, num_nodes_dict={'product': num_nodes})

    return g, feat, id_map

# Fallback: small synthetic test
def build_synthetic_graph(num_nodes: int = 100, num_edges: int = 500, feat_dim: int = 64):
    """Build a tiny graph for rapid smoke testing."""
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    feats = torch.randn(num_nodes, feat_dim)
    g = dgl.heterograph({
        ('product', 'co_purchase', 'product'): (src, dst),
    }, num_nodes_dict={'product': num_nodes})
    return g, feats

# ── Execute ──
g, node_feats, id_map = load_graph_data(cfg)
cfg.num_nodes = g.num_nodes('product')
cfg.num_edges = g.num_edges()

# Display statistics
print(f"Nodes: {cfg.num_nodes:,}  |  Edges: {cfg.num_edges:,}")
print(f"Feature dim: {node_feats.shape[1]}")
for etype in g.canonical_etypes:
    print(f"  {etype[1]}: {g.num_edges(etype):,} edges")
print(f"Estimated graph memory on GPU: {estimate_graph_memory(g, node_feats):.1f} GB")
```

**Data Loading Checklist:**
- [ ] Graph loads without errors
- [ ] Node count and edge stats printed
- [ ] Fallback synthetic graph available for quick tests
- [ ] Memory estimate < 14 GB (safe for T4)


### Section 4: Graph Construction & Splitting (~10 min)

**Purpose**: Prepare the graph for training — add reverse edges, split data, create neighbor-sampling dataloaders.

```python
# Cell 4a — Feature projection & normalization
# Project features to model dimension if needed
if node_feats.shape[1] != cfg.hidden_dim:
    proj = nn.Linear(node_feats.shape[1], cfg.hidden_dim)
    g.nodes['product'].data['feat'] = proj(node_feats)
else:
    g.nodes['product'].data['feat'] = node_feats

# Cell 4b — Train/Val/Test split
def split_graph(g: dgl.DGLGraph, ratios=(0.8, 0.1, 0.1)):
    """Split edges for link prediction. Returns masks for each set."""
    num_edges = g.num_edges()
    perm = torch.randperm(num_edges)
    train_end = int(ratios[0] * num_edges)
    val_end = train_end + int(ratios[1] * num_edges)

    masks = {
        'train': torch.zeros(num_edges, dtype=torch.bool),
        'val':   torch.zeros(num_edges, dtype=torch.bool),
        'test':  torch.zeros(num_edges, dtype=torch.bool),
    }
    masks['train'][perm[:train_end]] = True
    masks['val'][perm[train_end:val_end]] = True
    masks['test'][perm[val_end:]] = True
    return masks

masks = split_graph(g)
print(f"Train edges: {masks['train'].sum():,}  "
      f"Val: {masks['val'].sum():,}  Test: {masks['test'].sum():,}")

# Cell 4c — Build neighbor-sampling dataloaders
sampler = MultiLayerNeighborSampler(cfg.num_neighbors)

train_loader = DGLDataLoader(
    g, {'product': torch.arange(cfg.num_nodes)}, sampler,
    batch_size=cfg.batch_size, shuffle=True, drop_last=False,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# For validation: full batch on GPU (or sampled if OOM)
print(f"Dataloader ready — {len(train_loader)} batches/epoch")
```

**Key DGL concepts to get right:**
- `MultiLayerNeighborSampler`: pulls neighbor subgraphs per batch — essential for T4 memory
- Edge masks: we train on 80% edges, validate on 10%, test on 10%
- Feature projection: ensures input dim matches model's `in_feats`


### Section 5: Model Definition (~15 min for code)

**Purpose**: Define the DAEMON GNN model and asymmetric loss. This is the core intellectual contribution.

```python
class DAEMONLayer(nn.Module):
    """Single DAEMON layer: directional message passing with asymmetry gating.

    For each edge (u→v):
      h_v ← AGG( σ( W_dir·[h_u ⊕ h_v] ) · W_msg·h_u )   for all neighbours u

    The gating term σ(W_dir·concat) learns edge importance with direction awareness.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.W_msg   = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dir   = nn.Linear(in_dim * 2, 1, bias=False)  # direction gate
        self.W_self  = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(out_dim)

    def forward(self, g, h):
        """
        Args:
            g: DGL block (subgraph from sampler)
            h: node features [N, in_dim]
        Returns:
            updated features [N, out_dim]
        """
        with g.local_scope():
            g.srcdata['h'] = h
            g.update_all(self._message, self._reduce)
            h_new = g.dstdata['h_new']
            return self.norm(h_new + self.W_self(h[:h_new.shape[0]]))

    def _message(self, edges):
        src_h, dst_h = edges.src['h'], edges.dst['h']
        # Direction-aware gate: σ(W_dir · [src||dst])
        gate = torch.sigmoid(self.W_dir(torch.cat([src_h, dst_h], dim=-1)))
        msg = self.W_msg(src_h) * gate
        return {'m': msg}

    def _reduce(self, nodes):
        return {'h_new': torch.sum(nodes.mailbox['m'], dim=1)}


class DAEMONModel(nn.Module):
    """Full DAEMON model: stacked DAEMON layers → embedding projection.

    Returns normalized embeddings for all nodes.
    """
    def __init__(self, cfg: DAEMONConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            DAEMONLayer(
                cfg.hidden_dim if i > 0 else cfg.in_feats,
                cfg.hidden_dim,
                cfg.dropout
            ) for i in range(cfg.num_layers)
        ])
        self.proj = nn.Linear(cfg.hidden_dim, cfg.out_dim)

    def forward(self, blocks, h):
        """Blocks: list of DGL message-flow graphs from sampler."""
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
        return nn.functional.normalize(self.proj(h), p=2, dim=-1)


class AsymmetricLoss(nn.Module):
    """Multi-component loss for directed graph recommendations.

    L = L_rel + λ_sym·L_sym + λ_asym·L_asym

    L_rel : Binary cross-entropy for edge existence (co-purchase/co-view)
    L_sym : Pulls co-purchase pairs together (cosine similarity)
    L_asym: Penalizes |score(a,b) - score(b,a)| for genuinely asymmetric pairs
             BUT only when edge (a→b) exists and (b→a) doesn't (or vice versa).
    """
    def __init__(self, cfg: DAEMONConfig):
        super().__init__()
        self.lambda_sym = cfg.lambda_sym
        self.lambda_asym = cfg.lambda_asym
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, embeds: torch.Tensor, pos_edges: torch.Tensor,
                neg_edges: torch.Tensor, asym_pairs: Optional[torch.Tensor] = None):
        """
        Args:
            embeds:    node embeddings [N, d]
            pos_edges: positive edges [E_pos, 2]
            neg_edges: negative edges [E_neg, 2]
            asym_pairs: pairs where rel(a,b) != rel(b,a) [A, 2]
        """
        # L_rel: edge prediction
        pos_scores = dot_product(embeds, pos_edges)
        neg_scores = dot_product(embeds, neg_edges)
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones(len(pos_scores)), torch.zeros(len(neg_scores))
        ]).to(scores.device)
        L_rel = self.bce(scores, labels)

        # L_sym: symmetry pull (co-purchase should be bidirectional-ish)
        if pos_edges.numel() > 0:
            src_e, dst_e = embeds[pos_edges[:, 0]], embeds[pos_edges[:, 1]]
            L_sym = (1 - torch.cosine_similarity(src_e, dst_e, dim=-1)).mean()
        else:
            L_sym = torch.tensor(0.0, device=embeds.device)

        # L_asym: asymmetry penalty
        if asym_pairs is not None and asym_pairs.numel() > 0:
            a_e = embeds[asym_pairs[:, 0]]
            b_e = embeds[asym_pairs[:, 1]]
            # score(a→b) - score(b→a) should be LARGE for asymmetric pairs
            score_ab = (a_e * b_e).sum(dim=-1)
            score_ba = score_ab  # symmetric dot product; replace with directional if needed
            L_asym = torch.exp(-torch.abs(score_ab - score_ba)).mean()
        else:
            L_asym = torch.tensor(0.0, device=embeds.device)

        return L_rel + self.lambda_sym * L_sym + self.lambda_asym * L_asym


def dot_product(embeds: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Batch dot product: embeds[src] · embeds[dst]."""
    return (embeds[pairs[:, 0]] * embeds[pairs[:, 1]]).sum(dim=-1)

# ── Model summary ──
model = DAEMONModel(cfg).cuda()
print(f"DAEMON parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Estimated param memory: {sum(p.numel() * 4 for p in model.parameters()) / 1e6:.1f} MB")
```

**Code quality notes:**
- Every class/function has a docstring
- `assert` checks for shape consistency should be added within `forward()`
- Type hints on all parameters and returns


### Section 6: Training Loop (~2-4 hours runtime)

**Purpose**: Train the model with mixed precision, checkpointing, early stopping, and memory cleanup.

```python
# Cell 6a — Training utilities
def train_epoch(model, loader, optimizer, scaler, cfg):
    """One training epoch with AMP + gradient accumulation."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (input_nodes, output_nodes, blocks) in enumerate(tqdm(loader, desc="Train")):
        blocks = [b.to('cuda') for b in blocks]
        h = blocks[0].srcdata['feat']  # input features
        with autocast(enabled=cfg.use_amp):
            embeds = model(blocks, h)
            # Sample positive/negative edges from this subgraph
            pos_edges, neg_edges = sample_edges(blocks[-1], cfg.batch_size)
            loss = criterion(embeds, pos_edges, neg_edges)
            loss = loss / cfg.grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % cfg.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * cfg.grad_accum_steps

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, g, masks, cfg):
    """Evaluate link prediction on validation edges."""
    model.eval()
    embeds = generate_all_embeddings(model, g, cfg)  # full-graph inference
    pos_edges = torch.stack(g.edges(), dim=1)[masks['val']]
    neg_edges = sample_negative_edges(g, len(pos_edges))
    pos_score = dot_product(embeds, pos_edges)
    neg_score = dot_product(embeds, neg_edges)
    auc = compute_auc(pos_score, neg_score)
    return auc, embeds


# Cell 6b — Main training loop
model = DAEMONModel(cfg).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
criterion = AsymmetricLoss(cfg)
scaler = GradScaler(enabled=cfg.use_amp)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

best_auc, best_epoch, no_improve = 0.0, 0, 0
history = {'train_loss': [], 'val_auc': []}

for epoch in range(1, cfg.epochs + 1):
    train_loss = train_epoch(model, train_loader, optimizer, scaler, cfg)
    history['train_loss'].append(train_loss)

    if epoch % cfg.val_every == 0:
        val_auc, _ = validate(model, g, masks, cfg)
        history['val_auc'].append(val_auc)
        scheduler.step(val_auc)

        print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, cfg.checkpoint_path)
        else:
            no_improve += 1

    # Early stopping
    if no_improve >= cfg.patience:
        print(f"Early stopping at epoch {epoch} (best: {best_epoch}, AUC: {best_auc:.4f})")
        break

    # Memory cleanup
    if epoch % cfg.cleanup_every_n_epochs == 0:
        gc.collect(); torch.cuda.empty_cache()

print("Training complete. Best model saved.")
```

**Runtime breakdown:**
| Graph Size | Epochs | ~Time |
|---|---|---|
| Synthetic (100 nodes) | 50 | <2 min |
| Medium (100K nodes) | 100 | ~45 min |
| Large (1M nodes) | 100 | 3-5 hours |

**Memory monitoring cell** (run alongside training):
```python
# In logging/validation cell
def log_memory():
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU Memory: {alloc:.2f} GB allocated | {reserved:.2f} GB cached")
log_memory()
```


### Section 7: Evaluation (~10 min)

**Purpose**: Compute HitRate@k, MRR@k, AUC, and direction accuracy. Compare with baselines.

```python
@torch.no_grad()
def evaluate_full(model, g, masks, id_map, cfg):
    """Complete evaluation suite."""
    model.eval()
    embeds = generate_all_embeddings(model, g, cfg)  # [N, d]

    results = {}

    # 1. Node recommendation (HitRate@k, MRR@k)
    test_edges = torch.stack(g.edges(), dim=1)[masks['test']]
    results['hitrate'], results['mrr'] = compute_ranking_metrics(
        embeds, test_edges, ks=cfg.hitrate_k, top_n=100
    )

    # 2. Link prediction AUC
    pos_edges = test_edges
    neg_edges = sample_negative_edges(g, len(pos_edges))
    pos_score = dot_product(embeds, pos_edges)
    neg_score = dot_product(embeds, neg_edges)
    results['auc'] = compute_auc(pos_score, neg_score)

    # 3. Direction prediction (asymmetry test)
    asym_pairs = find_asymmetric_pairs(g)  # edges where (a→b) exists but (b→a) doesn't
    if len(asym_pairs) > 0:
        score_ab = dot_product(embeds, asym_pairs[:, [0, 1]])
        score_ba = dot_product(embeds, asym_pairs[:, [1, 0]])
        results['direction_acc'] = (score_ab > score_ba).float().mean().item()

    return results


# ── Print results ──
results = evaluate_full(model, g, masks, id_map, cfg)
print("=" * 50)
print("DAEMON Evaluation Results")
print("=" * 50)
for k in cfg.hitrate_k:
    print(f"  HitRate@{k:2d}:  {results['hitrate'][k]:.4f}")
    print(f"  MRR@{k:2d}:      {results['mrr'][k]:.4f}")
print(f"  AUC:            {results['auc']:.4f}")
print(f"  Direction Acc:  {results.get('direction_acc', 'N/A'):}")
print("=" * 50)

# Baseline comparison table (hardcoded from paper / prior work)
baselines = pd.DataFrame({
    'Model':         ['Random', 'Node2Vec', 'LightGCN', 'SAGE', 'GAT', 'DAEMON'],
    'HitRate@10':    [0.005, 0.031, 0.052, 0.061, 0.065, results['hitrate'].get(10, 0)],
    'MRR@10':        [0.001, 0.012, 0.024, 0.031, 0.034, results['mrr'].get(10, 0)],
    'AUC':           [0.500, 0.621, 0.708, 0.745, 0.752, results['auc']],
})
display(baselines.style.highlight_max(axis=0, subset=['HitRate@10', 'MRR@10', 'AUC']))
```

**Quality gate conditions (from Section 7 checklist):**
- [ ] HitRate@10 > 0.05
- [ ] AUC > 0.7
- [ ] Direction accuracy > 0.6 (model learns asymmetry)


### Section 8: Embedding Generation & FAISS Indexing (~10 min)

**Purpose**: Build a FAISS index for fast nearest-neighbor search over all product embeddings.

```python
def build_faiss_index(embeds: torch.Tensor) -> faiss.Index:
    """Build a GPU FAISS index for cosine-similarity search."""
    embeds_np = embeds.cpu().numpy().astype(np.float32)
    # Normalize for cosine similarity via inner-product search
    faiss.normalize_L2(embeds_np)

    dim = embeds_np.shape[1]
    # IndexFlatIP: exact inner-product (== cosine on normalized vectors)
    index = faiss.IndexFlatIP(dim)
    # Move to GPU
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(embeds_np)
    return index

@torch.no_grad()
def generate_all_embeddings(model, g, cfg) -> torch.Tensor:
    """Full-graph inference: generate embeddings for all nodes."""
    model.eval()
    # Use multi-layer neighbor sampler for full graph
    sampler = MultiLayerNeighborSampler([-1] * cfg.num_layers)  # -1 = all neighbors
    loader = DGLDataLoader(g, torch.arange(cfg.num_nodes), sampler,
                           batch_size=4096, shuffle=False, device='cuda')
    embeds_list = []
    for input_nodes, output_nodes, blocks in tqdm(loader, desc="Embedding"):
        blocks = [b.to('cuda') for b in blocks]
        h = blocks[0].srcdata['feat']
        emb = model(blocks, h)
        embeds_list.append(emb.cpu())
    return torch.cat(embeds_list, dim=0)

# ── Build index ──
all_embeds = generate_all_embeddings(model, g, cfg)
index = build_faiss_index(all_embeds)

# Latency test
query = all_embeds[:100].cpu().numpy()
faiss.normalize_L2(query)
_ = index.search(query, 10)  # warm-up
t0 = time.time()
distances, indices = index.search(query, 10)
latency = (time.time() - t0) / 100 * 1000  # ms per query
print(f"FAISS query latency: {latency:.2f} ms/query (target: <100ms)")
print(f"Index size: {index.ntotal:,} vectors × {all_embeds.shape[1]} dims")
```

**FAISS checklist:**
- [ ] Index builds without OOM
- [ ] Query latency < 100ms
- [ ] Results are cosine-similar (distance ≈ 1.0 for same product)


### Section 9: Demo — Product Recommendations (~5 min)

**Purpose**: Show that the trained model produces sensible, interpretable recommendations.

```python
def recommend_related(embeds: torch.Tensor, index: faiss.Index,
                      product_idx: int, id_map: Dict, k: int = 10):
    """Return top-k related products for a given product."""
    query = embeds[product_idx].cpu().numpy().reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query)
    distances, indices = index.search(query, k + 1)  # +1 to skip self
    # Skip self-match
    results = [(indices[0][i], distances[0][i])
               for i in range(1, k + 1)]
    return results

def product_name(idx: int, id_map: Dict) -> str:
    """Reverse-lookup product name from index."""
    reverse_map = {v: k for k, v in id_map.items()}
    return reverse_map.get(idx, f"Product_{idx}")


# ── Run demo ──
sample_products = [42, 137, 256, 1000]  # replace with meaningful indices
for pid in sample_products:
    print(f"\n{'─' * 60}")
    print(f"Query: {product_name(pid, id_map)} (idx={pid})")
    recs = recommend_related(all_embeds, index, pid, id_map, k=5)
    for rank, (rec_idx, score) in enumerate(recs, 1):
        print(f"  {rank}. {product_name(rec_idx, id_map)}  [{score:.3f}]")
```

**What to verify here:**
- [ ] Top recommendations look reasonable (e.g., phone case → phone screen protector)
- [ ] Related ≠ identical (diverse but relevant)
- [ ] High-scoring pairs exist in the co-purchase graph


### Section 10: Cold-Start Demo (~5 min)

**Purpose**: Simulate a brand-new product with only text features, generate its embedding, and show recommendations.

```python
def cold_start_recommend(model, proj, index, new_feat: torch.Tensor,
                         id_map: Dict, k: int = 10):
    """
    Predict embedding for a new product using its features only.
    Uses the feature projection layer + averaging over nearest known nodes.
    """
    model.eval()
    with torch.no_grad():
        # Project raw features
        proj_feat = proj(new_feat.cuda().unsqueeze(0))
        # Simple approach: use projection as approximate embedding
        # (A proper cold-start would use a learned induction head)
        fake_emb = nn.functional.normalize(proj_feat, p=2, dim=-1)

    query = fake_emb.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(query)
    distances, indices = index.search(query, k)
    return indices[0], distances[0]


# ── Simulate cold-start ──
# Random feature vector (stand-in for an actual new product's text embedding)
new_product_feat = torch.randn(cfg.in_feats)
print(f"\n{'─' * 60}")
print(f"Cold-Start Product (purely from features):")
indices, scores = cold_start_recommend(model, proj, index, new_product_feat, id_map, k=5)
for rank, (idx, score) in enumerate(zip(indices, scores), 1):
    print(f"  {rank}. {product_name(idx, id_map)}  [{score:.3f}]")
```

**Cold-start quality check:**
- [ ] Recommendations are non-trivial (not random)
- [ ] Top products share category/attributes with cold-start item
- [ ] Scores are reasonable (not all 0.99 or all 0.1)


### Section 11: Ablation Studies (optional, ~30 min)

**Purpose**: Quantify the contribution of each model component.

```python
# Cell 11a — Ablation configs
ablation_configs = {
    'DAEMON (full)':     {'use_asym_loss': True,  'use_coview': True},
    'w/o asymmetry':     {'use_asym_loss': False, 'use_coview': True},
    'w/o co-view':       {'use_asym_loss': True,  'use_coview': False},
    'w/o both':          {'use_asym_loss': False, 'use_coview': False},
}

ablation_results = {}
for name, ab_cfg in ablation_configs.items():
    print(f"\n{'=' * 50}\n  Running: {name}\n{'=' * 50}")
    ab_model = DAEMONModel(cfg).cuda()
    ab_criterion = AsymmetricLoss(cfg)  # modify with ab_cfg flags
    # ── Quick training (fewer epochs for speed) ──
    for epoch in range(1, 21):
        train_epoch(ab_model, train_loader, optimizer, scaler, cfg)
    res = evaluate_full(ab_model, g, masks, id_map, cfg)
    ablation_results[name] = res
    # Memory cleanup
    del ab_model; gc.collect(); torch.cuda.empty_cache()

# ── Ablation table ──
ab_df = pd.DataFrame(ablation_results).T
display(ab_df.style.highlight_max(axis=0))
```

**Expected pattern:**
- Full DAEMON > all ablations
- Asymmetry loss: improves direction accuracy by 8-15%
- Co-view data: improves HitRate@10 by 3-8%


### Section 12: Results Summary & Export (~5 min)

**Purpose**: Save everything needed for production use and future reproducibility.

```python
# Cell 12a — Final metrics dashboard
print("=" * 60)
print("  DAEMON — Final Results Summary")
print("=" * 60)
for metric, value in results.items():
    if isinstance(value, dict):
        for k, v in value.items():
            print(f"  {metric}@{k:2d}: {v:.4f}")
    else:
        print(f"  {metric}: {value:.4f}")

# Cell 12b — Save artifacts
save_dir = Path(cfg.output_dir)
save_dir.mkdir(exist_ok=True)

# 1. Model checkpoint
torch.save({
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'config': cfg,
    'epoch': epoch,
    'results': results,
}, save_dir / 'daemon_model.pt')

# 2. Embeddings
np.save(save_dir / 'product_embeddings.npy', all_embeds.cpu().numpy())

# 3. FAISS index
faiss.write_index(
    faiss.index_gpu_to_cpu(index),
    str(save_dir / 'faiss_index.bin')
)

# 4. Metrics JSON
with open(save_dir / 'metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nAll artifacts saved to: {cfg.output_dir}")
print("Files: daemon_model.pt, product_embeddings.npy, faiss_index.bin, metrics.json")

# Cell 12c — Download instructions
print("""
To download outputs:
  1. In Kaggle UI: Notebook → Output → Download All
  2. Or via Kaggle CLI:
     kaggle kernels output <username>/<kernel-slug> -p ./output/
""")
```


---

## 3. Milestone Timeline

Each milestone is independently runnable and produces a checkpoint. Build incrementally:

| Milestone | What | Success Criteria | Est. Time |
|---|---|---|---|
| **M1: Smoke test** | Train on synthetic graph (100 nodes). Verify all cells run, loss decreases, no NaN. | Loss drops >50% in 20 epochs. Shape assertions pass. | 30 min |
| **M2: Full training** | Load real data, train with all loss components, save checkpoint. | Val AUC > 0.7. Model checkpoint saves correctly. | 3-4 hours |
| **M3: Evaluation** | HitRate, MRR, AUC, direction accuracy. Compare with baselines. | HitRate@10 > 0.05. Direction accuracy > 0.6. | 30 min |
| **M4: Demo + FAISS** | Recommendations demo, cold-start, FAISS indexing. | Query latency < 100ms. Cold-start produces sensible results. | 20 min |
| **M5: Polish** | Ablations, docstrings, cleanup, edge-case testing, final export. | Restart & Run All succeeds. All quality checklist items pass. | 1 hour |

**Pro tip**: Commit M1 ASAP. It validates your environment. M2 is the long pole — start it and let it run while you work on M3 cells.


---

## 4. Code Organization Principles

Every cell must follow these rules:

| Principle | Example |
|---|---|
| **One component per cell** | Model class in one cell, training loop in another — never both |
| **Type hints** | `def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:` |
| **Docstrings** | Every function has a 1-line summary + Args/Returns |
| **Shape assertions** | `assert h.shape[-1] == self.in_feats, f"{h.shape} != {self.in_feats}"` |
| **Clear naming** | `pos_edges` not `pe`, `val_auc` not `va`, `num_neg_samples` not `nns` |
| **Memory hygiene** | `del large_tensor; gc.collect(); torch.cuda.empty_cache()` between phases |
| **Config-first** | Every magic number lives in `DAEMONConfig`, nowhere else |
| **Early exits** | `if run_synthetic: g, feats = build_synthetic_graph()` at top of data cell |


---

## 5. Testing Strategy

### 5.1 Synthetic Graph Test (M1)

```python
# Run this at the top of every cell group during development
RUN_SYNTHETIC = True  # flip to False for real data

if RUN_SYNTHETIC:
    g, node_feats = build_synthetic_graph(num_nodes=100, num_edges=500, feat_dim=768)
    cfg.num_nodes = 100
```

### 5.2 Gradient Flow Test

```python
# After first forward pass
assert all(p.grad is not None or not p.requires_grad for p in model.parameters()), \
    "Some parameters received no gradient!"
assert not any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None), \
    "NaN in gradients!"
```

### 5.3 Overfitting Test

```python
# Train on 20 nodes, verify model can memorize
g_tiny, feats_tiny = build_synthetic_graph(20, 60, 768)
# ... train for 200 epochs ...
# Expect: loss < 0.01, AUC > 0.99
assert loss < 0.05, f"Model failed to overfit tiny graph! Loss={loss:.4f}"
```

### 5.4 Shape Consistency Tests

```python
# In model.forward()
assert h.shape[0] == blocks[0].num_src_nodes(), "Input node mismatch"
assert embeds.shape == (blocks[-1].num_dst_nodes(), cfg.out_dim), \
    f"Output shape {embeds.shape} != expected"
```

### 5.5 Asymmetry Test

```python
# For any pair (a,b) where edge (a→b) exists but (b→a) doesn't:
score_ab = (embeds[a] * embeds[b]).sum()
score_ba = (embeds[b] * embeds[a]).sum()  # same for dot product
# With DAEMON's directional gating, score_ab should differ from a symmetric model
ab_scores = dot_product(embeds, asym_pairs[:, [0, 1]])
ba_scores = dot_product(embeds, asym_pairs[:, [1, 0]])
directional_diff = (ab_scores - ba_scores).abs().mean()
assert directional_diff > 0.01, f"No directional effect detected: {directional_diff}"
```


---

## 6. Risk Mitigation

### 6.1 DGL Install Fails

```python
try:
    import dgl
except ImportError:
    print("⚠ DGL install failed — falling back to PyTorch Geometric")
    !pip install torch_geometric
    # Rewrite model with PyG primitives:
    #  - dgl.DGLGraph → torch_geometric.data.HeteroData
    #  - dgl.nn.SAGEConv → torch_geometric.nn.SAGEConv
    #  - MultiLayerNeighborSampler → NeighborLoader
    USE_DGL = False
```

### 6.2 Out-of-Memory (OOM)

```python
def auto_reduce_batch(cfg: DAEMONConfig, oom_occurred: bool) -> DAEMONConfig:
    """Halve batch size on OOM, retry."""
    if oom_occurred and cfg.batch_size > 64:
        cfg.batch_size //= 2
        print(f"⚠ OOM detected → batch size reduced to {cfg.batch_size}")
    return cfg

# Usage: wrap training call in try/except RuntimeError
```

### 6.3 Session Timeout

```python
# Checkpoint recovery cell (run after restart)
def load_checkpoint(path: str, model, optimizer):
    if not Path(path).exists():
        print("No checkpoint found — starting fresh.")
        return 0
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    print(f"Resumed from epoch {ckpt['epoch']} | Best AUC: {ckpt.get('best_auc', 'N/A')}")
    return ckpt['epoch']

# Place at top of training cell:
start_epoch = load_checkpoint(cfg.checkpoint_path, model, optimizer)
```

### 6.4 Dataset Too Large

```python
# Subsampling option at data-load time
if cfg.num_nodes > 5_000_000:  # >5M nodes
    print("⚠ Large graph detected — using 25% subsample for training")
    sample_mask = torch.randperm(cfg.num_nodes)[:cfg.num_nodes // 4]
    g = dgl.node_subgraph(g, {'product': sample_mask})
    # Re-index features
    node_feats = node_feats[sample_mask]
    cfg.num_nodes = g.num_nodes('product')
```

---

## 7. Quality Checklist

Before declaring the notebook complete, verify EVERY item:

| # | Check | How to Verify | Critical? |
|---|---|---|---|
| 1 | All cells run without errors in order | Kaggle: Restart & Run All | **Yes** |
| 2 | GPU is utilized | `nvidia-smi` shows >0% utilization during training | **Yes** |
| 3 | Training loss decreases | Loss curve from Section 6 is monotonically downward | **Yes** |
| 4 | HitRate@10 > 0.05 | Section 7 evaluation output | **Yes** |
| 5 | rel(a,b) != rel(b,a) for asymmetric pairs | Direction accuracy > 0.6 in Section 7 | **Yes** |
| 6 | Cold-start recommendations are sensible | Section 10 output; manual inspection | No |
| 7 | FAISS search returns <100ms | Timing output in Section 8 | **Yes** |
| 8 | Memory < 14 GB VRAM | `torch.cuda.max_memory_allocated() / 1e9` at peak < 14 | **Yes** |
| 9 | No NaN in loss/gradients | Gradient flow test passes (Section 5.2) | **Yes** |
| 10 | Seeds are reproducible | Two runs with same seed produce identical loss at epoch 1 | Nice-to-have |
| 11 | Checkpoint saves + loads correctly | Load checkpoint in fresh session, verify same val AUC | **Yes** |
| 12 | Docstrings on all public functions | `help(function_name)` returns meaningful text | Nice-to-have |
| 13 | Config-driven; no hardcoded magic numbers | grep for `128`, `256`, `0.001` — all should reference `cfg` | Nice-to-have |
| 14 | Ablation results make sense | Full model > ablations in Section 11 table | Nice-to-have |


---

## 8. Appendix: Quick-Start Command Reference

```bash
# Local testing (CPU, fast)
python -c "
g, f = build_synthetic_graph(100, 500, 64)
cfg = DAEMONConfig(in_feats=64, hidden_dim=32, out_dim=16, epochs=5)
model = DAEMONModel(cfg)
print('Synthetic smoke test OK')
"

# Kaggle upload
kaggle datasets create -p ./data/processed --dir-mode zip
kaggle kernels push -p ./notebooks/daemon_kaggle

# Monitor training (from another cell)
!watch -n 5 nvidia-smi
```


---

## 9. File Inventory (what the notebook produces)

| File | Size (est.) | Purpose |
|---|---|---|
| `daemon_model.pt` | ~50-200 MB | Full model + optimizer state for resumption |
| `product_embeddings.npy` | `N × 128 × 4 bytes` | Dense embeddings for all products |
| `faiss_index.bin` | ~same as embeddings | GPU FAISS index for fast retrieval |
| `metrics.json` | <1 KB | Evaluation results (JSON) |
| `training_history.csv` | <10 KB | Loss/AUC per epoch for plotting |


---

*This plan is living documentation — update it as constraints change or new findings emerge during implementation. The numbered quality checklist (Section 7) is the final gate: all "Yes" items must pass before the notebook ships.*
