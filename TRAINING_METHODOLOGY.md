# TRAINING_METHODOLOGY.md — DAEMON Training Loop, Loss, and Evaluation

> **Project:** DAEMON-Kaggle — Related Product Recommendation via GNNs on Directed Graphs
> **Paper:** Virinchi et al., ECML-PKDD 2022

---

## 1. Training Overview

DAEMON is trained in an **unsupervised, self-supervised** manner. There are no explicit labels — the training signal comes entirely from the graph structure:

- **Co-purchase edges (E_cp):** Products bought together → should be recommended together
- **Co-view edges (E_cv):** Products viewed in the same session → similar items
- **Edge direction:** One-way co-purchase edges define asymmetry (phone→case, not case→phone)

The model learns embeddings that maximize the likelihood of observed edges while preserving directionality.

---

## 2. Loss Function — Complete Specification

### 2.1 Full Equation 2 from the Paper

```
L = - Σ_{(u,v)∈E_cp}           [ log σ(θ_u^s · θ_v^t) + Σ_{k=1}^{n_k} log σ(1 - θ_u^s · θ_z^t) ]
    - Σ_{(u,v)∈E_cp∧(v,u)∉E_cp} [ log σ(θ_u^s · θ_v^t) + log σ(1 - θ_v^s · θ_u^t) ]
    - Σ_{(u,v)∈E_cv}           [ log σ(θ_u^s · θ_v^s) + log σ(θ_u^t · θ_v^t) ]
```

Where:
- `σ(x) = 1 / (1 + exp(-x))` — sigmoid function
- `z ∼ P_r(P)` — negative sample drawn uniformly from all products
- `n_k` — number of negative samples per positive pair (typically 5)
- `θ_u^s, θ_u^t` — source and target embeddings of product u

### 2.2 Component 1: Co-purchase Likelihood with Negative Sampling

```python
def co_purchase_loss(src_emb, tgt_emb, pos_u, pos_v, neg_samples, num_neg=5):
    """
    For each co-purchase pair (u, v):
      + Maximize σ(θ_u^s · θ_v^t)         [u's source should match v's target]
      + Minimize σ(θ_u^s · θ_z^t) for random neg samples [shouldn't match random products]
    """
    # Positive term
    u_src = src_emb[pos_u]           # [B, d]
    v_tgt = tgt_emb[pos_v]           # [B, d]
    pos_score = (u_src * v_tgt).sum(dim=1)  # [B]
    pos_loss = -F.logsigmoid(pos_score).mean()
    
    # Negative term
    # neg_samples: [B, num_neg] — random product indices
    z_tgt = tgt_emb[neg_samples]     # [B, num_neg, d]
    neg_score = (u_src.unsqueeze(1) * z_tgt).sum(dim=2)  # [B, num_neg]
    neg_loss = -F.logsigmoid(-neg_score).mean()  # log(1 - σ(score)) = logsigmoid(-score)
    
    return pos_loss + neg_loss
```

### 2.3 Component 2: Asymmetry Enforcement

```python
def asymmetry_loss(src_emb, tgt_emb, one_way_u, one_way_v):
    """
    For one-way edges (u→v exists, v→u does NOT):
      + Maximize σ(θ_u^s · θ_v^t)   [correct direction]
      + Minimize σ(θ_v^s · θ_u^t)   [wrong direction — should be low]
    """
    # Correct direction
    u_src = src_emb[one_way_u]
    v_tgt = tgt_emb[one_way_v]
    forward_score = (u_src * v_tgt).sum(dim=1)
    forward_loss = -F.logsigmoid(forward_score).mean()
    
    # Wrong direction (should NOT recommend)
    v_src = src_emb[one_way_v]
    u_tgt = tgt_emb[one_way_u]
    reverse_score = (v_src * u_tgt).sum(dim=1)
    reverse_loss = -F.logsigmoid(-reverse_score).mean()
    
    return forward_loss + reverse_loss
```

### 2.4 Component 3: Co-view Similarity

```python
def co_view_loss(src_emb, tgt_emb, cv_u, cv_v):
    """
    For co-view pairs (u, v):
      + Maximize σ(θ_u^s · θ_v^s)   [source embeddings should be similar]
      + Maximize σ(θ_u^t · θ_v^t)   [target embeddings should be similar]
    """
    # Source similarity
    u_src = src_emb[cv_u]
    v_src = src_emb[cv_v]
    src_score = (u_src * v_src).sum(dim=1)
    src_loss = -F.logsigmoid(src_score).mean()
    
    # Target similarity
    u_tgt = tgt_emb[cv_u]
    v_tgt = tgt_emb[cv_v]
    tgt_score = (u_tgt * v_tgt).sum(dim=1)
    tgt_loss = -F.logsigmoid(tgt_score).mean()
    
    return src_loss + tgt_loss
```

### 2.5 Combined Asymmetric Loss

```python
class AsymmetricLoss(nn.Module):
    def __init__(self, num_neg=5):
        super().__init__()
        self.num_neg = num_neg
    
    def forward(self, src_emb, tgt_emb, batch_data):
        """
        batch_data: dict containing:
          - cp_u, cp_v: co-purchase positive pairs
          - ow_u, ow_v: one-way edge pairs (for asymmetry)
          - cv_u, cv_v: co-view pairs
          - num_nodes: total nodes for negative sampling
        """
        # Generate negative samples
        neg = sample_negatives(batch_data['num_nodes'], len(batch_data['cp_u']), self.num_neg)
        
        loss_cp = co_purchase_loss(src_emb, tgt_emb, 
                                    batch_data['cp_u'], batch_data['cp_v'], neg, self.num_neg)
        loss_asym = asymmetry_loss(src_emb, tgt_emb, 
                                    batch_data['ow_u'], batch_data['ow_v'])
        loss_cv = co_view_loss(src_emb, tgt_emb, 
                                batch_data['cv_u'], batch_data['cv_v'])
        
        # Paper combines them with equal weight (sum)
        total_loss = loss_cp + loss_asym + loss_cv
        
        return total_loss, {
            'loss_cp': loss_cp.item(),
            'loss_asym': loss_asym.item(),
            'loss_cv': loss_cv.item(),
        }
```

### 2.6 Negative Sampling Implementation

```python
def sample_negatives(num_nodes, num_positive_pairs, num_neg=5, device='cuda'):
    """
    Uniform negative sampling.
    For each positive pair (u, v), sample `num_neg` random nodes z.
    
    Note: The paper uses a simple uniform distribution Pr(P).
    Some negatives might accidentally be real positives — 
    this is acceptable noise for large graphs (probability is tiny).
    """
    neg = torch.randint(0, num_nodes, (num_positive_pairs, num_neg), device=device)
    return neg
```

---

## 3. Training Algorithm (Per-Epoch)

```
For each epoch:
  1. Shuffle training edge set
  2. For each mini-batch of edges (seed nodes):
     a. Sample subgraph using neighbor sampler [20, 10, 10]
     b. Move subgraph + features to GPU
     c. Forward pass through DAEMON (3 layers)
     d. Extract embeddings for nodes in this batch
     e. Compute AsymmetricLoss (components 1+2+3)
     f. Backward pass + optimizer step
  3. After all batches:
     a. (Optional) Update full embedding matrices
     b. Evaluate on validation set
     c. Save checkpoint
     d. Check early stopping criteria
```

---

## 4. Hyperparameters

| Parameter | Paper Value | Kaggle Default | Notes |
|-----------|-------------|----------------|-------|
| **Learning rate** | 1e-4 | 1e-4 | Adam optimizer; grid searched {1e-1, 1e-2, 1e-3, 1e-4} |
| **Optimizer** | Adam | Adam | β1=0.9, β2=0.999, ε=1e-8 |
| **Batch size** | 1024 | 1024 (reducible to 512) | Reduce if OOM |
| **Embedding dim** | 64 | 64 | Final output dim of dual embeddings |
| **Hidden dim** | 128–256 | 128 | Intermediate layer dim; paper varies by dataset |
| **GNN layers (L)** | 3 | 3 | More layers = exponential neighbor explosion |
| **Neighbor sampling** | [20, 10, 10] | [20, 10, 10] | Layer 1: 20, Layer 2: 10, Layer 3: 10 |
| **Negative samples (n_k)** | ~5 | 5 | Per positive co-purchase edge |
| **Epochs** | 30 | 20–30 | With early stopping (patience=5) |
| **Weight decay** | Not specified | 1e-5 | L2 regularization |
| **Gradient clipping** | Not specified | 1.0 | Max norm clipping |
| **Dropout** | Not specified | 0.1 | On embedding outputs before next layer |
| **AMP** | Not used | Yes (default) | Mixed precision for T4 |

---

## 5. Optimizer & Scheduler

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-5
)

# Cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

# Alternative: plateau-based reduction
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='max', factor=0.5, patience=3
# )
```

Gradient clipping:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 6. Training Loop Pseudocode

```python
def train_epoch(model, dataloader, optimizer, scaler, config):
    model.train()
    total_loss = 0
    loss_components = {'cp': 0, 'asym': 0, 'cv': 0}
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (input_nodes, output_nodes, blocks) in enumerate(pbar):
        # blocks: list of DGL blocks, one per layer
        # input_nodes: all nodes needed for this subgraph
        # output_nodes: seed nodes
        
        optimizer.zero_grad()
        
        with autocast():
            # Forward through 3-layer model
            # blocks[0] = layer 1 subgraph, blocks[1] = layer 2, blocks[2] = layer 3
            src_emb, tgt_emb = model(blocks, blocks[0].srcdata['feat'])
            
            # Only compute loss on output nodes (seed batch)
            # Extract relevant pairs for this batch
            batch_pairs = extract_batch_pairs(blocks, output_nodes)
            
            loss, losses_dict = asymmetric_loss(src_emb, tgt_emb, batch_pairs)
        
        # Backward with AMP
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # If using per-step scheduler
        
        # Logging
        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += losses_dict.get(f'loss_{k}', 0)
        
        if batch_idx % 50 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'vram': f'{torch.cuda.memory_allocated()/1e9:.1f}GB'
            })
        
        # Intermediate checkpoint
        if batch_idx > 0 and batch_idx % 500 == 0:
            save_checkpoint(model, optimizer, scaler, epoch, batch_idx)
    
    n_batches = len(dataloader)
    return total_loss / n_batches, {k: v / n_batches for k, v in loss_components.items()}
```

---

## 7. Evaluation Protocol

### 7.1 Train/Val/Test Split

Per the paper (Section 5.1.4):

```
Total edges:      100%
├── Training:      75%  (non-overlapping)
├── Validation:     5%  (for early stopping / hyperparameter tuning)
└── Test:          20%  (for final evaluation)

Split is on EDGES, not nodes — ensures no training edge leaks into evaluation.
```

```python
def split_edges(edge_index, num_edges):
    """Split edges into train/val/test with no overlap."""
    perm = torch.randperm(num_edges)
    train_end = int(num_edges * 0.75)
    val_end = train_end + int(num_edges * 0.05)
    
    train_edges = edge_index[:, perm[:train_end]]
    val_edges = edge_index[:, perm[train_end:val_end]]
    test_edges = edge_index[:, perm[val_end:]]
    
    return train_edges, val_edges, test_edges
```

### 7.2 EQ1: Node Recommendation (Primary Task)

**Goal:** Given query product q, find top-k products that are likely co-purchased.

**Metrics:**

#### HitRate@k
```
HitRate@k = (1 / |Q|) Σ_{q∈Q} 𝟙[ ground_truth_v ∈ R_k(q) ]

Where R_k(q) = top-k products by rel(q, v) = θ_q^s · θ_v^t
And ground_truth_v is the known co-purchase partner from test set
```

```python
def hit_rate_at_k(query_src_emb, candidate_tgt_emb, query_ids, true_candidate_ids, k=10):
    """
    For each query product, check if the true co-purchase partner is in top-k.
    """
    hits = 0
    for i, q_id in enumerate(query_ids):
        q_src = query_src_emb[q_id]  # [d]
        scores = q_src @ candidate_tgt_emb.T  # [num_candidates]
        _, top_k_indices = torch.topk(scores, k)
        if true_candidate_ids[i] in top_k_indices:
            hits += 1
    return hits / len(query_ids)
```

#### MRR@k (Mean Reciprocal Rank)
```
MRR@k = (1 / |Q|) Σ_{q∈Q} (1 / rank(ground_truth_v))
                                       if rank(ground_truth_v) ≤ k, else 0
```

```python
def mrr_at_k(query_src_emb, candidate_tgt_emb, query_ids, true_candidate_ids, k=10):
    reciprocal_ranks = []
    for i, q_id in enumerate(query_ids):
        q_src = query_src_emb[q_id]
        scores = q_src @ candidate_tgt_emb.T
        _, top_k_indices = torch.topk(scores, min(k, len(scores)))
        
        mask = top_k_indices == true_candidate_ids[i]
        if mask.any():
            rank = mask.nonzero(as_tuple=True)[0][0].item() + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)
```

**k values:** Evaluate at k = {5, 10, 20} (per paper).

**Optimization:** For large candidate sets, use FAISS for approximate top-k instead of exhaustive matrix multiplication.

### 7.3 EQ2: Existential Link Prediction

**Goal:** Does the edge (u, v) exist? A good model assigns higher scores to real edges than fake ones.

```python
def link_prediction_auc(src_emb, tgt_emb, pos_edges, neg_edges):
    """
    pos_edges: real co-purchase edges
    neg_edges: randomly generated non-edges
    
    Compute AUC: probability that a random positive has higher score
    than a random negative.
    """
    pos_scores = (src_emb[pos_edges[0]] * tgt_emb[pos_edges[1]]).sum(dim=1)
    neg_scores = (src_emb[neg_edges[0]] * tgt_emb[neg_edges[1]]).sum(dim=1)
    
    all_scores = torch.cat([pos_scores, neg_scores])
    all_labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    
    return roc_auc_score(all_labels.cpu(), all_scores.cpu())
```

### 7.4 EQ3: Direction Link Prediction

**Goal:** For one-way edges u→v, does the model assign higher score to the correct direction?

```python
def direction_prediction_auc(src_emb, tgt_emb, one_way_edges):
    """
    one_way_edges: edges where u→v exists but v→u does NOT
    
    Positive: forward direction (u as query, v as candidate)
    Negative: reverse direction (v as query, u as candidate)
    """
    u, v = one_way_edges
    
    forward_scores = (src_emb[u] * tgt_emb[v]).sum(dim=1)
    reverse_scores = (src_emb[v] * tgt_emb[u]).sum(dim=1)
    
    all_scores = torch.cat([forward_scores, reverse_scores])
    all_labels = torch.cat([torch.ones(len(u)), torch.zeros(len(u))])
    
    return roc_auc_score(all_labels.cpu(), all_scores.cpu())
```

### 7.5 EQ4: Cold-Start Recommendation

**Goal:** Can model recommend for products with zero purchase history?

- Split: hold out 20% of **nodes** entirely from training
- These nodes have catalog features but no edges
- Evaluate HitRate@k and MRR@k for their test edges

```python
def evaluate_cold_start(model, graph, cold_nodes, cold_features, 
                        existing_features, test_edges, k=10):
    """
    For each cold-start node:
    1. Find k-NN similar existing nodes via feature similarity
    2. Augment graph with similarity edges
    3. Generate embeddings
    4. Evaluate recommendations
    """
    # Similar to cold-start flow in ARCHITECTURE.md
    pass
```

### 7.6 EQ5: Selection Bias

**Goal:** Can model recommend products that were co-viewed with co-purchased products, even if never co-purchased directly?

- Add transitive edges to test set: aRcp_b ∧ bRcv_c → should recommend c for a
- These transitive relationships exist in training graph but are held out from test edges
- Evaluate HitRate@k and MRR@k on these transitive edges

---

## 8. Early Stopping & Model Selection

```python
patience = 5
best_val_hr = 0.0
patience_counter = 0
best_model_path = '/kaggle/working/best_model.pt'

for epoch in range(num_epochs):
    # Training...
    train_loss, train_components = train_epoch(...)
    
    # Validation
    val_hr = evaluate_hit_rate(model, val_edges, k=10)
    val_mrr = evaluate_mrr(model, val_edges, k=10)
    
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
          f"Val HR@10={val_hr:.4f}, Val MRR@10={val_mrr:.4f}")
    
    if val_hr > best_val_hr:
        best_val_hr = val_hr
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_hr': val_hr,
            'val_mrr': val_mrr,
        }, best_model_path)
        print(f"  → New best model (HR@10={val_hr:.4f})")
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Load best model for final evaluation
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 9. Regularization Techniques

### 9.1 Dropout

```python
class DAEMONLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, graph, h_s, h_t):
        # Apply dropout to input embeddings
        h_s = self.dropout(h_s)
        h_t = self.dropout(h_t)
        # ... aggregation logic ...
        return h_s_new, h_t_new
```

### 9.2 L2 Normalization (Built-in)

The paper normalizes after each layer — this acts as an implicit regularizer by bounding embedding magnitude to 1.

### 9.3 Edge Dropout

Randomly drop edges during training to prevent overfitting:

```python
def edge_dropout(graph, drop_prob=0.05):
    """Randomly drop edges for regularization."""
    if drop_prob > 0:
        mask = torch.rand(graph.num_edges()) > drop_prob
        graph = graph.edge_subgraph(mask)
    return graph
```

---

## 10. Monitoring & Logging

### 10.1 Per-Batch Logging

```python
# Live progress bar
pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
pbar.set_postfix({
    'loss': f'{loss.item():.3f}',
    'cp': f'{loss_cp:.3f}',
    'asym': f'{loss_asym:.3f}',
    'cv': f'{loss_cv:.3f}',
    'lr': f'{lr:.1e}',
    'VRAM': f'{torch.cuda.memory_allocated()/1e9:.1f}G'
})
```

### 10.2 GPU Memory Tracking

```python
def log_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    print(f"GPU Memory: allocated={allocated:.1f}GB, "
          f"reserved={reserved:.1f}GB, peak={max_allocated:.1f}GB")
```

### 10.3 Epoch Summary

```python
# After each epoch, print structured summary
print(f"{'='*60}")
print(f"Epoch {epoch:3d} Summary")
print(f"{'='*60}")
print(f"  Train Loss:     {train_loss:.4f}")
print(f"  Loss Components: CP={cp_loss:.4f}, Asym={asym_loss:.4f}, CV={cv_loss:.4f}")
print(f"  Validation:")
print(f"    HR@5={val_hr5:.4f}   HR@10={val_hr10:.4f}   HR@20={val_hr20:.4f}")
print(f"    MRR@5={val_mrr5:.4f}  MRR@10={val_mrr10:.4f}  MRR@20={val_mrr20:.4f}")
print(f"  Time: {epoch_time:.1f}s")
print(f"  LR: {current_lr:.2e}")
print(f"  GPU Peak: {peak_vram:.1f}GB")
print(f"  Best Val HR@10: {best_val_hr:.4f} (epoch {best_epoch})")
```

---

## 11. Quick Reference: Hyperparameter Grid

| Parameter | Search Range | Best (Paper) | Kaggle Start |
|-----------|-------------|--------------|--------------|
| lr | {1e-1, 1e-2, 1e-3, 1e-4} | 1e-4 | 1e-4 |
| batch_size | {256, 512, 1024, 2048} | 1024 | 1024 |
| embed_dim | {32, 64, 128} | 64 | 64 |
| hidden_dim | {64, 128, 256} | 128-256 | 128 |
| num_layers | {2, 3, 4} | 3 | 3 |
| fanouts_l1 | {10, 15, 20, 25} | 20 | 20 |
| fanouts_l2+ | {5, 10, 15} | 10 | 10 |
| num_neg | {1, 5, 10, 20} | ~5 | 5 |
| dropout | {0.0, 0.1, 0.2, 0.3} | - | 0.1 |
| weight_decay | {1e-6, 1e-5, 1e-4} | - | 1e-5 |

---

## References

- Virinchi et al. (2022). *Recommending Related Products Using Graph Neural Networks in Directed Graphs*. ECML-PKDD.
- PyTorch AMP Tutorial: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
- DGL Node Classification Tutorial (sampling): https://docs.dgl.ai/tutorials/blitz/1_introduction.html
