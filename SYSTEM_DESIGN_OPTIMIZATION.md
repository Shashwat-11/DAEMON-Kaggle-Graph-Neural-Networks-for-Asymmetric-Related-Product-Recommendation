# SYSTEM_DESIGN_OPTIMIZATION.md — Kaggle T4 Memory & Performance Optimization

> **Project:** DAEMON-Kaggle — Related Product Recommendation via GNNs on Directed Graphs
> **Constraint:** T4 GPU (16GB VRAM), ~13GB system RAM, ~9hr session, ~30hr/week GPU
> **Goal:** Train DAEMON on graphs of 100K–500K nodes within these constraints

---

## 1. Constraints Summary

| Resource | Limit | Impact on DAEMON |
|----------|-------|------------------|
| **GPU VRAM** | 16 GB (Tesla T4) | Must use mini-batch sampling; full graph won't fit |
| **GPU Session** | ~9 hours max | Need checkpointing every ~30 min |
| **GPU Weekly Quota** | ~30 hours | 1-2 full training runs per week |
| **System RAM** | ~13–16 GB | Full graph + features must fit in CPU RAM |
| **Disk (/kaggle/working)** | ~20 GB | Store compressed checkpoints, embeddings |
| **Internet** | OFF by default | Enable for `pip install` cells only |
| **DGL** | NOT pre-installed | `!pip install dgl` in first cell |
| **FAISS** | NOT pre-installed | `!pip install faiss-gpu` in first cell |

---

## 2. Memory Budget Analysis

### For a 500K node, 3M edge graph (our target scale)

| Component | Calculation | Memory (FP32) | Memory (FP16) |
|-----------|------------|---------------|---------------|
| **Node features** | 500K × 384d × 4B | 768 MB | 384 MB |
| **Edge list (COO)** | 3M × 2 × 8B | 48 MB | 48 MB |
| **Edge types** | 3M × 1B | 3 MB | 3 MB |
| **Model params (3 layers)** | 3 × (384×128) × 4B | ~1.5 MB | ~1.5 MB |
| **Embedding matrix (θ^s + θ^t)** | 500K × 64d × 2 × 4B | 256 MB | 128 MB |
| **Activations (per batch, 1024 nodes)** | Subgraph ~5000 nodes × 3 layers × 128d | ~30 MB | ~15 MB |
| **Gradient buffer** | ~params × 2 | ~3 MB | ~3 MB |
| **AMP scaler overhead** | ~10-20% VRAM buffer | ~1 GB | ~1 GB |
| **CUDA context + overhead** | Fixed overhead | ~500 MB | ~500 MB |

| Total (FP32) | Total (FP16) |
|-------------|-------------|
| **~2.6 GB peak** | **~2.1 GB peak** |

> **Verdict:** 500K nodes with FP16 easily fits within 16GB VRAM. Even at 2M nodes (paper G1 scale), peak would be ~6–8 GB — still feasible with careful optimization.

### Memory Budget for 2M node, 14M edge graph (paper G1)

| Component | FP16 Memory |
|-----------|------------|
| Node features (2M × 384d) | 1.5 GB |
| Edge list (14M × 2) | 224 MB |
| Embedding matrix (2M × 64d × 2) | 512 MB |
| Activations per batch | ~30 MB |
| AMP overhead + CUDA | ~1.5 GB |
| **Total peak** | **~3.8 GB** |

This still fits comfortably — the key is keeping the **full graph on CPU** and only **subgraphs on GPU**.

---

## 3. Mini-Batch Training Design

### 3.1 Neighbor Sampling Configuration

Matching the paper: **3-layer GNN with fanouts [20, 10, 10]**

```python
import dgl
from dgl.dataloading import MultiLayerNeighborSampler, DataLoader

# Layer-wise neighbor sampling
sampler = MultiLayerNeighborSampler(fanouts=[20, 10, 10])

# DataLoader configured for GPU
dataloader = DataLoader(
    g,                          # Full graph on CPU
    train_seed_nodes,           # Batch seed nodes (indices)
    sampler,
    batch_size=1024,            # Per paper
    shuffle=True,
    drop_last=False,
    num_workers=2,              # Kaggle CPUs are limited; 2 is safe
    device='cuda'               # Moves subgraph directly to GPU
)
```

### 3.2 Subgraph Size Analysis

With fanouts [20, 10, 10] and batch_size 1024:
- Layer 0 (seed): 1024 nodes
- Layer 1: up to 1024 × 20 = 20,480 nodes (actual less due to overlap)
- Layer 2: up to 20,480 × 10 = 204,800 (heavy overlap in practice)
- Typical subgraph: **5,000–15,000 nodes**

### 3.3 Batch Size Tuning

| Batch Size | Subgraph Size (approx) | VRAM (activations) | Training Speed |
|------------|----------------------|---------------------|----------------|
| 2048 | 10K–30K nodes | ~4 GB | Fastest |
| 1024 | 5K–15K nodes | ~2 GB | **Default (paper)** |
| 512 | 2.5K–8K nodes | ~1 GB | Safe fallback |
| 256 | 1K–4K nodes | ~500 MB | Emergency fallback |

Start at 1024. If OOM, halve. If plenty of headroom, try 2048.

### 3.4 Handling Both Edge Types in Sampling

```python
# DGL's neighbor sampler samples ALL edge types by default.
# We differentiate co-purchase (0) vs co-view (1) at the model level using edge_data['type'].

# During forward pass, we separate neighbors by edge type:
cp_edges = subgraph.edges(form='all', order='eid')[subgraph.edata['type'] == 0]
cv_edges = subgraph.edges(form='all', order='eid')[subgraph.edata['type'] == 1]
```

---

## 4. Mixed Precision Training (AMP)

### 4.1 Configuration

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():  # FP16 for forward pass
        src_emb, tgt_emb = model(batch_graph, batch_feat)
        loss = asymmetric_loss(src_emb, tgt_emb, pos_pairs, neg_pairs)
    
    scaler.scale(loss).backward()  # Scale loss, backward in FP16
    scaler.step(optimizer)          # Unscale gradients, update weights
    scaler.update()                  # Update scale factor
```

### 4.2 What Runs in FP16 vs FP32

| Operation | Precision | Reason |
|-----------|-----------|--------|
| Linear layers (W^l @ h) | FP16 | Matrix multiply — major speedup on T4 Tensor Cores |
| Embedding aggregations (mean/sum) | FP16 | Safe reduction |
| ReLU activation | FP16 | Safe |
| L2 normalization | FP32 | Numerical stability (norm computation) |
| Sigmoid in loss | FP32 | Critical for loss accuracy |
| Loss sum | FP32 | Precision matters for gradient |
| BatchNorm (if used) | FP32 | Known instability in FP16 |

### 4.3 Expected Benefits

- **Memory:** 40–50% reduction in activation memory
- **Speed:** ~3–5x faster matrix multiplications on T4 Tensor Cores
- **No quality loss:** DAEMON uses cosine similarity (after L2 norm), which is robust to FP16

---

## 5. Gradient Checkpointing

Trade compute for memory on the 3 DAEMON layers:

```python
from torch.utils.checkpoint import checkpoint

class DAEMONModel(nn.Module):
    def forward(self, graph, features):
        h_s, h_t = features, features
        
        for layer in self.layers:
            # Checkpoint each layer: recompute activations during backward
            h_s, h_t = checkpoint(
                self._layer_forward, layer, graph, h_s, h_t,
                use_reentrant=False  # PyTorch 2.0+ recommended
            )
        
        return h_s, h_t
    
    @staticmethod
    def _layer_forward(layer, graph, h_s, h_t):
        return layer(graph, h_s, h_t)
```

**Tradeoff:**
- Memory saved: ~30–50% (don't store intermediate activations for all 3 layers)
- Compute overhead: ~15–20% (recompute activations during backward pass)
- On T4 with AMP, the net effect is positive since memory is the bottleneck, not compute

---

## 6. Graph Storage Strategy

### 6.1 CPU-Resident Graph

```python
# Graph lives on CPU
g = dgl.graph((src_nodes, dst_nodes))  # On CPU RAM (~250 MB for 3M edges)
g.ndata['feat'] = features_fp16        # On CPU (~384 MB for 500K × 384d FP16)
g.edata['type'] = edge_types           # On CPU (~3 MB)

# Only subgraphs move to GPU
dataloader = DataLoader(
    g, train_nids, sampler,
    batch_size=1024, device='cuda'      # Subgraphs auto-moved to GPU
)
```

### 6.2 Pinned Memory for Faster Transfer

```python
# Pin feature tensor in CPU memory for faster GPU transfer
features_pinned = features_fp16.pin_memory()

# DGL DataLoader handles the transfer; pinning speeds up CPU→GPU DMA
```

### 6.3 Memory-Mapped Features (if needed)

For very large feature matrices that exceed RAM:

```python
# Store features as memory-mapped file
features_mmap = np.memmap(
    '/kaggle/working/features.dat',
    dtype='float16',
    mode='r',
    shape=(num_nodes, feature_dim)
)

# Load chunks as needed
batch_features = torch.from_numpy(features_mmap[batch_nids])
```

---

## 7. Embedding Storage Optimization

### 7.1 CPU-Side Embedding Table

During training, we need embeddings for all nodes for negative sampling and evaluation. Keep the full embedding table on CPU:

```python
# Maintain current embeddings on CPU (updated after each epoch or every N steps)
emb_src_cpu = torch.zeros(num_nodes, embed_dim, dtype=torch.float16)
emb_tgt_cpu = torch.zeros(num_nodes, embed_dim, dtype=torch.float16)

# After each epoch, run full inference and update:
with torch.no_grad():
    with autocast():
        for batch in full_dataloader:
            src, tgt = model(batch_graph, batch_feat)
            emb_src_cpu[batch_nids] = src.half()
            emb_tgt_cpu[batch_nids] = tgt.half()
```

### 7.2 FAISS Index Memory

For 500K products with 64-dim embeddings:
- Raw vectors: 500K × 64 × 2 bytes = **64 MB** (FP16)
- IVFFlat index: ~128 MB (with 4096 centroids)
- **Total FAISS memory:** ~200 MB — negligible compared to graph

---

## 8. Checkpointing & Resume Strategy

### 8.1 What to Save

```python
checkpoint = {
    'epoch': epoch,
    'batch_idx': batch_idx,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'emb_src_cpu': emb_src_cpu,
    'emb_tgt_cpu': emb_tgt_cpu,
    'best_val_hr': best_val_hr,
    'config': config_dict,
    'random_state': random.getstate(),
    'np_random_state': np.random.get_state(),
    'torch_random_state': torch.get_rng_state(),
    'cuda_random_state': torch.cuda.get_rng_state(),
}

torch.save(checkpoint, '/kaggle/working/checkpoint.pt')
```

### 8.2 Checkpoint Schedule

| Frequency | What | Why |
|-----------|------|-----|
| Every 500 batches | Intermediate checkpoint | Survive OOM/timeout mid-epoch |
| Every epoch | Epoch checkpoint + best model | Standard resume point |
| On KeyboardInterrupt | Emergency save | Session kill by Kaggle |

```python
import signal

def emergency_save(signum, frame):
    torch.save(checkpoint, '/kaggle/working/emergency_checkpoint.pt')
    print("Emergency checkpoint saved!")

signal.signal(signal.SIGTERM, emergency_save)
signal.signal(signal.SIGINT, emergency_save)
```

### 8.3 Resume Logic

```python
import os
ckpt_path = '/kaggle/working/checkpoint.pt'

if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resumed from epoch {start_epoch}, batch {checkpoint['batch_idx']}")
else:
    start_epoch = 0
    print("Starting fresh training")
```

### 8.4 Multi-Session Strategy

If training takes >9 hours:
1. **Session 1:** Train epochs 0–10, save checkpoint
2. **Session 2:** Load checkpoint, train epochs 11–20
3. **Session 3:** Load checkpoint, train epochs 21–30, final eval

Kaggle allows downloading outputs between sessions:
```bash
# In Kaggle notebook sidebar: "Save Version" → download checkpoint.pt
# Then re-attach as a dataset for the next session
```

---

## 9. Performance Optimization Checklist

| # | Optimization | How | Expected Gain |
|---|-------------|-----|---------------|
| 1 | **cuDNN benchmark** | `torch.backends.cudnn.benchmark = True` | Auto-tune conv kernels |
| 2 | **torch.compile** | `model = torch.compile(model, mode='reduce-overhead')` | 20–40% speedup (PyTorch 2.x) |
| 3 | **DataLoader workers** | `num_workers=2` | Overlap CPU sampling with GPU compute |
| 4 | **Prefetch** | DGL DataLoader auto-prefetches | Hidden by default |
| 5 | **Avoid sync** | Never call `.item()` or `.cpu()` mid-loop | Eliminates GPU stalls |
| 6 | **FP16 features** | Store as `torch.float16` | Half the memory bandwidth |
| 7 | **In-place ops** | Use `F.normalize(..., inplace=True)` where possible | Reduce memory fragmentation |
| 8 | **GC collect** | `gc.collect(); torch.cuda.empty_cache()` between epochs | Defragment VRAM |
| 9 | **Profile first** | `torch.profiler.profile()` on 2-3 batches | Find real bottlenecks |
| 10 | **CUDA graphs** | If PyTorch 2.x, `torch.compile` handles this | Reduce kernel launch overhead |

### 9.1 Quick Profiling Setup

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profile'),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for i, batch in enumerate(dataloader):
        # ... training step ...
        prof.step()
        if i >= 10:
            break

# View with: %load_ext tensorboard; %tensorboard --logdir ./log
```

---

## 10. Fallback Strategies

### Tiered OOM Response

If you hit `CUDA out of memory`:

```
Level 1: Reduce batch_size
  1024 → 512 → 256 → 128

Level 2: Reduce neighbor sampling
  [20, 10, 10] → [15, 10, 5] → [10, 5, 5] → [5, 5, 5]

Level 3: Reduce embedding/feature dims
  Feature: 384 → 256 → 128 (via PCA)
  Embedding: 64 → 48 → 32

Level 4: Reduce graph size
  Filter out nodes with degree < 3
  Subsample to 50% of original

Level 5: Disable components
  Turn off gradient checkpointing → if it's causing issues
  Switch to CPU-only FAISS → frees ~200MB GPU
```

### Automatic Fallback

```python
def train_with_fallback(config):
    try:
        return train(config)
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            gc.collect()
            
            if config.batch_size > 128:
                config.batch_size //= 2
                print(f"OOM: reducing batch_size to {config.batch_size}")
                return train_with_fallback(config)
            elif config.fanouts[0] > 5:
                config.fanouts = [max(f-5, 2) for f in config.fanouts]
                print(f"OOM: reducing fanouts to {config.fanouts}")
                return train_with_fallback(config)
            else:
                raise RuntimeError("Cannot reduce further — graph too large for T4")
        raise
```

---

## 11. Estimated Training Time

### Per-Epoch Estimate

For 500K nodes, 3M edges, batch_size=1024, 3 layers, fanouts [20,10,10]:

| Phase | Time (T4) |
|-------|-----------|
| DataLoader sampling per batch | ~50 ms |
| Forward pass (3 layers, AMP) | ~30 ms |
| Loss computation | ~5 ms |
| Backward pass | ~40 ms |
| Optimizer step | ~5 ms |
| **Total per batch** | **~130 ms** |
| Batches per epoch (75% of 3M edges) | ~2,200 |
| **Per epoch** | **~5 minutes** |
| **30 epochs** | **~2.5 hours** |
| Plus validation (per epoch) | ~15 seconds × 30 = 7.5 min |
| Plus final evaluation | ~10 min |
| **Total training** | **~3 hours** |

### Fits in 9-Hour Session?

**Yes, with large margin.** Even a 2M node graph (paper G1 scale, ~16K batches/epoch, ~8 min/epoch) would finish in ~4 hours.

### Multi-Session Note

Most Kaggle training fits in a single session. The bigger risk is the notebook kernel dying from idle timeout or OOM — which checkpointing handles.

---

## 12. Quick Startup Checklist

Before starting training, run this verification:

```python
# 1. Check GPU
assert torch.cuda.is_available(), "GPU not available!"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# 2. Memory budget
vram_free = torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated()
print(f"Free VRAM: {vram_free / 1e9:.1f} GB")
assert vram_free > 4e9, "Less than 4GB free — won't fit!"

# 3. Test single batch
test_batch = next(iter(dataloader))
with autocast():
    src, tgt = model(test_batch, test_batch.ndata['feat'])
    loss = asymmetric_loss(src, tgt, test_pos_pairs, test_neg_pairs)
loss.backward()
print(f"Single batch test passed. Batch VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# 4. Clear and proceed
del test_batch, src, tgt, loss
torch.cuda.reset_peak_memory_stats()
gc.collect()
torch.cuda.empty_cache()
```

---

## References

- Kaggle GPU Usage: https://www.kaggle.com/docs/efficient-gpu-usage
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html
- DGL Mini-batch Training: https://docs.dgl.ai/guide/minibatch.html
- PyTorch Gradient Checkpointing: https://pytorch.org/docs/stable/checkpoint.html
- FAISS GPU: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
