# DAEMON-Kaggle: Graph Neural Networks for Related Product Recommendation on Directed Graphs

> Reproducing Amazon's ECML-PKDD 2022 paper on Kaggle free tier (T4 GPU, 16GB VRAM)
>
> **Paper:** *"Recommending Related Products Using Graph Neural Networks in Directed Graphs"* — Srinivas Virinchi, Anoop Saladi, Abhirup Mondal (Amazon ML, Bengaluru)

---

## Elevator Pitch

DAEMON is a production-proven Graph Neural Network that powers Amazon's "Customers Who Bought This Also Bought" widget. It models e-commerce products as nodes in a **directed graph**, learns **dual embeddings** (source and target) per product to capture purchase asymmetry (phone→case, but not case→phone), and solves **selection bias** by leveraging browse co-view data to uncover products that should have been purchased together. In A/B tests, DAEMON delivered **+170% product sales and +190% profit gain** — making this one of the highest-impact GNN deployments in industry. This project replicates DAEMON on Kaggle's free T4 GPU, adapting a system originally trained on 64-core/488GB RAM machines to run within 16GB VRAM via mini-batch sampling, mixed precision, and aggressive memory optimization.

---

## Paper Summary

DAEMON (Direction AwarE Graph Neural Network MOdel for Node recommendation) jointly addresses **5 orthogonal challenges** in related product recommendation:

| # | Challenge | DAEMON's Approach |
|---|-----------|-------------------|
| 1 | **Product Asymmetry** | Dual embeddings (θ^s, θ^t); source embedding of query matched against target embeddings of candidates |
| 2 | **Co-purchase Likelihood** | Graph edges encode co-purchase relationships; GNN message passing aggregates neighborhood signals |
| 3 | **Selection Bias** | Co-view browse data mitigates historical bias; transitive reasoning (bought+viewed → should buy) |
| 4 | **Cold-Start Products** | Catalog metadata features allow embedding generation for products with zero purchase history |
| 5 | **Scalability** | Mini-batch neighbor sampling, FAISS approximate nearest neighbor search over millions of products |

**Key Results:**
- **30–160%** improvement in HitRate and MRR over SOTA baselines (APP, NERD, MagNet, DGGAN, Gravity GAE)
- **4–16%** AUC gains for link prediction tasks
- **+170%** product sales, **+190%** profit in A/B tests (p < 0.05)

---

## Why This Project Is High-Impact

### 1. Industry-Proven at Amazon Scale
This isn't a toy academic model — DAEMON runs in production on Amazon's marketplaces, serving millions of customers. Reproducing it demonstrates you understand systems that work at scale.

### 2. Novel Technical Contributions
- **Dual embeddings for directed graphs** — unlike undirected GNNs (GCN, GAT, GraphSage) that can't model asymmetry
- **Asymmetric loss function** — jointly optimizes co-purchase likelihood, asymmetry enforcement, and co-view similarity
- **Selection bias mitigation through co-view data** — no assumption of unbiased data required
- **Cold-start via metadata features** — works for brand-new products with zero interaction history

### 3. Portfolio Value
- Demonstrates GNN expertise on directed/heterogeneous graphs
- Shows ability to adapt large-scale industrial ML to consumer hardware
- Covers end-to-end ML: data pipeline, model architecture, training, evaluation, deployment (FAISS serving)

### 4. Reproducibility Achievement
The paper trained on 64-core machines with **488GB RAM** and graphs of 5.5M nodes, 31.7M edges. Making this work on a **T4 with 16GB VRAM** is a genuine engineering challenge that showcases system design skills.

---

## Kaggle Feasibility

| Aspect | Original Paper | Kaggle Adaptation |
|--------|---------------|-------------------|
| **Hardware** | 64-core CPU, 488GB RAM | T4 GPU (16GB VRAM), ~13GB system RAM |
| **Graph size** | 1.98M–5.5M nodes, 14–32M edges | 100K–500K nodes, 1–5M edges (subsampled) |
| **Training** | Full-batch on CPU cluster | Mini-batch neighbor sampling on GPU |
| **Embedding dim** | 64 | 64 (feasible) |
| **Feature dim** | 384–512 | 128–384 (reduced if needed) |
| **Precision** | FP32 | FP16 mixed precision (T4 Tensor Cores) |
| **Session** | Unlimited | ~9 hours (checkpoint & resume) |
| **Libraries** | DGL, PyTorch, FAISS | Same — all installable via pip |

### 5 Key Adaptations
1. **Mini-batch neighbor sampling** — train on subgraphs of 1024 seed nodes with layer-wise sampling [20, 10, 10]
2. **Mixed precision (AMP)** — 40-50% memory savings, ~8x faster on T4 Tensor Cores
3. **CPU-resident graph** — full graph lives on CPU RAM; only mini-batch subgraphs move to GPU
4. **Gradient checkpointing** — trades 20% compute for 30-50% activation memory
5. **Checkpoint & resume** — save every N batches to survive 9-hour session limit

---

## Project Structure

| Document | Purpose |
|----------|---------|
| [`SUMMARY.md`](./SUMMARY.md) | **This file** — project overview and context |
| [`ARCHITECTURE.md`](./ARCHITECTURE.md) | Model architecture, dual embeddings, forward pass, message-passing rules, asymmetric loss formulation |
| [`SYSTEM_DESIGN_OPTIMIZATION.md`](./SYSTEM_DESIGN_OPTIMIZATION.md) | Kaggle-specific memory budgets, batch strategies, AMP, checkpointing, fallback tiers |
| [`TRAINING_METHODOLOGY.md`](./TRAINING_METHODOLOGY.md) | Complete training loop, loss function details, hyperparameters, evaluation protocol (EQ1–EQ5) |
| [`DATA_PIPELINE.md`](./DATA_PIPELINE.md) | Data sourcing, feature engineering, graph construction, splits, negative sampling |
| [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) | Notebook cell-by-cell outline, milestones, testing strategy, risk mitigation, quality checklist |

---

## Quick Start

The final deliverable is a **single Jupyter notebook** (`daemon_kaggle.ipynb`) that runs end-to-end on Kaggle:

```
Cell 1:  pip install dgl faiss-gpu → imports → GPU check
Cell 2:  Configuration (all hyperparameters)
Cell 3:  Load preprocessed graph data (.npz → DGL graph)
Cell 4:  Graph construction + train/val/test split
Cell 5:  DAEMONLayer + DAEMONModel + AsymmetricLoss definitions
Cell 6:  Training loop (AMP, gradient checkpointing, tqdm, checkpointing)
Cell 7:  Evaluation: HitRate@k, MRR@k, AUC link prediction
Cell 8:  Generate all embeddings → build FAISS index
Cell 9:  Demo: query products → show related & similar recommendations
Cell 10: Cold-start demo: new product → recommendations
Cell 11: Ablation studies (optional)
Cell 12: Export results, save model/embeddings/FAISS index
```

---

## Technology Stack

| Component | Library | Why |
|-----------|---------|-----|
| **Deep Learning** | PyTorch 2.x | Pre-installed on Kaggle, AMP support, `torch.compile` |
| **Graph NN** | DGL 2.x (`pip install dgl`) | Mini-batch sampling, neighbor sampling, GPU graph storage |
| **Similarity Search** | FAISS-GPU (`pip install faiss-gpu`) | Billion-scale ANN, used by Amazon in production |
| **Text Encoding** | sentence-transformers | Encode product titles/descriptions → feature vectors |
| **Mixed Precision** | `torch.cuda.amp` | T4 Tensor Cores, 40-50% memory savings |
| **Monitoring** | tqdm, `torch.cuda.memory_stats` | Progress bars, VRAM tracking |

---

## Expected Results

For a graph of **100K–500K products** with **1–5M edges**, expect:

| Metric | Target | Paper Reference (G1, 1.98M nodes) |
|--------|--------|-----------------------------------|
| **HitRate@10** | 0.15–0.35 | DAEMON: 5.87x baseline |
| **HitRate@20** | 0.25–0.50 | DAEMON: 8.6x baseline |
| **MRR@10** | 0.10–0.25 | DAEMON: 2.02x baseline |
| **AUC (Link Existence)** | 0.85–0.95 | DAEMON: 40.31x baseline |
| **AUC (Direction)** | 0.70–0.85 | DAEMON: 14.72x baseline |
| **Cold-start HR@10** | 0.05–0.15 | DAEMON: 22.02x R-GCN |
| **Asymmetry check** | rel(a,b) ≠ rel(b,a) | Verified for one-way edges |
| **Inference latency** | <100ms per query | FAISS GPU index |

> **Note:** Paper reports relative gains (not absolute numbers) for confidentiality. Absolute values depend on dataset density and feature quality.

---

## References

- Virinchi, S., Saladi, A., & Mondal, A. (2022). *Recommending Related Products Using Graph Neural Networks in Directed Graphs.* ECML-PKDD.
- Kaggle Notebooks Documentation: https://www.kaggle.com/docs/notebooks
- DGL Documentation: https://docs.dgl.ai
- FAISS: https://github.com/facebookresearch/faiss
