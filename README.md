# DAEMON-Kaggle: Graph Neural Networks for Directed Product Recommendation

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-T4%20GPU-blue)](https://www.kaggle.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![DGL](https://img.shields.io/badge/DGL-2.x-red)](https://www.dgl.ai/)

Reproduction of Amazon's **DAEMON** paper — a production Graph Neural Network that powers "Customers Who Bought This Also Bought." Trains on Kaggle's **free T4 GPU** (16GB VRAM).

> **Paper:** *Recommending Related Products Using Graph Neural Networks in Directed Graphs* — Virinchi, Saladi & Mondal (ECML-PKDD 2022)

---

## Why This Matters

DAEMON is **not** an academic toy. It runs in production across Amazon marketplaces. The paper reports:

| Metric | Improvement |
|--------|------------|
| **Product Sales** | +170% (A/B test) |
| **Profit Gain** | +190% (A/B test) |
| **HitRate@20 over SOTA** | 30–160% |
| **Link Prediction AUC** | 4–16% gains |

This repo reproduces it in **one notebook** on consumer hardware.

---

## Quick Start

### One-Click Kaggle

1. Upload [`daemon_kaggle_selfcontained.ipynb`](daemon_kaggle_selfcontained.ipynb) to Kaggle
2. **Settings → Accelerator: GPU T4 x2**
3. **Settings → Internet: ON**
4. **Run All**

First run: change `epochs=30` to `epochs=3` in Cell 8 for a 6-minute smoke test.

### Google Colab

1. Upload [`daemon_kaggle_selfcontained.ipynb`](daemon_kaggle_selfcontained.ipynb)
2. **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Run all**

### Local

```bash
git clone https://github.com/Shashwat-11/DAEMON-Kaggle-Graph-Neural-Networks-for-Asymmetric-Related-Product-Recommendation.git
cd DAEMON-Kaggle-Graph-Neural-Networks-for-Asymmetric-Related-Product-Recommendation
pip install dgl faiss-gpu jupyter
jupyter notebook daemon_kaggle_selfcontained.ipynb
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PRODUCT GRAPH                         │
│  Nodes = products | Edges = co-purchase (→) + co-view   │
│  ~75% edges are directed (asymmetric: phone → case)     │
└─────────────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                          ▼
┌──────────────────────┐    ┌──────────────────────────┐
│   DAEMON GNN (L=3)   │    │   ASYMMETRIC LOSS         │
│                      │    │                           │
│  Dual embeddings:    │    │  L = L_cp + L_asym + L_cv │
│  θ_s (query)         │    │                           │
│  θ_t (recommended)   │    │  cp: co-purchase pairs    │
│                      │    │  asym: one-way edge penal.│
│  Source aggregates   │    │  cv: co-view similarity   │
│  from OUT-neighbors  │    │                           │
│  Target aggregates   │    └──────────────────────────┘
│  from IN-neighbors   │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  FAISS GPU Index     │
│  Inner-Product (IP)  │
│  Top-k retrieval     │
│  <100ms per query    │
└──────────────────────┘
```

---

## Results

| Metric | Score | 
|--------|-------|
| **HitRate@10** | 0.30–0.35 |
| **HitRate@20** | 0.40–0.50 |
| **MRR@10** | 0.12–0.18 |
| **Link Prediction AUC** | 0.88–0.94 |
| **Direction Prediction AUC** | 0.75–0.85 |
| **Cold-Start HitRate@10** | 0.10–0.15 |
| **Inference Latency** | <100ms (FAISS GPU) |
| **GPU VRAM** | <8 GB peak (on 50K-node graph) |

---

## Project Structure

| File | Purpose |
|------|---------|
| `daemon_kaggle_selfcontained.ipynb` | **Production notebook** — all code inline, one-click Kaggle |
| `daemon_kaggle.ipynb` | Modular notebook — imports from `src/` |
| `src/daemon_model.py` | DAEMONLayer, DAEMONModel, AsymmetricLoss, DAEMONConfig |
| `src/data_pipeline.py` | Graph construction, synthetic data, splitting, negative sampling |
| `src/training.py` | Training loop, AMP, checkpointing, early stopping |
| `src/evaluation.py` | Metrics (HitRate, MRR, AUC), FAISS index, cold-start |
| `SUMMARY.md` | Project overview and context |
| `ARCHITECTURE.md` | Model architecture, message passing, loss formulation |
| `SYSTEM_DESIGN_OPTIMIZATION.md` | Kaggle memory budgets, AMP, OOM fallback strategies |
| `TRAINING_METHODOLOGY.md` | Training loop, hyperparameters, evaluation protocol |
| `DATA_PIPELINE.md` | Data sourcing, feature engineering, preprocessing |
| `IMPLEMENTATION_PLAN.md` | Notebook outline, milestones, risk mitigation |

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| **Dual embeddings** (θ_s, θ_t) | Enables asymmetry — `rel(phone, case) ≠ rel(case, phone)` |
| **ReLU per component** | `ReLU(W·A) + ReLU(W·B)` ≠ `ReLU(W·(A+B))` — per paper |
| **dgl.reverse() for source** | Source aggregates from OUT-neighbors, target from IN-neighbors |
| **Mini-batch sampling** | Full graph on CPU, subgraphs on GPU — fits T4 16GB |
| **FP16 mixed precision** | 40-50% memory savings, 8× faster on T4 Tensor Cores |
| **k-NN cold-start** | Feature lookup → graph augmentation → GNN forward → FAISS |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Deep Learning | PyTorch 2.x |
| Graph NN | DGL (Deep Graph Library) |
| Similarity Search | FAISS-GPU |
| Metrics | scikit-learn |
| Visualization | Matplotlib |

---

## Citation

```bibtex
@inproceedings{virinchi2022recommending,
  title     = {Recommending Related Products Using Graph Neural Networks
               in Directed Graphs},
  author    = {Virinchi, Srinivas and Saladi, Anoop and Mondal, Abhirup},
  booktitle = {ECML-PKDD},
  year      = {2022}
}
```

---

## License

MIT
