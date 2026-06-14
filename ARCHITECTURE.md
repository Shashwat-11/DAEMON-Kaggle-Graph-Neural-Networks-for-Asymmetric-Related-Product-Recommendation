# ARCHITECTURE.md — DAEMON Model & System Architecture

> **Project:** DAEMON-Kaggle — Related Product Recommendation via GNNs on Directed Graphs
> **Paper:** Virinchi et al., ECML-PKDD 2022
> **Date:** June 2026

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │ Co-Purchase  │  │  Co-View     │  │  Product Catalog Metadata  │  │
│  │ Pairs (E_cp) │  │ Pairs (E_cv) │  │  (titles, desc, category)  │  │
│  └──────┬───────┘  └──────┬───────┘  └─────────────┬─────────────┘  │
└─────────┼──────────────────┼────────────────────────┼────────────────┘
          │                  │                        │
          ▼                  ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     GRAPH CONSTRUCTION                               │
│  Directed product graph G = (P, {E_cp ∪ E_cv})                      │
│  Nodes = products (0..N-1)                                          │
│  Edges = co-purchase (type 0) + co-view (type 1)                     │
│  Features X_i = text embeddings from product metadata               │
│  ~75% edges are directed (asymmetric)                                │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│     TRAINING PIPELINE         │   │     INFERENCE PIPELINE         │
│                               │   │                               │
│  Mini-batch Neighbor Sampling │   │  For query product q:         │
│  Subgraph → GPU               │   │  1. Lookup θ_q^s              │
│  Forward Pass (L=3 layers)    │   │  2. FAISS search on θ^t space │
│  Asymmetric Loss Computation  │   │  3. Return top-k results      │
│  Backward Pass + Adam         │   │                               │
│  Embedding L2 Normalization   │   │  (Also: θ_q^s · θ_v^s →       │
│                               │   │   similar/substitute products) │
└───────────────┬───────────────┘   └───────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   STORED ARTIFACTS                                   │
│  θ^s ∈ R^{N×d}  : source embedding matrix                           │
│  θ^t ∈ R^{N×d}  : target embedding matrix                           │
│  FAISS IndexIVFFlat on θ^t                                          │
│  Model checkpoint (.pt)                                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

| Component | Responsibility |
|-----------|---------------|
| **Data Ingestion** | Load co-purchase pairs, co-view pairs, product metadata features |
| **Graph Constructor** | Build DGL directed graph with edge types, assign node features |
| **Neighbor Sampler** | Layer-wise sampling: 20 neighbors (L1), 10 neighbors (L2+), per batch |
| **DAEMON Model** | L-layer GNN generating dual embeddings (θ^s, θ^t) per node |
| **Asymmetric Loss** | Jointly optimizes co-purchase likelihood, asymmetry, and co-view similarity |
| **Evaluator** | Computes HitRate@k, MRR@k, AUC for link prediction tasks |
| **FAISS Index** | GPU-accelerated nearest neighbor search for top-k retrieval |
| **Checkpoint Manager** | Saves/loads model state, embeddings, and training progress |

---

## 2. Model Architecture: DAEMON

### 2.1 Core Idea: Dual Embeddings

Every product `u` has **two embeddings**:

- **Source embedding** θ_u^s ∈ R^d — "As a query product, what do I want to recommend?"
- **Target embedding** θ_u^t ∈ R^d — "As a candidate product, when am I a good recommendation?"

**Recommendation relevance**:
```
rel(q, v) = (θ_q^s)^T · (θ_v^t)
```

This naturally enforces asymmetry: `rel(phone, case) >> rel(case, phone)`.

### 2.2 Message-Passing Rules

For **source embedding update** (what to recommend):
```
(h_u^s)^l = σ( Σ_{(u,v)∈E_cp} (h_v^t)^{l-1} · W^l )   ← co-purchase out-neighbors
          + σ( Σ_{(u,v)∈E_cv} (h_v^s)^{l-1} · W^l )   ← co-view out-neighbors
```

For **target embedding update** (when to be recommended):
```
(h_u^t)^l = σ( Σ_{(v,u)∈E_cp} (h_v^s)^{l-1} · W^l )   ← co-purchase in-neighbors
          + σ( Σ_{(v,u)∈E_cv} (h_v^t)^{l-1} · W^l )   ← co-view in-neighbors
```

Key insight: source aggregates from **target embeddings of out-neighbors**; target aggregates from **source embeddings of in-neighbors**. This is what makes the two embeddings learn different representations.

### 2.3 Forward Pass (Algorithm 1 from Paper)

```
Algorithm: DAEMON Forward Pass
──────────────────────────────────────────────────
Input:  G = (P, E_cp ∪ E_cv), node features X_u
        L layers, weight matrices W^1..W^L
Output: θ_u^s, θ_u^t for all u ∈ P

1. Initialize: (h_u^s)^0 = X_u, (h_u^t)^0 = X_u

2. For layer l = 1..L:
   a. For each node u:
      (h_u^s)^l = ReLU( Σ_out_cp (h_v^t)^{l-1} W^l )
                + ReLU( Σ_out_cv (h_v^s)^{l-1} W^l )
      
      (h_u^t)^l = ReLU( Σ_in_cp  (h_v^s)^{l-1} W^l )
                + ReLU( Σ_in_cv  (h_v^t)^{l-1} W^l )
   
   b. L2 Normalize:
      (h_u^s)^l = (h_u^s)^l / ||(h_u^s)^l||_2
      (h_u^t)^l = (h_u^t)^l / ||(h_u^t)^l||_2

3. Final embeddings:
   θ_u^s = (h_u^s)^L
   θ_u^t = (h_u^t)^L
```

### 2.4 Layer Architecture Detail

Each layer `l` has **one shared weight matrix W^l** (learned) and uses ReLU activation. The paper applies this same W^l for both co-purchase and co-view aggregation — the differentiation comes from the neighborhood and which embedding variant is aggregated.

```
class DAEMONLayer(nn.Module):
    """
    Single layer of DAEMON.
    
    For source embedding:
      h_s = ReLU( CP_out_neighbors(tgt_emb) · W )
          + ReLU( CV_out_neighbors(src_emb) · W )
    
    For target embedding:
      h_t = ReLU( CP_in_neighbors(src_emb) · W )
          + ReLU( CV_in_neighbors(tgt_emb) · W )
    
    Then: L2 normalize both outputs
    """
    
    def __init__(self, in_dim, out_dim):
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # Shared weight matrix for all 4 aggregation types
    
    def forward(self, graph, h_src, h_tgt):
        # graph must have edge_type: 0=cp, 1=cv
        # h_src: source embeddings from previous layer [N, d]
        # h_tgt: target embeddings from previous layer [N, d]
        
        cp_mask = graph.edata['type'] == 0
        cv_mask = graph.edata['type'] == 1
        
        # Source embedding update
        cp_out_msg_tgt = aggregate(h_tgt[graph.out_edges(cp_mask)], ...)
        cv_out_msg_src = aggregate(h_src[graph.out_edges(cv_mask)], ...)
        h_src_new = ReLU(cp_out_msg_tgt @ W) + ReLU(cv_out_msg_src @ W)
        
        # Target embedding update
        cp_in_msg_src = aggregate(h_src[graph.in_edges(cp_mask)], ...)
        cv_in_msg_tgt = aggregate(h_tgt[graph.in_edges(cv_mask)], ...)
        h_tgt_new = ReLU(cp_in_msg_src @ W) + ReLU(cv_in_msg_tgt @ W)
        
        # L2 normalize
        return F.normalize(h_src_new), F.normalize(h_tgt_new)
```

### 2.5 Why L2 Normalization After Each Layer

The paper normalizes embeddings to unit norm after **every layer** (not just at the end). This:
1. Prevents embedding magnitude explosion across layers
2. Makes the dot product in `rel(q,v)` a cosine similarity
3. Stabilizes training with the sigmoid-based loss function
4. Ensures FAISS inner-product search ≡ cosine similarity search

---

## 3. Graph Construction

### 3.1 Node and Edge Definitions

```
P   = {0, 1, ..., N-1}        # Product IDs
E_cp = {(u,v) | u, v co-purchased together}  # ~75% one-way
E_cv = {(u,v) | u, v co-viewed in same session}  
G   = (P, E_cp ∪ E_cv)        # Directed product graph

Edge types:
  type = 0  →  co-purchase edge
  type = 1  →  co-view edge

Feature per node:
  X_u ∈ R^d  →  text embedding from product title/description (384-dim)
```

### 3.2 Heterogeneous Edge Handling

The graph has two edge types but **one node type** (products). DGL can handle this as:
- **Option A (Recommended):** Single homogeneous graph with `edge_type` tensor
- **Option B:** Heterogeneous graph with relation types `('product', 'cp', 'product')` and `('product', 'cv', 'product')`

Option A is simpler and performs better for mini-batch sampling.

### 3.3 Expected Graph Statistics

| Statistic | Small Dataset | Medium Dataset | Paper (G1) |
|-----------|--------------|----------------|------------|
| Nodes | 10K–50K | 100K–500K | 1.98M |
| Edges (total) | 50K–500K | 1M–5M | 14.1M |
| Co-purchase edges | ~50% | ~50% | 7M |
| Co-view edges | ~50% | ~50% | 7.1M |
| % Directed edges | ~75% | ~75% | 76.33% |
| Avg degree | 5–10 | 5–10 | 7.3 |
| Feature dim | 384 | 384 | 384 |

### 3.4 Feature Engineering

Product features come from catalog metadata encoded as dense vectors:

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim output

# For each product:
text = f"{product['title']} {product['description']} {product['category']}"
feature = encoder.encode(text)  # [384] float32

# Store as FP16 for memory efficiency
features_fp16 = features.astype(np.float16)
```

**Alternative lightweight encoding** (if transformers use too much RAM):
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

tfidf = TfidfVectorizer(max_features=10000)
X_tfidf = tfidf.fit_transform(product_texts)
X_svd = TruncatedSVD(n_components=128).fit_transform(X_tfidf)
```

---

## 4. Asymmetric Loss Function

### 4.1 Full Formulation (Equation 2)

```
L = - [ Σ_{(u,v)∈E_cp} [ log σ(θ_u^s · θ_v^t) + Σ_{k=1}^{n_k} log σ(1 - θ_u^s · θ_z^t) ]    (1) Co-purchase
      + Σ_{(u,v)∈E_cp ∧ (v,u)∉E_cp} [ log σ(θ_u^s · θ_v^t) + log σ(1 - θ_v^s · θ_u^t) ]      (2) Asymmetry
      + Σ_{(u,v)∈E_cv} [ log σ(θ_u^s · θ_v^s) + log σ(θ_u^t · θ_v^t) ] ]                      (3) Co-view similarity
```

Where:
- σ(x) = 1/(1 + e^{-x}) is the sigmoid function
- z ~ P_r(P) is a randomly sampled negative product (uniform)
- n_k is the number of negative samples per positive pair

### 4.2 Component Breakdown

#### Component 1: Co-purchase Likelihood
```
For each co-purchase pair (u,v):
  → Maximize θ_u^s · θ_v^t   (source of u should match target of v)
  → Minimize θ_u^s · θ_z^t   (source of u should NOT match target of random z)
```
This learns: "products bought together should be recommended for each other."

#### Component 2: Asymmetry Enforcement
```
For each one-way co-purchase edge (u→v but v↛u):
  → Maximize θ_u^s · θ_v^t   (u can recommend v)
  → Minimize θ_v^s · θ_u^t   (v should NOT recommend u)
```
This learns: "phone can recommend case, but case cannot recommend phone."

#### Component 3: Co-view Similarity
```
For each co-view pair (u,v):
  → Maximize θ_u^s · θ_v^s   (source embeddings should be similar)
  → Maximize θ_u^t · θ_v^t   (target embeddings should be similar)
```
This learns: "products that are similar (co-viewed) should have similar representations." This enables cold-start and selection bias mitigation through transitivity.

### 4.3 Negative Sampling

```python
def sample_negatives(num_nodes, positive_pairs, num_neg=5):
    """
    For each positive pair (u, v), sample num_neg random nodes z
    where z ≠ u and (u,z) is not a known positive pair.
    """
    neg_samples = torch.randint(0, num_nodes, (len(positive_pairs), num_neg))
    # Filter out actual positives if needed
    return neg_samples
```

### 4.4 Theoretical Guarantees

**Lemma 1 (Asymmetry):** When (u,v) ∈ E_cp, DAEMON ensures θ_u^s · θ_v^t >> θ_v^s · θ_u^t for one-way edges.

**Lemma 2 (Selection Bias):** Through transitivity (bought→viewed→should buy), DAEMON assigns high scores to products that were never co-purchased together but are related.

---

## 5. Recommendation Engine

### 5.1 Related Product Retrieval

```python
def recommend_related(query_product_id, source_embeddings, target_embeddings, faiss_index, k=10):
    """
    Given query product q, find top-k products v that maximize θ_q^s · θ_v^t
    """
    query_vec = source_embeddings[query_product_id]  # [d]
    scores, indices = faiss_index.search(query_vec.reshape(1, -1), k + 1)
    # Remove self-match if present
    return indices[0], scores[0]
```

### 5.2 Substitute/Similar Product Retrieval (Byproduct)

The paper notes that DAEMON can also recommend similar products using source-source similarity:

```python
def recommend_similar(query_product_id, source_embeddings, faiss_src_index, k=10):
    """
    Find products v that maximize θ_q^s · θ_v^s (similar products)
    """
    query_vec = source_embeddings[query_product_id]
    scores, indices = faiss_src_index.search(query_vec.reshape(1, -1), k + 1)
    return indices[0], scores[0]
```

This is a free byproduct — no additional training needed.

### 5.3 FAISS Index Construction

```python
import faiss

# Build IVF index for fast approximate search
d = embedding_dim  # 64
nlist = min(4096, int(np.sqrt(num_products)))  # number of clusters

quantizer = faiss.IndexFlatIP(d)  # inner product = similarity
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

# Train the index (k-means clustering)
index.train(target_embeddings_np)
index.add(target_embeddings_np)

# Search
index.nprobe = 32  # number of clusters to search
```

---

## 6. Cold-Start Handling

### 6.1 Problem

New products (cold-start) have:
- Catalog metadata features (X_c)
- **Zero** co-purchase or co-view edges
- Need meaningful recommendations from day 1

### 6.2 DAEMON's Approach

```python
def handle_cold_start(cold_product_features, existing_features, existing_ids, k_nn=5):
    """
    For cold-start product c:
    1. Find k most similar existing products via feature similarity
    2. Add edges c → {similar_products} to graph
    3. Run forward pass to generate embeddings for c
    4. Use c's source embedding to search target space
    """
    # Step 1: k-NN lookup
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k_nn, metric='cosine')
    nn.fit(existing_features)
    distances, indices = nn.kneighbors(cold_product_features.reshape(1, -1))
    
    similar_ids = existing_ids[indices[0]]  # {c1, c2, ..., ck}
    
    # Step 2: Augment graph edges
    for sid in similar_ids:
        g.add_edges(cold_product_id, sid, {'type': 1})  # co-view type
        g.add_edges(sid, cold_product_id, {'type': 1})  # bidirectional
    
    # Step 3: Forward pass (model handles subgraph)
    with torch.no_grad():
        src_emb, tgt_emb = model(g, features)
    
    # Step 4: Recommend
    cold_src = src_emb[cold_product_id]
    scores, recs = faiss_index.search(cold_src, top_k)
    return recs
```

### 6.3 Why This Works

The co-view similarity term in the loss function (Component 3) forces similar products to have similar embeddings. Since the cold-start product is connected to existing products via feature similarity, the GNN aggregates their embeddings to produce a meaningful representation — even with zero purchase history.

---

## 7. Data Flow Diagrams

### 7.1 Training Flow

```
┌──────────┐    ┌──────────────┐    ┌───────────────┐
│ Raw Data │ →  │ Graph Build  │ →  │ Edge Split    │
│ (csv/parq│    │ DGL Graph    │    │ 75/5/20%      │
└──────────┘    └──────────────┘    └───────┬───────┘
                                            │
                    ┌───────────────────────▼───────────────────────┐
                    │          Per-Batch Training Loop              │
                    │                                               │
                    │  Seed nodes (batch_size=1024)                 │
                    │       ↓                                       │
                    │  NeighborSampler([20, 10, 10])                │
                    │       ↓                                       │
                    │  Subgraph + features → GPU                    │
                    │       ↓                                       │
                    │  DAEMON Forward Pass (3 layers)               │
                    │    (h_src^l, h_tgt^l) = f(subgraph, W^l)     │
                    │       ↓                                       │
                    │  L2 normalize all embeddings                  │
                    │       ↓                                       │
                    │  Asymmetric Loss Computation                  │
                    │    Component 1: Co-purchase (+ neg samples)   │
                    │    Component 2: Asymmetry (one-way edges)     │
                    │    Component 3: Co-view similarity            │
                    │       ↓                                       │
                    │  Loss.backward() + optimizer.step()           │
                    │       ↓                                       │
                    │  Per-step: log loss, VRAM usage               │
                    └───────────────────────────────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │          Per-Epoch Validation                 │
                    │  Full graph forward pass (or sampled)         │
                    │  Compute HitRate@k, MRR@k on val edges       │
                    │  Save checkpoint if best val HR              │
                    └───────────────────────────────────────────────┘
```

### 7.2 Inference Flow

```
┌──────────────┐
│ Query Product│
│      q       │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Lookup θ_q^s     │  ← from stored embedding matrix
│ shape: [64]      │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ FAISS Search     │  ← inner product on target embeddings
│ IndexIVFFlat     │     nprobe=32, k=10
│ on θ^t [N × 64]  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Top-k Product IDs│
│ + Relevance      │
│ Scores           │
└──────────────────┘
```

### 7.3 Cold-Start Flow

```
┌──────────────┐
│ New Product c│
│ Features X_c │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ k-NN Feature Search  │  ← cosine similarity on input features
│ Find {c1, c2, ...ck} │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Augment Graph        │  ← add edges c→c_i (co-view type)
│ with new edges       │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ DAEMON Forward Pass  │  ← generate θ_c^s, θ_c^t
│ (subgraph around c)  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ FAISS Search θ_c^s   │  ← recommend related products
│ → top-k results      │
└──────────────────────┘
```

---

## 8. Component Diagram (Code-Level)

```
┌─────────────────────────────────────────────────────────────┐
│                       daemon_kaggle.ipynb                    │
│                                                              │
│  ┌──────────────────┐    ┌────────────────────────────┐     │
│  │   DAEMONConfig   │    │     GraphDataModule        │     │
│  │  (dataclass)     │    │  - build_graph()           │     │
│  │  - embed_dim     │    │  - split_edges()           │     │
│  │  - num_layers    │    │  - create_dataloaders()    │     │
│  │  - batch_size    │    │  - cold_start_subgraph()   │     │
│  │  - lr            │    └────────────┬───────────────┘     │
│  │  - num_neighbors │                 │                      │
│  │  - num_neg       │                 ▼                      │
│  └────────┬─────────┘    ┌────────────────────────────┐     │
│           │              │     NeighborSampler        │     │
│           │              │  dgl.dataloading.          │     │
│           │              │  MultiLayerNeighborSampler │     │
│           ▼              │  fanouts=[20, 10, 10]      │     │
│  ┌──────────────────┐    └────────────┬───────────────┘     │
│  │   DAEMONLayer    │                 │                      │
│  │  - W (Linear)    │                 ▼                      │
│  │  - forward()     │    ┌────────────────────────────┐     │
│  │   → (h_s, h_t)   │    │     DataLoader             │     │
│  └────────┬─────────┘    │  batch_size=1024           │     │
│           │              │  shuffle=True              │     │
│           ▼              │  device='cuda'             │     │
│  ┌──────────────────┐    └────────────────────────────┘     │
│  │   DAEMONModel    │                                        │
│  │  - layers:       │    ┌────────────────────────────┐     │
│  │    ModuleList     │    │   AsymmetricLoss           │     │
│  │  - forward()     │    │  - cp_loss()               │     │
│  │   → (θ_s, θ_t)   │    │  - asym_loss()             │     │
│  └────────┬─────────┘    │  - cv_loss()               │     │
│           │              └────────────┬───────────────┘     │
│           │                           │                      │
│           └───────────┬───────────────┘                     │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 Training Loop                        │   │
│  │  for batch in loader:                                │   │
│  │    with autocast():                                  │   │
│  │      src, tgt = model(batch_graph, batch_feat)       │   │
│  │      loss = asym_loss(src, tgt, pairs, negs)         │   │
│  │    scaler.scale(loss).backward()                     │   │
│  │    scaler.step(optimizer)                            │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 Evaluator                            │   │
│  │  - hit_rate_at_k(src, tgt, test_edges, k)            │   │
│  │  - mrr_at_k(src, tgt, test_edges, k)                 │   │
│  │  - link_prediction_auc(src, tgt, pos_edges, neg_edges)│  │
│  │  - direction_prediction_auc(src, tgt, one_way_edges) │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 FAISS Retriever                      │   │
│  │  - build_index(target_embeddings)                    │   │
│  │  - search(query_src_emb, k) → top_k_ids, scores      │   │
│  │  - recommend_related(q) / recommend_similar(q)       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Dual embeddings (not single)** | Required for asymmetry; single embedding can't model phone→case vs case→phone |
| **Shared weight matrix per layer** | Paper uses one W^l for all 4 aggregations; reduces parameters significantly |
| **Separate treatment of E_cp and E_cv** | Co-purchase is about "buy together", co-view is about "similarity"; conflating them hurts performance |
| **L2 normalize every layer** | Prevents magnitude drift, stabilizes training, enables cosine similarity in FAISS |
| **Mini-batch neighbor sampling** | Essential for Kaggle; full-batch of 1M+ nodes would OOM immediately |
| **Sigmoid-based loss** | Binary cross-entropy style loss works better than margin-based losses for this domain |
| **FAISS inner product index** | Cosine similarity search after L2 normalization ≡ inner product; FAISS optimized for this |

---

## References

- Virinchi et al. (2022). *Recommending Related Products Using Graph Neural Networks in Directed Graphs*. ECML-PKDD.
- DGL Documentation: Neighbor Sampling — https://docs.dgl.ai/guide/minibatch.html
- FAISS: Inner Product Search — https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html
