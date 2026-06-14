# DATA_PIPELINE.md — Data Sourcing, Feature Engineering & Graph Construction

> **Project:** DAEMON-Kaggle — Related Product Recommendation via GNNs on Directed Graphs
> **Target Graphs:** 100K–500K nodes (products), 1M–5M edges (co-purchase + co-view)

---

## 1. Data Sourcing Strategy

### 1.1 Dataset Options (Ranked by Suitability)

| # | Dataset | Nodes | Features | Co-purchase? | Co-view? | Kaggle? |
|---|---------|-------|----------|-------------|----------|---------|
| 1 | **Amazon Product Reviews 2023** | 100K+ products | Title, description, category, price | ✗ (needs synth) | ✗ (needs synth) | ✓ Available |
| 2 | **Instacart Market Basket** | 50K products, 3M orders | Product name, aisle, dept | ✓ Via basket co-occurrence | ✗ (needs synth) | ✓ Available |
| 3 | **Olist Brazilian E-Commerce** | 33K products | Title, category, price | ✓ Via order co-occurrence | ✗ (needs synth) | ✓ Available |
| 4 | **Walmart Product 2019** | 100K+ products | Title, description, category | ✗ | ✗ | ✓ Available |
| 5 | **Synthetic** (script-generated) | 1K–100K | Random embeddings | ✓ Generated | ✓ Generated | N/A (local) |

### 1.2 Recommended Approach

**Primary path:** Use **Amazon Product Reviews 2023** (McAuley Lab) for real product metadata, then **synthesize co-purchase and co-view edges** based on realistic patterns. This mirrors the paper's approach of using real e-commerce data while being fully reproducible on Kaggle.

**Quick-test path:** Use the **synthetic data generator** (Section 1.4) with 5K–50K nodes for code development and hyperparameter tuning.

### 1.3 Edge Synthesis Strategy

Since public datasets rarely have explicit co-purchase/co-view pairs, we synthesize them realistically:

```
Co-purchase edge (u, v) is likely if:
  - u and v are in the same category
  - u and v have similar price point (±20%)
  - u/v is an accessory of v/u (detected via category hierarchy)
  - u and v have high co-occurrence in order baskets

Co-view edge (u, v) is likely if:
  - u and v are in the same subcategory
  - u and v have high text similarity (cosine > 0.7)
  - u and v share brand
  - u has high rating and v is similar (selection bias proxy)
```

### 1.4 Synthetic Data Generator (for rapid testing)

```python
import numpy as np
import torch

def generate_synthetic_graph(num_products=10000, feature_dim=384, 
                              avg_cp_degree=5, avg_cv_degree=8,
                              asymmetry_ratio=0.75):
    """
    Generate a synthetic product graph for testing DAEMON.
    
    Returns:
        edge_index_cp: [2, num_cp_edges] co-purchase edges
        edge_index_cv: [2, num_cv_edges] co-view edges
        features: [num_products, feature_dim] product features
        categories: [num_products] category labels
    """
    np.random.seed(42)
    
    # Generate product features (cluster-like to simulate categories)
    num_categories = num_products // 500
    cluster_centers = np.random.randn(num_categories, feature_dim).astype(np.float32)
    categories = np.random.randint(0, num_categories, num_products)
    features = cluster_centers[categories] + 0.3 * np.random.randn(num_products, feature_dim).astype(np.float32)
    
    # Generate co-purchase edges (within category, some asymmetric)
    cp_edges = []
    for cat in range(num_categories):
        cat_products = np.where(categories == cat)[0]
        for u in cat_products:
            # Co-purchase with products in same category
            num_neighbors = np.random.poisson(avg_cp_degree)
            neighbors = np.random.choice(cat_products, min(num_neighbors, len(cat_products)-1), replace=False)
            for v in neighbors:
                if u != v:
                    cp_edges.append([u, v])
                    # Asymmetry: ~75% are one-way
                    if np.random.random() > asymmetry_ratio:
                        cp_edges.append([v, u])  # bidirectional
    
    # Generate co-view edges (higher degree, more bidirectional, cross-category)
    cv_edges = []
    for u in range(num_products):
        num_neighbors = np.random.poisson(avg_cv_degree)
        # Co-view: similar items (within category) + cross-category (for bias simulation)
        cat_neighbors = np.random.choice(
            np.where(categories == categories[u])[0], 
            min(num_neighbors // 2, max(1, np.sum(categories == categories[u])-1)), 
            replace=False
        )
        for v in cat_neighbors:
            if u != v:
                cv_edges.append([u, v])
                cv_edges.append([v, u])  # Co-view is mostly bidirectional
    
    cp_edges = np.array(cp_edges).T if cp_edges else np.zeros((2, 0), dtype=np.int64)
    cv_edges = np.array(cv_edges).T if cv_edges else np.zeros((2, 0), dtype=np.int64)
    
    print(f"Generated: {num_products} nodes, {cp_edges.shape[1]} CP edges, {cv_edges.shape[1]} CV edges")
    print(f"Asymmetry: {estimate_asymmetry(cp_edges):.1f}% one-way edges")
    
    return cp_edges, cv_edges, features, categories

def estimate_asymmetry(edges):
    """Estimate % of one-way edges."""
    edge_set = set(zip(edges[0], edges[1]))
    one_way = sum(1 for u, v in zip(edges[0], edges[1]) if (v, u) not in edge_set)
    return 100 * one_way / edges.shape[1] if edges.shape[1] > 0 else 0
```

---

## 2. Required Data Components

### 2.1 Product Catalog

| Field | Type | Description | Required? |
|-------|------|-------------|-----------|
| `product_id` | int/str | Unique identifier | ✓ |
| `title` | str | Product title | ✓ |
| `description` | str | Product description | Optional |
| `category` | str | Primary category (e.g., "Electronics/Phones") | ✓ |
| `subcategory` | str | Fine-grained category | Optional |
| `brand` | str | Brand name | Optional |
| `price` | float | Price in local currency | Optional |
| `rating` | float | Average rating (1-5) | Optional |

### 2.2 Co-purchase Pairs (E_cp)

```
Format: CSV with columns [product_u, product_v]
Each row: these two products were bought together in the same order
Direction: u → v means "u is co-purchased with v"
~75% should be one-way (paper G1: 76.33% directed)
```

### 2.3 Co-view Pairs (E_cv)

```
Format: CSV with columns [product_u, product_v]
Each row: these two products were viewed in the same browsing session
Direction: Mostly bidirectional (similarity relationship)
```

### 2.4 Generating Co-view from Co-category (if unavailable)

```python
def generate_co_view_edges(product_df, similarity_threshold=0.7):
    """
    If no co-view data, generate from product similarity.
    Products in the same subcategory with high text similarity are "co-viewed".
    """
    # Encode product titles
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(product_df['title'].tolist())
    
    # Per category, find similar pairs
    cv_pairs = []
    for category in product_df['category'].unique():
        cat_mask = product_df['category'] == category
        cat_embeddings = embeddings[cat_mask]
        cat_indices = np.where(cat_mask)[0]
        
        # Cosine similarity within category
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(cat_embeddings)
        
        for i in range(len(cat_indices)):
            for j in range(i+1, len(cat_indices)):
                if sim_matrix[i, j] > similarity_threshold:
                    cv_pairs.append([cat_indices[i], cat_indices[j]])
                    cv_pairs.append([cat_indices[j], cat_indices[i]])  # Bidirectional
    
    return np.array(cv_pairs).T
```

---

## 3. Feature Engineering

### 3.1 Text Features (Primary)

```python
from sentence_transformers import SentenceTransformer

def encode_product_texts(product_df, model_name='all-MiniLM-L6-v2', batch_size=256):
    """
    Encode product titles + descriptions into dense vectors.
    
    Model: all-MiniLM-L6-v2 → 384-dimensional embeddings
    Lightweight (~80MB), fast inference, good semantic quality.
    """
    model = SentenceTransformer(model_name)
    
    # Combine text fields
    product_df['text'] = product_df['title'].fillna('')
    if 'description' in product_df.columns:
        product_df['text'] += ' ' + product_df['description'].fillna('')
    if 'category' in product_df.columns:
        product_df['text'] += ' [CATEGORY: ' + product_df['category'].fillna('') + ']'
    
    texts = product_df['text'].tolist()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    return embeddings.astype(np.float32)  # [N, 384]

# For missing descriptions, use category as fallback
product_df['text'] = np.where(
    product_df['description'].isna(),
    product_df['title'] + ' ' + product_df['category'],
    product_df['title'] + ' ' + product_df['description']
)
```

### 3.2 Alternative: Lightweight TF-IDF + SVD

```python
def encode_lightweight(texts, feature_dim=128):
    """Use TF-IDF + TruncatedSVD when transformers won't fit."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))),
        ('svd', TruncatedSVD(n_components=feature_dim, random_state=42))
    ])
    
    embeddings = pipeline.fit_transform(texts)
    return embeddings.astype(np.float32)
```

### 3.3 Numerical Features (Optional)

```python
def encode_numerical_features(product_df):
    """Encode price, rating as additional features."""
    features = []
    
    if 'price' in product_df.columns:
        # Log-normalize price (prices are long-tailed)
        log_price = np.log1p(product_df['price'].fillna(product_df['price'].median()))
        log_price = (log_price - log_price.mean()) / (log_price.std() + 1e-8)
        features.append(log_price.values.reshape(-1, 1))
    
    if 'rating' in product_df.columns:
        # Normalize rating to [-1, 1]
        rating = (product_df['rating'].fillna(3.0) - 3.0) / 2.0
        features.append(rating.values.reshape(-1, 1))
    
    if features:
        return np.concatenate(features, axis=1).astype(np.float32)
    return None
```

### 3.4 Feature Fusion

```python
def fuse_features(text_embeddings, num_features=None, method='concat'):
    """
    Fuse text and numerical features.
    
    For DAEMON, input features must be a single vector per node.
    Options:
      - concat: [text_emb | num_features] → larger dim, simplest
      - weighted: learnable weights per feature type
      - projection: project concatenated to fixed dim via Linear layer
    """
    if num_features is None:
        return text_embeddings
    
    if method == 'concat':
        return np.concatenate([text_embeddings, num_features], axis=1)
    
    # Default: concatenate
    return np.concatenate([text_embeddings, num_features], axis=1)
```

---

## 4. Graph Construction Pipeline

### 4.1 Complete Pipeline

```python
def build_product_graph(product_df, cp_edges, cv_edges, feature_dim=384):
    """
    Build DGL directed graph for DAEMON training.
    
    Args:
        product_df: DataFrame with product metadata
        cp_edges: [2, E_cp] co-purchase edge pairs
        cv_edges: [2, E_cv] co-view edge pairs
        feature_dim: output feature dimension
    
    Returns:
        g: DGL graph with edge types
        features: [N, feature_dim] node feature tensor
    """
    N = len(product_df)
    
    # Step 1: Generate features
    print("Encoding product features...")
    features = encode_product_texts(product_df, model_name='all-MiniLM-L6-v2')
    
    # Step 2: Combine edges
    # Co-purchase edges: type=0
    # Co-view edges: type=1
    all_src = np.concatenate([cp_edges[0], cv_edges[0]])
    all_dst = np.concatenate([cp_edges[1], cv_edges[1]])
    edge_types = np.concatenate([
        np.zeros(cp_edges.shape[1], dtype=np.int64),  # type 0 = co-purchase
        np.ones(cv_edges.shape[1], dtype=np.int64)     # type 1 = co-view
    ])
    
    # Step 3: Build DGL graph
    import dgl
    g = dgl.graph((all_src, all_dst), num_nodes=N)
    g.edata['type'] = torch.tensor(edge_types, dtype=torch.long)
    g.ndata['feat'] = torch.tensor(features, dtype=torch.float32)
    
    print(f"Graph built: {N} nodes, {g.num_edges()} edges")
    print(f"  Co-purchase: {cp_edges.shape[1]} edges")
    print(f"  Co-view: {cv_edges.shape[1]} edges")
    print(f"  Directed: {estimate_directed_pct(g):.1f}%")
    
    return g, features

def estimate_directed_pct(g):
    """Estimate percentage of directed (one-way) edges in graph."""
    src, dst = g.edges()
    edge_set = set(zip(src.tolist(), dst.tolist()))
    one_way = sum(1 for s, d in zip(src.tolist(), dst.tolist()) if (d, s) not in edge_set)
    return 100 * one_way / g.num_edges()
```

### 4.2 Graph Statistics

```python
def print_graph_stats(g):
    """Print detailed graph statistics."""
    in_degrees = g.in_degrees().float()
    out_degrees = g.out_degrees().float()
    
    cp_mask = g.edata['type'] == 0
    cv_mask = g.edata['type'] == 1
    
    print(f"{'='*50}")
    print(f"GRAPH STATISTICS")
    print(f"{'='*50}")
    print(f"  Nodes:           {g.num_nodes():,}")
    print(f"  Total edges:     {g.num_edges():,}")
    print(f"    Co-purchase:   {cp_mask.sum().item():,}")
    print(f"    Co-view:       {cv_mask.sum().item():,}")
    print(f"  Avg degree:      {in_degrees.mean().item():.1f}")
    print(f"  Max degree:      {in_degrees.max().item():.0f}")
    print(f"  Min degree:      {in_degrees.min().item():.0f}")
    print(f"  Isolated nodes:  {(in_degrees == 0).sum().item()}")
    print(f"  Feature dim:     {g.ndata['feat'].shape[1]}")
    
    # Edge direction analysis
    src, dst = g.edges()
    edge_set = set(zip(src.tolist(), dst.tolist()))
    bidirectional = sum(1 for s, d in zip(src.tolist(), dst.tolist()) 
                       if (d, s) in edge_set) // 2
    print(f"  Bidirectional:   {bidirectional:,}")
    print(f"  One-way:         {g.num_edges() - 2*bidirectional:,}")
```

---

## 5. Graph Representation in DGL

### 5.1 Edge Type Encoding

Two approaches for handling mixed co-purchase/co-view edges:

**Approach A: Edge feature tensor (Recommended)**
```python
# Single homogeneous graph, edge type as data
g = dgl.graph((src, dst), num_nodes=N)
g.edata['type'] = torch.cat([torch.zeros(n_cp), torch.ones(n_cv)]).long()

# In model: filter edges by type during message passing
cp_eid = (g.edata['type'] == 0).nonzero(as_tuple=True)[0]
cv_eid = (g.edata['type'] == 1).nonzero(as_tuple=True)[0]
```

**Approach B: Heterogeneous graph**
```python
# Separate edge types as different relations
graph_data = {
    ('product', 'cp', 'product'): (cp_src, cp_dst),
    ('product', 'cv', 'product'): (cv_src, cv_dst),
}
g = dgl.heterograph(graph_data, num_nodes_dict={'product': N})

# More complex model code, but DGL handles type routing automatically
```

**Recommendation: Approach A** — simpler code, better performance for mini-batch sampling, and matches the paper's mathematical formulation more directly.

### 5.2 Graph Serialization

```python
# Save graph for reuse (avoids rebuilding every session)
dgl.save_graphs('/kaggle/working/product_graph.bin', [g])

# Load
g_list, _ = dgl.load_graphs('/kaggle/working/product_graph.bin')
g = g_list[0]
```

---

## 6. Data Splitting Strategy

### 6.1 Edge-Based Split (Main Training)

```python
def split_edges_by_type(g, train_ratio=0.75, val_ratio=0.05):
    """
    Split all edges into train/val/test with no overlap.
    Maintains ratio within each edge type.
    """
    cp_mask = g.edata['type'] == 0
    cv_mask = g.edata['type'] == 1
    
    # Split co-purchase edges
    cp_eids = cp_mask.nonzero(as_tuple=True)[0]
    cp_perm = torch.randperm(len(cp_eids))
    cp_train_end = int(len(cp_eids) * train_ratio)
    cp_val_end = cp_train_end + int(len(cp_eids) * val_ratio)
    
    train_cp = cp_eids[cp_perm[:cp_train_end]]
    val_cp = cp_eids[cp_perm[cp_train_end:cp_val_end]]
    test_cp = cp_eids[cp_perm[cp_val_end:]]
    
    # Split co-view edges similarly
    cv_eids = cv_mask.nonzero(as_tuple=True)[0]
    cv_perm = torch.randperm(len(cv_eids))
    cv_train_end = int(len(cv_eids) * train_ratio)
    cv_val_end = cv_train_end + int(len(cv_eids) * val_ratio)
    
    train_cv = cv_eids[cv_perm[:cv_train_end]]
    val_cv = cv_eids[cv_perm[cv_train_end:cv_val_end]]
    test_cv = cv_eids[cv_perm[cv_val_end:]]
    
    return {
        'train_cp': train_cp, 'train_cv': train_cv,
        'val_cp': val_cp, 'val_cv': val_cv,
        'test_cp': test_cp, 'test_cv': test_cv,
    }
```

### 6.2 Node-Based Split (Cold-Start)

```python
def split_nodes_cold_start(g, holdout_ratio=0.20):
    """
    Hold out 20% of nodes entirely for cold-start evaluation.
    These nodes' edges are removed from training graph.
    """
    all_nodes = torch.arange(g.num_nodes())
    perm = torch.randperm(g.num_nodes())
    
    n_holdout = int(g.num_nodes() * holdout_ratio)
    warm_nodes = perm[n_holdout:]
    cold_nodes = perm[:n_holdout]
    
    # Remove edges involving cold nodes for training
    # (Implementation depends on how graph is structured)
    
    return warm_nodes, cold_nodes
```

### 6.3 Training/Inference Graph Construction

```python
def create_train_graph(g, train_cp_eids, train_cv_eids):
    """Create training graph with only training edges."""
    train_eids = torch.cat([train_cp_eids, train_cv_eids])
    train_g = g.edge_subgraph(train_eids, relabel_nodes=False)
    return train_g
```

---

## 7. Negative Sampling Pipeline

### 7.1 Training Negatives

```python
class NegativeSampler:
    """Generates negative samples for training."""
    
    def __init__(self, num_nodes, num_neg=5, device='cuda'):
        self.num_nodes = num_nodes
        self.num_neg = num_neg
        self.device = device
    
    def sample(self, num_positives):
        """Uniform random negative sampling."""
        return torch.randint(
            0, self.num_nodes,
            (num_positives, self.num_neg),
            device=self.device
        )
```

### 7.2 Evaluation Negatives (for AUC)

```python
def generate_eval_negatives(g, pos_edges, num_neg_ratio=1):
    """
    Generate negative edges for link prediction evaluation.
    Ensures negatives are NOT in the positive edge set.
    """
    pos_set = set(zip(pos_edges[0].tolist(), pos_edges[1].tolist()))
    num_nodes = g.num_nodes()
    num_neg = len(pos_edges[0]) * num_neg_ratio
    
    neg_src, neg_dst = [], []
    while len(neg_src) < num_neg:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u != v and (u, v) not in pos_set:
            neg_src.append(u)
            neg_dst.append(v)
            pos_set.add((u, v))  # Avoid duplicates
    
    return torch.tensor([neg_src, neg_dst])
```

---

## 8. Data Augmentation

### 8.1 Edge Reversal for Asymmetry Training

```python
def find_one_way_edges(g):
    """Identify one-way co-purchase edges for asymmetry loss."""
    src, dst = g.edges()
    cp_mask = g.edata['type'] == 0
    cp_src = src[cp_mask]
    cp_dst = dst[cp_mask]
    
    edge_set = set(zip(cp_src.tolist(), cp_dst.tolist()))
    
    one_way_mask = torch.tensor([
        (dst[i].item(), src[i].item()) not in edge_set
        for i in range(len(cp_src))
    ])
    
    one_way_u = cp_src[one_way_mask]
    one_way_v = cp_dst[one_way_mask]
    
    return one_way_u, one_way_v
```

### 8.2 Transitive Edge Generation (for selection bias test)

```python
def generate_transitive_test_edges(g):
    """
    Find paths: u --cp→ w --cv→ v, then add (u, v) as test edge.
    These represent products that SHOULD be recommended together
    based on transitive reasoning but aren't in the training data.
    """
    cp_mask = g.edata['type'] == 0
    cv_mask = g.edata['type'] == 1
    
    cp_edges = g.find_edges(torch.where(cp_mask)[0])
    cv_edges = g.find_edges(torch.where(cv_mask)[0])
    
    # Adjacency list for fast lookup
    cp_adj = defaultdict(set)
    for u, v in zip(cp_edges[0].tolist(), cp_edges[1].tolist()):
        cp_adj[u].add(v)
    
    cv_adj = defaultdict(set)
    for u, v in zip(cv_edges[0].tolist(), cv_edges[1].tolist()):
        cv_adj[u].add(v)
    
    transitive_edges = []
    for u, w in zip(cp_edges[0].tolist(), cp_edges[1].tolist()):
        for v in cv_adj.get(w, set()):
            if v != u and v not in cp_adj.get(u, set()):
                transitive_edges.append([u, v])
    
    return np.array(transitive_edges).T
```

---

## 9. Preprocessing Checklist

Before training, verify:

```
□ Remove products with < 2 co-purchase edges (insufficient signal)
□ Handle missing text: use category + title only, or drop
□ Filter non-ASCII/non-English if using English text encoder
□ Log-normalize price values (if using price as feature)
□ Remove duplicate edges (same (u,v) appearing twice)
□ Ensure graph is weakly connected (remove isolated components < 5 nodes)
□ Verify no test edges appear in training set
□ Verify ~75% of co-purchase edges are one-way (asymmetric)
□ Convert features to FP16 for storage if memory-constrained
□ Create node ID mapping (original IDs → 0..N-1 contiguous indices)
```

```python
def preprocess_graph(g, min_degree=2):
    """Clean and validate graph before training."""
    # Remove isolated and low-degree nodes
    degrees = g.in_degrees() + g.out_degrees()
    keep_mask = degrees >= min_degree
    
    if keep_mask.sum() < g.num_nodes():
        g = g.subgraph(torch.where(keep_mask)[0])
        print(f"Removed {g.num_nodes() - keep_mask.sum().item()} low-degree nodes")
    
    # Remove duplicate edges
    g = dgl.remove_self_loop(g)
    g = dgl.to_simple(g)  # Removes duplicate edges
    
    # Verify
    assert g.num_nodes() > 0, "Empty graph after preprocessing!"
    assert g.ndata['feat'].shape[0] == g.num_nodes(), "Feature count mismatch!"
    
    return g
```

---

## 10. Kaggle-Specific Data Pipeline

### 10.1 Offline Preprocessing (Run Locally or on Kaggle CPU)

```python
# === OFFLINE PREPROCESSING SCRIPT ===
# Run this once and upload output to Kaggle as a dataset

# 1. Load raw data
product_df = pd.read_csv('amazon_products.csv')
# cp_edges = load_co_purchase_data()

# 2. Generate features
features = encode_product_texts(product_df)
np.savez_compressed('features.npz', features=features.astype(np.float16))

# 3. Build graph
g = build_product_graph(product_df, cp_edges, cv_edges)

# 4. Save
dgl.save_graphs('product_graph.bin', [g])
product_df[['product_id', 'title', 'category']].to_csv('product_meta.csv', index=False)

# 5. Create Kaggle dataset from these files
# Upload to Kaggle: https://www.kaggle.com/datasets
```

### 10.2 Kaggle Notebook: Quick Load

```python
# === IN KAGGLE NOTEBOOK ===
# This is all the data loading needed after preprocessing

import kagglehub
import dgl
import numpy as np

# Option A: Load from attached Kaggle dataset
DATASET_PATH = '/kaggle/input/daemon-product-graph/'

g_list, _ = dgl.load_graphs(f'{DATASET_PATH}/product_graph.bin')
g = g_list[0]

features_npz = np.load(f'{DATASET_PATH}/features.npz')
g.ndata['feat'] = torch.from_numpy(features_npz['features'].astype(np.float32))

meta_df = pd.read_csv(f'{DATASET_PATH}/product_meta.csv')

print(f"Loaded graph: {g.num_nodes():,} nodes, {g.num_edges():,} edges")

# Option B: Generate synthetic data on the fly (for testing)
# from data_pipeline import generate_synthetic_graph
# cp, cv, features, cats = generate_synthetic_graph(num_products=50000)
```

---

## 11. Data Validation

### 11.1 Assertions at Load Time

```python
def validate_graph(g):
    """Comprehensive graph validation. Raises AssertionError on issues."""
    N = g.num_nodes()
    E = g.num_edges()
    
    # Node checks
    assert N > 0, "Graph has 0 nodes"
    assert g.ndata['feat'] is not None, "No node features"
    assert g.ndata['feat'].shape[0] == N, f"Feature rows ({g.ndata['feat'].shape[0]}) != nodes ({N})"
    assert not torch.isnan(g.ndata['feat']).any(), "NaN in node features"
    assert not torch.isinf(g.ndata['feat']).any(), "Inf in node features"
    
    # Edge checks
    assert E > 0, "Graph has 0 edges"
    assert g.edata['type'] is not None, "No edge types"
    assert g.edata['type'].shape[0] == E, "Edge type count mismatch"
    assert set(g.edata['type'].unique().tolist()).issubset({0, 1}), \
        f"Unexpected edge types: {g.edata['type'].unique().tolist()}"
    
    # Node ID continuity
    assert g.num_nodes() == N, "Node count mismatch"
    assert g.nodes().min() == 0, "Node IDs don't start at 0"
    assert g.nodes().max() == N - 1, "Node IDs not contiguous"
    
    # Edge validity
    src, dst = g.edges()
    assert src.min() >= 0 and src.max() < N, "Source node out of bounds"
    assert dst.min() >= 0 and dst.max() < N, "Destination node out of bounds"
    
    # Direction check
    edge_set = set(zip(src.tolist(), dst.tolist()))
    one_way = sum(1 for s, d in zip(src.tolist(), dst.tolist()) if (d, s) not in edge_set)
    one_way_pct = 100 * one_way / E
    assert one_way_pct > 10, f"Only {one_way_pct:.1f}% edges are directed — data may be problematic"
    
    print(f"✓ Graph validated: {N} nodes, {E} edges, {one_way_pct:.1f}% directed")
    return True
```

### 11.2 Split Validation

```python
def validate_splits(train_eids, val_eids, test_eids):
    """Ensure no edge overlap between splits."""
    train_set = set(train_eids.tolist())
    val_set = set(val_eids.tolist())
    test_set = set(test_eids.tolist())
    
    assert len(train_set & val_set) == 0, "Train-Val overlap!"
    assert len(train_set & test_set) == 0, "Train-Test overlap!"
    assert len(val_set & test_set) == 0, "Val-Test overlap!"
    
    total = len(train_set) + len(val_set) + len(test_set)
    print(f"✓ Splits validated: {total:,} total edges, no overlap")
```

---

## 12. Quick Reference: Dataset Size Guidelines

| Phase | Nodes | Edges | Where | Purpose |
|-------|-------|-------|-------|---------|
| **Dev** | 1K–5K | 5K–50K | Synthetic (in-notebook) | Code development + unit tests |
| **Tuning** | 10K–50K | 50K–500K | Synthetic or small real dataset | Hyperparameter tuning |
| **Main run** | 100K–500K | 1M–5M | Real product data | Final training + evaluation |
| **Ambitious** | 1M+ | 10M+ | Real large-scale data | Push T4 limits |

---

## References

- Amazon Product Reviews (McAuley Lab): https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
- Kaggle Amazon Dataset: https://www.kaggle.com/datasets/piyushjain16/amazon-product-data
- sentence-transformers: https://www.sbert.net/
- DGL Graph Construction: https://docs.dgl.ai/guide/graph-graphs.html
