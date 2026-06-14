"""
data_pipeline.py — Data loading, processing, and graph construction for DAEMON-Kaggle.

Project: DAEMON — Related Product Recommendation via GNNs on Directed Graphs
Paper: Virinchi et al., ECML-PKDD 2022
Target graphs: 100K–500K nodes (products), 1M–5M edges (co-purchase + co-view)

This module handles ALL data loading, processing, and graph construction:
  - Synthetic data generation with realistic graph statistics
  - DGL graph construction from co-purchase and co-view edge pairs
  - Feature encoding from product metadata (text → embeddings)
  - Edge-based and node-based data splitting
  - Negative sampling for link prediction training
  - Graph validation and preprocessing
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# ---------------------------------------------------------------------------
# DGL is imported at point of use so that documentation / import errors are
# localized.  We also provide a helper that raises a clear message when the
# library is absent.
# ---------------------------------------------------------------------------

_HAS_DGL: bool = False
try:
    import dgl  # noqa: F401

    _HAS_DGL = True
except ImportError:
    _HAS_DGL = False


def _require_dgl() -> None:
    """Raise ``ImportError`` if DGL is not installed."""
    if not _HAS_DGL:
        raise ImportError(
            "DGL (Deep Graph Library) is required.  Install it with:\n"
            "  pip install dgl   # CPU version\n"
            "  pip install dgl-cu118  # CUDA 11.8 version\n"
            "See https://www.dgl.ai/ for details."
        )


# ===================================================================
# 1.  TEXT FEATURE ENCODING
# ===================================================================


def encode_product_texts(
    product_df: "pd.DataFrame",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> np.ndarray:
    """Encode product titles (+ descriptions + category) into dense vectors.

    Uses SentenceTransformers (``all-MiniLM-L6-v2`` → 384-d embeddings).
    Falls back to random features if the library is unavailable.

    Parameters
    ----------
    product_df:
        DataFrame with at least a ``'title'`` column.  ``'description'``
        and ``'category'`` columns are used when present.
    model_name:
        SentenceTransformer model identifier.
    batch_size:
        Batch size for encoding.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``[N, 384]``.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        warnings.warn(
            "sentence_transformers not installed — returning random features.  "
            "Install with: pip install sentence-transformers"
        )
        n = len(product_df)
        rng = np.random.default_rng(42)
        return rng.standard_normal((n, 384), dtype=np.float32)

    model = SentenceTransformer(model_name)

    # Build text field
    texts = product_df["title"].fillna("")
    if "description" in product_df.columns:
        texts = texts + " " + product_df["description"].fillna("")
    if "category" in product_df.columns:
        texts = texts + " [CATEGORY: " + product_df["category"].fillna("") + "]"

    embeddings = model.encode(
        texts.tolist(), batch_size=batch_size, show_progress_bar=True
    )
    return np.asarray(embeddings, dtype=np.float32)


def encode_lightweight(
    texts: List[str], feature_dim: int = 128
) -> np.ndarray:
    """TF-IDF + TruncatedSVD encoding when SentenceTransformers won't fit.

    Parameters
    ----------
    texts:
        Raw text strings for each product.
    feature_dim:
        Target embedding dimension (after SVD).

    Returns
    -------
    np.ndarray
        Float32 array of shape ``[N, feature_dim]``.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    pipeline: Pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000, stop_words="english", ngram_range=(1, 2)
        )),
        ("svd", TruncatedSVD(n_components=feature_dim, random_state=42)),
    ])
    embeddings = pipeline.fit_transform(texts)
    return np.asarray(embeddings, dtype=np.float32)


def encode_numerical_features(
    product_df: "pd.DataFrame",
) -> Optional[np.ndarray]:
    """Encode ``price`` (log-normalised) and ``rating`` (scaled to [-1, 1]).

    Parameters
    ----------
    product_df:
        DataFrame that may contain ``'price'`` and / or ``'rating'`` columns.

    Returns
    -------
    np.ndarray or None
        Float32 array of shape ``[N, K]``, or ``None`` if no numeric columns
        exist.
    """
    features: List[np.ndarray] = []

    if "price" in product_df.columns:
        log_price = np.log1p(product_df["price"].fillna(product_df["price"].median()))
        log_price = (log_price - log_price.mean()) / (log_price.std() + 1e-8)
        features.append(np.asarray(log_price, dtype=np.float32).reshape(-1, 1))

    if "rating" in product_df.columns:
        rating = (product_df["rating"].fillna(3.0) - 3.0) / 2.0
        features.append(np.asarray(rating, dtype=np.float32).reshape(-1, 1))

    if not features:
        return None
    return np.concatenate(features, axis=1).astype(np.float32)


def fuse_features(
    text_embeddings: np.ndarray,
    num_features: Optional[np.ndarray] = None,
    method: str = "concat",
) -> np.ndarray:
    """Fuse text and numerical features into a single node feature matrix.

    Parameters
    ----------
    text_embeddings:
        Dense text embeddings  ``[N, d_text]``.
    num_features:
        Optional numerical features ``[N, d_num]``.
    method:
        Fusion method — only ``'concat'`` is currently supported.

    Returns
    -------
    np.ndarray
        Fused feature matrix ``[N, d_out]``.
    """
    if num_features is None:
        return text_embeddings
    if method == "concat":
        return np.concatenate([text_embeddings, num_features], axis=1).astype(
            np.float32
        )
    raise ValueError(f"Unknown fusion method: {method!r}")


# ===================================================================
# 2.  SYNTHETIC DATA GENERATOR
# ===================================================================


def _power_law_degrees(
    n: int, avg_degree: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample a power-law degree sequence with target mean.

    Uses the Zipf distribution (a = 2.2) clipped to ``[1, n-1]``, then
    rescaled to match *avg_degree*.  The result is guaranteed to be a
    valid degree sequence (sum even, each entry ≤ n-1).

    Returns
    -------
    np.ndarray[int64]
        Length-*n* degree sequence.
    """
    # Zipf with a=2.2 gives a realistic heavy tail.
    a = 2.2
    raw = rng.zipf(a, size=n).astype(np.float64)
    # Clip and rescale to match target average.
    raw = np.clip(raw, 1, n - 1)
    degrees = (raw / raw.mean() * avg_degree).round().astype(np.int64)
    degrees = np.clip(degrees, 1, n - 1)
    # Ensure even sum (required for a valid simple directed graph).
    if degrees.sum() % 2 != 0:
        degrees[0] += 1
        degrees = np.clip(degrees, 1, n - 1)
    return degrees


def generate_synthetic_graph(
    num_products: int = 10000,
    feature_dim: int = 384,
    avg_cp_degree: int = 5,
    avg_cv_degree: int = 8,
    asymmetry_ratio: float = 0.75,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic product graph for testing DAEMON.

    Products are assigned to categories (cluster centres) and features are
    sampled as ``centre + noise``, simulating realistic product categories.
    Co-purchase edges are primarily *within-category* and asymmetric
    (one-way with probability *asymmetry_ratio*).  Co-view edges have a
    higher degree and are mostly bidirectional, with some cross-category
    connectivity.

    Parameters
    ----------
    num_products:
        Number of product nodes.
    feature_dim:
        Dimensionality of product features.
    avg_cp_degree:
        Average out-degree of co-purchase edges.
    avg_cv_degree:
        Average out-degree of co-view edges.
    asymmetry_ratio:
        Fraction of co-purchase edges that are one-way (~75% matches the
        paper's empirical finding).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    edge_index_cp:
        Integer array of shape ``[2, num_cp_edges]`` — co-purchase edges.
    edge_index_cv:
        Integer array of shape ``[2, num_cv_edges]`` — co-view edges.
    features:
        Float32 array of shape ``[num_products, feature_dim]``.
    categories:
        Integer array of shape ``[num_products]`` — category label per node.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 2a. Product features — cluster-like to simulate categories
    # ------------------------------------------------------------------
    num_categories = max(1, num_products // 500)  # ~500 products / category
    cluster_centers: np.ndarray = rng.standard_normal(
        (num_categories, feature_dim)
    ).astype(np.float32)

    categories = rng.integers(0, num_categories, size=num_products)
    features = (
        cluster_centers[categories]
        + 0.3 * rng.standard_normal((num_products, feature_dim)).astype(np.float32)
    )

    # Normalise features to unit length (cosine-similarity friendly).
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.clip(norms, 1e-8, None)

    # ------------------------------------------------------------------
    # 2b. Per-category product lists (for fast indexing)
    # ------------------------------------------------------------------
    cat_to_products: List[np.ndarray] = [
        np.where(categories == c)[0] for c in range(num_categories)
    ]

    # Pre-compute power-law degree sequences.
    cp_deg: np.ndarray = _power_law_degrees(num_products, avg_cp_degree, rng)
    cv_deg: np.ndarray = _power_law_degrees(num_products, avg_cv_degree, rng)

    # ------------------------------------------------------------------
    # 2c. Co-purchase edges (within-category, asymmetric)
    # ------------------------------------------------------------------
    cp_src: List[int] = []
    cp_dst: List[int] = []

    for u in range(num_products):
        cat = categories[u]
        neighbours = cat_to_products[cat]
        # Remove self from candidates.
        candidates = neighbours[neighbours != u]
        if len(candidates) == 0:
            continue

        deg = int(cp_deg[u])
        if deg == 0:
            continue

        # Sample distinct targets with replacement if degree > candidates.
        k = min(deg, len(candidates))
        targets: np.ndarray = rng.choice(candidates, size=k, replace=False)

        for v in targets.tolist():
            cp_src.append(u)
            cp_dst.append(v)
            # Asymmetry: with probability asymmetry_ratio, the reverse edge
            # is NOT added (making this a one-way edge).
            if rng.random() > asymmetry_ratio:
                cp_src.append(v)
                cp_dst.append(u)

    # ------------------------------------------------------------------
    # 2d. Co-view edges (bidirectional + cross-category)
    # ------------------------------------------------------------------
    cv_src: List[int] = []
    cv_dst: List[int] = []

    for u in range(num_products):
        cat = categories[u]
        deg = int(cv_deg[u])
        if deg == 0:
            continue

        # ~60% of co-view connections are within-category.
        n_within = max(0, deg * 3 // 5)
        neighbours = cat_to_products[cat]
        candidates_within = neighbours[neighbours != u]

        if len(candidates_within) > 0:
            k = min(n_within, len(candidates_within))
            within_targets = rng.choice(candidates_within, size=k, replace=False)
            for v in within_targets.tolist():
                # Co-view is bidirectional.
                cv_src.append(u)
                cv_dst.append(v)
                cv_src.append(v)
                cv_dst.append(u)

        # ~40% cross-category.
        n_cross = deg - n_within
        other_cats = [c for c in range(num_categories) if c != cat]
        if other_cats and n_cross > 0:
            # Pick a random other category and sample from it.
            other_cat = rng.choice(other_cats)
            other_neighbours = cat_to_products[other_cat]
            if len(other_neighbours) > 0:
                k = min(n_cross, len(other_neighbours))
                cross_targets = rng.choice(
                    other_neighbours, size=k, replace=False
                )
                for v in cross_targets.tolist():
                    cv_src.append(u)
                    cv_dst.append(v)
                    # Cross-category co-view edges are also bidirectional.
                    cv_src.append(v)
                    cv_dst.append(u)

    # ------------------------------------------------------------------
    # 2e. Convert to numpy arrays
    # ------------------------------------------------------------------
    edge_index_cp = (
        np.array([cp_src, cp_dst], dtype=np.int64)
        if cp_src
        else np.zeros((2, 0), dtype=np.int64)
    )
    edge_index_cv = (
        np.array([cv_src, cv_dst], dtype=np.int64)
        if cv_src
        else np.zeros((2, 0), dtype=np.int64)
    )

    asymmetry = estimate_asymmetry(edge_index_cp)

    print(f"Generated: {num_products} nodes, "
          f"{edge_index_cp.shape[1]:,} CP edges, "
          f"{edge_index_cv.shape[1]:,} CV edges")
    print(f"  Co-purchase asymmetry: {asymmetry:.1f}% one-way edges")
    print(f"  Features shape: {features.shape}")

    return edge_index_cp, edge_index_cv, features, categories


# ===================================================================
# 3.  ASYMMETRY ESTIMATION
# ===================================================================


def estimate_asymmetry(edges: np.ndarray) -> float:
    """Estimate the percentage of one-way (directed) edges.

    Parameters
    ----------
    edges:
        Integer array of shape ``[2, E]``.

    Returns
    -------
    float
        Percentage of edges whose reverse does **not** appear in the set.
    """
    if edges.shape[1] == 0:
        return 0.0
    edge_set = set(zip(edges[0].tolist(), edges[1].tolist()))
    one_way = sum(
        1 for u, v in zip(edges[0].tolist(), edges[1].tolist())
        if (v, u) not in edge_set
    )
    return 100.0 * one_way / edges.shape[1]


def estimate_directed_pct(g: "dgl.DGLGraph") -> float:
    """Percentage of directed (one-way) edges in a DGL graph.

    Parameters
    ----------
    g:
        DGL graph (may contain both one-way and bidirectional edges).

    Returns
    -------
    float
        Percentage of edges whose reverse edge is **not** present.
    """
    src, dst = g.edges()
    edge_set = set(zip(src.tolist(), dst.tolist()))
    one_way = sum(
        1 for s, d in zip(src.tolist(), dst.tolist()) if (d, s) not in edge_set
    )
    return 100.0 * one_way / g.num_edges()


# ===================================================================
# 4.  GRAPH CONSTRUCTION
# ===================================================================


def build_product_graph(
    product_df: "pd.DataFrame",
    cp_edges: np.ndarray,
    cv_edges: np.ndarray,
    feature_dim: int = 384,
    features: Optional[np.ndarray] = None,
) -> Tuple["dgl.DGLGraph", np.ndarray]:
    """Build a DGL directed graph from co-purchase and co-view edges.

    The graph is a single homogeneous graph with edge types stored as
    ``g.edata['type']``:
        - ``0`` = co-purchase
        - ``1`` = co-view

    Node features are stored as ``g.ndata['feat']``.

    Parameters
    ----------
    product_df:
        DataFrame with product metadata.  If *features* is ``None``,
        the function attempts to encode text features.
    cp_edges:
        Integer array ``[2, E_cp]`` — co-purchase edge pairs.
    cv_edges:
        Integer array ``[2, E_cv]`` — co-view edge pairs.
    feature_dim:
        Target feature dimension (used only when encoding fallback text
        features).
    features:
        Optional pre-computed feature matrix ``[N, feature_dim]``.
        If provided, text encoding is skipped.

    Returns
    -------
    g:
        DGL graph with edge types and node features.
    features_np:
        Feature matrix used (for reference).
    """
    _require_dgl()
    import dgl  # pylint: disable=import-outside-toplevel

    N = len(product_df)

    # ------------------------------------------------------------------
    # 4a. Obtain node features
    # ------------------------------------------------------------------
    if features is not None:
        features_np = np.asarray(features, dtype=np.float32)
    else:
        print("Encoding product features via sentence-transformers …")
        features_np = encode_product_texts(product_df)

    if features_np.shape[1] != feature_dim:
        warnings.warn(
            f"Feature dimension {features_np.shape[1]} != requested "
            f"{feature_dim}.  Using actual dimension."
        )

    # ------------------------------------------------------------------
    # 4b. Combine edges with type annotations
    # ------------------------------------------------------------------
    all_src: np.ndarray = np.concatenate([cp_edges[0], cv_edges[0]])
    all_dst: np.ndarray = np.concatenate([cp_edges[1], cv_edges[1]])
    edge_types: np.ndarray = np.concatenate([
        np.zeros(cp_edges.shape[1], dtype=np.int64),
        np.ones(cv_edges.shape[1], dtype=np.int64),
    ])

    # ------------------------------------------------------------------
    # 4c. Build DGL graph
    # ------------------------------------------------------------------
    g = dgl.graph((all_src, all_dst), num_nodes=N)
    g.edata["type"] = torch.as_tensor(edge_types, dtype=torch.long)
    g.ndata["feat"] = torch.as_tensor(features_np, dtype=torch.float32)

    directed_pct = estimate_directed_pct(g)

    print(f"Graph built: {N:,} nodes, {g.num_edges():,} edges")
    print(f"  Co-purchase: {cp_edges.shape[1]:,} edges")
    print(f"  Co-view:     {cv_edges.shape[1]:,} edges")
    print(f"  Directed:    {directed_pct:.1f}%")

    return g, features_np


# ===================================================================
# 5.  GRAPH STATISTICS
# ===================================================================


def print_graph_stats(g: "dgl.DGLGraph") -> None:
    """Print detailed graph statistics to stdout.

    Parameters
    ----------
    g:
        DGL graph with ``'type'`` in ``edata`` and ``'feat'`` in ``ndata``.
    """
    in_degrees = g.in_degrees().float()
    out_degrees = g.out_degrees().float()

    cp_mask = g.edata["type"] == 0
    cv_mask = g.edata["type"] == 1

    # Edge direction analysis
    src, dst = g.edges()
    edge_set = set(zip(src.tolist(), dst.tolist()))
    bidirectional = sum(
        1 for s, d in zip(src.tolist(), dst.tolist()) if (d, s) in edge_set
    ) // 2
    one_way = g.num_edges() - 2 * bidirectional

    print("=" * 52)
    print("  GRAPH STATISTICS")
    print("=" * 52)
    print(f"  Nodes:           {g.num_nodes():>12,}")
    print(f"  Total edges:     {g.num_edges():>12,}")
    print(f"    Co-purchase:   {int(cp_mask.sum()):>12,}")
    print(f"    Co-view:       {int(cv_mask.sum()):>12,}")
    print(f"  ─────────────────────────────────────")
    print(f"  Avg degree:      {float(in_degrees.mean()):>12.1f}")
    print(f"  Median degree:   {float(in_degrees.median()):>12.1f}")
    print(f"  Max degree:      {int(in_degrees.max()):>12,}")
    print(f"  Min degree:      {int(in_degrees.min()):>12,}")
    print(f"  Isolated nodes:  {int((in_degrees == 0).sum()):>12,}")
    print(f"  ─────────────────────────────────────")
    print(f"  Bidirectional:   {bidirectional:>12,}")
    print(f"  One-way:         {one_way:>12,}")
    print(f"  Directed:        {100.0 * one_way / g.num_edges():>11.1f}%")
    print(f"  ─────────────────────────────────────")
    print(f"  Feature dim:     {g.ndata['feat'].shape[1]:>12}")
    print(f"  Feature NaN:     {bool(torch.isnan(g.ndata['feat']).any()):>12}")
    print(f"  Feature Inf:     {bool(torch.isinf(g.ndata['feat']).any()):>12}")
    print("=" * 52)


# ===================================================================
# 6.  DATA SPLITTING
# ===================================================================


def split_edges_by_type(
    g: "dgl.DGLGraph",
    train_ratio: float = 0.75,
    val_ratio: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Split edges into train / val / test sets, stratified by edge type.

    The split is edge-ID-based (no overlap between sets).  The remaining
    20% (by default) go to the test set.

    Parameters
    ----------
    g:
        DGL graph with ``'type'`` in edge data.
    train_ratio:
        Fraction of edges used for training.
    val_ratio:
        Fraction of edges used for validation.

    Returns
    -------
    dict
        Keys: ``'train_cp', 'train_cv', 'val_cp', 'val_cv',
        'test_cp', 'test_cv'`` — each a 1-D ``torch.Tensor`` of edge IDs.
    """
    cp_mask = g.edata["type"] == 0
    cv_mask = g.edata["type"] == 1

    def _split_mask(mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        eids = mask.nonzero(as_tuple=True)[0]
        perm = eids[torch.randperm(len(eids))]
        n_train = int(len(perm) * train_ratio)
        n_val = int(len(perm) * val_ratio)
        return (
            perm[:n_train],
            perm[n_train : n_train + n_val],
            perm[n_train + n_val :],
        )

    train_cp, val_cp, test_cp = _split_mask(cp_mask)
    train_cv, val_cv, test_cv = _split_mask(cv_mask)

    return {
        "train_cp": train_cp,
        "train_cv": train_cv,
        "val_cp": val_cp,
        "val_cv": val_cv,
        "test_cp": test_cp,
        "test_cv": test_cv,
    }


def validate_splits(
    train_eids: torch.Tensor,
    val_eids: torch.Tensor,
    test_eids: torch.Tensor,
) -> bool:
    """Assert that the three edge-ID sets are pairwise disjoint.

    Parameters
    ----------
    train_eids:
        1-D tensor of training edge IDs.
    val_eids:
        1-D tensor of validation edge IDs.
    test_eids:
        1-D tensor of test edge IDs.

    Returns
    -------
    bool
        ``True`` if all checks pass.

    Raises
    ------
    AssertionError
        If any overlap is detected.
    """
    train_set: set = set(train_eids.tolist())
    val_set: set = set(val_eids.tolist())
    test_set: set = set(test_eids.tolist())

    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set

    assert not overlap_tv, f"Train-Val overlap: {len(overlap_tv)} edges"
    assert not overlap_tt, f"Train-Test overlap: {len(overlap_tt)} edges"
    assert not overlap_vt, f"Val-Test overlap: {len(overlap_vt)} edges"

    total = len(train_set) + len(val_set) + len(test_set)
    print(f"✓ Splits validated: {total:,} total edges, no overlap.")
    return True


def create_train_graph(
    g: "dgl.DGLGraph",
    train_cp_eids: torch.Tensor,
    train_cv_eids: torch.Tensor,
) -> "dgl.DGLGraph":
    """Create a training subgraph containing only the training edges.

    Node IDs are preserved (``relabel_nodes=False``).

    Parameters
    ----------
    g:
        Full graph.
    train_cp_eids:
        Edge IDs of co-purchase training edges.
    train_cv_eids:
        Edge IDs of co-view training edges.

    Returns
    -------
    dgl.DGLGraph
        Subgraph containing only the specified edges.
    """
    _require_dgl()
    import dgl  # pylint: disable=import-outside-toplevel

    train_eids = torch.cat([train_cp_eids, train_cv_eids])
    train_g = dgl.edge_subgraph(g, train_eids, relabel_nodes=False)
    return train_g


# ===================================================================
# 7.  COLD-START SPLIT
# ===================================================================


def split_nodes_cold_start(
    g: "dgl.DGLGraph",
    holdout_ratio: float = 0.20,
) -> Tuple[torch.Tensor, torch.Tensor, "dgl.DGLGraph"]:
    """Hold out a fraction of nodes for cold-start evaluation.

    The returned training graph has **no edges** incident to the cold
    nodes.  Cold nodes themselves are removed from the training graph
    via ``dgl.node_subgraph``.

    Parameters
    ----------
    g:
        Full DGL graph.
    holdout_ratio:
        Fraction of nodes to hold out (default 20%).

    Returns
    -------
    warm_nodes:
        1-D tensor of warm (training) node IDs.
    cold_nodes:
        1-D tensor of cold (held-out) node IDs.
    cold_g:
        Subgraph containing **only** cold nodes (for evaluation).
    """
    _require_dgl()
    import dgl  # pylint: disable=import-outside-toplevel

    all_nodes = torch.arange(g.num_nodes())
    perm = all_nodes[torch.randperm(g.num_nodes())]

    n_holdout = int(g.num_nodes() * holdout_ratio)
    warm_nodes = perm[n_holdout:]
    cold_nodes = perm[:n_holdout]

    # Build training graph: subgraph containing only warm nodes — this
    # automatically drops all edges incident to cold nodes.
    train_g = dgl.node_subgraph(g, warm_nodes)

    # Build a separate cold graph (if needed for evaluation).
    cold_g = dgl.node_subgraph(g, cold_nodes)

    print(f"Cold-start split: {len(warm_nodes):,} warm, {len(cold_nodes):,} cold")
    print(f"  Training graph: {train_g.num_nodes():,} nodes, "
          f"{train_g.num_edges():,} edges")

    return warm_nodes, cold_nodes, train_g


# ===================================================================
# 8.  ONE-WAY EDGES (ASYMMETRY LOSS)
# ===================================================================


def find_one_way_edges(
    g: "dgl.DGLGraph",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find one-way co-purchase edges for asymmetry loss computation.

    An edge ``(u, v)`` of type ``0`` (co-purchase) is considered one-way
    if the reverse edge ``(v, u)`` does **not** exist in the graph.

    Parameters
    ----------
    g:
        DGL graph with ``'type'`` in edge data.

    Returns
    -------
    one_way_u:
        Source nodes of one-way co-purchase edges.
    one_way_v:
        Destination nodes of one-way co-purchase edges.
    """
    src, dst = g.edges()
    cp_mask = g.edata["type"] == 0
    cp_src = src[cp_mask]
    cp_dst = dst[cp_mask]

    edge_set: set = set(zip(cp_src.tolist(), cp_dst.tolist()))

    one_way_mask = torch.tensor(
        [
            (dst_i.item(), src_i.item()) not in edge_set
            for src_i, dst_i in zip(cp_src, cp_dst)
        ],
        device=src.device,
    )

    one_way_u = cp_src[one_way_mask]
    one_way_v = cp_dst[one_way_mask]

    return one_way_u, one_way_v


# ===================================================================
# 9.  NEGATIVE SAMPLER
# ===================================================================


class NegativeSampler:
    """Uniform random negative sampler for link prediction training.

    Generates ``num_neg`` negative destination nodes per positive pair by
    sampling uniformly from all nodes.

    Parameters
    ----------
    num_nodes:
        Total number of nodes in the graph.
    num_neg:
        Number of negative samples per positive pair.
    device:
        Device for the output tensors.
    """

    def __init__(
        self,
        num_nodes: int,
        num_neg: int = 5,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.num_nodes = num_nodes
        self.num_neg = num_neg
        self.device = torch.device(device)

    def sample(self, num_positives: int) -> torch.Tensor:
        """Generate negative destination nodes.

        Parameters
        ----------
        num_positives:
            Number of positive pairs in the current batch.

        Returns
        -------
        torch.Tensor
            Long tensor of shape ``(num_positives, num_neg)`` containing
            uniformly-sampled negative node indices.
        """
        return torch.randint(
            0,
            self.num_nodes,
            (num_positives, self.num_neg),
            device=self.device,
        )


def generate_eval_negatives(
    g: "dgl.DGLGraph",
    pos_edges: torch.Tensor,
    num_neg_ratio: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Generate negative edges for link prediction evaluation (AUC).

    Ensures that generated negative edges are **not** in the positive
    edge set (and are not self-loops).

    Parameters
    ----------
    g:
        DGL graph (used only for ``num_nodes``).
    pos_edges:
        Tensor of shape ``[2, E_pos]`` — positive edge pairs.
    num_neg_ratio:
        Number of negatives per positive edge (default 1).
    rng:
        Optional ``numpy.random.Generator`` for reproducibility.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``[2, num_neg]`` with negative edge pairs.
    """
    if rng is None:
        rng = np.random.default_rng()

    pos_set: set = set(
        zip(pos_edges[0].tolist(), pos_edges[1].tolist())
    )
    num_nodes = g.num_nodes()
    num_neg = pos_edges.shape[1] * num_neg_ratio

    neg_src: List[int] = []
    neg_dst: List[int] = []

    while len(neg_src) < num_neg:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        if u != v and (u, v) not in pos_set:
            neg_src.append(u)
            neg_dst.append(v)
            pos_set.add((u, v))  # avoid duplicates

    return torch.tensor([neg_src, neg_dst], dtype=torch.long)


# ===================================================================
# 10. GRAPH VALIDATION
# ===================================================================


def validate_graph(g: "dgl.DGLGraph") -> bool:
    """Comprehensive graph validation.

    Checks:
    - Node count > 0
    - Node features present, matching node count, no NaN/Inf
    - Edge count > 0, edge types present
    - Node IDs contiguous starting at 0
    - All source / destination indices in bounds
    - At least 10% of edges are directed (heuristic)

    Parameters
    ----------
    g:
        DGL graph with ``'feat'`` in ``ndata`` and ``'type'`` in ``edata``.

    Returns
    -------
    bool
        ``True`` if all checks pass.

    Raises
    ------
    AssertionError
        On any validation failure.
    """
    N = g.num_nodes()
    E = g.num_edges()

    # Node checks
    assert N > 0, "Graph has 0 nodes"
    assert "feat" in g.ndata, "No node features ('feat' missing from ndata)"
    feat = g.ndata["feat"]
    assert feat.shape[0] == N, (
        f"Feature rows ({feat.shape[0]}) != nodes ({N})"
    )
    assert not torch.isnan(feat).any(), "NaN in node features"
    assert not torch.isinf(feat).any(), "Inf in node features"

    # Edge checks
    assert E > 0, "Graph has 0 edges"
    assert "type" in g.edata, "No edge types ('type' missing from edata)"
    assert g.edata["type"].shape[0] == E, (
        f"Edge type count ({g.edata['type'].shape[0]}) != edges ({E})"
    )
    unique_types = set(g.edata["type"].unique().tolist())
    assert unique_types.issubset({0, 1}), (
        f"Unexpected edge types: {unique_types}"
    )

    # Node ID continuity
    assert g.nodes().min() == 0, "Node IDs do not start at 0"
    assert g.nodes().max() == N - 1, "Node IDs are not contiguous"

    # Edge bounds
    src, dst = g.edges()
    assert src.min() >= 0 and src.max() < N, (
        f"Source node out of bounds [{src.min()}, {src.max()}] for [0, {N})"
    )
    assert dst.min() >= 0 and dst.max() < N, (
        f"Destination node out of bounds [{dst.min()}, {dst.max()}] for [0, {N})"
    )

    # Direction heuristic: at least 10% one-way edges (graphs with *no*
    # directionality are likely mis-configured).
    edge_set = set(zip(src.tolist(), dst.tolist()))
    one_way = sum(
        1 for s, d in zip(src.tolist(), dst.tolist()) if (d, s) not in edge_set
    )
    one_way_pct = 100.0 * one_way / E
    assert one_way_pct > 10, (
        f"Only {one_way_pct:.1f}% edges are directed — "
        f"data may be problematic"
    )

    print(f"✓ Graph validated: {N:,} nodes, {E:,} edges, "
          f"{one_way_pct:.1f}% directed")
    return True


# ===================================================================
# 11. PREPROCESSING
# ===================================================================


def preprocess_graph(
    g: "dgl.DGLGraph",
    min_degree: int = 2,
) -> "dgl.DGLGraph":
    """Clean and validate a graph before training.

    Steps:
    1. Remove nodes with total degree < *min_degree*.
    2. Remove self-loops.
    3. Deduplicate edges (``dgl.to_simple``).
    4. Assert non-empty and feature count matches.

    Parameters
    ----------
    g:
        DGL graph with ``'feat'`` in ``ndata``.
    min_degree:
        Minimum total (in + out) degree for a node to be retained.

    Returns
    -------
    dgl.DGLGraph
        Pre-processed graph.
    """
    _require_dgl()
    import dgl  # pylint: disable=import-outside-toplevel

    degrees = g.in_degrees() + g.out_degrees()
    keep_mask = degrees >= min_degree

    if keep_mask.sum() < g.num_nodes():
        removed = g.num_nodes() - int(keep_mask.sum())
        g = dgl.node_subgraph(g, torch.where(keep_mask)[0])
        print(f"Removed {removed} low-degree nodes (< {min_degree} edges)")

    g = dgl.remove_self_loop(g)
    g = dgl.to_simple(g)  # removes duplicate edges

    assert g.num_nodes() > 0, "Empty graph after preprocessing!"
    assert g.ndata["feat"].shape[0] == g.num_nodes(), (
        "Feature count mismatch after preprocessing"
    )

    print(f"Preprocessed graph: {g.num_nodes():,} nodes, {g.num_edges():,} edges")
    return g


# ===================================================================
# 12. TRANSITIVE TEST EDGES (SELECTION BIAS)
# ===================================================================


def generate_transitive_test_edges(
    g: "dgl.DGLGraph",
) -> np.ndarray:
    """Generate transitive test edges for selection-bias evaluation.

    Finds paths ``u --cp→ w --cv→ v`` where ``(u, v)`` is not already a
    co-purchase edge.  These represent products that *should* be
    recommended together based on transitive reasoning.

    Parameters
    ----------
    g:
        DGL graph with ``'type'`` in edge data.

    Returns
    -------
    np.ndarray
        Integer array of shape ``[2, T]`` — transitive test edge pairs.
    """
    src, dst = g.edges()
    cp_mask = g.edata["type"] == 0
    cv_mask = g.edata["type"] == 1

    cp_src = src[cp_mask].tolist()
    cp_dst = dst[cp_mask].tolist()
    cv_src = src[cv_mask].tolist()
    cv_dst = dst[cv_mask].tolist()

    cp_adj: Dict[int, set] = defaultdict(set)
    for u, v in zip(cp_src, cp_dst):
        cp_adj[u].add(v)

    cv_adj: Dict[int, set] = defaultdict(set)
    for u, v in zip(cv_src, cv_dst):
        cv_adj[u].add(v)

    transitive_edges: List[List[int]] = []
    for u, w in zip(cp_src, cp_dst):
        for v in cv_adj.get(w, set()):
            if v != u and v not in cp_adj.get(u, set()):
                transitive_edges.append([u, v])

    result = (
        np.array(transitive_edges, dtype=np.int64).T
        if transitive_edges
        else np.zeros((2, 0), dtype=np.int64)
    )

    print(f"Generated {result.shape[1]:,} transitive test edges")
    return result


# ===================================================================
# 13. CO-VIEW EDGE GENERATION (FROM CATEGORY SIMILARITY)
# ===================================================================


def generate_co_view_edges(
    product_df: "pd.DataFrame",
    similarity_threshold: float = 0.7,
) -> np.ndarray:
    """Generate co-view edges from product similarity within categories.

    Uses sentence-transformer embeddings and cosine similarity.
    Products in the same subcategory with high text similarity are
    considered "co-viewed".

    Parameters
    ----------
    product_df:
        DataFrame with ``'title'`` and ``'category'`` columns.
    similarity_threshold:
        Minimum cosine similarity for a co-view pair.

    Returns
    -------
    np.ndarray
        Integer array of shape ``[2, E_cv]`` — bidirectional co-view edges.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence_transformers is required for generate_co_view_edges. "
            "Install with: pip install sentence-transformers"
        )

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(product_df["title"].tolist())

    from sklearn.metrics.pairwise import cosine_similarity

    cv_pairs: List[List[int]] = []
    for category in product_df["category"].unique():
        cat_mask = product_df["category"] == category
        cat_embeddings = embeddings[cat_mask]
        cat_indices = np.where(cat_mask)[0]

        sim_matrix = cosine_similarity(cat_embeddings)
        n = len(cat_indices)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > similarity_threshold:
                    cv_pairs.append([cat_indices[i], cat_indices[j]])
                    cv_pairs.append([cat_indices[j], cat_indices[i]])

    if not cv_pairs:
        return np.zeros((2, 0), dtype=np.int64)

    return np.array(cv_pairs, dtype=np.int64).T


# ===================================================================
# 14. SERIALISATION HELPERS
# ===================================================================


def save_graph(g: "dgl.DGLGraph", path: str) -> None:
    """Save a DGL graph to disk.

    Parameters
    ----------
    g:
        DGL graph to save.
    path:
        Destination file path (e.g. ``'/kaggle/working/product_graph.bin'``).
    """
    _require_dgl()
    import dgl  # pylint: disable=import-outside-toplevel

    dgl.save_graphs(path, [g])
    print(f"Graph saved to {path}")


def load_graph(path: str) -> "dgl.DGLGraph":
    """Load a DGL graph from disk.

    Parameters
    ----------
    path:
        Path to the saved graph file.

    Returns
    -------
    dgl.DGLGraph
        Loaded graph.
    """
    _require_dgl()
    import dgl  # pylint: disable=import-outside-toplevel

    g_list, _ = dgl.load_graphs(path)
    g = g_list[0]
    print(f"Graph loaded from {path}: {g.num_nodes():,} nodes, "
          f"{g.num_edges():,} edges")
    return g
