"""Evaluation module for the DAEMON directed-graph GNN recommendation model.

Provides all evaluation, FAISS indexing, and recommendation demo functions
corresponding to the evaluation protocol (EQ1-EQ5) from:

    Virinchi et al., "Recommending Related Products Using Graph Neural Networks
    in Directed Graphs", ECML-PKDD 2022.

Core metrics:
    - HitRate@k / MRR@k  (EQ1: Node recommendation)
    - Link prediction AUC  (EQ2: Existential link prediction)
    - Direction prediction AUC  (EQ3: Direction link prediction)
    - Cold-start evaluation  (EQ4)
    - Selection-bias transitive evaluation  (EQ5)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Utility: AUC from score tensors
# ---------------------------------------------------------------------------


def compute_auc(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> float:
    """Compute ROC-AUC from positive and negative score tensors.

    Args:
        pos_scores: Scores assigned to positive (true) edges  [num_pos].
        neg_scores: Scores assigned to negative (fake) edges  [num_neg].

    Returns:
        AUC value in [0, 1] (1 = perfect separation).
    """
    scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    labels = np.concatenate(
        [np.ones(pos_scores.shape[0]), np.zeros(neg_scores.shape[0])]
    )
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# EQ1 — Node recommendation metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_hit_rate_at_k(
    src_emb: torch.Tensor,
    tgt_emb: torch.Tensor,
    query_ids: torch.Tensor,
    true_candidate_ids: torch.Tensor,
    k: int = 10,
) -> float:
    """HitRate@k for node recommendation (EQ1).

    For each query product *q*, the true co-purchase partner *v* is considered
    "hit" if it appears in the top-*k* products ranked by
    ``rel(q, v) = θ_q^s · θ_v^t``.

    Args:
        src_emb: Source embedding matrix  [num_nodes, dim].
        tgt_emb: Target embedding matrix  [num_nodes, dim].
        query_ids: Query node indices  [num_queries].
        true_candidate_ids: Ground-truth target node for each query  [num_queries].
        k: Number of top candidates to consider.

    Returns:
        HitRate@k as a float in [0, 1].
    """
    device = src_emb.device
    query_ids = query_ids.to(device)
    true_candidate_ids = true_candidate_ids.to(device)

    # Source embeddings for queries  [num_queries, dim]
    q_src = src_emb[query_ids]

    # Dot-product scores against all target embeddings  [num_queries, num_nodes]
    scores = q_src @ tgt_emb.T

    # Top-k indices per query
    _, topk_indices = scores.topk(k=k, dim=1, largest=True, sorted=True)

    # Check if true candidate is in top-k
    hits = (topk_indices == true_candidate_ids.unsqueeze(1)).any(dim=1).float()
    return float(hits.mean().item())


@torch.no_grad()
def compute_mrr_at_k(
    src_emb: torch.Tensor,
    tgt_emb: torch.Tensor,
    query_ids: torch.Tensor,
    true_candidate_ids: torch.Tensor,
    k: int = 10,
) -> float:
    """MRR@k (Mean Reciprocal Rank) for node recommendation (EQ1).

    Reciprocal rank is ``1 / rank(v)`` if the true candidate appears at
    rank ≤ *k*, otherwise 0.

    Args:
        src_emb: Source embedding matrix  [num_nodes, dim].
        tgt_emb: Target embedding matrix  [num_nodes, dim].
        query_ids: Query node indices  [num_queries].
        true_candidate_ids: Ground-truth target node for each query  [num_queries].
        k: Cutoff rank.

    Returns:
        MRR@k as a float in [0, 1].
    """
    device = src_emb.device
    query_ids = query_ids.to(device)
    true_candidate_ids = true_candidate_ids.to(device)

    q_src = src_emb[query_ids]
    scores = q_src @ tgt_emb.T

    # Get top-k scores (need k up to full candidate set if k > N)
    effective_k = min(k, scores.size(1))
    _, topk_indices = scores.topk(k=effective_k, dim=1, largest=True, sorted=True)

    # Find rank of the true candidate for each query
    # Expand true IDs to compare against every position in top-k
    match = topk_indices == true_candidate_ids.unsqueeze(1)  # [num_queries, k]

    # Get the first (lowest) rank where match occurs, 1-indexed
    # argmax returns 0 if no match -> rank becomes 1 incorrectly; we fix below
    first_match_pos = match.int().argmax(dim=1)  # [num_queries]
    found = match.any(dim=1)  # [num_queries]

    ranks = torch.where(found, first_match_pos + 1, torch.zeros_like(first_match_pos))
    reciprocal_ranks = torch.where(found, 1.0 / ranks.float(), torch.zeros_like(ranks.float()))

    return float(reciprocal_ranks.mean().item())


# ---------------------------------------------------------------------------
# EQ2 — Existential link prediction
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_link_prediction_auc(
    src_emb: torch.Tensor,
    tgt_emb: torch.Tensor,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
) -> float:
    """ROC-AUC for existential link prediction (EQ2).

    A higher score means the model assigns larger ``rel(u, v) = θ_u^s · θ_v^t``
    to real edges than to randomly generated non-edges.

    Args:
        src_emb: Source embedding matrix  [num_nodes, dim].
        tgt_emb: Target embedding matrix  [num_nodes, dim].
        pos_edges: Positive (real) edge index pairs  [2, num_pos] or [num_pos, 2].
        neg_edges: Negative (fake) edge index pairs  [2, num_neg] or [num_neg, 2].

    Returns:
        AUC as a float in [0, 1].
    """
    pos_edges = _ensure_2col(pos_edges)
    neg_edges = _ensure_2col(neg_edges)

    pos_scores = (src_emb[pos_edges[:, 0]] * tgt_emb[pos_edges[:, 1]]).sum(dim=1)
    neg_scores = (src_emb[neg_edges[:, 0]] * tgt_emb[neg_edges[:, 1]]).sum(dim=1)

    return compute_auc(pos_scores, neg_scores)


# ---------------------------------------------------------------------------
# EQ3 — Direction link prediction
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_direction_prediction_auc(
    src_emb: torch.Tensor,
    tgt_emb: torch.Tensor,
    one_way_edges: torch.Tensor,
) -> float:
    """ROC-AUC for direction prediction (EQ3).

    For each one-way edge ``u → v`` (where ``v → u`` does **not** exist):
        - Positive: ``rel(u, v) = θ_u^s · θ_v^t``  (correct direction)
        - Negative: ``rel(v, u) = θ_v^s · θ_u^t``  (wrong direction)

    A well-trained DAEMON model assigns higher scores to the correct direction.

    Args:
        src_emb: Source embedding matrix  [num_nodes, dim].
        tgt_emb: Target embedding matrix  [num_nodes, dim].
        one_way_edges: One-way edge index pairs  [2, num_edges] or [num_edges, 2].

    Returns:
        AUC as a float in [0, 1].
    """
    one_way_edges = _ensure_2col(one_way_edges)
    u = one_way_edges[:, 0]
    v = one_way_edges[:, 1]

    forward_scores = (src_emb[u] * tgt_emb[v]).sum(dim=1)
    reverse_scores = (src_emb[v] * tgt_emb[u]).sum(dim=1)

    return compute_auc(forward_scores, reverse_scores)


# ---------------------------------------------------------------------------
# Comprehensive ranking metrics over multiple k
# ---------------------------------------------------------------------------


def compute_ranking_metrics(
    embeds: torch.Tensor,
    test_edges: torch.Tensor,
    ks: Tuple[int, ...] = (1, 5, 10, 20),
    top_n: int = 100,
) -> Dict[str, Any]:
    """Compute HitRate@k and MRR@k for multiple cutoffs.

    Uses source and target halves of the unified embedding matrix.
    When embeddings are coupled (single matrix), the first half is treated
    as source and the second as target.  If the matrix is 2× stacked
    ``[θ^s; θ^t]`` of shape ``[2 * N, d]``, the first ``N`` rows are source
    and the last ``N`` rows are target.

    *NOTE*: If your model returns separate ``src_emb`` and ``tgt_emb``
    tensors, use ``compute_hit_rate_at_k`` / ``compute_mrr_at_k`` directly.

    Args:
        embeds: Unified embedding matrix  [N, d] (single) or [2*N, d] (stacked).
        test_edges: Test edge index pairs  [2, num_test] or [num_test, 2].
        ks: Tuple of cutoffs to evaluate.
        top_n: Number of top candidates to rank (speed / memory trade-off).

    Returns:
        Dict with keys ``"hitrate"`` and ``"mrr"``, each containing a dict
        mapping cutoff ``k`` to the metric value.
    """
    test_edges = _ensure_2col(test_edges)
    num_nodes = embeds.size(0)

    # Detect stacked vs. single embedding shape
    if test_edges[:, 0].max() < num_nodes and test_edges[:, 1].max() < num_nodes:
        # Single matrix — all nodes share same embedding for src & tgt
        src_emb = embeds
        tgt_emb = embeds
    else:
        # Stacked [θ^s; θ^t] — split in half
        mid = num_nodes // 2
        src_emb = embeds[:mid]
        tgt_emb = embeds[mid:]

    query_ids = test_edges[:, 0]
    true_ids = test_edges[:, 1]

    hitrate: Dict[int, float] = {}
    mrr: Dict[int, float] = {}

    for k in ks:
        hitrate[k] = compute_hit_rate_at_k(src_emb, tgt_emb, query_ids, true_ids, k=min(k, top_n))
        mrr[k] = compute_mrr_at_k(src_emb, tgt_emb, query_ids, true_ids, k=min(k, top_n))

    return {"hitrate": hitrate, "mrr": mrr}


# ---------------------------------------------------------------------------
# Full-graph embedding generation via neighbor sampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_all_embeddings(
    model: torch.nn.Module,
    g: "dgl.DGLGraph",  # noqa: F821 — lazy import; type kept for clarity
    batch_size: int = 4096,
    device: str = "cuda",
) -> "torch.Tensor | Tuple[torch.Tensor, torch.Tensor]":
    """Run full-graph inference to generate embeddings for all nodes.

    Uses a DGL ``MultiLayerNeighborSampler`` with fanouts ``[-1]`` (all
    neighbors) per layer, batching over seed nodes to avoid OOM on large
    graphs.  Each subgraph is moved to ``device`` for the forward pass.

    The returned embeddings are L2-normalized (consistent with the DAEMON
    forward pass).  If the model's ``forward`` returns a tuple
    ``(src_emb, tgt_emb)``, both are returned as a tuple.

    Args:
        model: Trained DAEMON model (or any ``nn.Module`` that accepts
            ``(blocks, features)`` and returns a ``[batch_size, dim]``
            tensor or a ``(src, tgt)`` tuple).
        g: Full DGL graph — must have ``g.ndata['feat']`` containing node
            features  [num_nodes, in_feats].
        batch_size: Number of seed nodes per batch.
        device: Target device for computation.

    Returns:
        - Single tensor  [num_nodes, out_dim] on CPU **or**
        - Tuple ``(src_emb, tgt_emb)`` each  [num_nodes, out_dim] on CPU.

    Raises:
        ImportError: If DGL is not installed.
    """
    try:
        import dgl
        from dgl.dataloading import DataLoader as DGLDataLoader
        from dgl.dataloading import MultiLayerNeighborSampler
    except ImportError as exc:
        raise ImportError(
            "DGL is required for generate_all_embeddings. "
            "Install it via: pip install dgl"
        ) from exc

    model = model.to(device)
    model.eval()

    # Determine number of layers from model
    num_layers = len(model.layers) if hasattr(model, "layers") else 3

    # Full-neighbor sampling (all neighbors per layer)
    sampler = MultiLayerNeighborSampler([-1] * num_layers)

    num_nodes = g.num_nodes()
    all_nodes = torch.arange(num_nodes)

    loader = DGLDataLoader(
        g,
        all_nodes,
        sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        device=device,
    )

    src_list: List[torch.Tensor] = []
    tgt_list: List[torch.Tensor] = []
    is_dual: bool | None = None

    for input_nodes, output_nodes, blocks in loader:
        # blocks[0].srcdata['feat'] already transferred by DGLDataLoader
        h = blocks[0].srcdata["feat"]
        out = model(blocks, h)

        if isinstance(out, (tuple, list)) and len(out) == 2:
            src_list.append(out[0].cpu())
            tgt_list.append(out[1].cpu())
            is_dual = True
        else:
            src_list.append(out.cpu())
            is_dual = False if is_dual is None else is_dual

    if is_dual:
        return (torch.cat(src_list, dim=0), torch.cat(tgt_list, dim=0))
    return torch.cat(src_list, dim=0)


# ---------------------------------------------------------------------------
# FAISS index construction and search
# ---------------------------------------------------------------------------


def build_faiss_index(
    embeds: torch.Tensor,
    use_gpu: bool = True,
    nlist: Optional[int] = None,
    nprobe: int = 32,
) -> "faiss.Index":  # noqa: F821
    """Build a GPU FAISS ``IndexIVFFlat`` for cosine-similarity search.

    Embeddings are L2-normalised so that inner-product search is equivalent
    to cosine-similarity search.

    Args:
        embeds: Embedding matrix  [num_vectors, dim] — must be dense float32.
        use_gpu: If True, build the index on the first available GPU.
        nlist: Number of IVF centroids (clusters).  Defaults to
            ``min(4096, int(sqrt(num_vectors)))``.
        nprobe: Number of clusters to probe during search (higher = slower
            but more accurate).

    Returns:
        A trained FAISS index (GPU if ``use_gpu=True``) with vectors added.

    Raises:
        ImportError: If FAISS is not installed.
    """
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "FAISS is required for build_faiss_index. "
            "Install it via: pip install faiss-gpu"
        ) from exc

    embeds_np = embeds.cpu().numpy().astype(np.float32)
    # Normalise in-place for cosine similarity via inner product
    faiss.normalize_L2(embeds_np)

    dim = embeds_np.shape[1]
    num_vectors = embeds_np.shape[0]

    if nlist is None:
        nlist = min(4096, int(np.sqrt(num_vectors)))
    nlist = max(nlist, 1)  # must be at least 1

    # Use IndexFlatIP as quantizer for IVF
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nprobe

    # Train (k-means clustering)
    index.train(embeds_np)
    # Add vectors
    index.add(embeds_np)

    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        # CoarseParams: use float16 for faster distance computation
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)

    return index


def recommend_related(
    embeds: torch.Tensor,
    index: "faiss.Index",  # noqa: F821
    product_idx: int,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the top-*k* most related products for a given query product.

    Relevance is measured as ``θ_q^s · θ_v^t`` (source embedding of query ×
    target embedding of candidate), which equates to cosine similarity after
    L2 normalisation.

    The query's own index is excluded from results (self-match removal).

    Args:
        embeds: Full embedding matrix  [num_nodes, dim].
        index: Trained FAISS index built over target embeddings (or unified
            embeddings).
        product_idx: Index of the query product.
        k: Number of recommendations to return.

    Returns:
        Tuple of ``(indices, distances)`` where:
            - ``indices``: array of shape ``(k,)`` with product indices.
            - ``distances``: array of shape ``(k,)`` with cosine similarities.
    """
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("FAISS is required for recommend_related.") from exc

    query_vec = embeds[product_idx].cpu().numpy().astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query_vec)

    # Search k+1 to account for possible self-match
    distances, indices = index.search(query_vec, k + 1)

    # Remove self-match if present
    mask = indices[0] != product_idx
    result_indices = indices[0][mask][:k]
    result_distances = distances[0][mask][:k]

    # Pad with fallback if self-match consumed a slot
    if len(result_indices) < k:
        extra_needed = k - len(result_indices)
        # Take the next available results
        alt_mask = ~mask
        extra_indices = indices[0][alt_mask][:extra_needed]
        extra_distances = distances[0][alt_mask][:extra_needed]
        result_indices = np.concatenate([result_indices, extra_indices])
        result_distances = np.concatenate([result_distances, extra_distances])

    return result_indices, result_distances


# ---------------------------------------------------------------------------
# Cold-start recommendation
# ---------------------------------------------------------------------------


@torch.no_grad()
def cold_start_recommend(
    model: torch.nn.Module,
    index: "faiss.Index",  # noqa: F821
    new_features: torch.Tensor,
    g: "dgl.DGLGraph",  # noqa: F821
    existing_features: torch.Tensor,
    cold_node_id: Optional[int] = None,
    k_nn: int = 5,
    k: int = 10,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate recommendations for a cold-start product using the paper's
    approach (Virinchi et al., 2022, lines 367-378):

        1. k-NN feature lookup to find similar existing products.
        2. Add bidirectional co-view edges between the cold node and its
           neighbours to augment the graph.
        3. Run a GNN forward pass on the augmented graph to obtain the
           cold node's source embedding θ_c^s.
        4. Search the FAISS index with θ_c^s for top-k recommendations.

    Args:
        model: Trained DAEMON model.
        index: Trained FAISS index (built on target embeddings).
        new_features: Raw feature vector for the new product  [in_feats].
        g: Full DGL graph (CPU-resident).  Node features must be stored in
            ``g.ndata['feat']``.
        existing_features: Feature matrix of all existing nodes  [N, in_feats].
        cold_node_id: Explicit ID to assign to the cold node.  If ``None``,
            ``g.num_nodes()`` is used (i.e. the next available index).
        k_nn: Number of feature-space neighbours for graph augmentation.
        k: Number of recommendations to return.
        device: Device for GNN forward pass.

    Returns:
        Tuple of ``(indices, distances)`` arrays of shape ``(k,)``.
    """
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("FAISS is required for cold_start_recommend.") from exc

    try:
        import dgl
        from dgl.dataloading import DataLoader as DGLDataLoader
        from dgl.dataloading import MultiLayerNeighborSampler
    except ImportError as exc:
        raise ImportError("DGL is required for cold_start_recommend.") from exc

    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize as sk_normalize

    model = model.to(device).eval()

    # Ensure 2D
    if new_features.dim() == 1:
        new_features = new_features.unsqueeze(0)

    # ------------------------------------------------------------------
    # Step 1: k-NN feature lookup
    # ------------------------------------------------------------------
    nn_model = NearestNeighbors(n_neighbors=k_nn, metric="cosine")
    nn_model.fit(existing_features.cpu().numpy())
    _distances, knn_indices = nn_model.kneighbors(
        new_features.cpu().numpy().reshape(1, -1)
    )
    similar_ids = knn_indices[0]

    # ------------------------------------------------------------------
    # Step 2: Augment graph with cold-start edges
    # ------------------------------------------------------------------
    cs_id = cold_node_id if cold_node_id is not None else g.num_nodes()
    g_aug = dgl.add_nodes(g, 1)
    g_aug.ndata["feat"] = torch.cat(
        [g.ndata["feat"], new_features.unsqueeze(0).cpu()]
    )

    for sid in similar_ids:
        g_aug.add_edges(cs_id, int(sid), {"type": torch.tensor([1])})  # co-view type
        g_aug.add_edges(int(sid), cs_id, {"type": torch.tensor([1])})

    # ------------------------------------------------------------------
    # Step 3: GNN forward pass on augmented graph
    # ------------------------------------------------------------------
    num_layers = len(model.layers) if hasattr(model, "layers") else 3
    # Use paper's fanouts or full-neighbor if needed; scale down for speed
    sampler = MultiLayerNeighborSampler([20, 10, 10][:num_layers])
    loader = DGLDataLoader(
        g_aug,
        torch.arange(g_aug.num_nodes()),
        sampler,
        batch_size=4096,
        shuffle=False,
        device=device,
    )

    all_src: List[torch.Tensor] = []
    all_tgt: List[torch.Tensor] = []

    for _input_nodes, _output_nodes, blocks in loader:
        blocks = [b.to(device) for b in blocks]
        h = blocks[0].srcdata["feat"].to(device)
        out = model(blocks, h)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            all_src.append(out[0].cpu())
            all_tgt.append(out[1].cpu())
        else:
            all_src.append(out.cpu())

    src_emb = torch.cat(all_src, dim=0)

    # ------------------------------------------------------------------
    # Step 4: Search with cold node's source embedding
    # ------------------------------------------------------------------
    query = src_emb[cs_id].cpu().numpy().reshape(1, -1).astype(np.float32)
    query = sk_normalize(query, norm="l2")
    distances, indices = index.search(query, k)

    return indices[0], distances[0]


def cold_start_augment_graph(
    g: "dgl.DGLGraph",  # noqa: F821
    cold_idx: int,
    new_features: torch.Tensor,
    existing_features: torch.Tensor,
    k_nn: int = 5,
    edge_type: int = 1,
) -> "dgl.DGLGraph":  # noqa: F821
    """Augment the graph with similarity edges for a cold-start product.

    Finds *k* nearest neighbours of the new product in feature space and adds
    bidirectional co-view edges (type=1) to the graph so the GNN can
    aggregate meaningful embeddings during the next forward pass.

    Args:
        g: The full DGL graph.
        cold_idx: Index assigned to the new product (must be ``g.num_nodes()``
            before adding).
        new_features: Feature vector of the cold-start product  [in_feats].
        existing_features: Feature matrix of all existing nodes  [N, in_feats].
        k_nn: Number of feature-space neighbours to connect.
        edge_type: Edge type label to assign (default 1 = co-view).

    Returns:
        Graph with new nodes and edges added.

    Raises:
        ImportError: If DGL or sklearn is not installed.
    """
    try:
        import dgl
    except ImportError as exc:
        raise ImportError("DGL is required for cold_start_augment_graph.") from exc

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError("scikit-learn is required for cold_start_augment_graph.") from exc

    nn_model = NearestNeighbors(n_neighbors=k_nn, metric="cosine")
    nn_model.fit(existing_features.cpu().numpy())

    feat_np = new_features.cpu().numpy().reshape(1, -1)
    _distances, indices = nn_model.kneighbors(feat_np)

    similar_ids = indices[0].tolist()

    # Add bidirectional co-view edges
    src = torch.full((k_nn,), cold_idx, dtype=torch.long)
    dst = torch.tensor(similar_ids, dtype=torch.long)
    # Forward edges: cold -> similar
    g.add_edges(src, dst, {"type": torch.full((k_nn,), edge_type, dtype=torch.long)})
    # Backward edges: similar -> cold
    g.add_edges(dst, src, {"type": torch.full((k_nn,), edge_type, dtype=torch.long)})

    return g


# ---------------------------------------------------------------------------
# EQ4 + EQ5 — Full evaluation suite
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_full(
    model: torch.nn.Module,
    g: "dgl.DGLGraph",  # noqa: F821
    masks: Dict[str, torch.Tensor],
    cfg: Any,
    device: str = "cuda",
    batch_size: int = 4096,
) -> Dict[str, Any]:
    """Run the complete DAEMON evaluation suite (EQ1–EQ5).

    This function:
        1. Generates embeddings for all nodes via ``generate_all_embeddings``.
        2. Computes node recommendation metrics (HitRate@k, MRR@k) on test
           edges (EQ1).
        3. Computes link prediction AUC on test edges vs. random negatives
           (EQ2).
        4. Computes direction prediction AUC on one-way test edges (EQ3).
        5. Optionally evaluates cold-start performance if ``cold_nodes`` is
           provided in ``masks`` (EQ4).
        6. Optionally evaluates transitive (selection bias) edges if
           ``trans_edges`` is provided in ``masks`` (EQ5).

    Args:
        model: Trained DAEMON model instance.
        g: Full DGL graph with edge type information in ``g.edata['type']``.
        masks: Dictionary of boolean edge masks — must include ``'test'``.
            May also include ``'cold_nodes'`` (node indices for EQ4) and
            ``'trans_edges'`` (edge indices for EQ5).
        cfg: Configuration object (or dict) with attribute ``hitrate_k``
            (iterable of int cutoffs) and ``num_nodes``.
        device: Device for inference.
        batch_size: Batch size for embedding generation.

    Returns:
        Dictionary containing:
            - ``"hitrate"``: dict mapping k → HitRate@k
            - ``"mrr"``: dict mapping k → MRR@k
            - ``"auc_link"``: link prediction AUC (EQ2)
            - ``"auc_direction"``: direction prediction AUC (EQ3)
            - ``"direction_accuracy"``: fraction where forward > reverse
            - ``"cold_hitrate"``: (if EQ4 run) HitRate for cold-start nodes
            - ``"cold_mrr"``: (if EQ4 run) MRR for cold-start nodes
            - ``"trans_hitrate"``: (if EQ5 run) HitRate for transitive edges
            - ``"trans_mrr"``: (if EQ5 run) MRR for transitive edges
    """
    model = model.to(device).eval()

    # ------------------------------------------------------------------
    # 1. Full-graph embedding generation
    # ------------------------------------------------------------------
    raw_embeds = generate_all_embeddings(
        model, g, batch_size=batch_size, device=device
    )

    src_emb: torch.Tensor
    tgt_emb: torch.Tensor

    # If generate_all_embeddings returns a tuple (from a dual-output model),
    # split into source and target.  Otherwise share a single embedding space.
    if isinstance(raw_embeds, (tuple, list)) and len(raw_embeds) == 2:
        src_emb, tgt_emb = raw_embeds[0].to(device), raw_embeds[1].to(device)
    else:
        src_emb = raw_embeds.to(device)
        tgt_emb = raw_embeds.to(device)

    num_nodes = g.num_nodes()
    ks = list(getattr(cfg, "hitrate_k", [1, 5, 10, 20]))

    results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 2. EQ1 — Node recommendation on test edges
    # ------------------------------------------------------------------
    test_edge_mask = masks.get("test", masks.get("test_mask"))
    if test_edge_mask is not None:
        all_edges = torch.stack(g.edges(), dim=1)
        test_edges = all_edges[test_edge_mask].to(device)

        query_ids = test_edges[:, 0]
        true_ids = test_edges[:, 1]

        hitrate: Dict[int, float] = {}
        mrr: Dict[int, float] = {}
        for k in ks:
            hitrate[k] = compute_hit_rate_at_k(
                src_emb, tgt_emb, query_ids, true_ids, k=k
            )
            mrr[k] = compute_mrr_at_k(
                src_emb, tgt_emb, query_ids, true_ids, k=k
            )

        results["hitrate"] = hitrate
        results["mrr"] = mrr

    # ------------------------------------------------------------------
    # 3. EQ2 — Link prediction AUC
    # ------------------------------------------------------------------
    if test_edge_mask is not None:
        # Keep test_edges on device (already moved above in EQ1 block)
        # Sample same number of negative edges
        neg_src = torch.randint(0, num_nodes, (test_edges.size(0),), device=device)
        neg_dst = torch.randint(0, num_nodes, (test_edges.size(0),), device=device)
        neg_edges = torch.stack([neg_src, neg_dst], dim=1)

        results["auc_link"] = compute_link_prediction_auc(
            src_emb, tgt_emb, test_edges, neg_edges
        )

    # ------------------------------------------------------------------
    # 4. EQ3 — Direction prediction AUC / Accuracy
    # ------------------------------------------------------------------
    if test_edge_mask is not None and "type" in g.edata:
        edge_types = g.edata["type"]
        # Find one-way test edges: co-purchase (type=0) edges where the
        # reverse does NOT exist in the graph.
        cp_test_mask = test_edge_mask & (edge_types == 0)  # co-purchase edges
        cp_test_edges = all_edges[cp_test_mask].to(device)

        if cp_test_edges.size(0) > 0:
            # Check which pairs are one-way (reverse edge absent)
            # Keep u, v on CPU for _has_edge which queries the CPU-resident graph
            u_cpu, v_cpu = cp_test_edges[:, 0].cpu(), cp_test_edges[:, 1].cpu()
            has_reverse = _has_edge(g, v_cpu, u_cpu)  # boolean per pair
            has_reverse = has_reverse.to(device)
            one_way_mask = ~has_reverse
            one_way_edges = cp_test_edges[one_way_mask]

            if one_way_edges.size(0) > 0:
                results["auc_direction"] = compute_direction_prediction_auc(
                    src_emb, tgt_emb, one_way_edges
                )

                # Direction accuracy: fraction where forward > reverse
                u_ids = one_way_edges[:, 0]
                v_ids = one_way_edges[:, 1]
                forward_scores = (src_emb[u_ids] * tgt_emb[v_ids]).sum(dim=1)
                reverse_scores = (src_emb[v_ids] * tgt_emb[u_ids]).sum(dim=1)
                results["direction_accuracy"] = (
                    (forward_scores > reverse_scores).float().mean().item()
                )

    # ------------------------------------------------------------------
    # 5. EQ4 — Cold-start recommendation (if cold node indices provided)
    # ------------------------------------------------------------------
    cold_node_mask = masks.get("cold_nodes")
    if cold_node_mask is not None:
        cold_node_indices = cold_node_mask.nonzero(as_tuple=True)[0]
        if cold_node_indices.numel() > 0 and test_edge_mask is not None:
            test_edges = all_edges[test_edge_mask].to(device)

            # Find test edges where the query is a cold-start node
            cold_query_mask = torch.isin(test_edges[:, 0], cold_node_indices)
            cold_test_edges = test_edges[cold_query_mask]

            if cold_test_edges.size(0) > 0:
                cold_q = cold_test_edges[:, 0]
                cold_t = cold_test_edges[:, 1]

                cold_hitrate: Dict[int, float] = {}
                cold_mrr: Dict[int, float] = {}
                for k in ks:
                    cold_hitrate[k] = compute_hit_rate_at_k(
                        src_emb, tgt_emb, cold_q, cold_t, k=k
                    )
                    cold_mrr[k] = compute_mrr_at_k(
                        src_emb, tgt_emb, cold_q, cold_t, k=k
                    )

                results["cold_hitrate"] = cold_hitrate
                results["cold_mrr"] = cold_mrr

    # ------------------------------------------------------------------
    # 6. EQ5 — Selection bias (transitive edges)
    # ------------------------------------------------------------------
    trans_edge_mask = masks.get("trans_edges")
    if trans_edge_mask is not None and test_edge_mask is not None:
        trans_edges = all_edges[trans_edge_mask].to(device)
        if trans_edges.size(0) > 0:
            trans_q = trans_edges[:, 0]
            trans_t = trans_edges[:, 1]

            trans_hitrate: Dict[int, float] = {}
            trans_mrr: Dict[int, float] = {}
            for k in ks:
                trans_hitrate[k] = compute_hit_rate_at_k(
                    src_emb, tgt_emb, trans_q, trans_t, k=k
                )
                trans_mrr[k] = compute_mrr_at_k(
                    src_emb, tgt_emb, trans_q, trans_t, k=k
                )

            results["trans_hitrate"] = trans_hitrate
            results["trans_mrr"] = trans_mrr

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_2col(edges: torch.Tensor) -> torch.Tensor:
    """Ensure edge tensor has shape ``[num_edges, 2]``.

    Accepts either ``[2, num_edges]`` or ``[num_edges, 2]``.
    """
    if edges.dim() == 2 and edges.size(0) == 2 and edges.size(1) != 2:
        edges = edges.T
    return edges


def _has_edge(
    g: "dgl.DGLGraph", src: torch.Tensor, dst: torch.Tensor  # noqa: F821
) -> torch.BoolTensor:
    """Check which edges ``(src[i], dst[i])`` exist in the graph.

    Returns a boolean tensor per pair. Handles batches where some edges
    exist and others don't by checking each pair individually.
    """
    try:
        eid = g.edge_ids(src, dst)
        return eid >= 0
    except (ValueError, RuntimeError, AssertionError):
        # DGL raises when some edges don't exist — fall back to per-pair check
        result = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for i in range(src.size(0)):
            try:
                g.edge_ids(src[i:i+1], dst[i:i+1])
                result[i] = True
            except (ValueError, RuntimeError, AssertionError):
                result[i] = False
        return result


def _probe_model_output(
    model: torch.nn.Module,
    g: "dgl.DGLGraph",  # noqa: F821
    device: str = "cuda",
) -> torch.Tensor:
    """Probe the model output shape with a tiny subgraph.

    Returns a single output tensor to determine whether the model returns
    a single embedding or ``(src_emb, tgt_emb)``.
    """
    try:
        import dgl
    except ImportError:
        return torch.empty(0)

    model = model.to(device).eval()
    # Pick 2 seed nodes, sample 1-hop neighbors
    sampler = dgl.dataloading.MultiLayerNeighborSampler([5])
    loader = dgl.dataloading.DataLoader(
        g,
        torch.arange(min(g.num_nodes(), 10), device="cpu"),
        sampler,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        device=device,
    )
    for _input_nodes, _output_nodes, blocks in loader:
        h = blocks[0].srcdata["feat"]
        out = model(blocks, h)
        return out  # type: ignore[no-any-return]
    return torch.empty(0)
