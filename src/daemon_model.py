"""
DAEMON — Directed Asymmetric GNN Model for Product Recommendation.

Core model classes:

    DAEMONLayer   — Single GNN layer with dual-embedding message passing
    DAEMONModel   — Stacked DAEMON layers + final projection
    AsymmetricLoss— Three-component loss (co-purchase, asymmetry, co-view)
    DAEMONConfig  — All hyperparameters in one dataclass

References:
    Virinchi et al. (2022). Recommending Related Products Using Graph Neural
    Networks in Directed Graphs. ECML-PKDD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl
from dgl import function as fn


# ============================================================================
# DAEMONConfig
# ============================================================================

@dataclass
class DAEMONConfig:
    """Single source of truth for all hyperparameters."""
    # Graph
    num_nodes: int = 0
    num_edges: int = 0
    num_relations: int = 2

    # Model architecture
    in_feats: int = 384
    hidden_dim: int = 128
    out_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1

    # Training
    epochs: int = 30
    batch_size: int = 1024
    num_neighbors: Tuple[int, ...] = (20, 10, 10)
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_accum_steps: int = 1
    use_amp: bool = True
    patience: int = 5

    # Loss
    num_neg: int = 5

    # Evaluation
    hitrate_k: Tuple[int, ...] = (5, 10, 20)
    val_every: int = 1

    # Paths
    data_dir: str = "/kaggle/input/daemon-data"
    output_dir: str = "/kaggle/working"
    checkpoint_path: str = "/kaggle/working/daemon_best.pt"

    # Memory
    cleanup_every_n_epochs: int = 4

    def __post_init__(self):
        assert self.out_dim <= self.hidden_dim, \
            f"out_dim ({self.out_dim}) should be <= hidden_dim ({self.hidden_dim})"


# ============================================================================
# DAEMONLayer
# ============================================================================

class DAEMONLayer(nn.Module):
    """Single DAEMON layer with dual-embedding message passing.

    For each node u, the layer produces:

    .. code-block:: text

        h_s^l = ReLU(W^l @ AGG_cp_out(h_t^{l-1})) + ReLU(W^l @ AGG_cv_out(h_s^{l-1}))
        h_t^l = ReLU(W^l @ AGG_cp_in( h_s^{l-1})) + ReLU(W^l @ AGG_cv_in( h_t^{l-1}))

    where:
        AGG_cp_out : mean over co-purchase **out**-neighbors' target embeddings
        AGG_cv_out : mean over co-view    **out**-neighbors' source embeddings
        AGG_cp_in  : mean over co-purchase **in**-neighbors'  source embeddings
        AGG_cv_in  : mean over co-view    **in**-neighbors'  target embeddings

    The shared weight matrix ``W`` is applied to **each** aggregation
    independently, and ReLU is applied per-component before summing
    (matching Algorithm 1, line 4 of the paper).

    Source embedding aggregates from OUT-neighbors (via
    :func:`dgl.reverse`), while target embedding aggregates from
    IN-neighbors (the original sampled block).  Both outputs are
    L2-normalised before return.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    dropout : float
        Dropout probability applied to inputs.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        block: dgl.DGLBlock,
        h_src: torch.Tensor,
        h_tgt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for one layer on a sampled block.

        Parameters
        ----------
        block : dgl.DGLBlock
            Sampled message-flow graph (in-neighbour edges).
            Must contain ``edata['type']`` where 0 = co-purchase,
            1 = co-view.
        h_src : torch.Tensor  [num_src_nodes, in_dim]
            Source embeddings of all nodes in the block's source frontier.
        h_tgt : torch.Tensor  [num_src_nodes, in_dim]
            Target embeddings of all nodes in the block's source frontier.

        Returns
        -------
        (h_src_new, h_tgt_new) : Tuple[torch.Tensor, torch.Tensor]
            Updated embeddings of shape [num_dst_nodes, out_dim] each.
        """
        h_src = self.dropout(h_src)
        h_tgt = self.dropout(h_tgt)

        with block.local_scope():
            block.srcdata['h_src'] = h_src
            block.srcdata['h_tgt'] = h_tgt

            # Edge-type masks on the original (in-neighbour) block
            cp_eid = (block.edata['type'] == 0).nonzero(as_tuple=True)[0]
            cv_eid = (block.edata['type'] == 1).nonzero(as_tuple=True)[0]

            # ---- Source embedding update (OUT-neighbours) ----
            # Reverse the block so edges point from batch nodes *to* their
            # out-neighbours.  We then aggregate dst→src (see _agg_rev).
            rev_block = dgl.reverse(block)
            rev_cp_eid = (rev_block.edata['type'] == 0).nonzero(as_tuple=True)[0]
            rev_cv_eid = (rev_block.edata['type'] == 1).nonzero(as_tuple=True)[0]

            h_s_cp = self._agg_rev(rev_block, rev_cp_eid, 'h_tgt')
            h_s_cv = self._agg_rev(rev_block, rev_cv_eid, 'h_src')

            # ---- Target embedding update (IN-neighbours) ----
            h_t_cp = self._aggregate(block, cp_eid, 'h_src', 'tmp_t_cp')
            h_t_cv = self._aggregate(block, cv_eid, 'h_tgt', 'tmp_t_cv')

            dst_ids = block.dstdata['_ID']

            # Apply shared weight + ReLU **per component** (Algorithm 1, line 4)
            h_s_new = F.relu(self.W(h_s_cp[dst_ids])) + F.relu(self.W(h_s_cv[dst_ids]))
            h_t_new = F.relu(self.W(h_t_cp[dst_ids])) + F.relu(self.W(h_t_cv[dst_ids]))

        # L2 normalise
        h_s_new = F.normalize(h_s_new, p=2, dim=-1)
        h_t_new = F.normalize(h_t_new, p=2, dim=-1)

        return h_s_new, h_t_new

    def _aggregate(
        self,
        block: dgl.DGLBlock,
        eid: torch.Tensor,
        src_feat_name: str,
        mail_name: str,
    ) -> torch.Tensor:
        """Message passing over edges ``eid`` (src → dst).

        Copies source features as messages, reduces by mean at
        destination nodes.
        """
        if eid.numel() == 0:
            return torch.zeros(
                block.num_dst_nodes(), self.W.out_features,
                device=block.device
            )
        subg = block.edge_subgraph(eid, preserve_nodes=True)
        subg.srcdata['x'] = subg.srcdata[src_feat_name]
        subg.update_all(
            fn.copy_u('x', 'm'),
            fn.mean('m', 'agg'),
        )
        return subg.dstdata['agg']

    def _agg_rev(
        self,
        block: dgl.DGLBlock,
        eid: torch.Tensor,
        feat_name: str,
    ) -> torch.Tensor:
        """Message passing aggregating from **dst** to **src** (dst → src).

        Used for source-embedding updates where out-neighbour features
        reside in ``block.dstdata`` after a :func:`dgl.reverse`.
        """
        if eid.numel() == 0:
            return torch.zeros(
                block.num_src_nodes(), self.W.out_features,
                device=block.device
            )
        subg = block.edge_subgraph(eid, preserve_nodes=True)
        # Features live on dst side → store before reversing
        feat = subg.dstdata[feat_name]
        subg_rev = dgl.reverse(subg)
        subg_rev.srcdata['x'] = feat
        subg_rev.update_all(
            fn.copy_u('x', 'm'),
            fn.mean('m', 'agg'),
        )
        return subg_rev.dstdata['agg']


# ============================================================================
# DAEMONModel
# ============================================================================

class DAEMONModel(nn.Module):
    """Full DAEMON model: stacked :class:`DAEMONLayer` → projection → L2 norm.

    Expects DGL ``blocks`` (list of ``DGLBlock``) from ``MultiLayerNeighborSampler``.
    Produces dual embeddings (source, target) for the output nodes.

    Parameters
    ----------
    cfg : DAEMONConfig
        Model configuration.
    """
    def __init__(self, cfg: DAEMONConfig):
        super().__init__()
        self.cfg = cfg

        # Input projection (if features differ from hidden dim)
        self.input_proj: Optional[nn.Linear] = None
        if cfg.in_feats != cfg.hidden_dim:
            self.input_proj = nn.Linear(cfg.in_feats, cfg.hidden_dim)

        # GNN layers
        self.layers = nn.ModuleList()
        for i in range(cfg.num_layers):
            in_dim = cfg.hidden_dim
            out_dim = cfg.hidden_dim
            self.layers.append(
                DAEMONLayer(in_dim, out_dim, dropout=cfg.dropout)
            )

        # Output projection to final embedding dimension
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.out_dim)

    def forward(
        self,
        blocks: List[dgl.DGLBlock],
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        blocks : list of dgl.DGLBlock
            Message-flow computation graphs, one per layer (L=3).
            ``blocks[0].srcdata`` contains the input node features.
        h : torch.Tensor
            Input node features [num_src_nodes, in_feats].

        Returns
        -------
        (h_src_all, h_tgt_all) : Tuple[torch.Tensor, torch.Tensor]
            Source and target embeddings for ALL nodes
            (including those from ``blocks[-1].dstdata``).
        """
        # Input projection
        if self.input_proj is not None:
            h = self.input_proj(h)

        h_src, h_tgt = h, h  # initialise both from same input

        for layer, block in zip(self.layers, blocks):
            h_src, h_tgt = layer(block, h_src, h_tgt)

        # Output projection + L2 norm
        h_src = F.normalize(self.out_proj(h_src), p=2, dim=-1)
        h_tgt = F.normalize(self.out_proj(h_tgt), p=2, dim=-1)

        return h_src, h_tgt


# ============================================================================
# AsymmetricLoss
# ============================================================================

class AsymmetricLoss(nn.Module):
    """Three-component loss for directed graph recommendation (Eq. 2 in paper).

    Components:
        1. Co-purchase likelihood (positive sampling + negative sampling)
        2. Asymmetry enforcement (one-way edge penalty)
        3. Co-view similarity (source + target alignment)

    Parameters
    ----------
    cfg : DAEMONConfig
        Configuration (uses ``num_neg``).
    """
    def __init__(self, cfg: DAEMONConfig):
        super().__init__()
        self.num_neg = cfg.num_neg

    def forward(
        self,
        src_emb: torch.Tensor,
        tgt_emb: torch.Tensor,
        cp_u: torch.Tensor,
        cp_v: torch.Tensor,
        ow_u: torch.Tensor,
        ow_v: torch.Tensor,
        cv_u: torch.Tensor,
        cv_v: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss.

        Parameters
        ----------
        src_emb : torch.Tensor  [N, d]
            Source embeddings for all nodes in batch.
        tgt_emb : torch.Tensor  [N, d]
            Target embeddings for all nodes in batch.
        cp_u, cp_v : torch.Tensor  [E_cp]
            Co-purchase positive pair indices.
        ow_u, ow_v : torch.Tensor  [E_ow]
            One-way edges (for asymmetry component).
        cv_u, cv_v : torch.Tensor  [E_cv]
            Co-view pair indices.
        num_nodes : int
            Total nodes for negative sampling.

        Returns
        -------
        (total_loss, components) : Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            Total loss + per-component breakdown.
        """
        device = src_emb.device
        components = {}

        # ---- Component 1: Co-purchase likelihood ----
        loss_cp = self._co_purchase_loss(src_emb, tgt_emb, cp_u, cp_v, num_nodes)
        components['cp'] = loss_cp.detach()

        # ---- Component 2: Asymmetry enforcement ----
        loss_ow = self._asymmetry_loss(src_emb, tgt_emb, ow_u, ow_v)
        components['ow'] = loss_ow.detach()

        # ---- Component 3: Co-view similarity ----
        loss_cv = self._co_view_loss(src_emb, tgt_emb, cv_u, cv_v)
        components['cv'] = loss_cv.detach()

        total = loss_cp + loss_ow + loss_cv
        return total, components

    def _co_purchase_loss(
        self,
        src_emb: torch.Tensor,
        tgt_emb: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Loss for co-purchase likelihood.

        -log σ(θ_u^s·θ_v^t) - Σ_{z~P_r} log σ(1 - θ_u^s·θ_z^t)
        """
        if u.numel() == 0:
            return torch.tensor(0.0, device=src_emb.device)

        pos = (src_emb[u] * tgt_emb[v]).sum(dim=1)
        loss = -F.logsigmoid(pos).mean()

        # Negative sampling
        neg_z = torch.randint(0, num_nodes, (u.numel(), self.num_neg), device=src_emb.device)
        neg = (src_emb[u].unsqueeze(1) * tgt_emb[neg_z]).sum(dim=2)
        loss = loss + (-F.logsigmoid(-neg)).mean()

        return loss

    def _asymmetry_loss(
        self,
        src_emb: torch.Tensor,
        tgt_emb: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Asymmetry enforcement for one-way edges.

        -log σ(θ_u^s·θ_v^t) - log σ(1 - θ_v^s·θ_u^t)
        """
        if u.numel() == 0:
            return torch.tensor(0.0, device=src_emb.device)

        forward = (src_emb[u] * tgt_emb[v]).sum(dim=1)
        reverse = (src_emb[v] * tgt_emb[u]).sum(dim=1)

        loss = -F.logsigmoid(forward).mean() - F.logsigmoid(-reverse).mean()
        return loss

    def _co_view_loss(
        self,
        src_emb: torch.Tensor,
        tgt_emb: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Co-view similarity loss.

        -log σ(θ_u^s·θ_v^s) - log σ(θ_u^t·θ_v^t)
        """
        if u.numel() == 0:
            return torch.tensor(0.0, device=src_emb.device)

        src_sim = (src_emb[u] * src_emb[v]).sum(dim=1)
        tgt_sim = (tgt_emb[u] * tgt_emb[v]).sum(dim=1)

        loss = -F.logsigmoid(src_sim).mean() - F.logsigmoid(tgt_sim).mean()
        return loss


# ============================================================================
# Utility: count parameters
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



