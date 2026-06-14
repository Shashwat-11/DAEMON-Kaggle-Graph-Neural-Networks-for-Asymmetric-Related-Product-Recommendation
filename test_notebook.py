#!/usr/bin/env python3
"""
test_notebook.py — Comprehensive test script for DAEMON-Kaggle project.

Tests:
1. Module imports (all 4 src modules)
2. Synthetic graph generation and DGL graph construction
3. DAEMONModel forward pass shape verification
4. AsymmetricLoss forward with correct arguments
5. Training functions (setup_training, train_epoch, train_model)
6. Evaluation functions (evaluate_full, compute_* metrics)
7. Device placement consistency
8. Edge case handling (empty edges, missing types)

Usage:
    python test_notebook.py

Exits with code 0 if ALL tests pass, 1 if any test fails.
"""

import sys
import os
import math
import traceback
import warnings
from typing import Any, Dict, List, Optional, Tuple

import inspect
import numpy as np

# ============================================================================
# Test configuration
# ============================================================================
NUM_SYNTHETIC_NODES = 1000
FEATURE_DIM = 384
HIDDEN_DIM = 128
OUT_DIM = 64
BATCH_SIZE = 64
NUM_EPOCHS = 2
DEVICE = "cpu"  # Test on CPU to avoid CUDA dependency

# ============================================================================
# Test result tracking
# ============================================================================
passed = 0
failed = 0
errors: List[str] = []


def test(name: str, condition: bool, detail: str = "") -> None:
    """Assert a test condition and track pass/fail."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ PASS: {name}")
    else:
        failed += 1
        msg = f"  ❌ FAIL: {name}" + (f" — {detail}" if detail else "")
        print(msg)
        errors.append(msg)


def test_raises(name: str, fn, exception_type=Exception) -> None:
    """Assert that a function raises a specific exception."""
    global passed, failed
    try:
        fn()
        failed += 1
        msg = f"  ❌ FAIL: {name} — expected {exception_type.__name__} but no exception raised"
        print(msg)
        errors.append(msg)
    except exception_type:
        passed += 1
        print(f"  ✅ PASS: {name} (raised {exception_type.__name__} as expected)")
    except Exception as e:
        failed += 1
        msg = f"  ❌ FAIL: {name} — expected {exception_type.__name__} but got {type(e).__name__}: {e}"
        print(msg)
        errors.append(msg)


# ============================================================================
# 1. Import all modules
# ============================================================================
print("=" * 70)
print("  TEST 1: Module imports")
print("=" * 70)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Try importing DGL first (critical dependency)
try:
    import dgl
    from dgl.dataloading import DataLoader as DGLDataLoader
    from dgl.dataloading import MultiLayerNeighborSampler
    print("  DGL imported successfully")
except ImportError as e:
    print(f"  ⚠️  DGL import failed: {e}")
    print("  Some tests will be skipped.")
    dgl = None  # type: ignore
    DGLDataLoader = None  # type: ignore
    MultiLayerNeighborSampler = None  # type: ignore

try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast, GradScaler
    print(f"  PyTorch {torch.__version__} imported")
except ImportError as e:
    print(f"  CRITICAL: PyTorch import failed: {e}")
    sys.exit(1)

# Import DAEMON modules
import_results: Dict[str, bool] = {}

try:
    from daemon_model import (
        DAEMONConfig,
        DAEMONModel,
        DAEMONLayer,
        AsymmetricLoss,
        count_parameters,
    )
    import_results["daemon_model"] = True
    print("  ✅ daemon_model.py — all symbols found")
except ImportError as e:
    import_results["daemon_model"] = False
    print(f"  ❌ daemon_model.py import failed: {e}")

try:
    from data_pipeline import (
        generate_synthetic_graph,
        build_product_graph,
        split_edges_by_type,
        find_one_way_edges,
        print_graph_stats,
        NegativeSampler,
        validate_graph,
        generate_eval_negatives,
    )
    import_results["data_pipeline"] = True
    print("  ✅ data_pipeline.py — all symbols found")
except ImportError as e:
    import_results["data_pipeline"] = False
    print(f"  ❌ data_pipeline.py import failed: {e}")

try:
    from training import (
        setup_training,
        train_model,
        train_epoch,
        save_checkpoint,
        load_checkpoint,
        memory_summary,
        print_memory_summary,
        log_epoch,
        generate_embeddings,
        _build_default_batch_data,
    )
    import_results["training"] = True
    print("  ✅ training.py — all symbols found")
except ImportError as e:
    import_results["training"] = False
    print(f"  ❌ training.py import failed: {e}")

try:
    from evaluation import (
        generate_all_embeddings,
        build_faiss_index,
        evaluate_full,
        recommend_related,
        cold_start_recommend,
        compute_hit_rate_at_k,
        compute_mrr_at_k,
        compute_link_prediction_auc,
        compute_direction_prediction_auc,
        _ensure_2col,
        _has_edge,
    )
    import_results["evaluation"] = True
    print("  ✅ evaluation.py — all symbols found")
except ImportError as e:
    import_results["evaluation"] = False
    print(f"  ❌ evaluation.py import failed: {e}")

test("All 4 src modules importable",
     all(import_results.values()),
     f"Failed modules: {[k for k, v in import_results.items() if not v]}")

# ============================================================================
# 2. Check function signatures match between modules and notebook usage
# ============================================================================
print("\n" + "=" * 70)
print("  TEST 2: API signature consistency")
print("=" * 70)

if import_results["daemon_model"]:
    # Check AsymmetricLoss.forward signature
    import inspect
    asig = inspect.signature(AsymmetricLoss.forward)
    asig_params = list(asig.parameters.keys())
    # Should be: self, src_emb, tgt_emb, cp_u, cp_v, ow_u, ow_v, cv_u, cv_v, num_nodes
    expected_params = ['self', 'src_emb', 'tgt_emb', 'cp_u', 'cp_v', 'ow_u', 'ow_v', 'cv_u', 'cv_v', 'num_nodes']
    test("AsymmetricLoss.forward() has correct params",
         asig_params == expected_params,
         f"Got {asig_params}")

    # Check DAEMONModel.forward signature
    msig = inspect.signature(DAEMONModel.forward)
    mparams = list(msig.parameters.keys())
    test("DAEMONModel.forward() has correct params",
         mparams == ['self', 'blocks', 'h'],
         f"Got {mparams}")

if import_results["data_pipeline"]:
    # Check generate_synthetic_graph return type
    gsig = inspect.signature(generate_synthetic_graph)
    test("generate_synthetic_graph has correct params",
         'num_products' in gsig.parameters and 'feature_dim' in gsig.parameters,
         f"Got {list(gsig.parameters.keys())}")

    # Check split_edges_by_type return keys
    ssig = inspect.signature(split_edges_by_type)
    test("split_edges_by_type has correct params",
         'train_ratio' in ssig.parameters and 'val_ratio' in ssig.parameters,
         f"Got {list(ssig.parameters.keys())}")

    # Check build_product_graph params
    bsig = inspect.signature(build_product_graph)
    test("build_product_graph has correct params",
         all(k in bsig.parameters for k in ['product_df', 'cp_edges', 'cv_edges', 'features']),
         f"Got {list(bsig.parameters.keys())}")

if import_results["training"]:
    # Check setup_training signature
    stsig = inspect.signature(setup_training)
    test("setup_training has correct params (cfg, model_class, criterion_class)",
         list(stsig.parameters.keys()) == ['cfg', 'model_class', 'criterion_class'],
         f"Got {list(stsig.parameters.keys())}")

    # Check train_model signature
    tmsig = inspect.signature(train_model)
    expected_tm = ['model', 'train_loader', 'val_data', 'criterion', 'optimizer',
                   'scheduler', 'scaler', 'cfg', 'resume_epoch']
    test("train_model has correct params",
         list(tmsig.parameters.keys()) == expected_tm,
         f"Got {list(tmsig.parameters.keys())}")

if import_results["evaluation"]:
    # Check evaluate_full signature
    efsig = inspect.signature(evaluate_full)
    expected_ef = ['model', 'g', 'masks', 'cfg', 'device', 'batch_size']
    test("evaluate_full has correct params",
         list(efsig.parameters.keys()) == expected_ef,
         f"Got {list(efsig.parameters.keys())}")

# ============================================================================
# 3. Check for hardcoded "cuda" and device issues
# ============================================================================
print("\n" + "=" * 70)
print("  TEST 3: Device placement analysis")
print("=" * 70)

# Check training.py for hardcoded "cuda"
training_path = os.path.join(os.path.dirname(__file__), "src", "training.py")
if os.path.exists(training_path):
    with open(training_path) as f:
        training_code = f.read()
    
    # Check for .cuda() calls (not via .to())
    cuda_calls = []
    for i, line in enumerate(training_code.split('\n'), 1):
        if '.cuda()' in line and not line.strip().startswith('#'):
            cuda_calls.append((i, line.strip()))
    
    test("training.py has no hardcoded .cuda() calls",
         len(cuda_calls) == 0,
         f"Found {len(cuda_calls)} hardcoded .cuda() calls: {cuda_calls}")
    
    # Check for hardcoded "cuda" string
    hardcoded_cuda = []
    for i, line in enumerate(training_code.split('\n'), 1):
        if '"cuda"' in line or "'cuda'" in line:
            hardcoded_cuda.append((i, line.strip()))
    
    # Line 120 has "cuda" hardcoded in train_epoch
    cuda_in_train_epoch = any('to("cuda")' in line or "to('cuda')" in line for _, line in hardcoded_cuda)
    test("training.py doesn't hardcode 'cuda' in .to()",
         not cuda_in_train_epoch,
         "Found hardcoded 'cuda' in train_epoch() - should use model device")

# Check evaluation.py for hardcoded "cuda"
eval_path = os.path.join(os.path.dirname(__file__), "src", "evaluation.py")
if os.path.exists(eval_path):
    with open(eval_path) as f:
        eval_code = f.read()
    
    # Check for g.edges() used without .to(device) before passing to metric functions
    print("  (evaluation.py device analysis - see runtime tests)")

# ============================================================================
# 4. Create synthetic graph and build DGL graph
# ============================================================================
print("\n" + "=" * 70)
print("  TEST 4: Synthetic graph generation + DGL construction")
print("=" * 70)

if dgl is None or not all(import_results.get(m, False) for m in ['daemon_model', 'data_pipeline']):
    print("  ⚠️  Skipping graph tests — missing dependencies")
else:
    import pandas as pd

    # 4a. Generate synthetic graph
    print(f"  Generating synthetic graph ({NUM_SYNTHETIC_NODES} nodes) ...")
    try:
        edges_cp, edges_cv, features_np, categories = generate_synthetic_graph(
            num_products=NUM_SYNTHETIC_NODES,
            feature_dim=FEATURE_DIM,
            avg_cp_degree=5,
            avg_cv_degree=8,
            asymmetry_ratio=0.75,
            seed=42,
        )
        test("generate_synthetic_graph returns 4 values",
             len([edges_cp, edges_cv, features_np, categories]) == 4)
        test("edges_cp is 2D with shape [2, E]",
             edges_cp.ndim == 2 and edges_cp.shape[0] == 2,
             f"Got shape {edges_cp.shape}")
        test("edges_cv is 2D with shape [2, E]",
             edges_cv.ndim == 2 and edges_cv.shape[0] == 2,
             f"Got shape {edges_cv.shape}")
        test("features_np has correct shape",
             features_np.shape == (NUM_SYNTHETIC_NODES, FEATURE_DIM),
             f"Got shape {features_np.shape}")
        test("categories is 1D",
             categories.ndim == 1 and categories.shape[0] == NUM_SYNTHETIC_NODES,
             f"Got shape {categories.shape}")
    except Exception as e:
        test("generate_synthetic_graph succeeds", False, str(e))
        traceback.print_exc()
        edges_cp = edges_cv = features_np = categories = None  # type: ignore

    # 4b. Build product graph
    if edges_cp is not None:
        try:
            product_df = pd.DataFrame({
                "title": [f"product_{i}" for i in range(NUM_SYNTHETIC_NODES)],
            })
            product_df["description"] = product_df["title"]
            product_df["category"] = "default"

            g, _ = build_product_graph(
                product_df=product_df,
                cp_edges=edges_cp,
                cv_edges=edges_cv,
                feature_dim=FEATURE_DIM,
                features=features_np,
            )
            test("build_product_graph returns DGL graph",
                 hasattr(g, 'num_nodes') and hasattr(g, 'num_edges'))
            test("graph has correct num_nodes",
                 g.num_nodes() == NUM_SYNTHETIC_NODES,
                 f"Got {g.num_nodes()}")
            test("graph has > 0 edges",
                 g.num_edges() > 0,
                 f"Got {g.num_edges()} edges")
            test("graph has ndata['feat']",
                 'feat' in g.ndata,
                 f"Keys: {list(g.ndata.keys())}")
            test("graph has edata['type']",
                 'type' in g.edata,
                 f"Keys: {list(g.edata.keys())}")
            test("feature shape matches",
                 g.ndata['feat'].shape == (NUM_SYNTHETIC_NODES, FEATURE_DIM),
                 f"Got {g.ndata['feat'].shape}")
        except Exception as e:
            test("build_product_graph succeeds", False, str(e))
            traceback.print_exc()
            g = None  # type: ignore

# ============================================================================
# 5. Edge splitting and validation
# ============================================================================
print("\n" + "=" * 70)
print("  TEST 5: Edge splitting and graph validation")
print("=" * 70)

if dgl is not None and import_results.get("data_pipeline") and g is not None:
    try:
        # 5a. Split edges
        split = split_edges_by_type(g, train_ratio=0.75, val_ratio=0.05)
        
        expected_keys = {'train_cp', 'train_cv', 'val_cp', 'val_cv', 'test_cp', 'test_cv'}
        test("split_edges_by_type returns all expected keys",
             set(split.keys()) == expected_keys,
             f"Got keys {set(split.keys())}")
        
        train_eids = torch.cat([split["train_cp"], split["train_cv"]])
        val_eids = torch.cat([split["val_cp"], split["val_cv"]])
        test_eids = torch.cat([split["test_cp"], split["test_cv"]])
        
        total_edges = len(train_eids) + len(val_eids) + len(test_eids)
        test("split covers all edges",
             total_edges == g.num_edges(),
             f"Split sum {total_edges} != total {g.num_edges()}")
        
        # Check for overlap
        train_set = set(train_eids.tolist())
        val_set = set(val_eids.tolist())
        test_set_set = set(test_eids.tolist())
        
        test("train/val have no overlap",
             len(train_set & val_set) == 0,
             f"Overlap: {len(train_set & val_set)}")
        test("train/test have no overlap",
             len(train_set & test_set_set) == 0,
             f"Overlap: {len(train_set & test_set_set)}")
        test("val/test have no overlap",
             len(val_set & test_set_set) == 0,
             f"Overlap: {len(val_set & test_set_set)}")
        
        # 5b. Validate graph
        try:
            result = validate_graph(g)
            test("validate_graph returns True", result is True)
        except AssertionError as e:
            test("validate_graph passes", False, str(e))
        
        # 5c. Find one-way edges
        one_way_u, one_way_v = find_one_way_edges(g)
        test("find_one_way_edges returns two tensors",
             isinstance(one_way_u, torch.Tensor) and isinstance(one_way_v, torch.Tensor))
        test("one-way edges have matching lengths",
             len(one_way_u) == len(one_way_v),
             f"len(ow_u)={len(one_way_u)}, len(ow_v)={len(one_way_v)}")
        
        # 5d. Negative sampler
        neg_sampler = NegativeSampler(num_nodes=g.num_nodes(), num_neg=5, device=DEVICE)
        neg_samples = neg_sampler.sample(10)
        test("NegativeSampler returns correct shape",
             neg_samples.shape == (10, 5),
             f"Got shape {neg_samples.shape}")
        test("NegativeSampler samples in valid range",
             neg_samples.min() >= 0 and neg_samples.max() < g.num_nodes(),
             f"Range: [{neg_samples.min()}, {neg_samples.max()}]")
        
        # 5e. Generate eval negatives
        pos_edges = torch.stack(g.edges(), dim=1)[test_eids]  # [E_test, 2]
        if pos_edges.shape[0] > 0:
            eval_neg = generate_eval_negatives(
                g, pos_edges.T, num_neg_ratio=1  # type: ignore
            )
            test("generate_eval_negatives returns 2-row tensor",
                 eval_neg.shape[0] == 2,
                 f"Got shape {eval_neg.shape}")
        
        # 5f. Build train subgraph + DataLoader
        train_g = dgl.edge_subgraph(g, train_eids, relabel_nodes=False)
        sampler = MultiLayerNeighborSampler([5, 5])
        train_loader = DGLDataLoader(
            train_g,
            torch.arange(train_g.num_nodes()),
            sampler,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            device=DEVICE,
            use_uva=False,
        )
        n_batches = math.ceil(train_g.num_nodes() / BATCH_SIZE)
        test("DataLoader yields batches",
             n_batches > 0,
             f"Expected ~{n_batches} batches")
        
    except Exception as e:
        test("edge splitting suite", False, str(e))
        traceback.print_exc()

# ============================================================================
# 6. DAEMONModel forward pass shape verification
# ============================================================================
print("\n" + "=" * 70)
print("  TEST 6: Model forward pass shape verification")
print("=" * 70)

if dgl is not None and all(import_results.get(m, False) for m in ['daemon_model', 'data_pipeline']) and g is not None:
    try:
        cfg = DAEMONConfig(
            in_feats=FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            out_dim=OUT_DIM,
            num_layers=2,  # Use 2 layers for faster test
            dropout=0.1,
            num_neighbors=(5, 5),
        )
        cfg.num_nodes = g.num_nodes()
        cfg.num_edges = g.num_edges()

        model = DAEMONModel(cfg).to(DEVICE)
        test("DAEMONModel instantiates", True)
        
        n_params = count_parameters(model)
        test("count_parameters returns > 0", n_params > 0, f"Got {n_params}")
        
        # 6a. Test forward pass with a single batch
        sampler = MultiLayerNeighborSampler([5, 5])
        loader = DGLDataLoader(
            g,
            torch.arange(min(g.num_nodes(), 100)),  # First 100 nodes
            sampler,
            batch_size=10,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            device=DEVICE,
        )
        
        model.eval()
        found_batch = False
        with torch.no_grad():
            for batch in loader:
                input_nodes, output_nodes, blocks = batch
                blocks = [b.to(DEVICE) for b in blocks]
                batch_inputs = blocks[0].srcdata["feat"]
                
                src_emb, tgt_emb = model(blocks, batch_inputs)
                
                test("src_emb is 2D",
                     src_emb.dim() == 2,
                     f"Got shape {src_emb.shape}")
                test("tgt_emb is 2D",
                     tgt_emb.dim() == 2,
                     f"Got shape {tgt_emb.shape}")
                test("src_emb has out_dim columns",
                     src_emb.shape[1] == OUT_DIM,
                     f"Got {src_emb.shape[1]}, expected {OUT_DIM}")
                test("tgt_emb has out_dim columns",
                     tgt_emb.shape[1] == OUT_DIM,
                     f"Got {tgt_emb.shape[1]}, expected {OUT_DIM}")
                test("src_emb has same number of rows as output_nodes",
                     src_emb.shape[0] == len(output_nodes),
                     f"Got {src_emb.shape[0]} rows, expected {len(output_nodes)}")
                test("tgt_emb has same number of rows as output_nodes",
                     tgt_emb.shape[0] == len(output_nodes),
                     f"Got {tgt_emb.shape[0]} rows, expected {len(output_nodes)}")
                
                # Check L2 normalization
                src_norms = src_emb.norm(dim=1)
                tgt_norms = tgt_emb.norm(dim=1)
                test("src_emb is L2-normalized (close to 1.0)",
                     torch.allclose(src_norms, torch.ones_like(src_norms), atol=1e-5),
                     f"Mean norm: {src_norms.mean().item():.4f}")
                test("tgt_emb is L2-normalized (close to 1.0)",
                     torch.allclose(tgt_norms, torch.ones_like(tgt_norms), atol=1e-5),
                     f"Mean norm: {tgt_norms.mean().item():.4f}")
                
                found_batch = True
                break
        
        test("DataLoader yields at least one batch", found_batch)
        
    except Exception as e:
        test("model forward pass suite", False, str(e))
        traceback.print_exc()

# ============================================================================
# 7. AsymmetricLoss forward pass test
# ============================================================================
print("\n" + "=" * 70)
print("  TEST 7: AsymmetricLoss forward pass")
print("=" * 70)

if import_results.get("daemon_model") and dgl is not None and 'g' in dir() and g is not None:
    try:
        criterion = AsymmetricLoss(cfg).to(DEVICE)
        
        # Create dummy embeddings
        N = 128  # batch size
        src_emb = torch.randn(N, OUT_DIM, device=DEVICE)
        src_emb = src_emb / src_emb.norm(dim=1, keepdim=True)
        tgt_emb = torch.randn(N, OUT_DIM, device=DEVICE)
        tgt_emb = tgt_emb / tgt_emb.norm(dim=1, keepdim=True)
        
        # Test with all three components having data
        cp_u = torch.randint(0, N, (20,), device=DEVICE)
        cp_v = torch.randint(0, N, (20,), device=DEVICE)
        ow_u = torch.randint(0, N, (10,), device=DEVICE)
        ow_v = torch.randint(0, N, (10,), device=DEVICE)
        cv_u = torch.randint(0, N, (15,), device=DEVICE)
        cv_v = torch.randint(0, N, (15,), device=DEVICE)
        
        loss, components = criterion(src_emb, tgt_emb, cp_u, cp_v, ow_u, ow_v, cv_u, cv_v, N)
        
        test("AsymmetricLoss returns finite loss",
             torch.isfinite(loss).item(),
             f"Loss = {loss.item()}")
        test("Loss has all 3 components",
             set(components.keys()) == {'cp', 'ow', 'cv'},
             f"Got keys: {set(components.keys())}")
        test("All components are finite",
             all(torch.isfinite(v).item() for v in components.values()),
             f"Components: { {k: v.item() for k, v in components.items()} }")
        
        # Test with empty components
        loss_empty, comp_empty = criterion(
            src_emb, tgt_emb,
            torch.empty(0, dtype=torch.long, device=DEVICE),
            torch.empty(0, dtype=torch.long, device=DEVICE),
            torch.empty(0, dtype=torch.long, device=DEVICE),
            torch.empty(0, dtype=torch.long, device=DEVICE),
            torch.empty(0, dtype=torch.long, device=DEVICE),
            torch.empty(0, dtype=torch.long, device=DEVICE),
            N,
        )
        test("Empty edges return zero loss",
             loss_empty.item() == 0.0,
             f"Loss = {loss_empty.item()}")
        
        # Test device consistency
        test("Loss is on correct device",
             loss.device.type == DEVICE,
             f"Got {loss.device}")
        
    except Exception as e:
        test("AsymmetricLoss suite", False, str(e))
        traceback.print_exc()

# ============================================================================
# 8. Training function tests
# ============================================================================
print("\n" + "=" * 70)
print("  TEST 8: Training functions")
print("=" * 70)

if all(import_results.get(m, False) for m in ['daemon_model', 'training']) and dgl is not None and 'g' in dir() and g is not None:
    try:
        # 8a. setup_training
        print("  8a. Testing setup_training() ...")
        train_setup = setup_training(cfg, DAEMONModel, AsymmetricLoss)
        
        expected_keys = {'model', 'optimizer', 'criterion', 'scheduler', 'scaler', 'start_epoch'}
        test("setup_training returns all expected keys",
             set(train_setup.keys()) == expected_keys,
             f"Got keys: {set(train_setup.keys())}")
        test("setup_training['model'] is nn.Module",
             isinstance(train_setup['model'], nn.Module))
        test("setup_training['start_epoch'] is 0",
             train_setup['start_epoch'] == 0)
        
        model = train_setup['model']
        optimizer = train_setup['optimizer']
        criterion = train_setup['criterion']
        scheduler = train_setup['scheduler']
        scaler = train_setup['scaler']
        
        # 8b. Test _build_default_batch_data for edge-type awareness
        print("  8b. Testing _build_default_batch_data() ...")
        # Build a block with mixed edge types
        sampler = MultiLayerNeighborSampler([5])
        loader = DGLDataLoader(
            g,
            torch.arange(min(g.num_nodes(), 20)),
            sampler,
            batch_size=5,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            device=DEVICE,
        )
        
        for batch in loader:
            _in, _out, blocks = batch
            blocks = [b.to(DEVICE) for b in blocks]
            batch_data = blocks[-1].dstdata.get("batch_data", None)
            if batch_data is None:
                batch_data = _build_default_batch_data(blocks[-1])
            
            # Check what keys it returns
            test("batch_data contains all expected keys",
                 all(k in batch_data for k in ['cp_u', 'cp_v', 'ow_u', 'ow_v', 'cv_u', 'cv_v', 'num_nodes']),
                 f"Got keys: {set(batch_data.keys())}")
            
            # CRITICAL BUG CHECK: _build_default_batch_data should respect edge types
            # but it currently treats ALL edges as co-purchase
            blk = blocks[-1]
            if 'type' in blk.edata:
                num_cp = (blk.edata['type'] == 0).sum().item()
                num_cv = (blk.edata['type'] == 1).sum().item()
                has_cv = num_cv > 0
                batch_has_cv = batch_data['cv_u'].shape[0] > 0
                if has_cv:
                    test("batch_data includes CV edges when block has CV edges",
                         batch_has_cv,
                         f"Block has {num_cv} CV edges but batch_data has 0 CV edges (BUG!)")
            
            # Check num_nodes is reasonable
            test("batch_data['num_nodes'] > 0",
                 batch_data['num_nodes'] > 0,
                 f"Got {batch_data['num_nodes']}")
            break
        
        # 8c. Test train_epoch (single epoch on small graph)
        print("  8c. Testing train_epoch() ...")
        train_g = dgl.edge_subgraph(g, train_eids[:min(len(train_eids), 200)], relabel_nodes=False)
        small_loader = DGLDataLoader(
            train_g,
            torch.arange(train_g.num_nodes()),
            MultiLayerNeighborSampler([3, 3]),
            batch_size=32,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            device=DEVICE,
        )
        
        # Check that train_epoch accepts the signature correctly
        # Note: train_epoch hardcodes "cuda" in blocks.to("cuda"), which will fail on CPU
        # So we test with a wrapped version that fixes the device
        try:
            avg_loss = train_epoch(
                model=model,
                loader=small_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                cfg=cfg,
                epoch_idx=0,
            )
            test("train_epoch produces finite loss",
                 math.isfinite(avg_loss),
                 f"Loss = {avg_loss}")
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                test("train_epoch (noted: hardcoded 'cuda' would fail on CPU)", False,
                     "BUG: train_epoch hardcodes 'cuda' at line 120 - fix needed")
            else:
                test("train_epoch raises error", False, str(e))
        
        # 8d. Test train_model
        print("  8d. Testing train_model() ...")
        # Reset model and optimizer for clean training
        model2 = DAEMONModel(cfg).to(DEVICE)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=5)
        scaler2 = GradScaler(enabled=False)
        
        # Create masks
        val_mask = torch.zeros(g.num_edges(), dtype=torch.bool)
        val_mask[val_eids[:min(len(val_eids), 10)]] = True  # Use first 10 val edges
        masks2 = {"val": val_mask}
        
        # Run train_model with resume_epoch=0, 1 epoch
        try:
            history, best = train_model(
                model=model2,
                train_loader=small_loader,
                val_data=(g, masks2),
                criterion=criterion,
                optimizer=opt2,
                scheduler=sched2,
                scaler=scaler2,
                cfg=cfg,
                resume_epoch=0,
            )
            test("train_model returns history with train_loss",
                 'train_loss' in history and len(history['train_loss']) > 0,
                 f"Got keys: {set(history.keys())}")
            test("train_model returns best_metrics with auc",
                 'auc' in best,
                 f"Got keys: {set(best.keys())}")
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                test("train_model (noted: device issue)", False,
                     "BUG: device mismatch in validate() or hardcoded 'cuda' in train_epoch")
            else:
                test("train_model completes", False, str(e))
                traceback.print_exc()
        except Exception as e:
            test("train_model completes", False, str(e))
            traceback.print_exc()
        
    except Exception as e:
        test("training suite", False, str(e))
        traceback.print_exc()

# ============================================================================
# 9. Evaluation function tests
# ============================================================================
print("\n" + "=" * 70)
print("  TEST 9: Evaluation functions (device consistency)")
print("=" * 70)

if all(import_results.get(m, False) for m in ['evaluation', 'daemon_model']) and dgl is not None and 'g' in dir() and g is not None:
    try:
        # 9a. compute_hit_rate_at_k and compute_mrr_at_k
        print("  9a. Testing compute_hit_rate_at_k / compute_mrr_at_k ...")
        N = 100
        d = OUT_DIM
        src_emb = torch.randn(N, d)
        src_emb = src_emb / src_emb.norm(dim=1, keepdim=True)
        tgt_emb = torch.randn(N, d)
        tgt_emb = tgt_emb / tgt_emb.norm(dim=1, keepdim=True)
        
        query_ids = torch.arange(10)
        true_ids = torch.arange(10) + 5  # offset so they're not position 0
        
        hr10 = compute_hit_rate_at_k(src_emb, tgt_emb, query_ids, true_ids, k=10)
        test("compute_hit_rate_at_k returns float",
             isinstance(hr10, float),
             f"Got {type(hr10)}: {hr10}")
        test("compute_hit_rate_at_k in valid range",
             0.0 <= hr10 <= 1.0,
             f"Got {hr10}")
        
        mrr10 = compute_mrr_at_k(src_emb, tgt_emb, query_ids, true_ids, k=10)
        test("compute_mrr_at_k returns float",
             isinstance(mrr10, float),
             f"Got {type(mrr10)}: {mrr10}")
        test("compute_mrr_at_k in valid range",
             0.0 <= mrr10 <= 1.0,
             f"Got {mrr10}")
        
        # 9b. _ensure_2col
        print("  9b. Testing _ensure_2col ...")
        t2xN = torch.zeros(2, 50)
        tNx2 = torch.zeros(50, 2)
        t2x2 = torch.zeros(2, 2)
        
        test("_ensure_2col transposes [2, N] to [N, 2]",
             _ensure_2col(t2xN).shape == (50, 2))
        test("_ensure_2col leaves [N, 2] unchanged",
             _ensure_2col(tNx2).shape == (50, 2))
        # Edge case: [2, 2] is ambiguous but should not crash
        test("_ensure_2col handles [2, 2]",
             _ensure_2col(t2x2).shape == (2, 2))
        
        # 9c. _has_edge with batch testing
        print("  9c. Testing _has_edge ...")
        if g.num_edges() > 10:
            src, dst = g.edges()
            # Pick some existing edges
            existing_src = src[:5]
            existing_dst = dst[:5]
            # Pick some non-existing pairs (use out-of-range or random)
            non_existing_src = torch.tensor([0, 1, 2])
            non_existing_dst = torch.tensor([NUM_SYNTHETIC_NODES + 10] * 3)  # out of range
            
            # Test with all existing
            result_all_exist = _has_edge(g, existing_src, existing_dst)
            test("_has_edge returns True for existing edges",
                 result_all_exist.all() if result_all_exist.numel() > 0 else False)
            
            # Note: _has_edge has a known bug - it catches exceptions and returns all False
            # for the entire batch if any single edge doesn't exist
            print("  ⚠️  _has_edge BUG NOTE: catches exceptions for entire batch")
        
        # 9d. Test evaluate_full with device handling
        print("  9d. Testing evaluate_full structure ...")
        eval_model = DAEMONModel(cfg).to(DEVICE)
        eval_masks = {"test": test_mask[:min(len(test_mask), 10)]} if 'test_mask' in dir() else None
        
        if eval_masks is not None:
            try:
                # This tests the critical device flow:
                # - generate_all_embeddings returns CPU tensors
                # - evaluate_full moves them to device
                # - but test_edges from g.edges() are CPU
                # - compute_link_prediction_auc receives CPU test_edges but CUDA embeddings
                
                # On CPU this should work fine (everything is CPU)
                results = evaluate_full(
                    model=eval_model,
                    g=g,
                    masks=eval_masks,
                    cfg=cfg,
                    device=DEVICE,
                    batch_size=64,
                )
                test("evaluate_full returns dict",
                     isinstance(results, dict),
                     f"Got {type(results)}")
                test("evaluate_full returns non-empty dict",
                     len(results) > 0,
                     f"Got empty dict")
            except Exception as e:
                test("evaluate_full completes", False, str(e))
                # Check if it's the device mismatch bug
                if 'device' in str(e).lower() or 'cuda' in str(e).lower() or 'same device' in str(e).lower():
                    print("  ⚠️  Detected device mismatch bug in evaluate_full")
                traceback.print_exc()
        
    except Exception as e:
        test("evaluation suite", False, str(e))
        traceback.print_exc()

# ============================================================================
# 10. Edge case tests
# ============================================================================
print("\n" + "=" * 70)
print("  TEST 10: Edge case handling")
print("=" * 70)

if import_results.get("data_pipeline") and import_results.get("daemon_model") and dgl is not None:
    try:
        import pandas as pd
        
        # 10a. Empty co-purchase edges
        print("  10a. Testing empty co-purchase edges ...")
        empty_cp = np.zeros((2, 0), dtype=np.int64)
        cv_with_data = edges_cv if 'edges_cv' in dir() else np.zeros((2, 10), dtype=np.int64)
        
        try:
            g_no_cp, _ = build_product_graph(
                product_df=pd.DataFrame({"title": [f"p{i}" for i in range(NUM_SYNTHETIC_NODES)]}),
                cp_edges=empty_cp,
                cv_edges=cv_with_data,
                feature_dim=FEATURE_DIM,
                features=features_np if 'features_np' in dir() else None,
            )
            test("build_product_graph with 0 CP edges", True)
        except Exception as e:
            test("build_product_graph with 0 CP edges", False, str(e))
        
        # 10b. Empty co-view edges
        print("  10b. Testing empty co-view edges ...")
        cp_with_data = edges_cp if 'edges_cp' in dir() else np.zeros((2, 10), dtype=np.int64)
        empty_cv = np.zeros((2, 0), dtype=np.int64)
        
        try:
            g_no_cv, _ = build_product_graph(
                product_df=pd.DataFrame({"title": [f"p{i}" for i in range(NUM_SYNTHETIC_NODES)]}),
                cp_edges=cp_with_data,
                cv_edges=empty_cv,
                feature_dim=FEATURE_DIM,
                features=features_np if 'features_np' in dir() else None,
            )
            test("build_product_graph with 0 CV edges", True)
        except Exception as e:
            test("build_product_graph with 0 CV edges", False, str(e))
        
        # 10c. Split edges from graph with only one edge type
        print("  10c. Testing split_edges_by_type with one type ...")
        try:
            split_one = split_edges_by_type(g_no_cv, train_ratio=0.75, val_ratio=0.05)
            test("split_edges_by_type with only CP edges returns all keys",
                 set(split_one.keys()) == {'train_cp', 'train_cv', 'val_cp', 'val_cv', 'test_cp', 'test_cv'})
            test("split with only CP has empty CV splits",
                 split_one['train_cv'].numel() == 0 and split_one['val_cv'].numel() == 0 and split_one['test_cv'].numel() == 0,
                 f"train_cv={split_one['train_cv'].numel()}, val_cv={split_one['val_cv'].numel()}, test_cv={split_one['test_cv'].numel()}")
        except Exception as e:
            test("split_edges_by_type with one type", False, str(e))
        
        # 10d. Find one-way edges in graph with no CP edges
        if 'g_no_cp' in dir() and g_no_cp is not None:
            print("  10d. Testing find_one_way_edges with no CP edges ...")
            try:
                ow_u, ow_v = find_one_way_edges(g_no_cp)
                test("find_one_way_edges with no CP returns empty tensors",
                     ow_u.numel() == 0 and ow_v.numel() == 0,
                     f"Got {ow_u.numel()} one-way edges")
            except Exception as e:
                test("find_one_way_edges with no CP", False, str(e))
        
        # 10e. AsymmetricLoss with empty tensors
        if import_results.get("daemon_model"):
            print("  10e. Testing AsymmetricLoss with all-empty edges ...")
            criterion_empty = AsymmetricLoss(cfg).to(DEVICE)
            empty_src = torch.randn(10, OUT_DIM, device=DEVICE)
            empty_tgt = torch.randn(10, OUT_DIM, device=DEVICE)
            empty = torch.empty(0, dtype=torch.long, device=DEVICE)
            
            try:
                loss_all_empty, comp_all_empty = criterion_empty(
                    empty_src, empty_tgt, empty, empty, empty, empty, empty, empty, 10
                )
                test("AsymmetricLoss with all-empty edges returns 0",
                     loss_all_empty.item() == 0.0,
                     f"Loss = {loss_all_empty.item()}")
            except Exception as e:
                test("AsymmetricLoss with all-empty edges", False, str(e))
        
    except Exception as e:
        test("edge case suite", False, str(e))
        traceback.print_exc()

# ============================================================================
# 11. Notebook syntax check
# ============================================================================
print("\n" + "=" * 70)
print("  TEST 11: Notebook static analysis")
print("=" * 70)

notebook_path = os.path.join(os.path.dirname(__file__), "daemon_kaggle.ipynb")
if os.path.exists(notebook_path):
    try:
        import json
        
        with open(notebook_path) as f:
            nb = json.load(f)
        
        # Extract all code cells
        code_cells = []
        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                # Skip pip install cells
                if '!pip install' not in source and '!pip3 install' not in source:
                    code_cells.append((i, source))
        
        test("Notebook has code cells (non-pip)", len(code_cells) > 0, f"Found {len(code_cells)} cells")
        
        # Check for indentation errors
        for cell_idx, source in code_cells:
            lines = source.split('\n')
            for line_idx, line in enumerate(lines, 1):
                if line.strip() and not line.strip().startswith('#'):
                    # Check for inconsistent indentation (odd number of extra spaces)
                    stripped = line.lstrip()
                    indent = len(line) - len(stripped)
                    if indent > 0 and indent % 4 != 0 and indent % 2 != 0:
                        print(f"  ⚠️  Possible indentation issue in cell ~{cell_idx}, line {line_idx}: "
                              f"indent={indent} spaces")
        
        # Check for specific known bugs in the notebook source
        full_source = '\n'.join(code_source for _, code_source in code_cells)
        
        # Bug check: Cell 1c indentation error (extra indented print)
        has_bad_indent = '        print("         Enable a GPU accelerator' in full_source
        test("No indentation error in Cell 1c (CPU warning)",
             not has_bad_indent,
             "BUG: Line 153 has extra indentation - will cause SyntaxError")
        
        # Bug check: Cell 5b uses g.num_nodes() for negative sampling with batch embeddings
        has_g_num_nodes_in_loss = 'g.num_nodes()' in full_source and 'criterion(' in full_source
        # This needs more context to detect properly - look for the specific pattern
        cell_5b_source = ""
        for cell_idx, source in code_cells:
            if 'Smoke test' in source or 'smoke test' in source:
                cell_5b_source = source
                break
        
        if cell_5b_source:
            has_bug_5b = 'g.num_nodes()' in cell_5b_source and 'criterion(' in cell_5b_source
            test("Cell 5b uses correct num_nodes (not g.num_nodes())",
                 not has_bug_5b,
                 "BUG: Cell 5b passes g.num_nodes() to criterion but embeddings are batch-sized")
        
        # Check for missing validate_graph import
        has_validate = 'validate_graph(g)' in full_source
        test("Notebook calls validate_graph",
             has_validate,
             "Missing validate_graph call after graph construction")
        
    except json.JSONDecodeError as e:
        test("Notebook JSON is valid", False, str(e))
    except Exception as e:
        test("notebook analysis", False, str(e))

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("  TEST SUMMARY")
print("=" * 70)
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Total:  {passed + failed}")
print("=" * 70)

if failed > 0:
    print("\n  ERRORS FOUND:")
    for i, err in enumerate(errors, 1):
        print(f"  {i}. {err}")
    print(f"\n  ❌ {failed} test(s) FAILED — see above for details")
    sys.exit(1)
else:
    print("\n  ✅ ALL TESTS PASSED")
    sys.exit(0)
