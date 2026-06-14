#!/usr/bin/env python3
"""Generate daemon_kaggle.ipynb directly as JSON."""
import json

cells = []

def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': [text]})

def code(text):
    cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 
                  'outputs': [], 'source': [l + '\n' for l in text.split('\n')]})

# ===== CELL GROUP 1: Environment Setup =====
md("## Cell Group 1: Environment Setup\n\nInstall dependencies, import all modules, verify GPU, set seeds.")

code("""# Cell 1a: Install DGL and FAISS
# WARNING: Enable Internet in Kaggle settings (Settings > Internet > ON)

# Detect PyTorch/CUDA for correct DGL wheel
import torch
tv = torch.__version__.split("+")[0]
cv = "cu" + torch.version.cuda.replace(".", "")
dgl_url = f"https://data.dgl.ai/wheels/torch-{tv}/{cv}/repo.html"
print(f"PyTorch {tv}, CUDA {cv}")
print(f"DGL URL: {dgl_url}")

# Install DGL
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "dgl", "-f", dgl_url, "-q"])

# Install FAISS-GPU
subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-gpu", "-q"])

# Core packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "matplotlib", "tqdm", "scikit-learn", "-q"])

# Optional: for real product text features
# subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "-q"])

print("Dependencies installed")""")

code("""# Cell 1b: Imports
import sys, os, json, math, gc, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("__file__")), "src"))
if not os.path.isdir("src"):
    sys.path.insert(0, "/kaggle/input/daemon-src/src")

# Try DGL import
try:
    import dgl
    from dgl.dataloading import DataLoader as DGLDataLoader, MultiLayerNeighborSampler
    from dgl import function as fn
    print(f"DGL {dgl.__version__} imported")
except ImportError as e:
    raise ImportError(f"DGL not found: {e}. Check Cell 1a installation.")

# FAISS import
try:
    import faiss
    print(f"FAISS imported (GPU: {faiss.get_num_gpus()} GPUs)")
except ImportError:
    print("FAISS not available - will use brute-force fallback")

# Project imports
from daemon_model import DAEMONConfig, DAEMONModel, AsymmetricLoss, count_parameters
from data_pipeline import (generate_synthetic_graph, build_product_graph,
                            print_graph_stats, validate_graph, split_edges_by_type,
                            find_one_way_edges, NegativeSampler, estimate_asymmetry)
from training import (setup_training, train_model, load_checkpoint, 
                      memory_summary, log_epoch)
from evaluation import (generate_all_embeddings, build_faiss_index,
                         evaluate_full, recommend_related, cold_start_recommend,
                         compute_hit_rate_at_k, compute_mrr_at_k)

print("All imports successful")""")

code("""# Cell 1c: GPU verification and reproducibility
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("WARNING: Running on CPU - training will be very slow")
    print("Enable GPU accelerator on Kaggle (Accelerator > GPU T4 x2)")

# Reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

memory_summary()""")

# ===== CELL GROUP 2: Configuration =====
md("## Cell Group 2: Configuration\n\nAll hyperparameters in one place.")

code("""# Cell 2: Configuration
cfg = DAEMONConfig(
    num_nodes=0, num_edges=0, num_relations=2,
    in_feats=384, hidden_dim=128, out_dim=64, num_layers=3, dropout=0.1,
    epochs=30, batch_size=1024, num_neighbors=(20, 10, 10),
    lr=1e-4, weight_decay=1e-5, grad_accum_steps=1, use_amp=True, patience=5,
    num_neg=5, hitrate_k=(5, 10, 20), val_every=1,
    data_dir="/kaggle/input/daemon-data",
    output_dir="/kaggle/working/",
    checkpoint_path="/kaggle/working/daemon_best.pt",
    cleanup_every_n_epochs=4,
)

# Local fallback
if not os.path.isdir("/kaggle/working"):
    cfg.output_dir = "./output/"
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.checkpoint_path = "./output/daemon_best.pt"

print("Configuration:")
for f in sorted(cfg.__dataclass_fields__):
    print(f"  {f}: {getattr(cfg, f)}")""")

# ===== CELL GROUP 3: Data =====
md("## Cell Group 3: Data Loading\n\nLoad real data or generate synthetic.")

code("""# Cell 3a: Data loading with synthetic fallback
RUN_SYNTHETIC = True  # Set False for real Kaggle data

if RUN_SYNTHETIC:
    print("Generating synthetic product graph (50K nodes)...")
    cp_edges, cv_edges, features, categories = generate_synthetic_graph(
        num_products=50000, feature_dim=cfg.in_feats,
        avg_cp_degree=5, avg_cv_degree=8, asymmetry_ratio=0.75
    )
    np.random.seed(42)
    print(f"CP edges: {cp_edges.shape[1]:,}, CV edges: {cv_edges.shape[1]:,}")
else:
    import kagglehub
    path = kagglehub.dataset_download("your-dataset/daemon-products")
    features = np.load(f"{path}/features.npy").astype(np.float32)
    cp_edges = np.load(f"{path}/cp_edges.npy")
    cv_edges = np.load(f"{path}/cv_edges.npy")""")

code("""# Cell 3b: Build DGL graph
product_df = pd.DataFrame({
    'title': [f'Product_{i}' for i in range(features.shape[0])],
    'category': [f'Cat_{c}' for c in categories] if 'categories' in dir() 
                  else ['General'] * features.shape[0]
})

g = build_product_graph(product_df, cp_edges, cv_edges, feature_dim=cfg.in_feats)
cfg.num_nodes = g.num_nodes()
cfg.num_edges = g.num_edges()
print_graph_stats(g)
validate_graph(g)""")

code("""# Cell 3c: Save graph
try:
    dgl.save_graphs(f"{cfg.output_dir}/product_graph.bin", [g])
    print(f"Graph saved to {cfg.output_dir}")
except Exception as e:
    print(f"Could not save graph: {e}")""")

# ===== CELL GROUP 4: Graph Splitting =====
md("## Cell Group 4: Graph Splitting and DataLoaders")

code("""# Cell 4a: Split edges
split = split_edges_by_type(g, train_ratio=0.75, val_ratio=0.05)
print(f"Train CP: {len(split['train_cp']):,}, CV: {len(split['train_cv']):,}")
print(f"Val CP:   {len(split['val_cp']):,}, CV: {len(split['val_cv']):,}")
print(f"Test CP:  {len(split['test_cp']):,}, CV: {len(split['test_cv']):,}")

train_eids = torch.cat([split['train_cp'], split['train_cv']])
train_g = dgl.edge_subgraph(g, train_eids, relabel_nodes=False)

one_way_u, one_way_v = find_one_way_edges(train_g)
print(f"One-way CP edges: {len(one_way_u):,}")""")

code("""# Cell 4b: Create DataLoaders
sampler = MultiLayerNeighborSampler(list(cfg.num_neighbors))
train_seeds = torch.arange(cfg.num_nodes)
train_loader = DGLDataLoader(
    train_g, train_seeds, sampler,
    batch_size=cfg.batch_size, shuffle=True, drop_last=False,
    num_workers=0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"Training batches: {len(train_loader)}")

neg_sampler = NegativeSampler(
    num_nodes=cfg.num_nodes, num_neg=cfg.num_neg,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)""")

code("""# Cell 4c: Test single batch
test_inputs, test_outputs, test_blocks = next(iter(train_loader))
test_blocks = [b.to('cuda') for b in test_blocks]
print(f"Subgraph: {test_blocks[0].num_src_nodes()} src -> {test_blocks[-1].num_dst_nodes()} dst")
print(f"Features: {test_blocks[0].srcdata['feat'].shape}")
cp_n = (test_blocks[-1].edata['type'] == 0).sum().item()
cv_n = (test_blocks[-1].edata['type'] == 1).sum().item()
print(f"Edges: {cp_n} CP + {cv_n} CV")
memory_summary()""")

# ===== CELL GROUP 5: Model =====
md("## Cell Group 5: Model Definition")

code("""# Cell 5a: Instantiate model
model = DAEMONModel(cfg)
criterion = AsymmetricLoss(cfg)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

n_params = count_parameters(model)
print(f"DAEMON parameters: {n_params:,}")
print(f"Model size: {sum(p.numel() * 4 for p in model.parameters() if p.requires_grad) / 1e6:.1f} MB")
memory_summary()""")

code("""# Cell 5b: Smoke test forward pass
model.train()
test_blocks_gpu = [b.to(device) for b in test_blocks]
h = test_blocks_gpu[0].srcdata['feat']

with autocast(enabled=cfg.use_amp):
    src_emb, tgt_emb = model(test_blocks_gpu, h)
    block = test_blocks_gpu[-1]
    dst_nodes = block.dstdata['_ID']
    
    cp_mask = block.edata['type'] == 0
    cv_mask = block.edata['type'] == 1
    cp_eids = torch.where(cp_mask)[0]
    cv_eids = torch.where(cv_mask)[0]
    
    if len(cp_eids) > 0 and len(cv_eids) > 0:
        cp_src, cp_dst = block.find_edges(cp_eids[:100])
        cv_src, cv_dst = block.find_edges(cv_eids[:100])
        loss, components = criterion(
            src_emb, tgt_emb,
            cp_u=cp_dst[:50], cp_v=cp_dst[:50],
            ow_u=cp_dst[:20], ow_v=cp_dst[:20],
            cv_u=cv_dst[:50], cv_v=cv_dst[:50],
            num_nodes=tgt_emb.size(0)
        )
        print(f"Smoke test OK - Loss: {loss.item():.4f}")
        print(f"  CP: {components['cp'].item():.4f}")
        print(f"  Asym: {components['ow'].item():.4f}")
        print(f"  CV: {components['cv'].item():.4f}")
    else:
        print("Smoke test skipped - insufficient edge types in batch")

del test_blocks_gpu, h; gc.collect()
torch.cuda.empty_cache()""")

# ===== CELL GROUP 6: Training =====
md("## Cell Group 6: Training")

code("""# Cell 6a: Training setup
def validate_wrapper(model, val_data, cfg):
    model.eval()
    from sklearn.metrics import roc_auc_score
    with torch.no_grad():
        src_emb, tgt_emb = generate_all_embeddings(model, g, batch_size=4096, device=device)
    
    val_cp_src, val_cp_dst = g.find_edges(split['val_cp'])
    n_val = len(val_cp_src)
    neg_src = torch.randint(0, cfg.num_nodes, (n_val,))
    neg_dst = torch.randint(0, cfg.num_nodes, (n_val,))
    
    pos_scores = (src_emb[val_cp_src] * tgt_emb[val_cp_dst]).sum(dim=1)
    neg_scores = (src_emb[neg_src] * tgt_emb[neg_dst]).sum(dim=1)
    
    scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    labels = np.concatenate([np.ones(n_val), np.zeros(n_val)])
    auc = roc_auc_score(labels, scores)
    val_hr = compute_hit_rate_at_k(src_emb, tgt_emb, val_cp_src, val_cp_dst, k=10)
    model.train()
    return {'val_auc': auc, 'val_hr10': val_hr}

setup = setup_training(cfg, DAEMONModel, AsymmetricLoss, device=device)
model = setup['model']
optimizer = setup['optimizer']
criterion_wrapped = setup['criterion']
scheduler = setup['scheduler']
scaler = setup['scaler']
print(f"Training setup complete. Device: {device}")""")

code("""# Cell 6b: Checkpoint resume
resume_path = cfg.checkpoint_path
start_epoch = 0
if os.path.isfile(resume_path):
    start_epoch, _ = load_checkpoint(
        resume_path, model=model, optimizer=optimizer,
        scaler=scaler, scheduler=scheduler
    )
    print(f"Resumed from checkpoint at epoch {start_epoch}")
else:
    print("No checkpoint found - starting fresh training")""")

code("""# Cell 6c: Train model
history = train_model(
    model=model, train_loader=train_loader, val_data=g,
    criterion=criterion_wrapped, optimizer=optimizer,
    scheduler=scheduler, scaler=scaler, cfg=cfg,
    val_fn=validate_wrapper, start_epoch=start_epoch
)
print(f"Training complete")
print(f"Best val AUC: {history.get('best_val_auc', 'N/A'):.4f}")
print(f"Best val HR@10: {history.get('best_val_hr10', 'N/A'):.4f}")""")

code("""# Cell 6d: Training history plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(history['train_loss'], label='Train Loss', color='#2196F3')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Training Loss'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
if 'val_auc' in history and history['val_auc']:
    ax.plot(history['val_auc'], label='Val AUC', color='#4CAF50', marker='o')
if 'val_hr10' in history:
    ax.plot(history['val_hr10'], label='Val HR@10', color='#FF9800', marker='s')
ax.set_xlabel('Epoch'); ax.set_ylabel('Metric')
ax.set_title('Validation Metrics'); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{cfg.output_dir}/training_history.png", dpi=100, bbox_inches='tight')
plt.show()""")

# ===== CELL GROUP 7: Evaluation =====
md("## Cell Group 7: Evaluation")

code("""# Cell 7a: Load best model
best_path = cfg.checkpoint_path
if os.path.isfile(best_path):
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print("Loaded best model")
else:
    print("Using current model state")""")

code("""# Cell 7b: Generate embeddings
print("Generating embeddings for all products...")
src_emb, tgt_emb = generate_all_embeddings(model, g, batch_size=4096, device=device)
print(f"Source: {src_emb.shape}, Target: {tgt_emb.shape}")
memory_summary()""")

code("""# Cell 7c: Evaluate
from evaluation import compute_link_prediction_auc, compute_direction_prediction_auc

results = {}
cp_test_src, cp_test_dst = g.find_edges(split['test_cp'])

for k in cfg.hitrate_k:
    results[f'HR@{k}'] = compute_hit_rate_at_k(src_emb, tgt_emb, cp_test_src, cp_test_dst, k=k)
    results[f'MRR@{k}'] = compute_mrr_at_k(src_emb, tgt_emb, cp_test_src, cp_test_dst, k=k)

n_test = len(cp_test_src)
neg_src = torch.randint(0, cfg.num_nodes, (n_test,))
neg_dst = torch.randint(0, cfg.num_nodes, (n_test,))
results['AUC'] = compute_link_prediction_auc(
    src_emb, tgt_emb, torch.stack([cp_test_src, cp_test_dst]),
    torch.stack([neg_src, neg_dst])
)

n_ow = min(10000, len(one_way_u))
results['Direction_AUC'] = compute_direction_prediction_auc(
    src_emb, tgt_emb, torch.stack([one_way_u[:n_ow], one_way_v[:n_ow]])
)

print("=" * 60)
print("DAEMON Evaluation Results")
print("=" * 60)
for k in cfg.hitrate_k:
    print(f"  HitRate@{k:2d}: {results[f'HR@{k}']:.4f}")
    print(f"  MRR@{k:2d}:     {results[f'MRR@{k}']:.4f}")
print(f"  AUC:            {results['AUC']:.4f}")
print(f"  Direction AUC:  {results['Direction_AUC']:.4f}")
print("=" * 60)

with open(f"{cfg.output_dir}/metrics.json", 'w') as f:
    json.dump({k: float(v) if isinstance(v, (torch.Tensor, np.floating)) else v 
               for k, v in results.items()}, f, indent=2)""")

# ===== CELL GROUP 8: FAISS =====
md("## Cell Group 8: FAISS Indexing")

code("""# Cell 8a: Build FAISS index
try:
    faiss_index = build_faiss_index(tgt_emb.cpu().numpy(), use_gpu=torch.cuda.is_available())
    print(f"FAISS index: {faiss_index.ntotal:,} vectors x {tgt_emb.shape[1]} dims")
    use_faiss = True
except Exception as e:
    print(f"FAISS failed ({e}), using brute-force search")
    use_faiss = False

if use_faiss:
    query = src_emb[:100].cpu().numpy().astype(np.float32)
    _ = faiss_index.search(query, 10)
    t0 = time.time()
    for _ in range(10):
        faiss_index.search(query, 10)
    latency = (time.time() - t0) / 10 / 100 * 1000
    print(f"FAISS latency: {latency:.2f} ms/query")
    if latency < 100:
        print("Latency OK (< 100ms)")""")

code("""# Cell 8b: Save FAISS index
if use_faiss:
    cpu_index = faiss.index_gpu_to_cpu(faiss_index)
    faiss.write_index(cpu_index, f"{cfg.output_dir}/faiss_index.bin")
    print(f"FAISS index saved")""")

# ===== CELL GROUP 9: Demo =====
md("## Cell Group 9: Recommendation Demo")

code("""# Cell 9: Recommendation demo
sample_queries = list(range(min(10, cfg.num_nodes)))

for q_id in sample_queries[:5]:
    print(f"Query: Product_{q_id}")
    if use_faiss:
        recs = recommend_related(src_emb, faiss_index, q_id, k=5)
        for rank, (rec_id, score) in enumerate(recs, 1):
            print(f"  {rank}. Product_{rec_id} [{score:.3f}]")
    else:
        q_vec = src_emb[q_id]
        scores = q_vec @ tgt_emb.T
        scores[q_id] = -float('inf')
        top_k = torch.topk(scores, 5)
        for rank, (rec_id, score) in enumerate(zip(top_k.indices, top_k.values), 1):
            print(f"  {rank}. Product_{rec_id.item()} [{score.item():.3f}]")
    print()""")

# ===== CELL GROUP 10: Cold-Start =====
md("## Cell Group 10: Cold-Start Demo")

code("""# Cell 10: Cold-start demo
print("Simulating cold-start product...")
new_feat = torch.randn(cfg.in_feats)
existing_feats = g.ndata['feat'].cpu()

try:
    cold_recs = cold_start_recommend(
        model=model, index=faiss_index if use_faiss else None,
        new_features=new_feat, g=g, existing_features=existing_feats,
        k_nn=5, k=10, device=device
    )
    print("Cold-Start Recommendations:")
    for rank, rec_id in enumerate(cold_recs[0][:5], 1):
        print(f"  {rank}. Product_{rec_id}")
except Exception as e:
    print(f"Cold-start failed: {e}")
    print("Using feature similarity fallback...")
    feat_sim = F.cosine_similarity(new_feat.unsqueeze(0), existing_feats.float())
    top5 = torch.topk(feat_sim, 5)
    print("Top-5 by feature similarity:")
    for rank, (idx, sim) in enumerate(zip(top5.indices, top5.values), 1):
        print(f"  {rank}. Product_{idx.item()} [{sim.item():.3f}]")""")

# ===== CELL GROUP 11: Ablation =====
md("## Cell Group 11: Ablation Studies (Optional)")

code("""# Cell 11: Ablation studies
RUN_ABLATION = False

if RUN_ABLATION:
    ablation_results = {}
    for name in ['Full DAEMON', 'w/o Asymmetry']:
        print(f"Training {name}...")
        ab_model = DAEMONModel(cfg).to(device)
        ab_optimizer = torch.optim.Adam(ab_model.parameters(), lr=cfg.lr)
        for ep in range(1, 6):
            ab_model.train()
            for it, (inp, out, blocks) in enumerate(train_loader):
                blocks = [b.to(device) for b in blocks]
                h = blocks[0].srcdata['feat']
                with autocast(enabled=cfg.use_amp):
                    s, t = ab_model(blocks, h)
                    cp_mask = blocks[-1].edata['type'] == 0
                    if cp_mask.sum() == 0: continue
                    cp_eids = torch.where(cp_mask)[0]
                    cp_s, cp_d = blocks[-1].find_edges(cp_eids[:50])
                    loss, _ = criterion(s, t, cp_s, cp_d, cp_s[:10], cp_d[:10],
                                       cp_s[:10], cp_d[:10], t.size(0))
                loss.backward()
                ab_optimizer.step()
                ab_optimizer.zero_grad()
                if it >= 5: break
        ab_src, ab_tgt = generate_all_embeddings(ab_model, g, batch_size=4096, device=device)
        hr10 = compute_hit_rate_at_k(ab_src, ab_tgt, cp_test_src, cp_test_dst, k=10)
        ablation_results[name] = {'HR@10': hr10}
        del ab_model; gc.collect(); torch.cuda.empty_cache()
    
    ab_df = pd.DataFrame(ablation_results).T
    display(ab_df.style.highlight_max(axis=0))
else:
    print("Ablation studies skipped. Set RUN_ABLATION=True to run.")""")

# ===== CELL GROUP 12: Export =====
md("## Cell Group 12: Export Results")

code("""# Cell 12a: Save model checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': cfg,
    'results': results,
}, f"{cfg.output_dir}/daemon_model.pt")
print(f"Model saved to {cfg.output_dir}/daemon_model.pt")""")

code("""# Cell 12b: Save embeddings
np.save(f"{cfg.output_dir}/src_embeddings.npy", src_emb.cpu().numpy().astype(np.float16))
np.save(f"{cfg.output_dir}/tgt_embeddings.npy", tgt_emb.cpu().numpy().astype(np.float16))
print(f"Embeddings saved ({src_emb.shape})")""")

code("""# Cell 12c: Final summary
print("=" * 60)
print("  DAEMON - Final Results Summary")
print("=" * 60)
for k, v in results.items():
    if isinstance(v, (int, float, np.floating)):
        print(f"  {k}: {float(v):.4f}")
    else:
        print(f"  {k}: {v}")

print(f"Output directory: {cfg.output_dir}")
for fname in ['daemon_model.pt', 'src_embeddings.npy', 'tgt_embeddings.npy',
              'metrics.json', 'faiss_index.bin', 'training_history.png']:
    path = os.path.join(cfg.output_dir, fname)
    if os.path.isfile(path):
        print(f"  {fname} ({os.path.getsize(path)/1e6:.1f} MB)")

print("To download: Kaggle UI > Notebook > Output > Download All")
memory_summary()
print("Notebook execution complete!")""")

# ===== Write notebook =====
nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'}
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

with open('/home/shashwat-11/ML/daemon_kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f'Generated notebook with {len(cells)} cells')
