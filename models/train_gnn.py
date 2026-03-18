"""
Training Pipeline for 6 GNN Models — Football Match Prediction
===============================================================
Transductive edge classification: train/test on the same graph,
using masks to split edges by season.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from data.graph_builder import FootballGraphBuilder
from models.gnn_models import get_model

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "models" / "results"
MODELS_DIR = BASE_DIR / "models" / "saved"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 300
LR = 0.003
WEIGHT_DECAY = 5e-4
PATIENCE = 40
HIDDEN_DIM = 64
DROPOUT = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════

def ranked_probability_score(y_true, y_prob):
    n_classes = y_prob.shape[1]
    rps = 0.0
    for i in range(len(y_true)):
        cum_p, cum_t, rps_m = 0.0, 0.0, 0.0
        for j in range(n_classes - 1):
            cum_p += y_prob[i, j]
            cum_t += 1.0 if y_true[i] <= j else 0.0
            rps_m += (cum_p - cum_t) ** 2
        rps += rps_m / (n_classes - 1)
    return rps / len(y_true)


def evaluate(y_true, y_pred, y_prob, class_names):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'log_loss': log_loss(y_true, y_prob, labels=[0, 1, 2]),
        'rps': ranked_probability_score(y_true, y_prob),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'per_class_f1': {c: f1_score(y_true, y_pred, average=None, labels=[i])[0] 
                         for i, c in enumerate(class_names)},
    }


# ═══════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════

def train_one_model(model_name, graph_data):
    """Train a single GNN model using transductive edge classification."""
    
    print(f"\n{'─' * 50}")
    print(f"  Training: {model_name}")
    print(f"{'─' * 50}")
    
    num_nf = graph_data['num_node_features']
    num_ef = graph_data['num_edge_features']
    
    is_hybrid = (model_name == 'Hybrid')
    extra_kwargs = {}
    if is_hybrid:
        extra_kwargs['num_tabular_features'] = graph_data['num_tabular_features']
    
    model = get_model(model_name, num_nf, num_ef,
                      hidden_dim=HIDDEN_DIM, num_classes=3, dropout=DROPOUT, **extra_kwargs)
    model = model.to(DEVICE)
    
    # Move data to device
    x = graph_data['x'].to(DEVICE)
    ei = graph_data['edge_index'].to(DEVICE)
    ea = graph_data['edge_attr'].to(DEVICE)
    ey = graph_data['edge_y'].to(DEVICE)
    train_mask = graph_data['train_mask'].to(DEVICE)
    test_mask = graph_data['test_mask'].to(DEVICE)
    tabular = graph_data['tabular_features'].to(DEVICE) if is_hybrid else None
    
    # Class weights
    train_labels = ey[train_mask]
    class_counts = torch.bincount(train_labels, minlength=3).float()
    class_weights = (1.0 / class_counts.clamp(min=1)) * class_counts.sum() / 3.0
    class_weights = class_weights.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                            patience=15, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_losses = []
    
    start = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        optimizer.zero_grad()
        if is_hybrid:
            out = model(x, ei, ea, tabular_features=tabular)
        else:
            out = model(x, ei, ea)  # Predictions for ALL edges
        
        # Loss only on TRAIN edges
        loss = criterion(out[train_mask], ey[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())
        
        # ── Eval on test edges ──
        model.eval()
        with torch.no_grad():
            if is_hybrid:
                val_out = model(x, ei, ea, tabular_features=tabular)
            else:
                val_out = model(x, ei, ea)
            val_loss = criterion(val_out[test_mask], ey[test_mask])
        
        scheduler.step(val_loss)
        
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break
        
        if epoch % 50 == 0 or epoch == 1:
            t_pred = out[train_mask].argmax(dim=1)
            t_acc = (t_pred == ey[train_mask]).float().mean().item()
            v_pred = val_out[test_mask].argmax(dim=1)
            v_acc = (v_pred == ey[test_mask]).float().mean().item()
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                  f"Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")
    
    train_time = time.time() - start
    
    # ── Final eval with best model ──
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        if is_hybrid:
            final_out = model(x, ei, ea, tabular_features=tabular)
        else:
            final_out = model(x, ei, ea)
        test_logits = final_out[test_mask]
        y_prob = F.softmax(test_logits, dim=1).cpu().numpy()
        y_pred = test_logits.argmax(dim=1).cpu().numpy()
    
    y_true = ey[test_mask].cpu().numpy()
    class_names = ['A', 'D', 'H']
    
    metrics = evaluate(y_true, y_pred, y_prob, class_names)
    metrics['train_time'] = train_time
    metrics['epochs'] = epoch
    metrics['train_losses'] = train_losses
    
    print(f"\n  ✓ Results ({model_name}):")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Macro F1:  {metrics['f1_macro']:.4f}")
    print(f"    Log Loss:  {metrics['log_loss']:.4f}")
    print(f"    RPS:       {metrics['rps']:.4f}")
    print(f"    Per-class: A={metrics['per_class_f1']['A']:.3f} "
          f"D={metrics['per_class_f1']['D']:.3f} H={metrics['per_class_f1']['H']:.3f}")
    print(f"    Time:      {train_time:.1f}s ({epoch} epochs)")
    
    # Save
    model_path = MODELS_DIR / f'gnn_{model_name.lower()}.pt'
    torch.save({'model_state': best_state or model.state_dict(),
                'model_name': model_name, 'hidden_dim': HIDDEN_DIM,
                'num_node_features': num_nf, 'num_edge_features': num_ef}, model_path)
    
    return metrics


# ═══════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════

def plot_comparison(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [('accuracy', 'Accuracy ↑', '#2ecc71'),
               ('f1_macro', 'Macro F1 ↑', '#3498db'),
               ('log_loss', 'Log Loss ↓', '#e74c3c'),
               ('rps', 'RPS ↓', '#f39c12')]
    
    for ax, (m, title, color) in zip(axes.flatten(), metrics):
        data = df.sort_values(m, ascending=(m in ['log_loss', 'rps']))
        bars = ax.barh(data['model'], data[m], color=color, alpha=0.8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        for bar, val in zip(bars, data[m]):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)
    
    plt.suptitle('GNN Model Comparison — Football Match Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'gnn_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved comparison chart")


def plot_confusion(all_results, class_names):
    n = len(all_results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    for idx, (name, res) in enumerate(all_results.items()):
        cm = res['confusion_matrix']
        cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[idx], cbar=False)
        axes[idx].set_title(f'{name}\nAcc={res["accuracy"]:.3f}', fontsize=10)
        axes[idx].set_ylabel('True'); axes[idx].set_xlabel('Predicted')
    for idx in range(n, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('GNN Models — Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'gnn_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrices")


def plot_curves(all_results):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    for (name, res), c in zip(all_results.items(), colors):
        losses = res.get('train_losses', [])
        if losses:
            ax.plot(losses, label=name, color=c, alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Training Loss')
    ax.set_title('GNN Training Curves', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'gnn_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves")


def plot_vs_traditional(gnn_df):
    trad_path = RESULTS_DIR / 'tuned_comparison.csv'
    if not trad_path.exists():
        trad_path = RESULTS_DIR / 'model_comparison.csv'
    if not trad_path.exists():
        return
    
    trad_df = pd.read_csv(trad_path)
    best_gnn = gnn_df.sort_values('accuracy', ascending=False).iloc[0]
    best_trad = trad_df.sort_values('accuracy', ascending=False).iloc[0]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ['accuracy', 'f1_macro', 'rps']
    labels = ['Accuracy ↑', 'Macro F1 ↑', 'RPS ↓']
    x = np.arange(len(metrics))
    w = 0.35
    
    trad_vals = [best_trad[m] for m in metrics]
    gnn_vals = [best_gnn[m] for m in metrics]
    
    b1 = ax.bar(x-w/2, trad_vals, w, label=f'Trad: {best_trad["model"]}', color='#3498db', alpha=0.8)
    b2 = ax.bar(x+w/2, gnn_vals, w, label=f'GNN: {best_gnn["model"]}', color='#2ecc71', alpha=0.8)
    
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title('Best Traditional ML vs Best GNN', fontsize=14, fontweight='bold')
    ax.legend()
    for bar in b1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{bar.get_height():.4f}', ha='center', fontsize=9)
    for bar in b2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{bar.get_height():.4f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'gnn_vs_traditional.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved GNN vs Traditional comparison")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  GNN TRAINING — FOOTBALL MATCH PREDICTION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)
    
    builder = FootballGraphBuilder(
        data_path=str(BASE_DIR / "data" / "processed" / "processed_matches.csv"))
    graph_data = builder.build_train_test_graphs()
    
    models_to_train = ['GCN', 'GraphSAGE', 'GAT', 'GIN', 'EdgeConv', 'Hybrid']
    
    all_results = {}
    rows = []
    
    for i, name in enumerate(models_to_train, 1):
        print(f"\n{'=' * 70}")
        print(f"  [{i}/{len(models_to_train)}] {name}")
        print(f"{'=' * 70}")
        
        try:
            m = train_one_model(name, graph_data)
            all_results[name] = m
            rows.append({
                'model': name, 'accuracy': m['accuracy'],
                'f1_macro': m['f1_macro'], 'f1_weighted': m['f1_weighted'],
                'log_loss': m['log_loss'], 'rps': m['rps'],
                'train_time_s': round(m['train_time'], 1), 'epochs': m['epochs'],
                'f1_A': m['per_class_f1']['A'], 'f1_D': m['per_class_f1']['D'],
                'f1_H': m['per_class_f1']['H'],
            })
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            import traceback; traceback.print_exc()
    
    # Summary
    print(f"\n{'=' * 70}")
    print(f"  GNN RESULTS")
    print(f"{'=' * 70}\n")
    
    df = pd.DataFrame(rows).sort_values('accuracy', ascending=False).reset_index(drop=True)
    df.index += 1
    print(df[['model','accuracy','f1_macro','log_loss','rps','f1_D','train_time_s']].to_string())
    
    df.to_csv(RESULTS_DIR / 'gnn_comparison.csv', index=False)
    print(f"\n✓ Saved to {RESULTS_DIR / 'gnn_comparison.csv'}")
    
    # Plots
    plot_comparison(df)
    plot_confusion(all_results, ['A','D','H'])
    plot_curves(all_results)
    plot_vs_traditional(df)
    
    best = df.iloc[0]
    print(f"\n{'=' * 70}")
    print(f"  🏆 BEST GNN: {best['model']}")
    print(f"     Accuracy:  {best['accuracy']:.4f}")
    print(f"     Macro F1:  {best['f1_macro']:.4f}")
    print(f"     RPS:       {best['rps']:.4f}")
    print(f"     Draw F1:   {best['f1_D']:.4f}")
    print(f"{'=' * 70}\n")
    
    print("✓ GNN pipeline complete!")
    return df, all_results


if __name__ == '__main__':
    main()
