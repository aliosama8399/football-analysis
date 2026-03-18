"""
Fine-Tuning Dataset Builder
============================
Generates a JSONL dataset for fine-tuning an SLM on football tactical analysis.

Pipeline per match:
  1. Read pre-match stats from processed_matches.csv
  2. Run EdgeConv GNN prediction → outcome + probabilities
  3. Run GNNExplainer → top node features + influential edges
  4. Call a "teacher" LLM (e.g. Gemini Flash) to write the analysis
  5. Save (user_prompt, assistant_response) as a JSONL row

Usage:
  # Dry run — print first 3 samples without calling the LLM
  python data/build_finetune_dataset.py --dry-run --max-samples 3

  # Full run — generate all samples with Gemini teacher
  python data/build_finetune_dataset.py --provider gemini

  # Resume after a crash (skip already-generated rows)
  python data/build_finetune_dataset.py --provider gemini --resume
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from data.graph_builder import FootballGraphBuilder
from models.gnn_models import get_model
from models.llm_providers import get_llm_provider

BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "processed" / "processed_matches.csv"
OUTPUT_DIR = BASE_DIR / "data" / "finetune"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Feature name mapping: column suffix → plain English for the prompt
FEATURE_LABELS = {
    'Form_5':           'Points per match (last 5)',
    'GF_5':             'Goals scored per match (last 5)',
    'GA_5':             'Goals conceded per match (last 5)',
    'xG_5':             'Expected goals per match (last 5)',
    'xGA_5':            'Expected goals against per match (last 5)',
    'Shots_5':          'Shots per match (last 5)',
    'ShotsAgainst_5':   'Shots conceded per match (last 5)',
    'SOT_5':            'Shots on target per match (last 5)',
    'SOTAgainst_5':     'Shots on target conceded per match (last 5)',
    'Corners_5':        'Corners per match (last 5)',
    'CornersAgainst_5': 'Corners conceded per match (last 5)',
    'Fouls_5':          'Fouls committed per match (last 5)',
    'FoulsAgainst_5':   'Fouls suffered per match (last 5)',
    'Yellows_5':        'Yellow cards per match (last 5)',
    'Reds_5':           'Red cards per match (last 5)',
}


# ═══════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════

def load_model_and_graph():
    """Load the tuned EdgeConv model and the full graph."""
    model_path = BASE_DIR / "models" / "saved" / "gnn_edgeconv_tuned.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Tuned EdgeConv not found at {model_path}. Run tune_gnn.py first.")

    builder = FootballGraphBuilder(data_path=str(DATA_PATH))
    graph   = builder.build_train_test_graphs()

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    bp = checkpoint.get('best_params', {})
    model = get_model('EdgeConv', graph['num_node_features'], graph['num_edge_features'],
                      hidden_dim=bp.get('hidden_dim', 64), dropout=bp.get('dropout', 0.3))
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(DEVICE).eval()

    return model, graph, builder


# ═══════════════════════════════════════════════════════════
# PER-MATCH PROCESSING
# ═══════════════════════════════════════════════════════════

def build_user_prompt(row: pd.Series, builder, graph, model, edge_idx: int,
                      pred: str, probs: dict, gnn_explanation: dict) -> str:
    """
    Build the human-readable user prompt from raw match data + GNN output.
    This is what the fine-tuned SLM will receive at inference time.
    """
    home = row['HomeTeam']
    away = row['AwayTeam']

    # ── Home team stats (readable) ──
    home_stats = []
    for suffix, label in FEATURE_LABELS.items():
        col = f'Home{suffix}'
        val = row.get(col, None)
        if val is not None and not pd.isna(val):
            home_stats.append(f"  - {label}: {val:.2f}")
    home_block = "\n".join(home_stats) if home_stats else "  (no recent stats available)"

    # ── Away team stats (readable) ──
    away_stats = []
    for suffix, label in FEATURE_LABELS.items():
        col = f'Away{suffix}'
        val = row.get(col, None)
        if val is not None and not pd.isna(val):
            away_stats.append(f"  - {label}: {val:.2f}")
    away_block = "\n".join(away_stats) if away_stats else "  (no recent stats available)"

    # ── Head-to-head ──
    h2h_parts = []
    for col, label in [('H2H_Matches', 'Total meetings'), ('H2H_HomeWins', 'Home wins'),
                        ('H2H_AwayWins', 'Away wins'), ('H2H_Draws', 'Draws')]:
        val = row.get(col, None)
        if val is not None and not pd.isna(val):
            h2h_parts.append(f"{label}: {int(val)}")
    h2h_block = ", ".join(h2h_parts) if h2h_parts else "No head-to-head data"

    # ── Key historical matches from GNN ──
    hist_lines = []
    for m in gnn_explanation.get('top_influencing_matches', [])[:3]:
        hist_lines.append(f"  - {m['match']}")
    hist_block = "\n".join(hist_lines) if hist_lines else "  (none identified)"

    prompt = f"""Analyze the upcoming match: {home} (Home) vs {away} (Away).

{home} recent form (last 5 matches):
{home_block}

{away} recent form (last 5 matches):
{away_block}

Head-to-Head: {h2h_block}

Key recent encounters that shaped current form:
{hist_block}

Prediction: {pred} (Home {probs['H']:.1%} | Draw {probs['D']:.1%} | Away {probs['A']:.1%})"""

    return prompt


def get_edge_idx_for_match(graph, builder, home_team, away_team):
    """Find the graph edge index for a specific home/away pair."""
    ei = graph['edge_index']
    hi = builder.team_to_idx.get(home_team)
    ai = builder.team_to_idx.get(away_team)
    if hi is None or ai is None:
        return None
    mask = (ei[0] == hi) & (ei[1] == ai)
    edges = mask.nonzero(as_tuple=True)[0]
    if len(edges) == 0:
        return None
    return edges[-1].item()


def run_gnn_prediction(model, graph, edge_idx):
    """Get prediction + probabilities for a single edge."""
    x  = graph.get('x_test', graph['x']).to(DEVICE)
    ei = graph['edge_index'].to(DEVICE)
    ea = graph['edge_attr'].to(DEVICE)

    with torch.no_grad():
        out = model(x, ei, ea)

    logits = out[edge_idx]
    probs  = F.softmax(logits, dim=0).cpu().numpy()
    pred   = logits.argmax().item()

    class_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    return class_map[pred], {'A': float(probs[0]), 'D': float(probs[1]), 'H': float(probs[2])}


def run_gnn_explainer(model, graph, builder, edge_idx):
    """Extract structural explanation for one edge (lightweight version)."""
    from torch_geometric.explain import Explainer, GNNExplainer

    x  = graph.get('x_test', graph['x']).to(DEVICE)
    ei = graph['edge_index'].to(DEVICE)
    ea = graph['edge_attr'].to(DEVICE)

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),  # Fewer epochs for speed (100 vs 200)
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(mode='multiclass_classification', task_level='edge', return_type='raw'),
    )

    explanation = explainer(x, ei, edge_attr=ea, index=edge_idx)

    # Node feature importance
    node_mask = explanation.node_mask.cpu()
    home_idx  = ei[0, edge_idx].item()
    away_idx  = ei[1, edge_idx].item()
    home_imp  = node_mask[home_idx].numpy()
    away_imp  = node_mask[away_idx].numpy()

    feat_names = builder.NODE_FEATURE_SUFFIXES
    home_team  = builder.idx_to_team[home_idx]
    away_team  = builder.idx_to_team[away_idx]

    top_home = {f"{home_team}_{feat_names[i]}": float(home_imp[i])
                for i in np.argsort(home_imp)[-3:][::-1]}
    top_away = {f"{away_team}_{feat_names[i]}": float(away_imp[i])
                for i in np.argsort(away_imp)[-3:][::-1]}

    # Edge importance
    edge_mask = explanation.edge_mask.cpu().numpy()
    top_edges = np.argsort(edge_mask)[-6:][::-1]
    top_matches = []
    for e in top_edges:
        if int(e) == int(edge_idx):
            continue
        h = builder.idx_to_team[ei[0, e].item()]
        a = builder.idx_to_team[ei[1, e].item()]
        top_matches.append({"match": f"{h} vs {a}", "influence_score": round(float(edge_mask[e]), 6)})
        if len(top_matches) == 3:
            break

    return {
        'top_node_features':       {'home_team': top_home, 'away_team': top_away},
        'top_influencing_matches': top_matches,
    }


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build fine-tuning dataset for football SLM")
    parser.add_argument('--provider',     type=str, default='gemini',
                        help="Teacher LLM provider (default: gemini)")
    parser.add_argument('--model-name',   type=str, default='',
                        help="Teacher LLM model name (default: from config)")
    parser.add_argument('--max-samples',  type=int, default=0,
                        help="Max samples to generate (0 = all)")
    parser.add_argument('--dry-run',      action='store_true',
                        help="Print prompts without calling teacher LLM")
    parser.add_argument('--resume',       action='store_true',
                        help="Skip matches already in the output file")
    parser.add_argument('--output',       type=str, default='football_tactical.jsonl',
                        help="Output filename inside data/finetune/")
    parser.add_argument('--skip-explainer', action='store_true',
                        help="Skip GNNExplainer (faster, less rich prompts)")
    args = parser.parse_args()

    output_path = OUTPUT_DIR / args.output
    print("=" * 70)
    print("  FINE-TUNING DATASET BUILDER")
    print(f"  Provider:  {args.provider}")
    print(f"  Output:    {output_path}")
    print(f"  Dry run:   {args.dry_run}")
    print(f"  Device:    {DEVICE}")
    print("=" * 70)

    # ── Load model & graph ──
    model, graph, builder = load_model_and_graph()
    df = builder.df.copy()

    # Only process test-set matches (2024-25 season)
    test_seasons = [2425]
    test_df = df[df['Season'].isin(test_seasons)].copy()
    print(f"\n✓ Test-set matches to process: {len(test_df)}")

    # ── Resume support ──
    existing_keys = set()
    if args.resume and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    existing_keys.add(rec.get('match_id', ''))
                except json.JSONDecodeError:
                    pass
        print(f"✓ Resuming: {len(existing_keys)} already generated")

    # ── Teacher LLM ──
    teacher = None
    if not args.dry_run:
        teacher = get_llm_provider(provider_type=args.provider, model_name=args.model_name)
        print(f"✓ Teacher LLM ready: {args.provider}")

    # ── Process matches ──
    samples_written = 0
    errors = 0
    max_s = args.max_samples if args.max_samples > 0 else len(test_df)

    with open(output_path, 'a', encoding='utf-8') as out_f:
        for idx, (_, row) in enumerate(tqdm(test_df.iterrows(), total=min(len(test_df), max_s),
                                            desc="Building dataset")):
            if samples_written >= max_s:
                break

            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            match_id  = f"{row.get('Date', '')}_{home_team}_vs_{away_team}"

            if match_id in existing_keys:
                continue

            # Find edge in graph
            edge_idx = get_edge_idx_for_match(graph, builder, home_team, away_team)
            if edge_idx is None:
                errors += 1
                continue

            # GNN prediction
            try:
                pred, probs = run_gnn_prediction(model, graph, edge_idx)
            except Exception as e:
                errors += 1
                continue

            # GNN explanation (optional)
            gnn_exp = {'top_node_features': {}, 'top_influencing_matches': []}
            if not args.skip_explainer:
                try:
                    gnn_exp = run_gnn_explainer(model, graph, builder, edge_idx)
                except Exception:
                    pass  # Proceed without explanation

            # Build the user prompt
            user_prompt = build_user_prompt(row, builder, graph, model, edge_idx,
                                            pred, probs, gnn_exp)

            if args.dry_run:
                print(f"\n{'─' * 70}")
                print(f"MATCH: {home_team} vs {away_team}")
                print(f"{'─' * 70}")
                print(f"USER PROMPT:\n{user_prompt}")
                print(f"\n[Would call {args.provider} here for the assistant response]")
                samples_written += 1
                continue

            # Call teacher LLM
            match_context = {
                'home_team': home_team, 'away_team': away_team,
                'prediction': pred, 'probabilities': probs,
            }
            try:
                assistant_response = teacher.generate_explanation(match_context, gnn_exp)
            except Exception as e:
                print(f"  ✗ LLM error for {match_id}: {e}")
                errors += 1
                continue

            if not assistant_response or len(assistant_response.strip()) < 50:
                errors += 1
                continue

            # Build the JSONL record
            record = {
                "match_id": match_id,
                "actual_result": row.get('FTR', ''),
                "gnn_prediction": pred,
                "messages": [
                    {"role": "user",      "content": user_prompt},
                    {"role": "assistant", "content": assistant_response.strip()},
                ],
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            samples_written += 1

            # Rate limiting (avoid API throttling)
            time.sleep(0.5)

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  DATASET GENERATION COMPLETE")
    print(f"  Samples written:  {samples_written}")
    print(f"  Errors/skipped:   {errors}")
    print(f"  Output file:      {output_path}")
    print(f"{'=' * 70}")

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")


if __name__ == '__main__':
    main()
