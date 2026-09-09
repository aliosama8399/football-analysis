"""
MLflow bootstrap — pre-register experiments, models and dataset metadata so the
MLflow UI is populated WITHOUT running any training/Optuna experiments.

What it creates:
  * Experiments 'football-gnn-training' and 'football-gnn-tuning'
  * Experiment tags describing the dataset source + the 7-model GNN zoo
  * One DRAFT-ONLY placeholder run per GNN model (no metrics, no artifacts,
    tagged status='scaffold') so the UI lists every model card immediately.
    These never touch the real training loop and are skipped by the API
    (they have no .pt artifact).

Run:
    python models/mlflow_bootstrap.py                # local ./mlruns
    $env:MLFLOW_TRACKING_URI="http://localhost:5000"; python models/mlflow_bootstrap.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `python models/mlflow_bootstrap.py` (repo root may not be on sys.path)
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models.mlflow_config import setup_mlflow
DATASET_CSV = BASE_DIR / "data" / "processed" / "processed_matches.csv"

GNN_MODELS = ["GCN", "GraphSAGE", "GAT", "GIN", "EdgeConv", "Hybrid", "TEA-GNN"]

MODEL_NOTES = {
    "GCN":       "Spectral convolution (Kipf & Welling) — light baseline",
    "GraphSAGE": "Sampled aggregation (Hamilton) — inductive scaling baseline",
    "GAT":       "Multi-head attention on edges (Veličković) — attention baseline",
    "GIN":       "Graph isomorphism network — strongest WL baseline on structure",
    "EdgeConv":  "Edge-conditioned message passing (12 match features)",
    "Hybrid":    "GNN + tabular features (80) fused at the edge classifier",
    "TEA-GNN":   "Novel: temporal decay + cross-league context (tuned winner)",
}

EXPERIMENTS = ["football-gnn-training", "football-gnn-tuning"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-register MLflow experiments/models/dataset metadata")
    p.add_argument("--no-placeholders", action="store_true",
                   help="Skip creating the scaffold run rows (experiments only)")
    return p.parse_args()


def bootstrap() -> None:
    args = _parse_args()

    mlflow = setup_mlflow(EXPERIMENTS[0])
    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    print(f"\n[mlflow] Bootstrapping MLflow registry and experiments...")
    tracking_uri = mlflow.get_tracking_uri()
    print(f"[mlflow] Target Tracking URI: {tracking_uri}")

    # 1. Setup experiments + tags
    for exp in EXPERIMENTS:
        exp_obj = client.get_experiment_by_name(exp)
        if exp_obj is None:
            exp_id = client.create_experiment(
                exp,
                tags={
                    "project": "football-analysis",
                    "dataset": str(DATASET_CSV),
                    "models": ", ".join(GNN_MODELS),
                },
            )
            print(f"[mlflow] Created experiment: {exp} (ID: {exp_id})")
        else:
            client.set_experiment_tag(exp_obj.experiment_id, "project", "football-analysis")
            client.set_experiment_tag(exp_obj.experiment_id, "dataset", str(DATASET_CSV))
            client.set_experiment_tag(exp_obj.experiment_id, "models", ", ".join(GNN_MODELS))
            print(f"[mlflow] Experiment exists: {exp} (ID: {exp_obj.experiment_id})")

    # 2. Register models in Model Registry
    print("\n[mlflow] Checking Model Registry...")
    for model in GNN_MODELS:
        reg_name = f"football-gnn-{model.lower().replace(' ', '-')}"
        try:
            client.create_registered_model(
                reg_name,
                tags={
                    "project": "football-analysis",
                    "dataset": str(DATASET_CSV),
                    "notes": MODEL_NOTES[model],
                },
                description=f"GNN archetype {model} for football match outcome prediction."
            )
            print(f"[mlflow] Registered model: {reg_name}")
        except Exception as e:
            err_msg = str(e).lower()
            if "already exists" in err_msg or "resource_already_exists" in err_msg:
                print(f"[mlflow] Registered model exists: {reg_name}")
            else:
                print(f"[mlflow] Model registry note for {reg_name}: {e}")

    if args.no_placeholders:
        print("[mlflow] Bootstrap complete (experiments & registered models only, placeholders skipped).")
        return

    # 3. Create scaffold runs (one per model card) if not already created
    print("\n[mlflow] Checking scaffold run cards...")
    mlflow.set_experiment(EXPERIMENTS[0])
    training_exp = client.get_experiment_by_name(EXPERIMENTS[0])
    existing_runs = client.search_runs(
        experiment_ids=[training_exp.experiment_id],
        filter_string="tags.status = 'scaffold'",
    ) if training_exp else []
    existing_model_names = {r.data.tags.get("model", r.data.params.get("model")) for r in existing_runs}

    for model in GNN_MODELS:
        if model in existing_model_names:
            print(f"[mlflow] Scaffold run already exists for model: {model} (skipping duplicate)")
            continue

        with mlflow.start_run(run_name=model):
            mlflow.set_tag("status", "scaffold")   # NOT trained yet
            mlflow.set_tag("model", model)
            mlflow.set_tag("dataset", str(DATASET_CSV))
            mlflow.set_tag("notes", MODEL_NOTES[model])
            mlflow.log_params({
                "model": model,
                "trained": False,
                "dataset_path": str(DATASET_CSV),
            })
        print(f"[mlflow] Created scaffold run for model: {model}")

    print("\n" + "=" * 60)
    print(f"MLflow tracking URI: {tracking_uri}")
    print("Web UI available at http://localhost:5000 (Docker) or via `mlflow ui`.")
    print("Real runs from `python models/train_gnn.py` and `python models/tune_gnn.py`")
    print("will record metrics, parameters, and model artifacts directly.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    bootstrap()
