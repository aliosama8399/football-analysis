"""
MLflow tracking config for GNN experiments.

Backends:
  - Docker (auto-detected): if http://localhost:5000 is responding, it is
    automatically used. All runs appear in the web dashboard at http://localhost:5000.
  - Explicit URI: set MLFLOW_TRACKING_URI=http://localhost:5000 (or custom URL).
  - Local SQLite fallback: if Docker server is offline, logs to ./mlruns/mlflow.db
    (DB-backed so Model Registry versioning works locally via `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`).

Usage:
    from models.mlflow_config import setup_mlflow, log_graph_meta, safe_end_run
    mlflow = setup_mlflow("football-gnn-training")
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


def is_server_reachable(url: str, timeout: float = 0.8) -> bool:
    """Check if an HTTP MLflow server is responding."""
    try:
        req = urllib.request.Request(
            url.rstrip("/") + "/health",
            headers={"User-Agent": "mlflow-ping"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mlflow-ping"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False


def get_tracking_uri() -> str:
    """Return the active tracking URI.

    Priority:
      1. Explicit MLFLOW_TRACKING_URI environment variable if set.
      2. Active Docker MLflow server on port 5000 (auto-detected).
      3. Local SQLite database at ./mlruns/mlflow.db (Model Registry capable).
    """
    env_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if env_uri:
        return env_uri

    # Auto-detect running Docker server on localhost
    for candidate in ("http://127.0.0.1:5000", "http://localhost:5000"):
        if is_server_reachable(candidate):
            logger.info("Auto-detected active MLflow server at %s", candidate)
            return candidate

    # Local SQLite fallback
    store = BASE_DIR / "mlruns"
    (store / "artifacts").mkdir(parents=True, exist_ok=True)
    db_path = (store / "mlflow.db").as_posix()
    return f"sqlite:///{db_path}"


def setup_mlflow(experiment: str) -> "mlflow":
    """Configure tracking URI and experiment. Returns the mlflow module."""
    import mlflow

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)

    if uri.startswith("http"):
        print(f"[mlflow] Connected to server: {uri} | Experiment: {experiment}")
    else:
        print(f"[mlflow] Server offline — using local SQLite: {uri} | Experiment: {experiment}")
        print("         (Run 'docker compose up -d mlflow' to view runs at http://localhost:5000)")

    return mlflow


def safe_end_run() -> None:
    """Safely end any currently active MLflow run to prevent 'run already active' errors."""
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass


def log_graph_meta(graph_data: dict) -> None:
    """Log dataset shape metadata to the current MLflow run."""
    import mlflow

    meta = {
        "num_nodes": int(graph_data.get("num_nodes", 0)),
        "num_node_features": int(graph_data.get("num_node_features", 0)),
        "num_edge_features": int(graph_data.get("num_edge_features", 0)),
        "num_edges": int(graph_data["edge_index"].shape[1]) if "edge_index" in graph_data else 0,
        "train_edges": int(graph_data["train_mask"].sum()) if "train_mask" in graph_data else 0,
        "test_edges": int(graph_data["test_mask"].sum()) if "test_mask" in graph_data else 0,
    }
    if "num_tabular_features" in graph_data:
        meta["num_tabular_features"] = int(graph_data["num_tabular_features"])
    if "num_leagues" in graph_data:
        meta["num_leagues"] = int(graph_data["num_leagues"])
    mlflow.log_params({f"graph.{k}": v for k, v in meta.items()})

