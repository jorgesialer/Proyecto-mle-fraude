"""Publish actual baseline evidence to a separate DagsHub experiment, without retraining."""

import json
import os
from pathlib import Path

import pandas as pd

from .baseline import EXPERIMENT, TREE_PARAMS
from .prepare import ROOT
from .tracking import configure


def publish(root: Path = ROOT) -> dict:
    reports = root / "reports/baseline"
    summary = json.loads((reports / "summary.json").read_text(encoding="utf-8"))
    model_path = root / "artifacts/baseline/mlflow_model"
    if not (model_path / "MLmodel").is_file():
        raise FileNotFoundError("Primero ejecutar el baseline y verificar el modelo local")
    mlflow, experiment_id, backend, name = configure(root, experiment_name=EXPERIMENT, backend="dagshub")
    status_path = reports / "remote_tracking.json"
    # Avoid duplicate publication when re-running the notebook or CLI.
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("source_local_run_id") == summary["run_id"]:
            run = mlflow.MlflowClient().get_run(previous["run_id"])
            if run.info.status == "FINISHED":
                return previous
    with mlflow.start_run(experiment_id=experiment_id, run_name="baseline_temporal_comparison",
                          tags={"stage": "baseline", "execution_backend": "local",
                                "logging_backend": "dagshub", "source_local_run_id": summary["run_id"],
                                "model_role": "baseline_only", "offline_test_used": "false",
                                "production_used": "false"}) as parent:
        run_id = parent.info.run_id
        mlflow.log_params({"source_sha256": summary["source_sha256"],
                           "development_rows": summary["development_rows"],
                           "fixed_threshold": summary["fixed_threshold"], "cv_folds": summary["fold_count"],
                           "sampling": "none", "primary_metric": "average_precision", "prediction_moment": "post_transaction"})
        for row in summary["models"]:
            mlflow.log_metrics({f"{row['model']}_{k}": v for k, v in row.items() if k != "model"})
        mlflow.log_metrics({f"tree_pooled_{k}": v for k, v in summary["tree_pooled_validation"].items()})
        frame = pd.read_csv(reports / "fold_metrics.csv")
        metric_names = ["average_precision", "roc_auc", "precision", "recall", "f1", "f2", "alert_rate",
                        "prevalence", "tn", "fp", "fn", "tp", "fit_seconds", "predict_seconds"]
        for _, row in frame.iterrows():
            with mlflow.start_run(experiment_id=experiment_id,
                                  run_name=f"{row['model']}_fold_{int(row['fold']):02d}", nested=True):
                mlflow.log_params({k: row[k] for k in frame.columns if k not in metric_names})
                mlflow.log_params(TREE_PARAMS if row['model'] == 'decision_tree' else {"strategy": "prior"})
                mlflow.log_metrics({k: row[k] for k in metric_names})
        mlflow.log_artifacts(str(model_path), artifact_path="model")
        # Include only documented evidence, not raw rows or environment credentials.
        for filename in ["summary.json", "fold_metrics.csv", "comparison.csv", "baseline_validation.png", "tree_rules.txt", "STAGE2_RESULT.md"]:
            path = reports / filename
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="baseline_reports")
        mlflow.log_artifact(str(root / "reports/dataset_profile.json"), artifact_path="data_contract")
        mlflow.log_artifact(str(root / "reports/temporal_folds.csv"), artifact_path="data_contract")
    client = mlflow.MlflowClient()
    verified_run = client.get_run(run_id)
    model_files = {entry.path for entry in client.list_artifacts(run_id, "model")}
    if verified_run.info.status != "FINISHED" or not {"model/MLmodel", "model/model.pkl"}.issubset(model_files):
        raise RuntimeError("No se pudo verificar el run o sus artefactos remotos")
    uri = os.environ["MLFLOW_TRACKING_URI"].rstrip("/")
    result = {"backend": backend, "experiment_name": name, "experiment_id": experiment_id,
              "run_id": run_id, "source_local_run_id": summary["run_id"],
              "run_url": f"{uri}/#/experiments/{experiment_id}/runs/{run_id}",
              "model_uri": f"runs:/{run_id}/model", "remote_write_verified": True,
              "run_status": verified_run.info.status, "model_artifacts_verified": True,
              "model_role": "baseline_only", "registry_registration": False}
    status_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    try:
        result = publish()
        print(json.dumps(result, indent=2))
    except Exception as exc:
        # Do not dump transport exceptions that could contain authentication details.
        print(json.dumps({"remote_write_verified": False, "error_type": type(exc).__name__}))
        raise SystemExit(1)
