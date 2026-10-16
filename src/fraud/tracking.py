"""Tracking independiente de HMDA; backend explícito sin fallback silencioso."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_REMOTE = ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")


def configuration_status(root: Path = ROOT) -> dict:
    """Return presence only; never disclose values of authentication variables."""
    load_dotenv(root / ".env", override=False)
    present = {key: bool(os.getenv(key)) for key in REQUIRED_REMOTE}
    return {"backend": os.getenv("MLFLOW_BACKEND", "local"),
            "remote_config_present": present,
            "remote_ready_to_attempt": all(present.values()),
            "remote_connection_verified": False}


def configure(root: Path = ROOT):
    """Adapted from the V2 mlflow_utils pattern, with separate experiment."""
    import mlflow
    status = configuration_status(root)
    backend = status["backend"]
    if backend == "local":
        database = root / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{database.as_posix()}")
    elif backend == "dagshub":
        if not status["remote_ready_to_attempt"]:
            raise EnvironmentError("Faltan variables de autenticación DagsHub; configúralas localmente")
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    else:
        raise ValueError("MLFLOW_BACKEND debe ser local o dagshub")
    name = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-data-preparation")
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        kwargs = {}
        if backend == "local":
            artifacts = root / "mlartifacts"
            artifacts.mkdir(exist_ok=True)
            kwargs["artifact_location"] = artifacts.as_uri()
        experiment_id = mlflow.create_experiment(name, **kwargs)
    else:
        experiment_id = experiment.experiment_id
    return mlflow, experiment_id, backend, name


def log_preparation(root: Path = ROOT) -> dict:
    """Log data evidence only: this run is not a training experiment."""
    mlflow, experiment_id, backend, name = configure(root)
    reports = root / "reports"
    profile = json.loads((reports / "dataset_profile.json").read_text(encoding="utf-8"))
    with mlflow.start_run(experiment_id=experiment_id, run_name="paysim_data_preparation",
                          tags={"stage": "data_preparation", "prediction_moment": "post_transaction"}) as run:
        mlflow.log_params({"source_sha256": profile["source_sha256"], "development_end_step": 250,
                           "threshold_end_step": 300, "offline_end_step": 400, "production_start_step": 401})
        mlflow.log_metrics({"dataset_rows": profile["rows"], "dataset_frauds": profile["frauds"],
                            **{f"{k}_rows": d["rows"] for k, d in profile["partitions"].items()}})
        for filename in ["dataset_profile.json", "profile_by_step.csv", "profile_by_type.csv", "data_overview.png"]:
            path = reports / filename
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="data_preparation")
        result = {"run_id": run.info.run_id, "experiment_name": name,
                  "experiment_id": experiment_id, "backend": backend,
                  "remote_connection_verified": backend == "dagshub",
                  "stage": "data_preparation", "training_performed": False}
    (reports / "tracking_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(configuration_status(), indent=2))
