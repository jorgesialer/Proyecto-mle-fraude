"""Baseline temporal sobre development. No abre umbral, test ni producción."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
                            f1_score, fbeta_score, precision_recall_curve, precision_score,
                            recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from .prepare import ROOT, sha256
from .schema import FEATURES, TARGET, TYPES, temporal_folds
from .tracking import configure
from .modeling import FeatureContract

EXPERIMENT = "fraud-baseline-temporal"
TREE_PARAMS = {"max_depth": 6, "min_samples_leaf": 20, "class_weight": "balanced", "random_state": 42}


def make_pipeline(model_name: str) -> Pipeline:
    numeric = [name for name in FEATURES if name != "type"]
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), numeric),
        ("categorical", OneHotEncoder(categories=[list(TYPES)], handle_unknown="error", sparse_output=False), ["type"]),
    ], remainder="drop")
    if model_name == "dummy_prior":
        estimator = DummyClassifier(strategy="prior")
    elif model_name == "decision_tree":
        estimator = DecisionTreeClassifier(**TREE_PARAMS)
    else:
        raise ValueError("Modelo baseline desconocido")
    return Pipeline([("contract", FeatureContract()), ("preprocessor", preprocessor), ("model", estimator)])


def metrics(y, scores, threshold=0.5) -> dict:
    """Rank metrics and decision metrics; no calibration claim for weighted tree."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    if len(y) != len(scores) or set(np.unique(y)) != {0, 1}:
        raise ValueError("La evaluación necesita ambas clases y longitudes coincidentes")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores no finitos o fuera de [0,1]")
    pred = (scores >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "alert_rate": float(pred.mean()), "prevalence": float(y.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run_baseline(root: Path = ROOT) -> dict:
    import mlflow.sklearn
    from mlflow.models import infer_signature

    reports = root / "reports/baseline"
    artifacts = root / "artifacts/baseline"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    profile = json.loads((root / "reports/dataset_profile.json").read_text(encoding="utf-8"))
    path = root / "data/processed/development.parquet"
    if sha256(path) != profile["partitions"]["development"]["parquet_sha256"]:
        raise ValueError("El dataset de desarrollo no coincide con el manifiesto")
    summary_path = reports / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if (previous.get('source_sha256') == profile['source_sha256']
                and previous.get('tree_parameters') == TREE_PARAMS
                and sha256(artifacts/'pipeline_baseline.joblib') == previous.get('model_sha256')):
            print('Baseline existente verificado; no se repite el entrenamiento.', flush=True)
            return previous
        raise ValueError('Ya existe evidencia de otro baseline; conservarla antes de ejecutar una nueva versión')
    if (artifacts/'mlflow_model').exists() or (artifacts/'pipeline_baseline.joblib').exists():
        raise ValueError('Hay artefactos de una ejecución incompleta; revisarlos antes de reentrenar')
    data = pd.read_parquet(path)
    if not data.step.between(1, 250).all() or not data.row_id.is_unique:
        raise ValueError("Partición de desarrollo inválida")
    X, y = data.loc[:, list(FEATURES)], data[TARGET]
    folds = list(temporal_folds(data.step))
    mlflow, exp_id, backend, exp_name = configure(root, experiment_name=EXPERIMENT, backend="local")
    fold_results, oof_parts = [], []
    with mlflow.start_run(experiment_id=exp_id, run_name="baseline_temporal_comparison",
                          tags={"stage": "baseline", "test_evaluated": "false", "production_evaluated": "false"}) as parent:
        run_id = parent.info.run_id
        mlflow.log_params({"development_rows": len(data), "source_sha256": profile["source_sha256"],
                           "fixed_threshold": 0.5, "cv_folds": len(folds), "sampling": "none",
                           "primary_metric": "average_precision", "prediction_moment": "post_transaction"})
        for name in ["dummy_prior", "decision_tree"]:
            for fold_id, (train, valid) in enumerate(folds, start=1):
                if y.iloc[train].nunique() != 2 or y.iloc[valid].nunique() != 2:
                    raise ValueError("Fold con una sola clase")
                pipeline = make_pipeline(name)
                print(f"Entrenando {name}, fold {fold_id}, {len(train):,} filas", flush=True)
                start = perf_counter()
                pipeline.fit(X.iloc[train], y.iloc[train])
                fit_seconds = perf_counter() - start
                start = perf_counter()
                scores = pipeline.predict_proba(X.iloc[valid])[:, 1]
                predict_seconds = perf_counter() - start
                result = metrics(y.iloc[valid], scores)
                result.update({"fit_seconds": fit_seconds, "predict_seconds": predict_seconds})
                row = {"model": name, "fold": fold_id, "train_rows": len(train), "validation_rows": len(valid),
                       "train_max_step": int(data.step.iloc[train].max()),
                       "validation_min_step": int(data.step.iloc[valid].min()),
                       "validation_max_step": int(data.step.iloc[valid].max()), **result}
                fold_results.append(row)
                with mlflow.start_run(experiment_id=exp_id, run_name=f"{name}_fold_{fold_id:02d}", nested=True):
                    mlflow.log_params({k: v for k, v in row.items() if k not in result})
                    mlflow.log_params(TREE_PARAMS if name == "decision_tree" else {"strategy": "prior"})
                    mlflow.log_metrics(result)
                if name == "decision_tree":
                    oof_parts.append(pd.DataFrame({"row_id": data.row_id.iloc[valid].to_numpy(),
                                                    "step": data.step.iloc[valid].to_numpy(),
                                                    "fold": fold_id, TARGET: y.iloc[valid].to_numpy(), "score": scores}))
                print(f"  AP={result['average_precision']:.6f}; recall={result['recall']:.4f}; fit={fit_seconds:.1f}s", flush=True)
                del pipeline
        fold_frame = pd.DataFrame(fold_results)
        fold_frame.to_csv(reports / "fold_metrics.csv", index=False)
        metric_names = ["average_precision", "roc_auc", "precision", "recall", "f1", "f2", "alert_rate", "fit_seconds"]
        comparison = fold_frame.groupby("model")[metric_names].agg(["mean", "std"])
        comparison.columns = [f"{a}_{b}" for a, b in comparison.columns]
        comparison = comparison.reset_index()
        comparison.to_csv(reports / "comparison.csv", index=False)
        for _, row in comparison.iterrows():
            mlflow.log_metrics({f"{row['model']}_{k}": float(v) for k, v in row.items() if k != "model"})
        oof = pd.concat(oof_parts, ignore_index=True)
        if not oof.row_id.is_unique or not oof.step.between(101, 250).all():
            raise AssertionError("Las predicciones de validación se solapan o están fuera de desarrollo")
        oof.to_parquet(artifacts / "tree_temporal_oof.parquet", index=False)
        pooled = metrics(oof[TARGET], oof.score)
        mlflow.log_metrics({f"tree_pooled_{k}": v for k, v in pooled.items()})
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for fold_id, part in oof.groupby("fold"):
            precision, recall, _ = precision_recall_curve(part[TARGET], part.score)
            ap = average_precision_score(part[TARGET], part.score)
            axes[0].plot(recall, precision, label=f"Fold {fold_id}, AP={ap:.3f}")
        axes[0].set(xlabel="Recall", ylabel="Precision", title="Árbol baseline: validación temporal", xlim=(0, 1), ylim=(0, 1.03))
        axes[0].legend(loc="lower left", fontsize=8)
        ConfusionMatrixDisplay(confusion_matrix(oof[TARGET], oof.score >= 0.5, labels=[0, 1]),
                               display_labels=["Normal", "Fraude"]).plot(ax=axes[1], colorbar=False, values_format="d")
        axes[1].set_title("Validación combinada; umbral fijo 0,5")
        fig.savefig(reports / "baseline_validation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        # A usable baseline artifact, not a final champion. All steps 1–250 only.
        pipeline = make_pipeline("decision_tree")
        print("Ajustando baseline en development completo", flush=True)
        start = perf_counter()
        pipeline.fit(X, y)
        final_fit_seconds = perf_counter() - start
        rules = export_text(pipeline.named_steps['model'],
                            feature_names=list(pipeline.named_steps['preprocessor'].get_feature_names_out()),
                            decimals=6)
        (reports / 'tree_rules.txt').write_text(rules, encoding='utf-8')
        joblib.dump(pipeline, artifacts / "pipeline_baseline.joblib")
        example = X.head(8).copy()
        for col in example.select_dtypes(include="integer").columns:
            example[col] = example[col].astype(float)
        reloaded = joblib.load(artifacts / "pipeline_baseline.joblib")
        np.testing.assert_allclose(pipeline.predict_proba(example), reloaded.predict_proba(example))
        model_path = artifacts / "mlflow_model"
        if model_path.exists():
            raise FileExistsError("El modelo ya existe: preservar o mover el artefacto antes de repetir")
        # Save flavor locally then upload as run artifacts; works with MLflow 2-style tracking too.
        mlflow.sklearn.save_model(
            sk_model=pipeline, path=str(model_path), code_paths=[str(root / "src")], serialization_format="cloudpickle",
            signature=infer_signature(example, pipeline.predict(example)), input_example=example,
            pip_requirements=[f"{p}=={version(p)}" for p in ["mlflow", "scikit-learn", "pandas", "numpy", "cloudpickle", "joblib", "matplotlib", "pyarrow", "python-dotenv"]],
        )
        loaded = mlflow.sklearn.load_model(str(model_path))
        np.testing.assert_allclose(loaded.predict_proba(example), pipeline.predict_proba(example))
        mlflow.log_metric("baseline_full_development_fit_seconds", final_fit_seconds)
        mlflow.log_artifacts(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(root / "reports/dataset_profile.json"), artifact_path="data_contract")
        summary = {
            "stage": "baseline", "experiment_name": exp_name, "backend": backend, "run_id": run_id,
            "models": comparison.to_dict(orient="records"), "tree_pooled_validation": pooled,
            "fold_count": len(folds), "development_rows": len(data), "oof_rows": len(oof),
            "fixed_threshold": 0.5, "tree_parameters": TREE_PARAMS,
            "threshold_partition_used": False, "offline_test_used": False, "production_used": False,
            "model_role": "baseline_only", "scores_calibrated": False,
            "baseline_full_development_fit_seconds": final_fit_seconds,
            "source_sha256": profile["source_sha256"], "model_sha256": sha256(artifacts / "pipeline_baseline.joblib"),
            "model_reload_verified": True,
        }
        (reports / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifacts(str(reports), artifact_path="baseline_reports")
        mlflow.set_tag("baseline_model_uri", f"runs:/{run_id}/model")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run_baseline()
