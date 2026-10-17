"""Profile the full CSV and materialize independently verifiable partitions."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from .schema import FEATURES, MONEY, PARTITIONS, TARGET, engineer_features, partition_labels, validate_raw

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(csv_path: Path, root: Path = ROOT, chunksize: int = 200_000) -> dict:
    """Read full source without sampling; only stateless transforms are applied."""
    csv_path = Path(csv_path).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError("Configura PAYSIM_CSV con la ruta al CSV original")
    if chunksize < 1:
        raise ValueError("chunksize debe ser positivo")
    processed = root / "data/processed"
    reports = root / 'artifacts/run_reports'
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(csv_path)
    stat = csv_path.stat()
    partitions = {name: {"rows": 0, "frauds": 0, "step_min": None, "step_max": None}
                  for name in PARTITIONS}
    nulls, zeros, by_step, by_type, feature_nulls = {}, {}, [], [], {}
    samples, writers = {}, {}
    row_offset = 0
    flagged = 0
    previous_step = 0
    source_ordered = True
    try:
        for frame in pd.read_csv(csv_path, chunksize=chunksize):
            validate_raw(frame)
            source_ordered &= bool(frame.step.is_monotonic_increasing and frame.step.iloc[0] >= previous_step)
            previous_step = int(frame.step.iloc[-1])
            for col, value in frame.isna().sum().items():
                nulls[col] = nulls.get(col, 0) + int(value)
            for col in MONEY:
                zeros[col] = zeros.get(col, 0) + int(frame[col].eq(0).sum())
            flagged += int(frame.isFlaggedFraud.sum())
            by_step.append(frame.groupby("step")[TARGET].agg(rows="size", frauds="sum"))
            by_type.append(frame.groupby("type")[TARGET].agg(rows="size", frauds="sum"))
            x = engineer_features(frame)
            labels = partition_labels(frame.step)
            x.insert(0, "row_id", np.arange(row_offset, row_offset + len(frame), dtype="int64"))
            x["step"] = frame.step.astype("int16")
            x[TARGET] = frame[TARGET].astype("int8")
            for col, value in x.isna().sum().items():
                feature_nulls[col] = feature_nulls.get(col, 0) + int(value)
            for name in PARTITIONS:
                mask = labels.eq(name)
                if not mask.any():
                    continue
                part = x.loc[mask]
                d = partitions[name]
                d["rows"] += len(part)
                d["frauds"] += int(part[TARGET].sum())
                low, high = int(part.step.min()), int(part.step.max())
                d["step_min"] = low if d["step_min"] is None else min(low, d["step_min"])
                d["step_max"] = high if d["step_max"] is None else max(high, d["step_max"])
                table = pa.Table.from_pandas(part, preserve_index=False)
                if name not in writers:
                    writers[name] = pq.ParquetWriter(processed / f"{name}.parquet", table.schema, compression="zstd")
                writers[name].write_table(table)
                if name not in samples:
                    sample = frame.loc[mask].head(50).copy()
                    sample.insert(0, "row_id", x.loc[sample.index, "row_id"])
                    sample["partition"] = name
                    samples[name] = sample
            row_offset += len(frame)
            print(f"Preparadas {row_offset:,} filas", flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    if any(d["rows"] == 0 for d in partitions.values()):
        raise ValueError("El archivo no cubre las cuatro particiones")
    if csv_path.stat().st_size != stat.st_size or csv_path.stat().st_mtime_ns != stat.st_mtime_ns:
        raise RuntimeError("El CSV cambió durante la preparación")
    for name, d in partitions.items():
        d["fraud_rate"] = d["frauds"] / d["rows"]
        path = processed / f"{name}.parquet"
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != d["rows"]:
            raise AssertionError("El conteo Parquet no coincide")
        low, high = PARTITIONS[name]
        if not low <= d["step_min"] <= d["step_max"] <= high:
            raise AssertionError("Partición fuera de su ventana temporal")
        d["parquet_sha256"] = sha256(path)
    if sum(d["rows"] for d in partitions.values()) != row_offset:
        raise AssertionError("No se conservaron todas las filas")
    step_table = pd.concat(by_step).groupby(level=0).sum().sort_index()
    type_table = pd.concat(by_type).groupby(level=0).sum().sort_index()
    step_table["fraud_rate"] = step_table.frauds / step_table.rows
    type_table["fraud_rate"] = type_table.frauds / type_table.rows
    step_table.to_csv(reports / "profile_by_step.csv")
    type_table.to_csv(reports / "profile_by_type.csv")
    # Illustration only: first 50 rows from each partition, never training data.
    pd.concat(samples.values(), ignore_index=True).to_csv(root / "data/sample_illustrative.csv", index=False)
    summary = {
        "source_filename": csv_path.name, "source_sha256": source_hash,
        "source_bytes": stat.st_size, "rows": row_offset,
        "frauds": sum(d["frauds"] for d in partitions.values()),
        "source_ordered_by_step": source_ordered,
        "raw_nulls": nulls, "raw_zeros": zeros, "feature_nulls": feature_nulls,
        "existing_rule_alerts": flagged, "partitions": partitions,
        "features": list(FEATURES), "target": TARGET,
        "excluded_predictors": [TARGET, "isFlaggedFraud", "nameOrig", "nameDest", "step", "row_id"],
        "prediction_moment": "post_transaction",
        "sampling": "none; all rows retained",
        "sample_note": "First 50 rows per partition; illustrative, not representative or used for fitting",
    }
    (reports / "dataset_profile.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=os.getenv("PAYSIM_CSV"))
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    if not args.csv:
        parser.error("Indica --csv o PAYSIM_CSV en .env")
    path = Path(args.csv)
    if not path.is_absolute():
        path = ROOT / path
    summary = prepare(path, chunksize=args.chunksize)
    print(json.dumps(summary["partitions"], indent=2))


if __name__ == "__main__":
    main()
