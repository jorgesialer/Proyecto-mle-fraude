"""Contrato explícito: detección posterior al registro, target isFraud."""

import numpy as np
import pandas as pd

RAW_COLUMNS = (
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
)
TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
MONEY = ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
TARGET = "isFraud"
FEATURES = (
    "type", *MONEY, "origin_balance_change", "destination_balance_change",
    "amount_to_origin_balance", "origin_zero_balance", "destination_is_merchant",
    "origin_debit_discrepancy", "destination_credit_discrepancy",
)
PARTITIONS = {
    "development": (1, 250),
    "threshold_validation": (251, 300),
    "offline_test": (301, 400),
    "production": (401, 744),
}
# Use only on development. Never split transactions belonging to one step.
CV_WINDOWS = ((1, 100, 101, 150), (1, 150, 151, 200), (1, 200, 201, 250))


def validate_raw(frame: pd.DataFrame) -> None:
    """Fail on schema, target, nonfinite values or unknown categories."""
    if set(frame.columns) != set(RAW_COLUMNS):
        raise ValueError("El esquema no coincide con las 11 columnas esperadas")
    if frame.empty:
        raise ValueError("Bloque vacío")
    if frame.isna().any().any():
        raise ValueError("Hay nulos: investigar antes de preparar")
    numeric = frame.loc[:, ["step", *MONEY, TARGET, "isFlaggedFraud"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Hay valores no finitos")
    if not ((frame.step % 1 == 0) & frame.step.between(1, 744)).all():
        raise ValueError("step debe ser entero entre 1 y 744")
    if not frame.type.isin(TYPES).all():
        raise ValueError("Tipo de transacción desconocido")
    if (frame.loc[:, MONEY] < 0).any().any():
        raise ValueError("Monto o saldo negativo: investigar")
    for col in [TARGET, "isFlaggedFraud"]:
        if not frame[col].isin([0, 1]).all():
            raise ValueError(f"{col} debe ser binaria")


def partition_labels(steps: pd.Series) -> pd.Series:
    """Assign every step exactly once, including the 400/401 boundary."""
    if not np.isfinite(steps.to_numpy(dtype=float)).all():
        raise ValueError("step no finito")
    if not ((steps % 1 == 0) & steps.between(1, 744)).all():
        raise ValueError("step fuera del contrato")
    return pd.Series(
        np.select([steps.le(250), steps.le(300), steps.le(400)],
                  ["development", "threshold_validation", "offline_test"],
                  default="production"), index=steps.index, name="partition",
    )


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Stateless features: no fitting and no use of target or existing alert."""
    required = {"type", *MONEY, "nameDest"}
    if not required.issubset(frame.columns):
        raise ValueError("Faltan variables necesarias para las features")
    x = frame.loc[:, ["type", *MONEY]].copy()
    x["type"] = x.type.astype("string")
    x["origin_balance_change"] = frame.oldbalanceOrg - frame.newbalanceOrig
    x["destination_balance_change"] = frame.newbalanceDest - frame.oldbalanceDest
    # NaN means ratio undefined, not zero. Imputation belongs inside each fold.
    denominator = frame.oldbalanceOrg.where(frame.oldbalanceOrg.ne(0))
    x["amount_to_origin_balance"] = frame.amount.div(denominator)
    x["origin_zero_balance"] = frame.oldbalanceOrg.eq(0).astype("int8")
    x["destination_is_merchant"] = frame.nameDest.str.startswith("M").astype("int8")
    # These describe debit/credit consistency, not a universal accounting rule.
    # CASH_IN increases the origin balance; merchant final balances may be absent.
    x["origin_debit_discrepancy"] = x.origin_balance_change - frame.amount
    x["destination_credit_discrepancy"] = x.destination_balance_change - frame.amount
    if set(x.columns) != set(FEATURES):
        raise AssertionError("El contrato de features cambió")
    numeric = x.drop(columns="type").to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("Las variables derivadas contienen infinito")
    return x.loc[:, FEATURES]


def temporal_folds(steps: pd.Series):
    """Indices for expanding validation within steps 1–250 only."""
    if not steps.between(1, 250).all() or not (steps % 1 == 0).all():
        raise ValueError("Los folds solo admiten development (steps 1–250)")
    for low, high, vlow, vhigh in CV_WINDOWS:
        train = np.flatnonzero(steps.between(low, high).to_numpy())
        valid = np.flatnonzero(steps.between(vlow, vhigh).to_numpy())
        if not len(train) or not len(valid):
            raise ValueError("Fold sin datos de entrenamiento o validación")
        yield train, valid
