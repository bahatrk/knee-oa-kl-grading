"""results.csv yönetimi: her model ve seed için tek satır (proje belgesinin istediği sütunlar).

Aynı (model_name, seed) satırı tekrar yazılırsa günceller; test metrikleri yalnızca
final değerlendirme aşamasında ayrı olarak eklenir (update_test_metrics).
"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C

RESULTS_CSV = C.OUTPUT_DIR / "results.csv"
BASE_COLS = ["model_name", "model_family", "seed", "input_size", "param_count", "epochs", "batch_size",
             "optimizer", "learning_rate", "loss", "train_time_s", "model_size_mb",
             "latency_ms_median", "latency_ms_p95", "latency_device", "hyperparams", "notes"]


def _load() -> pd.DataFrame:
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame(columns=BASE_COLS)


def upsert_result(row: dict, val_metrics: dict | None = None) -> pd.DataFrame:
    row = dict(row)
    if isinstance(row.get("hyperparams"), dict):
        row["hyperparams"] = json.dumps(row["hyperparams"], ensure_ascii=False)
    for k, v in (val_metrics or {}).items():
        row[f"val_{k}"] = v
    df = _load()
    mask = (df.model_name == row["model_name"]) & (df.seed == row["seed"])
    if mask.any():
        for k, v in row.items():
            if k not in df.columns:
                df[k] = pd.NA
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    return df


def update_test_metrics(model_name: str, seed: int, test_metrics: dict) -> pd.DataFrame:
    """YALNIZCA final değerlendirmede çağrılır."""
    df = _load()
    mask = (df.model_name == model_name) & (df.seed == seed)
    if not mask.any():
        raise KeyError(f"{model_name}/{seed} results.csv içinde yok")
    for k, v in test_metrics.items():
        col = f"test_{k}"
        if col not in df.columns:
            df[col] = pd.NA
        df.loc[mask, col] = v
    df.to_csv(RESULTS_CSV, index=False)
    return df
