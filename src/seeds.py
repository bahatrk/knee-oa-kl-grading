"""Aşama 7a: en iyi 3 (Optuna ile ayarlı) model için seed 42 / 123 / 2026 tekrarları ve özet.

Veri bölmesi sabit (seed 42'de yapılmış resmi liste); yalnızca ağırlık başlatma, veri sırası ve augmentation
rastgeleliği seed ile değişir. Özet YALNIZCA validation metriklerini kullanır.
"En iyi mevcut model" kuralı (test görülmeden sabitlenir): 3 seed ortalama validation macro F1'i en yüksek model.
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from .optimize import retrain_best
from .utils import get_logger, save_json

log = get_logger("seeds", C.LOG_DIR)
METRICS = ["val_macro_f1", "val_balanced_accuracy", "val_accuracy", "val_qwk", "val_mae_grade",
           "val_roc_auc_macro", "val_mcc", "val_cohen_kappa", "val_distant_error_rate"]


def run_seeds(keys: list[str], seeds=(123, 2026)) -> None:
    for s in seeds:          # önce her seed turu, böylece kota biterse her modelin >= 2 seed'i olur
        for k in keys:
            retrain_best(k, seed=s)


def summarize(keys: list[str], seeds=tuple(C.SEEDS)) -> pd.DataFrame:
    r = pd.read_csv(C.OUTPUT_DIR / "results.csv")
    rows = []
    for k in keys:
        sub = r[(r.model_name == f"{k}__opt") & r.seed.isin(seeds)]
        if len(sub) < len(seeds):
            log.warning("%s: yalnızca %d/%d seed tamam", k, len(sub), len(seeds))
        row = {"model": f"{k}__opt", "n_seeds": len(sub)}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_std"] = sub[m].mean(), sub[m].std()
        row["param_count"] = int(sub.param_count.iloc[0])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("val_macro_f1_mean", ascending=False).reset_index(drop=True)
    df.to_csv(C.OUTPUT_DIR / "seed_summary.csv", index=False)
    complete = df[df.n_seeds == len(seeds)]
    pick = (complete if len(complete) else df).iloc[0]
    save_json({"best_existing_model": pick.model,
               "rule": "3 seed ortalama validation macro F1'i en yüksek model (test kullanılmadı)",
               "val_macro_f1_mean": pick.val_macro_f1_mean, "val_macro_f1_std": pick.val_macro_f1_std,
               "ranking": df[["model", "val_macro_f1_mean", "val_macro_f1_std", "val_qwk_mean"]].to_dict("records")},
              C.OUTPUT_DIR / "best_existing_model.json")
    return df
