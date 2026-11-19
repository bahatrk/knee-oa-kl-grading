"""Özgün model ablasyonları: her bileşen tek tek çıkarılır, 3 seed, McNemar testi.

Tüm varyantlar özgün modelin Optuna'dan çıkan hiperparametreleriyle (tuned) eğitilir; böylece fark yalnızca
çıkarılan bileşenden gelir. Bu modül yalnızca VALIDATION verisini kullanır (test final aşamasında aynı fonksiyonlarla).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from . import config as C
from .original import ABLATIONS, train_variant
from .utils import get_logger, save_json

log = get_logger("ablation", C.LOG_DIR)


def run_ablations(seeds=C.SEEDS, variants=None):
    variants = variants or list(ABLATIONS.values())
    for seed in seeds:
        for v in variants:
            train_variant(v, seed=seed, tuned=True)


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Eşleşmiş doğru/yanlış dizileri için kesin (exact) McNemar testi. b: A doğru-B yanlış, c: A yanlış-B doğru."""
    b = int(np.sum(correct_a & ~correct_b)); c = int(np.sum(~correct_a & correct_b))
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return {"a_only_correct": b, "b_only_correct": c, "p_value": p}


def _correct(name: str, seed: int, split: str = "val") -> np.ndarray:
    d = C.OUTPUT_DIR / "runs" / name / f"seed{seed}"
    return np.load(d / f"{split}_probs.npy").argmax(1) == np.load(d / f"{split}_labels.npy")


def summarize(seeds=C.SEEDS, reference: str = "Orig_full__opt") -> pd.DataFrame:
    r = pd.read_csv(C.OUTPUT_DIR / "results.csv")
    cols = ["val_macro_f1", "val_balanced_accuracy", "val_accuracy", "val_qwk", "val_mae_grade",
            "val_distant_error_rate", "val_roc_auc_macro"]
    rows = []
    for label, v in ABLATIONS.items():
        name = v + "__opt"
        s = r[(r.model_name == name) & r.seed.isin(seeds)]
        row = {"variant": label, "run": name, "n_seeds": len(s), "param_count": int(s.param_count.iloc[0]) if len(s) else None}
        for c in cols:
            row[c + "_mean"], row[c + "_std"] = s[c].mean(), s[c].std(ddof=1)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["delta_f1_vs_full"] = df.val_macro_f1_mean - df.loc[df.run == reference, "val_macro_f1_mean"].iloc[0]
    df.to_csv(C.OUTPUT_DIR / "ablation_summary.csv", index=False)

    # McNemar (proje kuralı): tam model ile EN ÇOK katkı veren bileşenin çıkarıldığı sürüm, yani doğrulama
    # macro F1'i tam modele göre en çok düşen ablasyon; seed başına
    abl = df[df.run != reference].sort_values("delta_f1_vs_full", ascending=True)
    best = abl.run.iloc[0]
    tests = {s: mcnemar(_correct(reference, s), _correct(best, s)) for s in seeds
             if (C.OUTPUT_DIR / "runs" / best / f"seed{s}" / "val_probs.npy").exists()}
    save_json({"reference": reference, "most_contributing_component_removed": best, "split": "val",
               "note": "seed başına ayrı test; seed'ler arası bağımsız değildir, p-değerleri birleştirilmez",
               "per_seed": tests}, C.OUTPUT_DIR / "ablation_mcnemar.json")
    return df


def run_control(seeds=C.SEEDS):
    """Kontrol: özgün bileşenleri olmayan ConvNeXt-Tiny, özgün modelin Optuna hiperparametreleriyle."""
    for seed in seeds:
        train_variant("Orig_base", seed=seed, tuned=True)


def control_summary(seeds=C.SEEDS, reference: str = "Orig_full__opt", control: str = "Orig_base__opt") -> pd.DataFrame:
    """Tam model ve kontrol (ve tüm ablasyonlar) için seed başına McNemar + kontrolün 3 seed ortalaması."""
    r = pd.read_csv(C.OUTPUT_DIR / "results.csv")
    cols = ["val_macro_f1", "val_qwk", "val_mae_grade", "val_distant_error_rate", "val_accuracy"]
    s = r[(r.model_name == control) & r.seed.isin(seeds)]
    out = {c: (s[c].mean(), s[c].std(ddof=1)) for c in cols}
    others = [v + "__opt" for v in ABLATIONS.values() if v + "__opt" != reference] + [control]
    rows = []
    for o in others:
        for sd in seeds:
            if (C.OUTPUT_DIR / "runs" / o / f"seed{sd}" / "val_probs.npy").exists():
                rows.append({"vs": o, "seed": sd, **mcnemar(_correct(reference, sd), _correct(o, sd))})
    mc = pd.DataFrame(rows)
    mc.to_csv(C.OUTPUT_DIR / "mcnemar_all.csv", index=False)
    save_json({"control": control, "n_seeds": len(s), "mean_std": out}, C.OUTPUT_DIR / "control_summary.json")
    return mc
