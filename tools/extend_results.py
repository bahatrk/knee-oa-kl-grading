"""results.csv'yi tamamlar: (1) hibrit modelin 3 seed satırını ekler, (2) CPU gecikme sütunlarını ekler.

Girdiler (hepsi önceki aşamaların çıktısıdır; yeni eğitim/değerlendirme yapılmaz):
  --results      mevcut results.csv (48 satır)
  --hybrid-val   outputs/hybrid/hybrid_val_runs.csv  (doğrulama metrikleri, 4 üyeli ortalama füzyonu)
  --test-runs    outputs/final/test_runs_all.csv     (test metrikleri; hibrit satırları 'hybrid_mean4')
  --cpu-a        outputs/final/cpu_latency.csv       (oturum A: ilk CPU ölçümü, ortam kaydı yok)
  --cpu-b        outputs/final/cpu_latency_extra.csv (oturum B: 1 iş parçacığı, torch 2.14.1+cpu, Xeon 2.20 GHz, 2 vCPU)
Kullanım: python tools/extend_results.py --results ... --out results.csv
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

MEMBERS = {"Orig": "Orig_full__opt", "DeiT": "DeiTSmall__opt", "ConvNeXt": "ConvNeXtTiny__opt", "ResNet50": "ResNet50__opt"}
VAL = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "cohen_kappa", "qwk",
       "mcc", "mae_grade", "adjacent_error_rate", "distant_error_rate", "adjacent_share_of_errors", "roc_auc_macro",
       "roc_auc_micro", "pr_auc_macro"]
TEST = VAL + ["ece15", "brier"]
CPU_A = "A: ilk CPU oturumu (ortam kaydedilmedi)"
CPU_B = "B: Intel Xeon 2.20 GHz, 2 vCPU, 1 iş parçacığı, torch 2.14.1+cpu"


def add_cpu(df: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("cpu_latency_ms_median", "cpu_latency_ms_p95", "cpu_latency_session"):
        df[c] = pd.NA
    for src, tag in ((a, CPU_A), (b, CPU_B)):          # B, A'nın üzerine yazar (aynı oturumda ölçülen tutarlı küme)
        for _, r in src.iterrows():
            m = df.model_name == r["run"]
            df.loc[m, "cpu_latency_ms_median"] = round(float(r["latency_ms_median"]), 2)
            df.loc[m, "cpu_latency_ms_p95"] = round(float(r["latency_ms_p95"]), 2)
            df.loc[m, "cpu_latency_session"] = tag
    return df


def hybrid_rows(df: pd.DataFrame, hv: pd.DataFrame, tr: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    hv = hv[(hv.method == "mean") & (hv.members == "+".join(MEMBERS))]
    tr = tr[tr.run == "hybrid_mean4"]
    ha = a[a.run == "hybrid_mean4"].iloc[0]
    rows = []
    for seed in sorted(hv.seed.unique()):
        mem = {k: df[(df.model_name == v) & (df.seed == seed)].iloc[0] for k, v in MEMBERS.items()}
        row = {"model_name": "Hybrid_mean4", "model_family": "Hibrit (4 üyenin olasılık ortalaması)", "seed": int(seed),
               "input_size": "224x224x3", "param_count": int(sum(m.param_count for m in mem.values())),
               "optimizer": "AdamW (üyeler)", "loss": "üye olasılıklarının ortalaması",
               "train_time_s": round(float(sum(m.train_time_s for m in mem.values())), 1),
               "model_size_mb": round(float(sum(m.model_size_mb for m in mem.values())), 2),
               "latency_ms_median": round(float(sum(m.latency_ms_median for m in mem.values())), 2),
               "latency_ms_p95": round(float(sum(m.latency_ms_p95 for m in mem.values())), 2),
               "latency_device": "Tesla T4 (üyelerin toplamı)",
               "hyperparams": json.dumps({"members": list(MEMBERS.values()), "fusion": "mean", "n_members": 4}, ensure_ascii=False),
               "notes": "Füzyon ve üyeler yalnızca doğrulama macro F1 ile seçildi (44 yapılandırma); test final aşamada",
               "cpu_latency_ms_median": round(float(ha.latency_ms_median), 2), "cpu_latency_ms_p95": round(float(ha.latency_ms_p95), 2),
               "cpu_latency_session": CPU_A + " (üyelerin toplamı)"}
        v = hv[hv.seed == seed].iloc[0]; t = tr[tr.seed == seed].iloc[0]
        row.update({f"val_{k}": v[k] for k in VAL}); row.update({f"test_{k}": t[k] for k in TEST})
        rows.append(row)
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    for k in ("results", "hybrid-val", "test-runs", "cpu-a", "cpu-b", "out"):
        p.add_argument(f"--{k}", required=True)
    g = p.parse_args()
    df = pd.read_csv(g.results)
    if (df.model_name == "Hybrid_mean4").any():
        raise SystemExit("Hibrit satırları zaten var; çıktı değiştirilmedi.")
    a, b = pd.read_csv(g.cpu_a), pd.read_csv(g.cpu_b)
    out = add_cpu(df, a[a.run != "hybrid_mean4"], b)
    out = hybrid_rows(out, pd.read_csv(g.hybrid_val), pd.read_csv(g.test_runs), a)
    cols = list(df.columns) + [c for c in out.columns if c not in df.columns]
    out[cols].to_csv(g.out, index=False)
    print(f"{len(df)} -> {len(out)} satır; CPU gecikme dolu: {out.cpu_latency_ms_median.notna().sum()}")


if __name__ == "__main__":
    main()
