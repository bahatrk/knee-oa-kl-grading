"""Aşama 8: Nihai test değerlendirmesi (test kümesi TEK sefer, tüm seçimler dondurulduktan sonra).

Akış
  freeze()       seçimleri (en iyi mevcut model, hibrit, ablasyon hedefi, dengeleme) test görülmeden diske yazar.
  predict_all()  her kayıtlı modelin test olasılıklarını bir kez hesaplayıp saklar (yeniden hesaplanmaz).
                 manifest: model.pt SHA256 + zaman damgası (modellerin test öncesi dondurulduğunun kanıtı).
  analyze()      yalnızca saklanan olasılıklarla: ana tablo, bootstrap GA (1000), kalibrasyon (ECE 15 kutu, Brier,
                 sıcaklık ölçekleme - sıcaklık YALNIZCA doğrulamada), eşleştirilmiş bootstrap, McNemar,
                 yinelenen görüntüler hariç duyarlılık analizi, sınıf başına raporlar, ROC/PR dizileri.
  cpu_latency()  CPU, batch 1, 10 ısınma, >=100 ölçüm, medyan + p95.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config as C
from .ablation import mcnemar
from .data import make_loaders, read_split
from .hybrid import MEMBERS, fuse
from .metrics import compute_metrics, confusion, per_class_report
from .models import build_model, count_params
from .results import update_test_metrics
from .train import measure_latency, predict
from .utils import get_logger, load_json, save_json

log = get_logger("final_eval", C.LOG_DIR)
RUNS = C.OUTPUT_DIR / "runs"
OUT = C.OUTPUT_DIR / "final"
K = C.NUM_CLASSES
N_BOOT = 1000

# Ana tablo: (görünen ad, koşu adı, seed'ler). Çok seed'li satırlarda ortalama ± std verilir.
MAIN = [
    ("SimpleCNN", "SimpleCNN", [42]),
    ("MiniResNet", "MiniResNet", [42]),
    ("ResNet50*", "ResNet50__opt", list(C.SEEDS)),
    ("DenseNet121", "DenseNet121", [42]),
    ("EfficientNetB0", "EfficientNetB0", [42]),
    ("MobileNetV3-Large", "MobileNetV3L", [42]),
    ("Xception", "Xception", [42]),
    ("InceptionV3", "InceptionV3", [42]),
    ("ConvNeXt-Tiny*", "ConvNeXtTiny__opt", list(C.SEEDS)),
    ("DeiT-Small*", "DeiTSmall__opt", list(C.SEEDS)),
    ("Original (KneeOrdinalNet)*", "Orig_full__opt", list(C.SEEDS)),
]
HYBRID_NAME = "Hybrid (mean of 4)"
APPENDIX = [  # test'te değerlendirilen ek koşular (seçim için kullanılmaz)
    ("ResNet50 (default hp)", "ResNet50", [42]), ("ConvNeXt-Tiny (default hp)", "ConvNeXtTiny", [42]),
    ("DeiT-Small (default hp)", "DeiTSmall", [42]),
    ("Ablation: - multiscale", "Orig_noMS__opt", list(C.SEEDS)), ("Ablation: - row attention", "Orig_noRow__opt", list(C.SEEDS)),
    ("Ablation: - ordinal loss", "Orig_noOrd__opt", list(C.SEEDS)),
    ("Ablation: - input standardization", "Orig_v3__opt", list(C.SEEDS)),
    ("Control: plain ConvNeXt, same hp", "Orig_base__opt", list(C.SEEDS)),
]
BOOT_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa", "qwk", "mcc", "mae_grade", "distant_error_rate"]


# --------------------------------------------------------------------------- dondurma
def freeze() -> dict:
    """Test öncesi seçimleri kaydeder; gerekli dosyalar yoksa durur."""
    need = ["best_existing_model.json", "balancing_choice.json", "hybrid_choice.json", "ablation_mcnemar.json"]
    miss = [f for f in need if not (C.OUTPUT_DIR / f).exists()]
    assert not miss, f"Eksik seçim dosyaları: {miss}"
    OUT.mkdir(parents=True, exist_ok=True)
    mp = C.OUTPUT_DIR / "ablation_mcnemar.json"
    if "most_contributing_component_removed" not in load_json(mp):  # eski sürümle üretilmiş dosya: doğrulamadan yeniden üret
        from .ablation import summarize
        summarize()
        log.info("ablation_mcnemar.json doğrulama verisiyle yeniden üretildi")
    fz = {"frozen_at": datetime.now().isoformat(timespec="seconds"),
          "best_existing_model": load_json(C.OUTPUT_DIR / "best_existing_model.json")["best_existing_model"],
          "balancing": load_json(C.OUTPUT_DIR / "balancing_choice.json"),
          "hybrid": load_json(C.OUTPUT_DIR / "hybrid_choice.json"),
          "mcnemar_target": load_json(C.OUTPUT_DIR / "ablation_mcnemar.json")["most_contributing_component_removed"],
          "note": "Bu seçimler test kümesi görülmeden yapıldı."}
    if not (OUT / "frozen_selections.json").exists():  # ikinci çağrıda ezilmez
        save_json(fz, OUT / "frozen_selections.json")
    return load_json(OUT / "frozen_selections.json")


# --------------------------------------------------------------------------- tahminler
def _sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_model(name: str, seed: int, device):
    d = RUNS / name / f"seed{seed}"
    meta = load_json(d / "meta.json")
    model = build_model(meta["model_key"], pretrained=False, drop=meta.get("drop", 0.3), **meta.get("model_kwargs", {}))
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    return model.to(device).eval(), meta


def predict_test(name: str, seed: int, device=None, batch_size: int = 64, num_workers: int = 2):
    d = RUNS / name / f"seed{seed}"
    if (d / "test_probs.npy").exists():
        return np.load(d / "test_probs.npy"), np.load(d / "test_labels.npy")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(name, seed, device)
    loader = make_loaders(tuple(meta["mean"]), tuple(meta["std"]), batch_size, num_workers, augment=False,
                          splits=("test",), seed=seed)["test"]
    logits, y = predict(model, loader, device, amp=True)
    probs = torch.softmax(logits, 1).numpy()
    np.save(d / "test_probs.npy", probs)
    np.save(d / "test_labels.npy", y)
    man = load_json(OUT / "test_eval_manifest.json") if (OUT / "test_eval_manifest.json").exists() else {}
    man[f"{name}/seed{seed}"] = {"model_sha256": _sha256(d / "model.pt"),
                                 "evaluated_at": datetime.now().isoformat(timespec="seconds"), "n_test": int(len(y))}
    save_json(man, OUT / "test_eval_manifest.json")
    del model
    return probs, y


def all_runs(include_appendix: bool = True):
    rows = MAIN + (APPENDIX if include_appendix else [])
    return [(label, name, s) for label, name, seeds in rows for s in seeds]


def predict_all(include_appendix: bool = True, **kw):
    freeze()
    for label, name, seed in all_runs(include_appendix):
        t = time.perf_counter()
        predict_test(name, seed, **kw)
        log.info("test olasılıkları: %s seed %d (%.0f sn)", name, seed, time.perf_counter() - t)


# --------------------------------------------------------------------------- hızlı metrikler (karışıklık matrisinden)
def metrics_from_cm(cm: np.ndarray) -> dict:
    cm = cm.astype(np.float64)
    n = cm.sum(); row, col, tp = cm.sum(1), cm.sum(0), np.diag(cm)
    rec = np.divide(tp, row, out=np.zeros(K), where=row > 0)
    prec = np.divide(tp, col, out=np.zeros(K), where=col > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    acc = tp.sum() / n
    pe = (row * col).sum() / n ** 2
    i, j = np.indices((K, K))
    w = (i - j) ** 2 / (K - 1) ** 2
    exp = np.outer(row, col) / n
    qwk = 1 - (w * cm).sum() / (w * exp).sum()
    den = np.sqrt((n ** 2 - (col ** 2).sum()) * (n ** 2 - (row ** 2).sum()))
    mcc = (tp.sum() * n - (row * col).sum()) / den if den > 0 else 0.0
    d = np.abs(i - j)
    return {"accuracy": acc, "balanced_accuracy": rec.mean(), "macro_f1": f1.mean(),
            "cohen_kappa": (acc - pe) / (1 - pe) if pe < 1 else 0.0, "qwk": qwk, "mcc": mcc,
            "mae_grade": (d * cm).sum() / n, "distant_error_rate": cm[d >= 2].sum() / n}


def _cm(y, pred):
    return np.bincount(y * K + pred, minlength=K * K).reshape(K, K)


def boot_indices(n: int, b: int = N_BOOT, seed: int = C.SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (b, n))


def boot_metrics(y, pred, idx) -> pd.DataFrame:
    return pd.DataFrame([metrics_from_cm(_cm(y[ix], pred[ix])) for ix in idx])[BOOT_METRICS]


def ci(series, lo=2.5, hi=97.5):
    return float(np.percentile(series, lo)), float(np.percentile(series, hi))


# --------------------------------------------------------------------------- kalibrasyon
def ece_score(probs, y, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def brier_score(probs, y) -> float:
    return float(((probs - np.eye(K)[y]) ** 2).sum(1).mean())


def temper(probs, T):
    z = np.log(np.clip(probs, 1e-12, 1)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(val_probs, val_y) -> float:
    """Sıcaklık YALNIZCA doğrulama kümesinde (NLL en küçük)."""
    def nll(logT):
        p = temper(val_probs, np.exp(logT))
        return -np.log(p[np.arange(len(val_y)), val_y] + 1e-12).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


# --------------------------------------------------------------------------- yardımcılar
def _val(name, seed):
    d = RUNS / name / f"seed{seed}"
    return np.load(d / "val_probs.npy"), np.load(d / "val_labels.npy")


def _test(name, seed):
    d = RUNS / name / f"seed{seed}"
    return np.load(d / "test_probs.npy"), np.load(d / "test_labels.npy")


def hybrid_probs(seed: int, split: str = "test", members=None):
    members = members or load_json(OUT / "frozen_selections.json")["hybrid"]["members"]
    f = _test if split == "test" else _val
    Ps = [f(MEMBERS[m], seed)[0] for m in members]
    y = f(MEMBERS[members[0]], seed)[1]
    return fuse("mean", Ps)[0], y


def dup_mask() -> np.ndarray:
    """Test kümesinde, doğrulanmış yinelenen görüntü çiftlerine karışanlar hariç tutulur (True = tut)."""
    te = read_split("test")
    p = C.AUDIT_DIR / "confirmed_duplicates.csv"
    keep = np.ones(len(te), bool)
    if not p.exists():
        return keep
    d = pd.read_csv(p)
    d = d[d.confirmed_duplicate.astype(bool)]
    drop = set()
    for _, r in d.iterrows():
        if r.split_i == "test" and r.split_j != "test":
            drop.add(r.relpath_i)
        elif r.split_j == "test" and r.split_i != "test":
            drop.add(r.relpath_j)
        elif r.split_i == "test" and r.split_j == "test":
            drop.add(r.relpath_j)
    return ~te.relpath.isin(drop).values


# --------------------------------------------------------------------------- analiz
def analyze(include_appendix: bool = True) -> dict:
    fz = freeze()
    OUT.mkdir(parents=True, exist_ok=True)
    labels = {name: label for label, name, _ in MAIN + APPENDIX}
    runs = [(l, n, s) for l, n, s in all_runs(include_appendix)]
    y0 = _test(runs[0][1], runs[0][2])[1]
    idx = boot_indices(len(y0))
    keep = dup_mask()
    rows, ci_rows, cal_rows, pc_rows, curves = [], [], [], [], {}

    def add(label, name, seed, probs, y, val_probs, val_y):
        pred = probs.argmax(1)
        m = compute_metrics(y, pred, probs)
        T = fit_temperature(val_probs, val_y)
        pt = temper(probs, T)
        cal = {"ece15": ece_score(probs, y), "brier": brier_score(probs, y), "T": T,
               "ece15_temp": ece_score(pt, y), "brier_temp": brier_score(pt, y)}
        bm = boot_metrics(y, pred, idx)
        for k in BOOT_METRICS:
            lo, hi = ci(bm[k])
            ci_rows.append({"model": label, "run": name, "seed": seed, "metric": k, "value": m[k], "ci_lo": lo, "ci_hi": hi})
        mk = keep & np.ones(len(y), bool)
        ms = compute_metrics(y[mk], pred[mk], probs[mk])
        rows.append({"model": label, "run": name, "seed": seed, **m, **{f"nodup_{k}": ms[k] for k in
                     ("macro_f1", "balanced_accuracy", "qwk", "accuracy", "mae_grade")}, **cal})
        cal_rows.append({"model": label, "run": name, "seed": seed, **cal})
        pc = per_class_report(y, pred)
        pc.insert(0, "seed", seed); pc.insert(0, "run", name); pc.insert(0, "model", label)
        pc_rows.append(pc)
        np.save(OUT / f"confusion_{name}_seed{seed}.npy", confusion(y, pred))
        if seed == C.SEED:
            curves[f"{name}"] = (probs, y)
        return m

    for label, name, seed in runs:
        probs, y = _test(name, seed); vp, vy = _val(name, seed)
        m = add(label, name, seed, probs, y, vp, vy)
        try:
            update_test_metrics(name, seed, {**m, "ece15": rows[-1]["ece15"], "brier": rows[-1]["brier"]})
        except KeyError:
            pass
    for seed in C.SEEDS:  # hibrit: üyelerin olasılık ortalaması
        probs, y = hybrid_probs(seed, "test"); vp, vy = hybrid_probs(seed, "val")
        add(HYBRID_NAME, "hybrid_mean4", seed, probs, y, vp, vy)
        np.save(OUT / f"hybrid_test_probs_seed{seed}.npy", probs)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "test_runs_all.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(OUT / "bootstrap_ci.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(OUT / "calibration.csv", index=False)
    pd.concat(pc_rows).to_csv(OUT / "per_class_test.csv", index=False)

    # ana tablo: seed ortalaması ± std
    keys = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "cohen_kappa", "mcc",
            "roc_auc_macro", "qwk", "mae_grade", "distant_error_rate", "ece15", "brier", "ece15_temp"]
    g = R.groupby(["model", "run"], sort=False)
    main = g[keys].mean().add_suffix("_mean").join(g[keys].std(ddof=1).add_suffix("_std")).reset_index()
    main["n_seeds"] = g.seed.nunique().values
    main.to_csv(OUT / "main_table_test_raw.csv", index=False)

    # eşleştirilmiş bootstrap (aynı yeniden örnekleme indeksleri) ve McNemar
    best = fz["best_existing_model"]
    pairs = [("Orig_full__opt", best), ("Orig_full__opt", "ConvNeXtTiny__opt"), ("Orig_full__opt", "Orig_base__opt"),
             ("hybrid_mean4", "Orig_full__opt"), ("hybrid_mean4", best)]
    pb = []
    for a, b in pairs:
        for seed in C.SEEDS:
            try:
                pa = hybrid_probs(seed)[0] if a == "hybrid_mean4" else _test(a, seed)[0]
                pbb = hybrid_probs(seed)[0] if b == "hybrid_mean4" else _test(b, seed)[0]
            except FileNotFoundError:
                continue
            y = y0
            da = boot_metrics(y, pa.argmax(1), idx); db = boot_metrics(y, pbb.argmax(1), idx)
            for k in ("macro_f1", "qwk", "mae_grade", "accuracy", "distant_error_rate"):
                d = da[k] - db[k]
                lo, hi = ci(d)
                pb.append({"A": a, "B": b, "seed": seed, "metric": k, "diff_A_minus_B": float(d.mean()),
                           "ci_lo": lo, "ci_hi": hi, "ci_excludes_0": bool(lo > 0 or hi < 0)})
    pd.DataFrame(pb).to_csv(OUT / "paired_bootstrap.csv", index=False)

    mc = {}
    tgt = fz["mcnemar_target"]
    for seed in C.SEEDS:
        a, y = _test("Orig_full__opt", seed); b, _ = _test(tgt, seed)
        mc[seed] = mcnemar(a.argmax(1) == y, b.argmax(1) == y)
    save_json({"reference": "Orig_full__opt", "component_removed_version": tgt, "split": "test",
               "chosen_on": "validation (en büyük macro F1 düşüşü)", "per_seed": mc}, OUT / "mcnemar_test.json")
    np.savez_compressed(OUT / "curves_seed42.npz", **{f"{k}__probs": v[0] for k, v in curves.items()},
                        **{f"{k}__y": v[1] for k, v in curves.items()})
    log.info("analiz bitti: %s", OUT)
    return {"n_runs": len(R), "n_dups_excluded": int((~keep).sum())}


def curve_arrays(probs, y):
    """Sınıf başına bire-karşı ROC ve PR eğrileri + makro ortalama için ortak ızgara (şekil aşaması için)."""
    out = {}
    for k in range(K):
        t = (y == k).astype(int)
        fpr, tpr, _ = roc_curve(t, probs[:, k]); pr, rc, _ = precision_recall_curve(t, probs[:, k])
        out[k] = {"fpr": fpr, "tpr": tpr, "precision": pr, "recall": rc}
    return out


# --------------------------------------------------------------------------- CPU gecikme
def cpu_latency(names=None, n: int = 110, warmup: int = 10) -> pd.DataFrame:
    """CPU, batch 1: 10 ısınma, 100 ölçüm; medyan + p95. Hibrit = üyelerin toplamı (sıralı çalıştırma)."""
    dev = torch.device("cpu")
    names = names or [(l, nm) for l, nm, _ in MAIN]
    rows = []
    for label, name in names:
        model, _ = load_model(name, C.SEED, dev)
        lat = measure_latency(model, dev, n=n, warmup=warmup)
        rows.append({"model": label, "run": name, "params": count_params(model),
                     "size_mb": round((RUNS / name / f"seed{C.SEED}" / "model.pt").stat().st_size / 2**20, 2), **lat})
        del model
    df = pd.DataFrame(rows).drop_duplicates("run")
    mem = [r for r in rows if r["run"] in {MEMBERS[m] for m in load_json(OUT / "frozen_selections.json")["hybrid"]["members"]}]
    if mem:
        df = pd.concat([df, pd.DataFrame([{"model": HYBRID_NAME, "run": "hybrid_mean4",
                                           "params": sum(r["params"] for r in mem), "size_mb": round(sum(r["size_mb"] for r in mem), 2),
                                           "latency_ms_median": sum(r["latency_ms_median"] for r in mem),
                                           "latency_ms_p95": sum(r["latency_ms_p95"] for r in mem),
                                           "latency_device": "CPU (sum of members)"}])])
    df.to_csv(OUT / "cpu_latency.csv", index=False)
    return df


# --------------------------------------------------------------------------- HOG+SVM (taban çizgisi) test özeti
def baseline_summary() -> pd.DataFrame:
    """HOG+SVM test metrikleri ve bootstrap GA. `python -m src.baseline --final-test` önce çalıştırılmış olmalı."""
    from .metrics import softmax
    d = C.OUTPUT_DIR / "models" / "baseline_hog_svm"
    sc = np.load(d / "test_scores.npy"); y = read_split("test").label.values
    pred = sc.argmax(1)
    m = compute_metrics(y, pred, softmax(sc))
    bm = boot_metrics(y, pred, boot_indices(len(y)))
    row = {"model": "HOG+LinearSVM", **{k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
                                                          "cohen_kappa", "mcc", "roc_auc_macro", "qwk", "mae_grade", "distant_error_rate")}}
    for k in BOOT_METRICS:
        lo, hi = ci(bm[k]); row[f"{k}_ci"] = f"[{lo:.4f}, {hi:.4f}]"
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row]); df.to_csv(OUT / "baseline_test.csv", index=False)
    return df

# --------------------------------------------------------------------------- ek CPU gecikme (ablasyon + kontrol)
EXTRA_LAT = [("Original (full)", "Orig_full__opt"), ("ConvNeXt-Tiny*", "ConvNeXtTiny__opt"), ("DeiT-Small*", "DeiTSmall__opt"),
             ("Control: plain ConvNeXt, same hp", "Orig_base__opt"), ("Ablation: - multiscale", "Orig_noMS__opt"),
             ("Ablation: - row attention", "Orig_noRow__opt"), ("Ablation: - ordinal loss", "Orig_noOrd__opt"),
             ("Ablation: - input standardization", "Orig_v3__opt")]


def cpu_latency_extra(n: int = 210, warmup: int = 10, repeats: int = 3) -> pd.DataFrame:
    """Ablasyon ve kontrol için CPU gecikmesi (batch 1). Aynı oturumda tam model, ConvNeXt ve DeiT de yeniden ölçülür
    (referans); modeller sırayla ``repeats`` tur ölçülür, tur medyanlarının medyanı raporlanır (sıra etkisini azaltır)."""
    import torch
    dev = torch.device("cpu")
    models = {nm: load_model(nm, C.SEED, dev)[0] for _, nm in EXTRA_LAT}
    acc = {nm: [] for nm in models}
    for _ in range(repeats):
        for nm, m in models.items():
            acc[nm].append(measure_latency(m, dev, n=n, warmup=warmup))
    rows = []
    for label, nm in EXTRA_LAT:
        med = [r["latency_ms_median"] for r in acc[nm]]; p95 = [r["latency_ms_p95"] for r in acc[nm]]
        rows.append({"model": label, "run": nm, "params": count_params(models[nm]),
                     "latency_ms_median": float(np.median(med)), "latency_ms_p95": float(np.median(p95)),
                     "round_medians": ";".join(f"{v:.1f}" for v in med), "threads": torch.get_num_threads(),
                     "latency_device": "CPU"})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "cpu_latency_extra.csv", index=False)
    return df
