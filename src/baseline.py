"""Aşama 3: Klasik karşılaştırma modeli — HOG öznitelikleri + doğrusal SVM.

Protokol
- Öznitelik: skimage HOG (9 yön, L2-Hys blok normalizasyonu), gri seviye 224x224 görüntü.
- Sınıflandırıcı: StandardScaler + LinearSVC.
- Seçim: HOG hücre boyutu x C x sınıf ağırlığı (yok / dengeli) ızgarası yalnızca
  DOĞRULAMA kümesinde macro F1 ile seçilir. Sınıf ağırlığı karşılaştırması, dengesizlik
  oranı >= 3 olduğu için belgede istenen "dengeli vs dengesiz" kıyasını da sağlar.
- Test kümesine burada DOKUNULMAZ; test metrikleri final değerlendirme aşamasında
  `python -m src.baseline --final-test` ile tek sefer hesaplanır.
- ROC AUC için SVM karar skorlarına softmax uygulanır (AUC sıralamaya dayalıdır).

Kullanım:  python -m src.baseline
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from . import config as C
from .data_prep import get_dataset_root
from .metrics import compute_metrics, confusion, per_class_report, softmax
from .results import update_test_metrics, upsert_result
from .utils import Timer, get_logger, load_json, save_json, set_seed

log = get_logger("baseline", C.LOG_DIR)
OUT = C.OUTPUT_DIR / "models" / "baseline_hog_svm"
MODEL_NAME = "HOG+LinearSVM"

PPC_GRID = [16, 24, 32]                      # HOG hücre boyutu (piksel)
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
CW_GRID = [None, "balanced"]


def hog_features(img: np.ndarray, ppc: int) -> np.ndarray:
    return hog(img, orientations=9, pixels_per_cell=(ppc, ppc), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True).astype(np.float32)


def load_gray(root: Path, rel: str) -> np.ndarray:
    return np.asarray(Image.open(root / rel).convert("L"), dtype=np.float32) / 255.0


def _feat(root: Path, rel: str, ppc: int) -> np.ndarray:
    return hog_features(load_gray(root, rel), ppc)


def extract(root: Path, rels, ppc: int) -> np.ndarray:
    feats = Parallel(n_jobs=-1, batch_size=64)(delayed(_feat)(root, r, ppc) for r in rels)
    return np.stack(feats)


def split_df(s: str) -> pd.DataFrame:
    return pd.read_csv(C.SPLITS_DIR / f"{s}.csv")


def measure_latency(model, root: Path, rels, ppc: int, n: int = 110, warmup: int = 10) -> dict:
    """Batch=1, uçtan uca (görüntü okuma + HOG + SVM). İlk 10 çalıştırma ölçüme katılmaz."""
    times = []
    for i in range(n):
        rel = rels[i % len(rels)]
        t0 = time.perf_counter()
        f = hog_features(load_gray(root, rel), ppc)[None]
        model.decision_function(f)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return {"latency_ms_median": float(np.median(times)), "latency_ms_p95": float(np.percentile(times, 95))}


def search(root: Path) -> dict:
    tr, va = split_df("train"), split_df("val")
    rows, best = [], None
    for ppc in PPC_GRID:
        with Timer() as t:
            Xtr, Xva = extract(root, tr.relpath, ppc), extract(root, va.relpath, ppc)
        log.info("HOG ppc=%d -> %d öznitelik (%.0f sn)", ppc, Xtr.shape[1], t.elapsed)
        for c, cw in itertools.product(C_GRID, CW_GRID):
            model = make_pipeline(StandardScaler(), LinearSVC(C=c, class_weight=cw, max_iter=5000,
                                                              random_state=C.SEED))
            with Timer() as tt:
                model.fit(Xtr, tr.label)
            scores = model.decision_function(Xva)
            m = compute_metrics(va.label, scores.argmax(1), softmax(scores))
            rows.append({"ppc": ppc, "C": c, "class_weight": str(cw), "n_features": Xtr.shape[1],
                         "fit_s": tt.elapsed, **{f"val_{k}": v for k, v in m.items()}})
            log.info("  ppc=%d C=%g cw=%s -> val macroF1=%.4f bAcc=%.4f QWK=%.4f",
                     ppc, c, cw, m["macro_f1"], m["balanced_accuracy"], m["qwk"])
            if best is None or m["macro_f1"] > best["metrics"]["macro_f1"]:
                best = {"ppc": ppc, "C": c, "class_weight": cw, "metrics": m, "fit_s": tt.elapsed,
                        "n_features": Xtr.shape[1]}
    grid = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT / "grid_search_val.csv", index=False)
    return best


def train_best(root: Path, best: dict) -> None:
    tr, va = split_df("train"), split_df("val")
    set_seed(C.SEED)
    Xtr = extract(root, tr.relpath, best["ppc"])
    model = make_pipeline(StandardScaler(), LinearSVC(C=best["C"], class_weight=best["class_weight"],
                                                      max_iter=5000, random_state=C.SEED))
    with Timer() as t:
        model.fit(Xtr, tr.label)
    Xva = extract(root, va.relpath, best["ppc"])
    scores = model.decision_function(Xva)
    pred = scores.argmax(1)
    m = compute_metrics(va.label, pred, softmax(scores))
    per_class_report(va.label, pred).to_csv(OUT / "val_per_class.csv", index=False)
    np.save(OUT / "val_confusion.npy", confusion(va.label, pred))
    np.save(OUT / "val_scores.npy", scores)

    path = OUT / "model.joblib"
    joblib.dump({"model": model, "ppc": best["ppc"], "class_names": C.CLASS_NAMES}, path)
    lat = measure_latency(model, root, list(va.relpath), best["ppc"])
    import platform
    svm = model[-1]
    row = {"model_name": MODEL_NAME, "model_family": "Klasik (HOG+SVM)", "seed": C.SEED,
           "input_size": "224x224x1", "param_count": int(svm.coef_.size + svm.intercept_.size),
           "epochs": None, "batch_size": None, "optimizer": "liblinear", "learning_rate": None,
           "loss": "squared hinge", "train_time_s": round(t.elapsed, 1),
           "model_size_mb": round(path.stat().st_size / 2**20, 2), **lat,
           "latency_device": f"CPU ({platform.processor() or platform.machine()})",
           "hyperparams": {"hog_ppc": best["ppc"], "hog_orient": 9, "hog_cpb": 2, "C": best["C"],
                           "class_weight": str(best["class_weight"])},
           "notes": "Seçim val macro F1 ile; test final aşamada"}
    upsert_result(row, m)
    save_json({"best": {k: v for k, v in best.items() if k != "metrics"}, "val_metrics": m, **lat},
              OUT / "summary.json")
    log.info("En iyi baseline: %s | val macroF1=%.4f", row["hyperparams"], m["macro_f1"])


def final_test(root: Path) -> None:
    """Final değerlendirme: test kümesinde TEK sefer."""
    obj = joblib.load(OUT / "model.joblib")
    te = split_df("test")
    scores = obj["model"].decision_function(extract(root, te.relpath, obj["ppc"]))
    pred = scores.argmax(1)
    m = compute_metrics(te.label, pred, softmax(scores))
    per_class_report(te.label, pred).to_csv(OUT / "test_per_class.csv", index=False)
    np.save(OUT / "test_confusion.npy", confusion(te.label, pred))
    np.save(OUT / "test_scores.npy", scores)
    update_test_metrics(MODEL_NAME, C.SEED, m)
    log.info("TEST: %s", {k: round(v, 4) for k, v in m.items()})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true", help="Yalnızca final değerlendirmede kullanın")
    args = ap.parse_args(argv)
    C.ensure_dirs()
    set_seed(C.SEED)
    root = get_dataset_root()
    if args.final_test:
        final_test(root)
        return
    best = search(root)
    train_best(root, best)


if __name__ == "__main__":
    main()
