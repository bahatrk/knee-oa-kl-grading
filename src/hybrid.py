"""Hibrit model: farklı mimarilerin olasılık düzeyinde birleştirilmesi (ensemble).

Üyeler (zaten eğitilmiş, `__opt` modelleri): Orig (özgün), DeiT (Transformer), ConvNeXt, ResNet50.
Yeni eğitim gerekmez: kayıtlı doğrulama olasılıkları (val_probs.npy) birleştirilir; her seed için o seed'in
üyeleri birleştirilir, böylece sonuç 3 seed ortalaması ± std olarak verilir.

Birleştirme (füzyon) yöntemleri:
  mean        eşit ağırlıklı olasılık ortalaması (yumuşak oylama), parametresiz
  weighted    üye ağırlıkları doğrulama kümesinde macro F1'i en büyükleyecek şekilde seçilir (adım 0.1'lik simpleks ızgarası)
  stack       üyelerin log-olasılıkları üzerinde çok sınıflı lojistik regresyon (meta-öğrenici), doğrulamada öğrenilir
  mean_ord    `mean` olasılıkları + sıralı karar kuralı: beklenen mutlak derece hatasını en küçükleyen sınıf (medyan)

Ağırlıklar ve meta-öğrenici YALNIZCA doğrulama verisinde öğrenilir (test kullanılmaz). Doğrulamada raporlanan sayı
iyimser olmasın diye öğrenilen yöntemler için 5 katlı çapraz doğrulama ile elde edilen "katman dışı" tahminler kullanılır.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from . import config as C
from .metrics import compute_metrics
from .utils import get_logger, save_json

log = get_logger("hybrid", C.LOG_DIR)
MEMBERS = {"Orig": "Orig_full__opt", "DeiT": "DeiTSmall__opt", "ConvNeXt": "ConvNeXtTiny__opt",
           "ResNet50": "ResNet50__opt"}
METHODS = ["mean", "weighted", "stack", "mean_ord"]
EPS = 1e-8


def load(name: str, seed: int, split: str = "val"):
    d = C.OUTPUT_DIR / "runs" / name / f"seed{seed}"
    return np.load(d / f"{split}_probs.npy"), np.load(d / f"{split}_labels.npy")


def ordinal_decode(p: np.ndarray) -> np.ndarray:
    """argmin_j Σ_k p_k |k-j|  (beklenen mutlak derece hatasını en küçükleyen sınıf)."""
    k = np.arange(p.shape[1])
    cost = (p[:, :, None] * np.abs(k[None, :, None] - k[None, None, :])).sum(axis=1)
    return cost.argmin(axis=1)


def _simplex_grid(m: int, step: float = 0.1):
    """Toplamı 1 olan, adımı `step` olan tüm ağırlık vektörleri (m üye)."""
    n = int(round(1 / step))
    for c in itertools.product(range(n + 1), repeat=m - 1):
        if sum(c) <= n:
            yield np.array(list(c) + [n - sum(c)]) / n


def fit_weights(Ps: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    """Ağırlıklar >=0, toplamı 1. Seçim ölçütü: doğrulama macro F1 (proje kuralı); eşitlikte eşit ağırlığa en yakın olan."""
    P = np.stack(Ps)  # (m,n,K)
    equal = np.full(len(Ps), 1 / len(Ps))
    best_w, best_key = equal, None
    for w in _simplex_grid(len(Ps)):
        pred = np.tensordot(w, P, axes=1).argmax(1)
        key = (f1_score(y, pred, average="macro", labels=list(range(C.NUM_CLASSES)), zero_division=0),
               -np.abs(w - equal).sum())
        if best_key is None or key > best_key:
            best_w, best_key = w, key
    return best_w


def _stack_features(Ps):
    return np.concatenate([np.log(p + EPS) for p in Ps], axis=1)


def fit_stack(Ps, y, C_grid=(0.01, 0.1, 1.0), seed: int = C.SEED):
    """Meta-öğrenici: C, doğrulama içi iç çapraz doğrulamayla macro F1'e göre seçilir (proje kuralı)."""
    X = _stack_features(Ps)
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    best, best_c = -np.inf, C_grid[0]
    for c in C_grid:
        scores = []
        for tr, te in skf.split(X, y):
            m = LogisticRegression(C=c, max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
            scores.append(f1_score(y[te], m.predict(X[te]), average="macro",
                                   labels=list(range(C.NUM_CLASSES)), zero_division=0))
        if np.mean(scores) > best:
            best, best_c = np.mean(scores), c
    return LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced").fit(X, y), best_c


def predict_stack(model, Ps):
    pr = np.zeros((len(Ps[0]), C.NUM_CLASSES))
    pr[:, model.classes_] = model.predict_proba(_stack_features(Ps))
    return pr


def fuse(method: str, Ps: list[np.ndarray], params: dict | None = None):
    """Eğitilmiş/parametresiz füzyonu uygular. Döndürür: (olasılıklar, tahmin sınıfları)."""
    if method in ("mean", "mean_ord"):
        p = np.mean(Ps, axis=0)
    elif method == "weighted":
        p = np.tensordot(np.asarray(params["weights"]), np.stack(Ps), axes=1)
    elif method == "stack":
        p = predict_stack(params["model"], Ps)
    else:
        raise ValueError(method)
    return p, (ordinal_decode(p) if method == "mean_ord" else p.argmax(1))


def fit(method: str, Ps, y, seed: int = C.SEED) -> dict:
    if method == "weighted":
        return {"weights": fit_weights(Ps, y)}
    if method == "stack":
        m, c = fit_stack(Ps, y, seed=seed)
        return {"model": m, "C": c}
    return {}


def oof_predict(method: str, Ps, y, seed: int = C.SEED, k: int = 5):
    """Doğrulama için dürüst tahmin: öğrenilen yöntemlerde 5 katlı çapraz doğrulama (katman dışı)."""
    if method in ("mean", "mean_ord"):
        return fuse(method, Ps)
    prob = np.zeros((len(y), C.NUM_CLASSES))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Ps[0], y):
        params = fit(method, [p[tr] for p in Ps], y[tr], seed)
        prob[te], _ = fuse(method, [p[te] for p in Ps], params)
    return prob, prob.argmax(1)


def member_sets(members=tuple(MEMBERS), min_size: int = 2):
    for r in range(min_size, len(members) + 1):
        yield from itertools.combinations(members, r)


def run_all(seeds=C.SEEDS, members=tuple(MEMBERS), methods=METHODS) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        data = {m: load(MEMBERS[m], seed) for m in members}
        y = data[members[0]][1]
        for ms in member_sets(members):
            Ps = [data[m][0] for m in ms]
            for meth in methods:
                prob, pred = oof_predict(meth, Ps, y, seed)
                rows.append({"method": meth, "members": "+".join(ms), "n_members": len(ms), "seed": seed,
                             **compute_metrics(y, pred, prob)})
    runs = pd.DataFrame(rows)
    runs.to_csv(C.OUTPUT_DIR / "hybrid_val_runs.csv", index=False)
    cols = ["macro_f1", "balanced_accuracy", "accuracy", "qwk", "mae_grade", "distant_error_rate", "roc_auc_macro"]
    g = runs.groupby(["method", "members", "n_members"])
    summ = g[cols].mean().add_suffix("_mean").join(g[cols].std(ddof=1).add_suffix("_std")).reset_index()
    summ["n_seeds"] = g.seed.nunique().values
    summ = summ.sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    summ.to_csv(C.OUTPUT_DIR / "hybrid_val_summary.csv", index=False)
    return summ


def choose_final(summ: pd.DataFrame | None = None) -> dict:
    """Kural (test görülmeden): 3 seed ortalama doğrulama (katman dışı) macro F1'i en yüksek (yöntem, üye kümesi)."""
    if summ is None:
        summ = pd.read_csv(C.OUTPUT_DIR / "hybrid_val_summary.csv")
    top = summ.iloc[0]
    out = {"method": top.method, "members": top.members.split("+"), "val_macro_f1_mean": float(top.macro_f1_mean),
           "val_macro_f1_std": float(top.macro_f1_std), "n_configs_compared": int(len(summ)),
           "rule": "en yüksek 3-seed ortalama doğrulama macro F1 (öğrenilen yöntemlerde katman dışı); test kullanılmadı"}
    save_json(out, C.OUTPUT_DIR / "hybrid_choice.json")
    return out
