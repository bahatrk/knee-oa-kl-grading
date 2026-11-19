"""Proje belgesinin zorunlu kıldığı tüm sınıflandırma metrikleri.

Her model için aynı fonksiyon kullanılır, böylece karşılaştırma adil olur.
- Genel: accuracy, balanced accuracy, macro/weighted precision-recall-F1, Cohen kappa, MCC
- Sıralı (KL): quadratic weighted kappa (QWK), ortalama mutlak derece hatası (MAE),
  komşu (|fark|=1) ve uzak (|fark|>=2) derece karışıklık oranları
- Olasılık tabanlı: one-vs-rest macro/micro ROC AUC, macro PR AUC (average precision)
- Sınıf bazında: precision, recall (= duyarlılık), özgüllük, F1, destek
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, precision_score, recall_score,
                             roc_auc_score)
from sklearn.preprocessing import label_binarize

from . import config as C


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ordinal_errors(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    d = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    n_err = int((d > 0).sum())
    return {"mae_grade": float(d.mean()),
            "adjacent_error_rate": float((d == 1).mean()),     # tüm örnekler içinde
            "distant_error_rate": float((d >= 2).mean()),
            "adjacent_share_of_errors": float((d == 1).sum() / n_err) if n_err else 0.0}


def compute_metrics(y_true, y_pred, y_prob: np.ndarray | None = None,
                    num_classes: int = C.NUM_CLASSES) -> dict:
    """Tek bir sözlükte tüm özet metrikler."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(num_classes))
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
        "qwk": cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic"),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    m.update(ordinal_errors(y_true, y_pred))
    if y_prob is not None:
        Y = label_binarize(y_true, classes=labels)
        try:
            m["roc_auc_macro"] = roc_auc_score(Y, y_prob, average="macro", multi_class="ovr")
            m["roc_auc_micro"] = roc_auc_score(Y, y_prob, average="micro", multi_class="ovr")
            m["pr_auc_macro"] = average_precision_score(Y, y_prob, average="macro")
        except ValueError:  # bir sınıf hiç yoksa
            m["roc_auc_macro"] = m["roc_auc_micro"] = m["pr_auc_macro"] = float("nan")
    return {k: float(v) for k, v in m.items()}


def per_class_report(y_true, y_pred, num_classes: int = C.NUM_CLASSES) -> pd.DataFrame:
    labels = list(range(num_classes))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    spec = []
    for k in labels:
        tn = cm.sum() - cm[k, :].sum() - cm[:, k].sum() + cm[k, k]
        fp = cm[:, k].sum() - cm[k, k]
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return pd.DataFrame({"class": C.CLASS_NAMES[:num_classes], "precision": p,
                         "recall_sensitivity": r, "specificity": spec, "f1": f, "support": s})


def confusion(y_true, y_pred, normalize: str | None = None,
              num_classes: int = C.NUM_CLASSES) -> np.ndarray:
    """Satırlar gerçek, sütunlar tahmin. normalize='true' -> satır bazında oran."""
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)), normalize=normalize)
