"""Rapor için İngilizce şekil seti (KNEE_LANG=en ile çalıştırılır; çıktı: outputs/figures_en).

Bu modül yalnızca kayıtlı tablolardan ve eğitilmiş modellerden şekil üretir; hiçbir model yeniden eğitilmez ve
test kümesi hakkında yeni bir seçim yapılmaz.
- denetim şekilleri (sınıf dağılımı, örnekler, yoğunluk) outputs/audit/*.csv'den
- veri artırma (augmentation) öncesi/sonrası şeması
- figures_final.run_all (Şekil 1-7)
"""
from __future__ import annotations

import os

assert os.environ.get("KNEE_LANG", "").lower() == "en", "Önce os.environ['KNEE_LANG']='en' ayarlayın (import'lardan önce)."

import numpy as np
import pandas as pd
from PIL import Image

from . import config as C
from . import plot_style as S
from .data import build_transforms
from .data_prep import get_dataset_root
from .utils import load_json


def audit_figures() -> list:
    from . import audit as A
    S.apply_style()
    root = get_dataset_root()
    ct = pd.read_csv(C.AUDIT_DIR / "class_distribution.csv", index_col=0)
    imb = load_json(C.AUDIT_DIR / "audit_summary.json")["imbalance"]
    d = pd.read_csv(C.AUDIT_DIR / "image_inspection.csv", dtype={"patient_id": str})
    d = d[d.ok].reset_index(drop=True)
    d["split"] = pd.Categorical(d["split"], C.SPLITS, ordered=True)
    out = [A.fig_class_distribution(ct, imb), A.fig_samples(d, root), A.fig_intensity(d)]
    return [str(o) for o in out]


def augmentation_figure(n_cols: int = 4, classes=(0, 2, 4), seed: int = C.SEED):
    """Her satırda bir sınıftan bir eğitim görüntüsü: orijinal + eğitimde kullanılan veri artırmanın rastgele örnekleri."""
    import matplotlib.pyplot as plt
    import torch
    import torchvision.transforms as T
    S.apply_style()
    root = get_dataset_root()
    tr = pd.read_csv(C.SPLITS_DIR / "train.csv", dtype={"patient_id": str})
    aug = T.Compose([T.RandomHorizontalFlip(0.5), T.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.93, 1.07)),
                     T.ColorJitter(brightness=0.2, contrast=0.2)])   # build_transforms(train=True) ile aynı, normalizasyonsuz
    fig, axes = plt.subplots(len(classes), n_cols + 1, figsize=(1.7 * (n_cols + 1), 1.8 * len(classes)))
    torch.manual_seed(seed)
    for r, k in enumerate(classes):
        row = tr[tr.label == k].sample(1, random_state=seed + k).iloc[0]
        img = Image.open(root / row.relpath).convert("RGB")
        for c in range(n_cols + 1):
            ax = axes[r, c]
            ax.imshow(img if c == 0 else aug(img), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title("Orijinal" if c == 0 else f"Artırılmış {c}", fontsize=8, color=S.INK_2)
            if c == 0:
                ax.set_ylabel(f"KL{k}", fontsize=9, color=S.INK)
    fig.tight_layout(w_pad=0.2, h_pad=0.3)
    # build_transforms ile tutarlılık kontrolü (parametreler değişirse uyarır)
    ref = build_transforms(True, (0.5,) * 3, (0.5,) * 3, True)
    assert "RandomAffine" in str(ref), "build_transforms değişmiş; augmentation_figure güncellenmeli"
    return str(S.save_fig(fig, C.FIG_DIR / "fig_augmentation.png"))


def final_figures():
    from . import figures_final as FF
    return FF.run_all()


def run_all():
    res = {"audit": audit_figures(), "augmentation": augmentation_figure(), "final": final_figures()}
    return res
