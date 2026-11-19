"""Tüm rapor figürleri için ortak görsel stil.

- KL dereceleri sıralı olduğu için tek tonlu (mavi) açık->koyu sıralı palet kullanılır.
- Bölmeler (train/val/test) kategorik renklerle ve ayrıca desenle (hatch) ayrılır,
  böylece ayrım yalnızca renge dayanmaz.
- Figürler en az 200 dpi kaydedilir.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# KL0 -> KL4 : açık -> koyu (sıralı)
KL_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
# Bölmeler: kategorik sabit sıra
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
SPLIT_HATCHES = {"train": "", "val": "//", "test": ".."}
SPLIT_LABELS_TR = {"train": "Eğitim", "val": "Doğrulama", "test": "Test"}

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

DPI = 300


def apply_style() -> None:
    from . import i18n
    i18n.install()   # KNEE_LANG=en ise şekil metinleri İngilizceye çevrilir
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "hatch.color": "white",
        "hatch.linewidth": 0.8,
    })


def save_fig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return path
