"""Aşama 9c: Zorlu görevler (en az 2). (1) Bozulma dayanıklılığı, (2) Seçici tahmin (risk-kapsam).
Her ikisi de özgün model ve en iyi mevcut model (DeiT-Small), ayrıca kontrol (düz ConvNeXt) ve hibrit için uygulanır.
Seçici tahmin eşikleri YALNIZCA doğrulama kümesinden belirlenir (hedef kapsama göre güven kantili)."""
from __future__ import annotations

import io
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

from . import config as C
from . import plot_style as S
from .data import KneeDataset, build_transforms, read_split
from .data_prep import get_dataset_root
from .final_eval import OUT, K, hybrid_probs, load_model, _test, _val, metrics_from_cm, _cm, temper, fit_temperature
from .train import predict
from .utils import get_logger

_trapz = getattr(np, "trapezoid", None) or np.trapz
log = get_logger("challenges", C.LOG_DIR)
CH = OUT / "challenges"
SEV = [1, 2, 3, 4, 5]


# ----------------------------------------------------------------------------- bozulmalar (PIL, 0-255 gri)
def gaussian_noise(img, s, rng):
    a = np.asarray(img, np.float32); sd = [4, 8, 14, 20, 28][s - 1]
    return Image.fromarray(np.clip(a + rng.normal(0, sd, a.shape), 0, 255).astype(np.uint8))


def gaussian_blur(img, s, rng):
    return img.filter(ImageFilter.GaussianBlur([0.7, 1.2, 1.8, 2.5, 3.5][s - 1]))


def contrast_loss(img, s, rng):
    a = np.asarray(img, np.float32); f = [0.8, 0.65, 0.5, 0.35, 0.25][s - 1]
    return Image.fromarray(np.clip((a - a.mean()) * f + a.mean(), 0, 255).astype(np.uint8))


def jpeg(img, s, rng):
    b = io.BytesIO(); img.save(b, "JPEG", quality=[60, 40, 25, 15, 8][s - 1]); b.seek(0)
    return Image.open(b).convert(img.mode).copy()


CORR = {"Gauss gürültüsü": gaussian_noise, "Gauss bulanıklığı": gaussian_blur, "Kontrast kaybı": contrast_loss, "JPEG sıkıştırma": jpeg}


class CorruptDS(KneeDataset):
    def __init__(self, base: KneeDataset, fn, sev, seed=C.SEED):
        self.base, self.fn, self.sev, self.seed = base, fn, sev, seed
        self.labels, self.transform = base.labels, base.transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 100003 + i)  # her görüntü için sabit rastgelelik
        img = self.base.cache[i]
        if self.sev > 0:  # gri görüntü üzerinde bozulma uygula (kanallara bağımsız gürültü olmasın)
            img = self.fn(img.convert("L"), self.sev, rng).convert("RGB")
        return self.transform(img), int(self.labels[i])


def robustness(models=(("Özgün", "Orig_full__opt"), ("DeiT-Small", "DeiTSmall__opt"), ("Kontrol (düz ConvNeXt)", "Orig_base__opt")),
               seed=C.SEED, device=None, batch=128):
    CH.mkdir(parents=True, exist_ok=True); S.apply_style()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for label, name in models:
        model, meta = load_model(name, seed, device)
        base = KneeDataset(read_split("test"), get_dataset_root(), build_transforms(False, tuple(meta["mean"]), tuple(meta["std"]), False))
        for cn, fn in CORR.items():
            for sev in [0] + SEV:
                if sev == 0 and cn != list(CORR)[0]:
                    continue  # temiz veri bir kez
                dl = DataLoader(CorruptDS(base, fn, sev), batch_size=batch, shuffle=False, num_workers=2)
                lg, y = predict(model, dl, device, amp=True)
                m = metrics_from_cm(_cm(np.asarray(y), lg.argmax(1).numpy()))
                rows.append({"model": label, "corruption": "temiz" if sev == 0 else cn, "severity": sev, **m})
                log.info("%s %s s%d F1=%.3f", label, cn, sev, m["macro_f1"])
        del model
    df = pd.DataFrame(rows)
    clean = df[df.severity == 0].set_index("model")
    d = df[df.severity > 0].copy()
    d["f1_clean"] = d.model.map(clean.macro_f1); d["f1_drop"] = d.f1_clean - d.macro_f1
    d["qwk_clean"] = d.model.map(clean.qwk); d["qwk_drop"] = d.qwk_clean - d.qwk
    df.to_csv(CH / "corruption_all.csv", index=False)
    summ = d.groupby(["model", "corruption"])[["macro_f1", "qwk", "f1_drop", "qwk_drop"]].mean().reset_index()
    summ.to_csv(CH / "corruption_summary.csv", index=False)
    fig, axes = plt.subplots(1, len(CORR), figsize=(11.5, 3.3), sharey=True)
    cols = {"Özgün": "#2a78d6", "DeiT-Small": "#eb6834", "Kontrol (düz ConvNeXt)": "#c9a227"}
    for ax, cn in zip(axes, CORR):
        for label, _ in models:
            s = df[(df.model == label) & (df.corruption == cn)].sort_values("severity")
            c0 = float(clean.loc[label, "macro_f1"])
            ax.plot([0] + list(s.severity), [c0] + list(s.macro_f1), "o-", ms=3, color=cols.get(label), label=label)
        ax.set_title(cn, fontsize=9); ax.set_xlabel("Şiddet")
    axes[0].set_ylabel("Test macro F1"); axes[0].legend(fontsize=7)
    fig.tight_layout(); S.save_fig(fig, C.FIG_DIR / "fig12_bozulma_dayaniklilik.png")
    return summ


# ----------------------------------------------------------------------------- seçici tahmin
def _conf(p, kind):
    if kind == "maks. olasılık":
        return p.max(1)
    return 1 + (p * np.log(np.clip(p, 1e-12, 1))).sum(1) / np.log(K)  # 1 - normalize entropi


def risk_coverage(p, y, conf):
    o = np.argsort(-conf); pred = p.argmax(1)[o]; yy = y[o]
    cov = np.arange(1, len(y) + 1) / len(y)
    risk = np.cumsum(pred != yy) / np.arange(1, len(y) + 1)
    mae = np.cumsum(np.abs(pred - yy)) / np.arange(1, len(y) + 1)
    return cov, risk, mae


def selective(seed=C.SEED, targets=(1.0, 0.9, 0.8, 0.7, 0.5)):
    CH.mkdir(parents=True, exist_ok=True); S.apply_style()
    entries = {"Özgün": "Orig_full__opt", "DeiT-Small": "DeiTSmall__opt", "Kontrol (düz ConvNeXt)": "Orig_base__opt", "Hibrit": "hybrid_mean4"}
    rows, curves = [], {}
    for label, name in entries.items():
        if name == "hybrid_mean4":
            pt, y = hybrid_probs(seed, "test"); pv, yv = hybrid_probs(seed, "val")
        else:
            pt, y = _test(name, seed); pv, yv = _val(name, seed)
        T = fit_temperature(pv, yv)
        variants = {"ham": (pv, pt)}
        if label == "Özgün":
            variants["sıcaklık"] = (temper(pv, T), temper(pt, T))
        for vn, (v, t) in variants.items():
            for kind in ("maks. olasılık", "entropi"):
                cv, ct = _conf(v, kind), _conf(t, kind)
                cov, risk, mae = risk_coverage(t, y, ct)
                aurc = float(_trapz(risk, cov))
                curves[(label, vn, kind)] = (cov, risk)
                for tg in targets:
                    tau = np.quantile(cv, 1 - tg) if tg < 1 else -np.inf   # eşik yalnızca doğrulamadan
                    keep = ct >= tau
                    if keep.sum() < 20: continue
                    m = metrics_from_cm(_cm(y[keep], t[keep].argmax(1)))
                    rows.append({"model": label, "olasılık": vn, "güven": kind, "hedef_kapsam(val)": tg, "test_kapsam": float(keep.mean()),
                                 "macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "qwk": m["qwk"], "mae_grade": m["mae_grade"],
                                 "distant_error_rate": m["distant_error_rate"], "AURC": aurc})
    df = pd.DataFrame(rows); df.to_csv(CH / "selective_prediction.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    cols = {"Özgün": "#2a78d6", "DeiT-Small": "#eb6834", "Kontrol (düz ConvNeXt)": "#c9a227", "Hibrit": "#1baf7a"}
    for (label, vn, kind), (cov, risk) in curves.items():
        if kind != "maks. olasılık":
            continue
        ls = "--" if vn == "sıcaklık" else "-"
        axes[0].plot(cov, risk, ls, color=cols[label], lw=1.5, label=f"{label}" + (" (sıcaklık)" if vn == "sıcaklık" else ""))
    axes[0].set_xlabel("Kapsam"); axes[0].set_ylabel("Hata oranı (kabul edilenlerde)"); axes[0].legend(fontsize=7)
    sub = df[(df.güven == "maks. olasılık") & (df.olasılık == "ham")]
    for label in cols:
        s = sub[sub.model == label].sort_values("test_kapsam")
        axes[1].plot(s.test_kapsam, s.macro_f1, "o-", ms=4, color=cols[label])
    axes[1].set_xlabel("Test kapsamı (eşik doğrulamadan)"); axes[1].set_ylabel("Kabul edilenlerde macro F1")
    fig.tight_layout(); S.save_fig(fig, C.FIG_DIR / "fig13_secici_tahmin.png")
    return df


def captions():
    cp = C.FIG_DIR / "captions.json"
    old = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}
    old["fig12_bozulma_dayaniklilik"] = "Şekil 12. Bozulma dayanıklılığı: dört bozulma türünün beş şiddetinde test macro F1 (seed 42)."
    old["fig13_secici_tahmin"] = "Şekil 13. Seçici tahmin: risk–kapsam eğrisi (sol) ve doğrulamada belirlenen eşiklerle test başarımı (sağ)."
    cp.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
