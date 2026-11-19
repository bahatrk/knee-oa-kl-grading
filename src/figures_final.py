"""Aşama 9a: Rapor şekilleri (test sonuçları ve eğitim kayıtlarından). Başlıksız/kısa; açıklamalar captions.json'da."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config as C
from . import plot_style as S
from .final_eval import OUT, RUNS, K, hybrid_probs, temper, fit_temperature, _val, _test
from .utils import load_json

FIG = C.FIG_DIR
PATIENCE = 6   # train.DEFAULTS["patience"]
CAP: dict[str, str] = {}
# kategorik sabit sıra: model kimliği -> renk (sıralamaya göre değil modele göre)
COL = {"DeiTSmall__opt": "#eb6834", "Orig_full__opt": "#2a78d6", "hybrid_mean4": "#1baf7a",
       "ConvNeXtTiny__opt": "#8a63d2", "ResNet50__opt": "#898781", "Orig_base__opt": "#c9a227"}
NAME = {"DeiTSmall__opt": "DeiT-Small (en iyi mevcut)", "Orig_full__opt": "Özgün model", "hybrid_mean4": "Hibrit",
        "ConvNeXtTiny__opt": "ConvNeXt-Tiny", "ResNet50__opt": "ResNet50", "Orig_base__opt": "Kontrol (düz ConvNeXt)"}


def _save(fig, name, caption):
    S.save_fig(fig, FIG / f"{name}.png")
    CAP[name] = caption


def _probs(name, seed=C.SEED, split="test"):
    if name == "hybrid_mean4":
        return hybrid_probs(seed, split)
    return (_test if split == "test" else _val)(name, seed)


# ---------------------------------------------------------------- akış şeması
def flowchart():
    steps = ["Veri seti\n(8.260 görüntü)", "Denetim:\nsızıntı, kopya,\nsınıf dengesizliği", "Ön işleme\n224×224",
             "HOG+SVM\nve 10 hazır model", "Optuna arama\n(≥20 deneme)", "Özgün model:\ntasarım günlüğü,\nablasyon", "Hibrit\nbirleştirme",
             "Test (tek sefer):\nmetrik, GA, kalibrasyon", "XAI ve\nzorlu görevler"]
    fig, ax = plt.subplots(figsize=(11, 3.4)); ax.axis("off"); ax.set_xlim(0, 9); ax.set_ylim(0, 1.4)
    for i, s in enumerate(steps):
        x = i
        ax.add_patch(FancyBboxPatch((x + 0.06, 0.45), 0.88, 0.8, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="#d9e6f7" if i < 7 else "#f7e1d6", ec=S.AXIS))
        ax.text(x + 0.5, 0.85, s, ha="center", va="center", fontsize=7.2, color=S.INK)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.95, 0.85), (x + 1.05, 0.85), arrowstyle="-|>", mutation_scale=9, color=S.INK_2))
    ax.text(0, 0.18, "Tüm seçimler (model, epoch, eşik, ağırlık) yalnızca doğrulama macro F1 ile yapıldı; test kümesi son aşamada bir kez kullanıldı.",
            fontsize=7.5, color=S.INK_2)
    _save(fig, "fig01_akis_semasi", "Şekil 1. Çalışmanın akışı. Mavi kutular doğrulamada yapılan seçimleri, turuncu kutular test sonrası analizleri gösterir.")


# ---------------------------------------------------------------- mimari şema
def architecture():
    """Ayrıntılı blok şeması: gövde (stem + 4 aşama, çıktı boyutları) ve aşama 2-4'ten çıkan dallar."""
    p = C.OUTPUT_DIR / "original_architecture.csv"
    df = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["block", "type", "output", "params"])
    fig, ax = plt.subplots(figsize=(11.5, 6.2)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 7.4)
    NEW, STD, IO = "#f7e1d6", "#d9e6f7", "#eeeeee"

    def prm(key):
        m = df[df.block.astype(str) == key]
        return f"{int(m.params.sum()):,}" if len(m) else "-"

    def box(x, y, w, h, t, fc=STD):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=S.AXIS))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.8, color=S.INK)

    def arr(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=S.INK_2))

    ax.text(0.1, 7.1, "ConvNeXt-Tiny gövdesi (ImageNet ağırlıkları)", fontsize=8, color=S.INK_2, weight="bold")
    # gövde (soldan sağa, tek sıra)
    back = [("Giriş\n3×224×224\n(A: görüntü başına\nstandardizasyon)", IO, "", 1.7),
            ("Stem\nConv 4×4, adım 4\n+ LayerNorm2d\n96×56×56", STD, "backbone.stem_0", 1.7),
            ("Aşama 1\n3 ConvNeXt bloğu\n96×56×56", STD, "backbone.stages_0", 1.7),
            ("Aşama 2\n3 blok (7×7 derinlik-\nbazlı konv.)\n192×28×28", STD, "backbone.stages_1", 1.9),
            ("Aşama 3\n9 blok\n384×14×14", STD, "backbone.stages_2", 1.7),
            ("Aşama 4\n3 blok\n768×7×7", STD, "backbone.stages_3", 1.7)]
    x = 0.1; xs = []
    for i, (t, fc, key, w) in enumerate(back):
        box(x, 5.4, w, 1.4, t, fc); xs.append((x, w))
        if i: arr((x - 0.18, 6.1), (x, 6.1))
        x += w + 0.18
    ax.add_patch(plt.Rectangle((0.5, 5.3), 0.01, 0.01, fc="none"))
    # dallar: aşama 2, 3, 4 çıktısı -> satır dikkati (C) -> havuz (B) -> LayerNorm -> birleştirme
    rows = [(4.0, "Aşama 2 çıktısı", "192×28×28", "row_attn[0]", "pool+norm[0]", 384),
            (2.7, "Aşama 3 çıktısı", "384×14×14", "row_attn[1]", "pool+norm[1]", 768),
            (1.4, "Aşama 4 çıktısı", "768×7×7", "row_attn[2]", "pool+norm[2]", 1536)]
    for y, src, shp, ra, pn, dim in rows:
        box(0.1, y, 3.0, 1.0, f"{src} ({shp})\nC: satır dikkati (eklenen)\n{prm(ra)} par.", NEW)
        arr((3.1, y + 0.5), (3.7, y + 0.5))
        box(3.7, y, 3.3, 1.0, f"B: ortalama + maks. havuz\n→ {dim}-d → LayerNorm\n{prm(pn)} par.", NEW)
        arr((7.0, y + 0.5), (8.0, 3.2))
    ax.text(0.1, 5.15, "Aşama 2, 3 ve 4 çıktıları üç dala ayrı ayrı beslenir (üstteki gövdeden):", fontsize=7.5, color=S.INK_2)
    box(8.0, 2.4, 2.4, 1.6, f"Birleştir\n384+768+1536\n= 2688 boyutlu vektör\nDropout → Linear(2688→5)\n{prm('head (Linear)')} par.", STD)
    arr((10.4, 3.2), (10.8, 3.2))
    box(10.8, 2.7, 2.1, 1.0, "Çıktı: 5 sınıf\n(KL0–KL4)\nD: ordinal kayıp\n(yalnız eğitim)", NEW)
    ax.text(0.1, 0.05, "Turuncu: bu çalışmada eklenen bileşenler (A, B, C, D). Mavi: hazır ağırlıklar / standart katmanlar.\nÇıktı boyutları kanal×yükseklik×genişlik.",
            fontsize=7, color=S.INK_2, va="bottom")
    _save(fig, "fig02_mimari", "Şekil 2. KneeOrdinalNet (özgün model) ayrıntılı blok şeması: stem ve dört aşamanın çıktı boyutları, aşama 2-4'ten çıkan satır dikkati (C) ve havuz (B) dalları, 2.688 boyutlu birleşik vektör ve sınıflandırma başı; her blokta parametre sayısı verilmiştir.")


# ---------------------------------------------------------------- eğitim eğrileri
def curves(names=("DeiTSmall__opt", "Orig_full__opt"), seed=C.SEED):
    fig, axes = plt.subplots(len(names), 2, figsize=(9, 3.2 * len(names)), squeeze=False)
    xmax = max(len(pd.read_csv(RUNS / n / f"seed{seed}" / "history.csv")) for n in names) + 0.5   # tüm panellerde aynı epoch aralığı
    for r, n in enumerate(names):
        d = RUNS / n / f"seed{seed}"
        h = pd.read_csv(d / "history.csv"); meta = load_json(d / "meta.json")
        ep = np.arange(1, len(h) + 1); be, last = meta["best_epoch"], len(h)
        for c, (cols, lab) in enumerate((((("train_loss", "val_loss")), "Kayıp"), ((("train_acc", "val_acc")), "Doğruluk"))):
            ax = axes[r, c]
            ax.plot(ep, h[cols[0]], color=S.SPLIT_COLORS["train"], label="Eğitim")
            ax.plot(ep, h[cols[1]], color=S.SPLIT_COLORS["val"], ls="--", label="Doğrulama")
            early = (last - be) >= PATIENCE   # sabır dolduysa gerçek erken durdurma; aksi halde epoch bütçesi bitti
            ax.axvline(be, color=S.INK, lw=1, ls=":"); ax.text(be - 0.3, ax.get_ylim()[1], f"en iyi epoch {be} ", fontsize=7, va="top", ha="right")
            if last != be:
                ax.axvline(last, color="#c0392b", lw=1, ls=":")
                ax.text(last - 0.3, ax.get_ylim()[0], ("erken durdurma (ep. %d) " if early else "epoch bütçesi bitti (ep. %d) ") % last,
                        fontsize=7, ha="right", va="bottom", color="#c0392b")
            ax.set_xlim(0.5, xmax); ax.set_xlabel("Epoch"); ax.set_ylabel(lab); ax.set_title(NAME[n], fontsize=9)
            if r == 0 and c == 0: ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig03_egitim_egrileri", "Şekil 3. DeiT-Small ve özgün modelin eğitim/doğrulama kayıp ve doğruluk eğrileri (eğitim ve doğrulama kümeleri, seed 42; tüm panellerde aynı epoch aralığı). Siyah noktalı çizgi en iyi epoch'u (geri yüklenen ağırlık), kırmızı çizgi eğitimin bittiği epoch'u gösterir; etiket, bitişin erken durdurma mı epoch bütçesi mi olduğunu belirtir. Çıkarım: iki model de doğrulama doğruluğu platoya girdiğinde durur; farklı bitiş epoch'ları modellerin farklı sürede yakınsadığını gösterir.")


# ---------------------------------------------------------------- karışıklık matrisleri
def confusions(names=("DeiTSmall__opt", "Orig_full__opt", "hybrid_mean4"), seed=C.SEED):
    fig, axes = plt.subplots(1, len(names), figsize=(4.1 * len(names), 4), squeeze=False)
    for ax, n in zip(axes[0], names):
        cm = np.load(OUT / f"confusion_{n}_seed{seed}.npy").astype(float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > .5 else S.INK)
        ax.set_xticks(range(K), C.CLASS_NAMES); ax.set_yticks(range(K), C.CLASS_NAMES)
        ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek"); ax.set_title(NAME[n], fontsize=9)
    fig.tight_layout()
    _save(fig, "fig04_karisiklik_matrisleri", "Şekil 4. DeiT-Small, özgün model ve hibritin test kümesindeki satır bazında normalize karışıklık matrisleri (seed 42). Çıkarım: hatalar ağırlıkla komşu derecelerde toplanır; KL1 en çok KL0 ve KL2 ile karışır.")


# ---------------------------------------------------------------- ROC / PR
def roc_pr(names=("DeiTSmall__opt", "Orig_full__opt"), seed=C.SEED):
    from sklearn.metrics import auc, average_precision_score, roc_auc_score
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4)); rows = []
    grid = np.linspace(0, 1, 201); rgrid = np.linspace(0, 1, 201)
    for n in names:
        p, y = _probs(n, seed); ls = "-" if n == "Orig_full__opt" else "--"
        tprs, precs = [], []
        for k in range(K):
            t = (y == k).astype(int)
            f, tp, _ = roc_curve(t, p[:, k]); pr, rc, _ = precision_recall_curve(t, p[:, k])
            axes[0].plot(f, tp, color=S.KL_COLORS[k], ls=ls, lw=1.0, alpha=.8, label=f"{C.CLASS_NAMES[k]}" if n == names[0] else None)
            axes[1].plot(rc, pr, color=S.KL_COLORS[k], ls=ls, lw=1.0, alpha=.8)
            tprs.append(np.interp(grid, f, tp)); precs.append(np.interp(rgrid, rc[::-1], pr[::-1]))
        axes[0].plot(grid, np.mean(tprs, 0), color="black", ls=ls, lw=2.2)
        axes[1].plot(rgrid, np.mean(precs, 0), color="black", ls=ls, lw=2.2)
        yb = np.eye(K)[y]
        fm, tm, _ = roc_curve(yb.ravel(), p.ravel())
        rows.append({"model": n, "seed": seed, "macro_auc_ovr": roc_auc_score(y, p, multi_class="ovr", average="macro"),
                     "micro_auc": auc(fm, tm), "macro_pr_auc": average_precision_score(yb, p, average="macro"),
                     "micro_pr_auc": average_precision_score(yb, p, average="micro")})
    pd.DataFrame(rows).to_csv(OUT / "roc_pr_summary.csv", index=False)
    axes[0].plot([0, 1], [0, 1], color=S.MUTED, lw=.8); axes[0].set_xlabel("Yanlış pozitif oranı"); axes[0].set_ylabel("Doğru pozitif oranı")
    axes[1].set_xlabel("Duyarlılık (recall)"); axes[1].set_ylabel("Kesinlik (precision)")
    axes[0].plot([], [], color="black", lw=2.2, label="Makro ortalama"); axes[0].legend(fontsize=7, title="Sınıf", title_fontsize=7)
    axes[1].plot([], [], "k-", label="Özgün model"); axes[1].plot([], [], "k--", label="DeiT-Small"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig05_roc_pr", "Şekil 5. Bire-karşı ROC (sol) ve kesinlik-duyarlılık (sağ) eğrileri, özgün model (düz) ve DeiT-Small (kesikli), test kümesi, seed 42; kalın siyah çizgi makro ortalamadır. Çıkarım: ara dereceler (KL1, KL2) uçlardan belirgin biçimde daha zor ayrılır; iki model makro düzeyde birbirine çok yakındır.")


# ---------------------------------------------------------------- kalibrasyon
def reliability(names=("DeiTSmall__opt", "Orig_full__opt"), seed=C.SEED, bins=15):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
    for ax, n in zip(axes, names):
        p, y = _probs(n, seed); vp, vy = _probs(n, seed, "val"); T = fit_temperature(vp, vy)
        for pp, lab, c in ((p, "ham", S.SPLIT_COLORS["train"]), (temper(p, T), f"sıcaklık (T={T:.2f})", S.SPLIT_COLORS["val"])):
            conf, ok = pp.max(1), (pp.argmax(1) == y)
            edges = np.linspace(0, 1, bins + 1); xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (conf > lo) & (conf <= hi)
                if m.sum() >= 5: xs.append(conf[m].mean()); ys.append(ok[m].mean())
            ax.plot(xs, ys, "o-", ms=3, color=c, label=lab)
        ax.plot([0, 1], [0, 1], color=S.MUTED, lw=.8); ax.set_xlabel("Ortalama güven"); ax.set_title(NAME[n], fontsize=9)
    axes[0].set_ylabel("Doğruluk"); axes[0].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig06_guvenilirlik", "Şekil 6. DeiT-Small ve özgün modelin güvenilirlik diyagramları (test kümesi, seed 42; 15 kutu, en az 5 örnekli kutular). Sıcaklık yalnızca doğrulama kümesinde öğrenildi. Çıkarım: sıcaklık ölçekleme güveni köşegene yaklaştırır.")


# ---------------------------------------------------------------- verimlilik
def tradeoff():
    r = pd.read_csv(OUT / "test_runs_all.csv"); lat = pd.read_csv(OUT / "cpu_latency.csv").set_index("run")
    g = r.groupby("run").macro_f1.agg(["mean", "std"])
    res = pd.read_csv(C.OUTPUT_DIR / "results.csv").drop_duplicates("model_name").set_index("model_name").param_count
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    # sol: ablasyonlar (parametre sayıları neredeyse eşit olduğundan başarım farkı çubuk + seed noktaları ile gösterilir)
    abl = [("Orig_full__opt", "Tam model"), ("Orig_noMS__opt", "- çok ölçekli"), ("Orig_noRow__opt", "- satır dikkati"),
           ("Orig_noOrd__opt", "- ordinal kayıp"), ("Orig_v3__opt", "- standardizasyon"), ("Orig_base__opt", "Kontrol (düz ConvNeXt)")]
    ax = axes[0]
    for i, (run, nm) in enumerate(abl[::-1]):
        v = r[r.run == run].macro_f1.values
        ax.barh(i, v.mean(), xerr=v.std(ddof=1), color=COL.get(run, "#7fb2ee"), alpha=.85, capsize=3, height=.6)
        ax.scatter(v, [i] * len(v), color="white", edgecolor=S.INK, s=14, zorder=3)
        pm = res.get(run, np.nan)
        ax.text(0.6405, i, "" if pd.isna(pm) else f"{pm / 1e6:.2f} M", va="center", fontsize=7, color=S.INK_2)
    ax.set_yticks(range(len(abl)), [n for _, n in abl[::-1]]); ax.set_xlim(0.64, 0.74)
    ax.set_xlabel("Test macro F1 (ort. ± std; noktalar: seed'ler)"); ax.set_title("Ablasyon (parametre sayısı çubuk içinde)", fontsize=9)
    # sağ: başarım - CPU gecikmesi
    ax = axes[1]
    sel = [("ResNet50__opt", "ResNet50", (4, -10)), ("ConvNeXtTiny__opt", "ConvNeXt-T", (4, -10)), ("DeiTSmall__opt", "DeiT-S", (-38, -12)),
           ("Orig_full__opt", "Özgün", (6, 4)), ("hybrid_mean4", "Hibrit", (-30, 6)), ("MobileNetV3L", "MobileNetV3", (4, 4)),
           ("EfficientNetB0", "EffNet-B0", (4, 4)), ("DenseNet121", "DenseNet121", (4, -10)), ("Xception", "Xception", (4, -10))]
    for run, nm, off in sel:
        if run not in g.index or run not in lat.index:
            continue
        x = lat.loc[run, "latency_ms_median"]
        c = COL.get(run, S.MUTED)
        std = 0 if np.isnan(g.loc[run, "std"]) else g.loc[run, "std"]
        ax.errorbar(x, g.loc[run, "mean"], yerr=std, fmt="o", color=c, ms=6, capsize=2)
        ax.annotate(nm, (x, g.loc[run, "mean"]), textcoords="offset points", xytext=off, fontsize=7)
    ax.set_xlabel("CPU gecikmesi (ms, batch 1, medyan)"); ax.set_ylabel("Test macro F1"); ax.set_title("Başarım - gecikme", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig07_ablasyon_verimlilik", "Şekil 7. Sol: ablasyon sürümlerinin test macro F1 değeri; hepsi yaklaşık 28 M parametreli olduğundan verimlilik farkı yoktur. Sağ: başarım ve CPU gecikmesi; hata çubuğu seed std'sidir (tek seed'li modellerde yoktur).")


def run_all():
    S.apply_style(); FIG.mkdir(parents=True, exist_ok=True)
    for f in (flowchart, architecture, curves, confusions, roc_pr, reliability, tradeoff):
        try:
            f()
        except Exception as e:  # tek şekil hatası diğerlerini durdurmasın
            print("HATA", f.__name__, repr(e))
    (FIG / "captions.json").write_text(json.dumps(CAP, ensure_ascii=False, indent=2), encoding="utf-8")
    return sorted(CAP)
