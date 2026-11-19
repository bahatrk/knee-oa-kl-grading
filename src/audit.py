"""Aşama 1: Veri denetimi (data audit).

Üretilenler (outputs/audit ve outputs/figures altında):
  - Dosya bütünlüğü, bozuk/boş dosyalar, görüntü boyutları ve renk modları
  - Bölme x sınıf dağılımı, beklenen sayılarla karşılaştırma, dengesizlik oranı
  - Yoğunluk (parlaklık/kontrast) istatistikleri ve aykırı görüntüler
  - Birebir kopyalar (piksel MD5) ve pHash + Hamming ile yakın kopya taraması
  - Dosya adından hasta kimliği çıkarımı ve bölmeler arası hasta çakışması
  - Rapor figürleri (>=200 dpi) ve özet (audit_summary.json, audit_summary.md)

Kullanım:  python -m src.audit
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from . import config as C
from . import plot_style as S
from .data_prep import get_dataset_root
from .utils import Timer, env_info, get_logger, save_json

log = get_logger("audit", C.LOG_DIR)


# --------------------------------------------------------------------------- görüntü bazlı ölçümler
def inspect_image(path: Path) -> dict:
    rec = {"ok": True, "error": None}
    try:
        with Image.open(path) as im:
            im.verify()  # yapısal bütünlük
        with Image.open(path) as im:
            im.load()
            rec.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
            arr = np.asarray(im)
            if arr.ndim == 3:
                ch = arr[..., :3]
                rec["rgb_channels_identical"] = bool(np.all(ch[..., 0] == ch[..., 1]) and
                                                     np.all(ch[..., 1] == ch[..., 2]))
            else:
                rec["rgb_channels_identical"] = None
            rec["dtype"] = str(arr.dtype)
            gray = np.asarray(im.convert("L"), dtype=np.float32)
            rec.update(mean=float(gray.mean()), std=float(gray.std()),
                       min=int(gray.min()), max=int(gray.max()),
                       frac_black=float((gray <= 5).mean()), frac_white=float((gray >= 250).mean()))
            rec["pixel_md5"] = hashlib.md5(arr.tobytes()).hexdigest()
            rec["phash"] = str(imagehash.phash(im.convert("L"), hash_size=C.PHASH_HASH_SIZE))
    except Exception as e:  # bozuk dosya
        rec.update(ok=False, error=f"{type(e).__name__}: {e}")
    return rec


def run_inspection(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    recs = []
    for i, rel in enumerate(df.relpath):
        recs.append(inspect_image(root / rel))
        if (i + 1) % 1000 == 0:
            log.info("  %d / %d görüntü incelendi", i + 1, len(df))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)


# --------------------------------------------------------------------------- dağılım
def class_distribution(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ct = pd.crosstab(df.class_name, df.split).reindex(index=C.CLASS_NAMES, columns=C.SPLITS, fill_value=0)
    out = ct.copy()
    for s in C.SPLITS:
        out[f"{s}_%"] = (ct[s] / ct[s].sum() * 100).round(2)
        out[f"{s}_expected"] = C.EXPECTED_CLASS_COUNTS[s]
        out[f"{s}_diff"] = ct[s] - out[f"{s}_expected"]
    out["total"] = ct.sum(axis=1)
    imb = {}
    for s in C.SPLITS:
        imb[s] = {"largest": ct[s].idxmax(), "smallest": ct[s].idxmin(),
                  "ratio": round(float(ct[s].max() / max(ct[s].min(), 1)), 2)}
    imb["balancing_required"] = imb["train"]["ratio"] >= C.IMBALANCE_RATIO_THRESHOLD
    return out, imb


# --------------------------------------------------------------------------- aykırı değerler
def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return (x - med) / (mad if mad > 0 else 1e-9)


def find_outliers(d: pd.DataFrame, z_thr: float = 3.5) -> pd.DataFrame:
    ok = d[d.ok].copy()
    ok["z_mean"] = robust_z(ok["mean"])
    ok["z_std"] = robust_z(ok["std"])
    reasons = []
    for _, r in ok.iterrows():
        rs = []
        if abs(r.z_mean) > z_thr:
            rs.append("çok koyu" if r.z_mean < 0 else "çok parlak")
        if abs(r.z_std) > z_thr:
            rs.append("düşük kontrast" if r.z_std < 0 else "yüksek kontrast")
        if r["std"] < 5:
            rs.append("neredeyse sabit görüntü")
        if r.frac_black > 0.3:
            rs.append("geniş siyah alan (>%30)")
        reasons.append(", ".join(rs))
    ok["outlier_reason"] = reasons
    return ok[ok.outlier_reason != ""].sort_values("z_std")


# --------------------------------------------------------------------------- kopyalar
def _popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int16)
    b = x.view(np.uint8).reshape(-1, 8)
    return np.unpackbits(b, axis=1).sum(axis=1).astype(np.int16)


def phash_pairs(d: pd.DataFrame, max_thr: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Tüm çiftler arasında Hamming mesafesi <= max_thr olanları ve her görüntünün en yakın
    komşu mesafesini döndürür."""
    h = np.array([int(x, 16) for x in d.phash], dtype=np.uint64)
    n = len(h)
    nn = np.full(n, 64, dtype=np.int16)
    I, J, D = [], [], []
    for i in range(n - 1):
        dist = _popcount64(h[i] ^ h[i + 1:])
        m = dist.min()
        if m < nn[i]:
            nn[i] = m
        np.minimum(nn[i + 1:], dist, out=nn[i + 1:])
        idx = np.nonzero(dist <= max_thr)[0]
        if idx.size:
            I.extend([i] * idx.size)
            J.extend((idx + i + 1).tolist())
            D.extend(dist[idx].tolist())
    pairs = pd.DataFrame({"i": I, "j": J, "hamming": D})
    if len(pairs):
        for k in ("i", "j"):
            pairs[f"split_{k}"] = d.split.values[pairs[k]]
            pairs[f"label_{k}"] = d.label.values[pairs[k]]
            pairs[f"relpath_{k}"] = d.relpath.values[pairs[k]]
            pairs[f"patient_{k}"] = d.patient_id.values[pairs[k]]
        pairs["cross_split"] = pairs.split_i.astype(str) != pairs.split_j.astype(str)
        pairs["label_conflict"] = pairs.label_i != pairs.label_j
        pairs["pixel_identical"] = d.pixel_md5.values[pairs.i] == d.pixel_md5.values[pairs.j]
    return pairs, nn


def duplicate_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in C.PHASH_THRESHOLDS:
        p = pairs[pairs.hamming <= t] if len(pairs) else pairs
        rows.append({"hamming_threshold": t, "pairs_total": len(p),
                     "pairs_cross_split": int(p.cross_split.sum()) if len(p) else 0,
                     "pairs_train_test": int(((p.split_i.astype(str) + p.split_j.astype(str))
                                              .isin(["traintest", "testtrain"])).sum()) if len(p) else 0,
                     "pairs_label_conflict": int(p.label_conflict.sum()) if len(p) else 0,
                     "images_involved": int(len(set(p.i) | set(p.j))) if len(p) else 0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- hasta analizi
def patient_analysis(d: pd.DataFrame) -> dict:
    parsed = d.patient_id.notna()
    res = {"filename_examples": d.filename.sample(min(10, len(d)), random_state=C.SEED).tolist(),
           "patient_id_parse_rate": round(float(parsed.mean()), 4)}
    if parsed.mean() < 0.9:
        res["conclusion"] = ("Dosya adlarında güvenilir bir hasta kimliği bulunamadı; hasta düzeyinde "
                             "çakışma kontrol edilemedi. Bu durum raporda kısıt olarak yazılmalıdır.")
        return res
    p = d[parsed]
    res["patients_per_split"] = p.groupby("split", observed=True).patient_id.nunique().to_dict()
    res["images_per_patient"] = p.groupby("patient_id").size().describe().round(2).to_dict()
    sets = {s: set(p[p.split == s].patient_id) for s in C.SPLITS}
    overlap = {f"{a}-{b}": len(sets[a] & sets[b]) for a, b in
               [("train", "val"), ("train", "test"), ("val", "test")]}
    res["patient_overlap"] = overlap
    multi = p.groupby("patient_id").split.nunique()
    leaked = set(multi[multi > 1].index)
    res["patients_in_multiple_splits"] = len(leaked)
    res["test_images_with_patient_in_train"] = int(p[(p.split == "test") & p.patient_id.isin(sets["train"])].shape[0])
    res["val_images_with_patient_in_train"] = int(p[(p.split == "val") & p.patient_id.isin(sets["train"])].shape[0])
    if p.side.notna().any():
        knee = p.patient_id + "_" + p.side.fillna("?")
        res["unique_knees"] = int(knee.nunique())
        res["images_per_knee_max"] = int(knee.value_counts().max())
    res["conclusion"] = ("Bölmeler arasında ortak hasta yok; resmi bölme hasta düzeyinde ayrık."
                         if not leaked else
                         f"{len(leaked)} hasta birden fazla bölmede yer alıyor. Resmi bölme (VS13 kuralı) "
                         "değiştirilmeden kullanılacak; etkisi raporda kısıt olarak tartışılmalıdır.")
    return res


# --------------------------------------------------------------------------- figürler
def fig_class_distribution(ct: pd.DataFrame, imb: dict) -> Path:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(C.NUM_CLASSES)
    w = 0.26
    for k, s in enumerate(C.SPLITS):
        for ax, col in ((axes[0], s), (axes[1], f"{s}_%")):
            bars = ax.bar(x + (k - 1) * w, ct[col].values, w - 0.02, color=S.SPLIT_COLORS[s],
                          hatch=S.SPLIT_HATCHES[s], label=f"{S.SPLIT_LABELS_TR[s]} (n={int(ct[s].sum())})",
                          edgecolor="white", linewidth=0.5)
            if col == s:
                ax.bar_label(bars, fmt="%d", fontsize=7, color=S.INK_2, padding=1)
    for ax, yl, t in ((axes[0], "Görüntü sayısı", "(a) Sınıf başına görüntü sayısı"),
                      (axes[1], "Bölme içindeki oran (%)", "(b) Bölme içi sınıf oranı")):
        ax.set_xticks(x, [f"{c}\n{d}" for c, d in zip(C.CLASS_NAMES, C.CLASS_DESCRIPTIONS_TR)])
        ax.set_ylabel(yl)
        ax.set_title(t, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper right")
    r = imb["train"]
    fig.suptitle(f"kneeKL224 sınıf dağılımı — eğitimde en büyük/en küçük sınıf oranı "
                 f"{r['largest']}/{r['smallest']} = {r['ratio']:.1f}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return S.save_fig(fig, C.FIG_DIR / "audit_class_distribution.png")


def fig_samples(d: pd.DataFrame, root: Path, n: int = 6) -> Path:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(C.NUM_CLASSES, n, figsize=(n * 1.6, C.NUM_CLASSES * 1.7))
    tr = d[(d.split == "train") & d.ok]
    for r, cname in enumerate(C.CLASS_NAMES):
        pool = tr[tr.class_name == cname]
        sub = pool.sample(min(n, len(pool)), random_state=C.SEED)
        for c, rel in enumerate(sub.relpath):
            ax = axes[r, c]
            ax.imshow(Image.open(root / rel).convert("L"), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(f"{cname}\n{C.CLASS_DESCRIPTIONS_TR[r]}", fontsize=9, color=S.INK)
    fig.suptitle("Eğitim kümesinden rastgele örnekler (seed=42), her satır bir KL derecesi",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return S.save_fig(fig, C.FIG_DIR / "audit_samples_per_class.png")


def fig_intensity(d: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt
    ok = d[d.ok]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for col, ax, xl in (("mean", axes[0], "Ortalama piksel yoğunluğu (0–255)"),
                        ("std", axes[1], "Piksel std (kontrast)")):
        bins = np.linspace(ok[col].min(), ok[col].max(), 50)
        for k, cname in enumerate(C.CLASS_NAMES):
            ax.hist(ok[ok.class_name == cname][col], bins=bins, histtype="step", density=True,
                    color=S.KL_COLORS[k], linewidth=1.8, label=cname,
                    linestyle=["-", "--", "-.", ":", "-"][k])
        ax.set_xlabel(xl); ax.set_ylabel("Yoğunluk")
    axes[0].legend(title="Sınıf (tüm bölmeler)")
    axes[0].set_title("(a) Parlaklık dağılımı", loc="left")
    axes[1].set_title("(b) Kontrast dağılımı", loc="left")
    fig.tight_layout()
    return S.save_fig(fig, C.FIG_DIR / "audit_intensity_distribution.png")


def fig_image_grid(items: list[tuple[Path, str]], title: str, out: Path, ncol: int = 6) -> Path | None:
    import matplotlib.pyplot as plt
    if not items:
        return None
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.8, nrow * 2.1), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (p, cap) in zip(axes.flat, items):
        ax.imshow(Image.open(p).convert("L"), cmap="gray", vmin=0, vmax=255)
        ax.set_title(cap, fontsize=7, color=S.INK_2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return S.save_fig(fig, out)


def fig_nn_hamming(nn: np.ndarray) -> Path:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(nn, bins=np.arange(-0.5, nn.max() + 1.5, 1), color=S.SPLIT_COLORS["train"],
            edgecolor="white", linewidth=0.5)
    ax.axvline(C.PHASH_PRIMARY_THRESHOLD + 0.5, color=S.INK, linestyle="--", linewidth=1.2)
    ax.text(C.PHASH_PRIMARY_THRESHOLD + 0.8, ax.get_ylim()[1] * 0.9,
            f"yakın kopya eşiği ≤ {C.PHASH_PRIMARY_THRESHOLD}", fontsize=8, color=S.INK)
    ax.set_xlabel("En yakın komşuya Hamming mesafesi (64 bit pHash)")
    ax.set_ylabel("Görüntü sayısı")
    ax.set_title("Her görüntünün en yakın komşusuna pHash mesafesi (tüm bölmeler)", loc="left")
    fig.tight_layout()
    return S.save_fig(fig, C.FIG_DIR / "audit_phash_nearest_neighbor.png")


# --------------------------------------------------------------------------- ana akış
def main(argv=None) -> None:
    argparse.ArgumentParser(description="VS13 veri denetimi").parse_args(argv)
    C.ensure_dirs()
    S.apply_style()
    root = get_dataset_root()
    df = pd.read_csv(C.SPLITS_DIR / "all_files.csv", dtype={"patient_id": str})
    df["split"] = pd.Categorical(df["split"], C.SPLITS, ordered=True)
    log.info("Veri kökü: %s | %d görüntü", root, len(df))

    with Timer() as t:
        d = run_inspection(df, root)
    log.info("Görüntü incelemesi %.1f sn sürdü", t.elapsed)
    d.to_csv(C.AUDIT_DIR / "image_inspection.csv", index=False)

    corrupt = d[~d.ok]
    corrupt.to_csv(C.AUDIT_DIR / "corrupt_files.csv", index=False)
    if len(corrupt):  # bozuk dosyalar sabit listelerden çıkarılır ve sayısı raporlanır
        bad = set(corrupt.relpath)
        for s in C.SPLITS:
            f = C.SPLITS_DIR / f"{s}.csv"
            sd = pd.read_csv(f, dtype={"patient_id": str})
            sd[~sd.relpath.isin(bad)].to_csv(f, index=False)
        log.warning("%d bozuk dosya bölme listelerinden çıkarıldı", len(corrupt))
    ok = d[d.ok].reset_index(drop=True)
    ok[["width", "height", "min", "max"]] = ok[["width", "height", "min", "max"]].astype(int)

    ct, imb = class_distribution(ok)
    ct.to_csv(C.AUDIT_DIR / "class_distribution.csv")

    sizes = ok.groupby(["width", "height"]).size().rename("count").reset_index()
    modes = ok.groupby(["mode", "dtype"]).size().rename("count").reset_index()
    sizes.to_csv(C.AUDIT_DIR / "image_sizes.csv", index=False)
    modes.to_csv(C.AUDIT_DIR / "image_modes.csv", index=False)

    outliers = find_outliers(ok)
    outliers.to_csv(C.AUDIT_DIR / "intensity_outliers.csv", index=False)

    md5_groups = ok.groupby("pixel_md5").filter(lambda g: len(g) > 1)
    md5_groups.sort_values("pixel_md5").to_csv(C.AUDIT_DIR / "exact_duplicates.csv", index=False)

    log.info("pHash çift taraması başlıyor...")
    with Timer() as t:
        pairs, nn = phash_pairs(ok, max(C.PHASH_THRESHOLDS))
    log.info("pHash taraması %.1f sn sürdü, %d aday çift", t.elapsed, len(pairs))
    pairs.to_csv(C.AUDIT_DIR / "phash_pairs.csv", index=False)
    dup_sum = duplicate_summary(pairs)
    dup_sum.to_csv(C.AUDIT_DIR / "phash_summary.csv", index=False)

    patient = patient_analysis(ok)

    # ---- figürler
    figs = [fig_class_distribution(ct, imb), fig_samples(ok, root), fig_intensity(ok), fig_nn_hamming(nn)]
    figs.append(fig_image_grid(
        [(root / r.relpath, f"{r.split}/{r.class_name}\n{r.outlier_reason}") for r in outliers.head(18).itertuples()],
        "Yoğunluk açısından aykırı görüntüler (robust z > 3.5)", C.FIG_DIR / "audit_outliers.png"))
    near = pairs[pairs.hamming <= C.PHASH_PRIMARY_THRESHOLD].sort_values(["cross_split", "hamming"],
                                                                         ascending=[False, True]).head(6)
    items = []
    for r in near.itertuples():
        items.append((root / r.relpath_i, f"{r.split_i}/KL{r.label_i}  d={r.hamming}"))
        items.append((root / r.relpath_j, f"{r.split_j}/KL{r.label_j}"))
    figs.append(fig_image_grid(items, f"pHash ile bulunan en yakın çiftler (Hamming ≤ {C.PHASH_PRIMARY_THRESHOLD}); "
                                      "yan yana iki görüntü bir çifttir", C.FIG_DIR / "audit_near_duplicates.png",
                               ncol=4))

    prim = dup_sum[dup_sum.hamming_threshold == C.PHASH_PRIMARY_THRESHOLD].iloc[0].to_dict()
    summary = {
        "dataset": {"code": C.DATASET_CODE, "name": C.DATASET_NAME, "page": C.DATASET_PAGE,
                    "doi": C.DATASET_DOI, "license": C.DATASET_LICENSE, "root": str(root)},
        "counts": {"total_listed": len(df), "readable": len(ok), "corrupt": len(corrupt),
                   "per_split": ok.split.value_counts().reindex(C.SPLITS).astype(int).to_dict(),
                   "expected_per_split": C.EXPECTED_SPLIT_COUNTS,
                   "matches_expected": ok.split.value_counts().reindex(C.SPLITS).astype(int).to_dict()
                   == C.EXPECTED_SPLIT_COUNTS},
        "class_distribution": ct[C.SPLITS].to_dict(),
        "imbalance": imb,
        "image_sizes": sizes.to_dict("records"),
        "image_modes": modes.to_dict("records"),
        "rgb_with_identical_channels": int(ok.rgb_channels_identical.fillna(False).sum()),
        "intensity": {"mean_of_means": round(ok["mean"].mean(), 2), "mean_of_stds": round(ok["std"].mean(), 2),
                      "n_outliers": len(outliers)},
        "exact_duplicates": {"images_in_groups": len(md5_groups),
                             "groups": int(md5_groups.pixel_md5.nunique()) if len(md5_groups) else 0},
        "phash": {"hash_bits": C.PHASH_HASH_SIZE ** 2, "primary_threshold": C.PHASH_PRIMARY_THRESHOLD,
                  "at_primary_threshold": prim, "by_threshold": dup_sum.to_dict("records"),
                  "nearest_neighbor_hamming_median": float(np.median(nn))},
        "patients": patient,
        "figures": [str(f) for f in figs if f],
        "environment": env_info(),
    }
    save_json(summary, C.AUDIT_DIR / "audit_summary.json")
    write_markdown(summary, ct)
    log.info("Denetim tamamlandı. Özet: %s", C.AUDIT_DIR / "audit_summary.md")


def write_markdown(s: dict, ct: pd.DataFrame) -> None:
    c, imb, ph, pa = s["counts"], s["imbalance"], s["phash"], s["patients"]
    lines = [
        "# Veri Denetimi Özeti (VS13 – kneeKL224)", "",
        f"- Listelenen görüntü: **{c['total_listed']}**, okunabilen: **{c['readable']}**, bozuk: **{c['corrupt']}**",
        f"- Bölme sayıları: {c['per_split']} (beklenen {c['expected_per_split']}) → "
        f"{'UYUMLU' if c['matches_expected'] else 'FARK VAR, raporda açıklanmalı'}",
        f"- Görüntü boyutları: {s['image_sizes']}",
        f"- Renk modları: {s['image_modes']}",
        "", "## Sınıf dağılımı", "", ct[C.SPLITS + ["total"]].to_markdown(), "",
        f"- Eğitimde en büyük/en küçük sınıf oranı: **{imb['train']['ratio']}** "
        f"({imb['train']['largest']}/{imb['train']['smallest']}) → dengeleme "
        f"{'ZORUNLU' if imb['balancing_required'] else 'zorunlu değil'} (eşik {C.IMBALANCE_RATIO_THRESHOLD})",
        "", "## Kopya taraması",
        f"- Birebir aynı piksel içeriğine sahip görüntü: {s['exact_duplicates']['images_in_groups']} "
        f"({s['exact_duplicates']['groups']} grup)",
        f"- pHash ({ph['hash_bits']} bit), Hamming ≤ {ph['primary_threshold']}: "
        f"{ph['at_primary_threshold']['pairs_total']} çift, bunların {ph['at_primary_threshold']['pairs_cross_split']}'i "
        f"farklı bölmelerde, {ph['at_primary_threshold']['pairs_train_test']}'i eğitim–test arasında, "
        f"{ph['at_primary_threshold']['pairs_label_conflict']}'inde etiketler farklı",
        f"- En yakın komşu Hamming mesafesi medyanı: {ph['nearest_neighbor_hamming_median']}",
        "", "## Hasta düzeyi analiz",
        f"- Dosya adı örnekleri: {pa['filename_examples']}",
        f"- Hasta kimliği çözümlenme oranı: {pa['patient_id_parse_rate']}",
    ]
    for k in ("patients_per_split", "patient_overlap", "patients_in_multiple_splits",
              "test_images_with_patient_in_train", "unique_knees"):
        if k in pa:
            lines.append(f"- {k}: {pa[k]}")
    lines += [f"- Sonuç: {pa['conclusion']}", "",
              f"- Yoğunluk aykırı görüntü sayısı: {s['intensity']['n_outliers']}"]
    (C.AUDIT_DIR / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
