"""pHash ile işaretlenen aday yakın-kopya çiftlerini piksel düzeyinde doğrular.

Neden gerekli? Tüm diz röntgenleri aynı şekilde ortalanmış benzer anatomiye sahip
olduğundan, düşük frekanslı DCT'ye dayanan pHash farklı hastaları da yakın gösterebilir.
Bir çiftin gerçek (yakın) kopya sayılması için iki ölçüt birlikte aranır:
  SSIM >= 0.90  ve  normalize çapraz korelasyon (NCC) >= 0.95
Ayna (yatay çevrilmiş) eşleşme de kontrol edilir.

Kullanım:  python -m src.dup_verify
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from . import config as C
from . import plot_style as S
from .data_prep import get_dataset_root
from .utils import get_logger, load_json, save_json

log = get_logger("dup_verify", C.LOG_DIR)
SSIM_THR, NCC_THR = 0.90, 0.95


def _load(root, rel) -> np.ndarray:
    return np.asarray(Image.open(root / rel).convert("L"), dtype=np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float((a * b).mean())


def main(argv=None) -> None:
    S.apply_style()
    root = get_dataset_root()
    pairs = pd.read_csv(C.AUDIT_DIR / "phash_pairs.csv", dtype={"patient_i": str, "patient_j": str})
    cand = pairs[pairs.hamming <= C.PHASH_PRIMARY_THRESHOLD].copy()
    log.info("%d aday çift doğrulanıyor (Hamming <= %d)", len(cand), C.PHASH_PRIMARY_THRESHOLD)
    s_list, n_list, sf_list, nf_list = [], [], [], []
    for r in cand.itertuples():
        a, b = _load(root, r.relpath_i), _load(root, r.relpath_j)
        bf = b[:, ::-1]
        s_list.append(ssim(a, b, data_range=255)); n_list.append(_ncc(a, b))
        sf_list.append(ssim(a, bf, data_range=255)); nf_list.append(_ncc(a, bf))
    cand["ssim"], cand["ncc"], cand["ssim_flip"], cand["ncc_flip"] = s_list, n_list, sf_list, nf_list
    cand["ssim_best"] = cand[["ssim", "ssim_flip"]].max(axis=1)
    cand["ncc_best"] = cand[["ncc", "ncc_flip"]].max(axis=1)
    cand["same_patient"] = cand.patient_i == cand.patient_j
    cand["confirmed_duplicate"] = (cand.ssim_best >= SSIM_THR) & (cand.ncc_best >= NCC_THR)
    cand.to_csv(C.AUDIT_DIR / "phash_pairs_verified.csv", index=False)

    conf = cand[cand.confirmed_duplicate]
    conf.to_csv(C.AUDIT_DIR / "confirmed_duplicates.csv", index=False)
    if len(conf):  # doğrulanan çiftleri görsel olarak göster
        from .audit import fig_image_grid
        items = []
        for r in conf.itertuples():
            items.append((root / r.relpath_i, f"{r.split_i}/KL{r.label_i}  SSIM={r.ssim_best:.2f}"))
            items.append((root / r.relpath_j, f"{r.split_j}/KL{r.label_j}  NCC={r.ncc_best:.2f}"))
        fig_image_grid(items, "Piksel düzeyinde doğrulanan yakın-kopya adayları (SSIM≥0.90 ve NCC≥0.95); "
                              "yan yana iki görüntü bir çifttir", C.FIG_DIR / "audit_confirmed_duplicates.png",
                       ncol=4)
    res = {
        "candidate_pairs": len(cand),
        "criteria": {"ssim_min": SSIM_THR, "ncc_min": NCC_THR, "mirror_checked": True},
        "confirmed_pairs": len(conf),
        "confirmed_cross_split": int(conf.cross_split.sum()),
        "candidates_same_patient": int(cand.same_patient.sum()),
        "ssim_best": cand.ssim_best.describe().round(3).to_dict(),
        "ncc_best": cand.ncc_best.describe().round(3).to_dict(),
        "max_ssim_pair": cand.loc[cand.ssim_best.idxmax(), ["relpath_i", "relpath_j", "hamming", "ssim_best",
                                                            "ncc_best"]].to_dict() if len(cand) else None,
    }
    save_json(res, C.AUDIT_DIR / "duplicate_verification.json")
    s = load_json(C.AUDIT_DIR / "audit_summary.json")
    s["duplicate_verification"] = res
    save_json(s, C.AUDIT_DIR / "audit_summary.json")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(cand.ssim_best, cand.ncc_best, s=12, color=S.SPLIT_COLORS["train"], alpha=0.6,
               label=f"Aday çiftler (pHash Hamming ≤ {C.PHASH_PRIMARY_THRESHOLD}, n={len(cand)})")
    ax.axvline(SSIM_THR, color=S.INK, ls="--", lw=1)
    ax.axhline(NCC_THR, color=S.INK, ls="--", lw=1)
    ax.text(SSIM_THR + 0.005, ax.get_ylim()[0] + 0.02, "kopya bölgesi →", fontsize=8, color=S.INK)
    ax.set_xlabel("SSIM (en iyi: normal / ayna)")
    ax.set_ylabel("Normalize çapraz korelasyon")
    ax.set_title("pHash adaylarının piksel düzeyinde doğrulanması", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    S.save_fig(fig, C.FIG_DIR / "audit_duplicate_verification.png")
    log.info("Sonuç: %s", res)


if __name__ == "__main__":
    main()
