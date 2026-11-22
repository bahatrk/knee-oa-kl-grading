"""Aşama 9b: Açıklanabilirlik. Grad-CAM + Integrated Gradients (iki yöntem), DeiT için dikkat yayılımı (rollout).

Seçimler test kümesinde yeniden yapılır ama sabit seed ile: 20 doğru sınıflandırılmış (sınıf başına 4) ve
12 yanlış sınıflandırılmış (özgün model hatası; önce uzak hatalar) örnek açıklanır.
Doğrulama testleri: (1) parametre rastgeleleştirme (kademeli, çıktıdan girdiye), (2) silme eğrisi (deletion; gerçek sınıfla), (3) kenar bandı atıf payı.
Baseline ve silme dolgusu: görüntünün bulanıklaştırılmış hali (özgün model görüntü başına standardizasyon
yaptığından sıfır/ölçekli baseline anlamsızdır: f(a·x) = f(x)).
"""
from __future__ import annotations

import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
import torchvision.transforms.functional as TF
from scipy.stats import spearmanr

from . import config as C
from . import plot_style as S
from .data import KneeDataset, build_transforms, read_split
from .data_prep import get_dataset_root
from .final_eval import OUT, RUNS, load_model, _test
from .utils import get_logger, load_json

_trapz = getattr(np, "trapezoid", None) or np.trapz
log = get_logger("xai", C.LOG_DIR)
XAI = OUT / "xai"
ORIG, DEIT = "Orig_full__opt", "DeiTSmall__opt"
SIZE = 224


# ----------------------------------------------------------------------------- veri
def test_dataset(meta) -> KneeDataset:
    df = read_split("test")
    return KneeDataset(df, get_dataset_root(), build_transforms(False, tuple(meta["mean"]), tuple(meta["std"]), False))


def to_gray(x, meta):
    """Normalize edilmiş tensörden gösterim için [0,1] gri görüntü."""
    m = torch.tensor(meta["mean"], device=x.device).view(1, 3, 1, 1); s = torch.tensor(meta["std"], device=x.device).view(1, 3, 1, 1)
    return (x * s + m).clamp(0, 1)[0].mean(0).cpu().numpy()


def blur_baseline(x):
    return TF.gaussian_blur(x, kernel_size=31, sigma=10.0)


# ----------------------------------------------------------------------------- yöntemler
def _layer_and_reshape(model, key):
    if key == "original":
        return list(dict(model.backbone.named_children()).values())[-1], None
    if hasattr(model, "blocks"):  # timm ViT
        def rs(t):  # (B,1+N,C) -> (B,C,h,w)
            n = int((t.shape[1] - 1) ** 0.5)
            return t[:, 1:].reshape(t.shape[0], n, n, -1).permute(0, 3, 1, 2)
        return model.blocks[-1].norm1, rs
    return list(model.stages)[-1], None


def gradcam(model, x, cls, key):
    layer, rs = _layer_and_reshape(model, key)
    store = {}
    def fwd(m, i, o):  # None döndürmeli; aksi halde çıktı değiştirilir
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))

    h1 = layer.register_forward_hook(fwd)
    try:
        model.zero_grad(set_to_none=True)
        out = model(x)
        out[torch.arange(len(x)), cls].sum().backward()
    finally:
        h1.remove()
    a, g = store["a"], store["g"]
    if rs is not None:
        a, g = rs(a), rs(g)
    w = g.mean((2, 3), keepdim=True)
    cam = Fn.relu((w * a).sum(1, keepdim=True))
    cam = Fn.interpolate(cam, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[:, 0]
    mx = cam.amax((1, 2), keepdim=True)
    return (cam / (mx + 1e-8)).detach()


def integrated_gradients(model, x, cls, steps=32):
    base = blur_baseline(x)
    alphas = torch.linspace(1 / steps, 1, steps, device=x.device).view(-1, 1, 1, 1)
    xi = (base + alphas * (x - base)).requires_grad_(True)
    out = model(xi)
    g = torch.autograd.grad(out[:, int(cls)].sum(), xi)[0]
    attr = ((x - base) * g.mean(0, keepdim=True)).abs().sum(1)
    attr = TF.gaussian_blur(attr[None], kernel_size=9, sigma=2.0)[0]
    return (attr / (attr.amax((1, 2), keepdim=True) + 1e-8)).detach()


def attention_rollout(model, x):
    """DeiT: katmanlar boyunca dikkat matrislerinin (kalıntı bağlantılı) çarpımı; [CLS] satırı -> yama haritası."""
    maps, hooks = [], []
    for blk in model.blocks:
        blk.attn.fused_attn = False
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o: maps.append(o.detach())))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    R = None
    for A in maps:
        a = A.mean(1)[0]
        a = a + torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        R = a if R is None else a @ R
    n = int((R.shape[0] - 1) ** 0.5)
    m = R[0, 1:].reshape(1, 1, n, n)
    m = Fn.interpolate(m, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0]
    return (m / (m.max() + 1e-8))


# ----------------------------------------------------------------------------- doğrulama testleri
def deletion_curve(model, x, cls, rank_map, fracs):
    """Haritaya göre en önemli piksellerden başlayarak bulanık değerle değiştir; tahmin edilen sınıfın olasılığı."""
    base = blur_baseline(x)
    order = torch.argsort(rank_map.flatten(), descending=True)
    xs = []
    for f in fracs:
        mask = torch.zeros(SIZE * SIZE, device=x.device)
        mask[order[: int(f * SIZE * SIZE)]] = 1
        mask = mask.view(1, 1, SIZE, SIZE)
        xs.append(x * (1 - mask) + base * mask)
    with torch.no_grad():
        p = torch.softmax(model(torch.cat(xs)), 1)[:, int(cls)]
    return p.cpu().numpy()


def _reinit(module):
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    for p in module.parameters(recurse=False):
        torch.nn.init.trunc_normal_(p, std=0.02)


def _stages(model, key):
    if key == "original":
        bb = list(dict(model.backbone.named_children()).items())
        grp = [("head", [model.head]), ("norm+satır dikkati", [model.norms] + ([model.row_attn] if model.row_attn is not None else []))]
        for nm, md in reversed(bb):
            grp.append((nm, [md]))
        return grp
    n = len(model.blocks)
    grp = [("head", [model.head, model.norm] + ([model.fc_norm] if hasattr(model, "fc_norm") else []))]
    cuts = [(n - 3, n), (n - 6, n - 3), (n - 9, n - 6), (0, n - 9)]
    for lo, hi in cuts:
        grp.append((f"bloklar {lo}-{hi - 1}", list(model.blocks[lo:hi])))
    grp.append(("patch_embed", [model.patch_embed]))
    return grp


def randomization_test(model, key, xs, cls, device):
    ref = gradcam(model, xs, cls, key).cpu().numpy().reshape(len(xs), -1)
    m2 = copy.deepcopy(model)
    rows = [{"stage": "orijinal", "spearman_mean": 1.0, "spearman_std": 0.0}]
    for nm, mods in _stages(m2, key):
        for md in mods:
            _reinit(md)
        m2.eval()
        cam = gradcam(m2, xs, cls, key).cpu().numpy().reshape(len(xs), -1)
        r = [spearmanr(ref[i], cam[i]).correlation for i in range(len(xs))]
        r = np.nan_to_num(r)
        rows.append({"stage": f"+ {nm} rastgele", "spearman_mean": float(np.mean(r)), "spearman_std": float(np.std(r))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- yardımcı
def map_stats(m):
    m = m.detach().cpu().numpy(); m = m / (m.sum() + 1e-8)
    yy, xx = np.indices(m.shape)
    thr = np.quantile(m, 0.8)
    return {"center_mass": float(m[56:168, 56:168].sum()), "peak_y": int(np.unravel_index(m.argmax(), m.shape)[0]),
            "peak_x": int(np.unravel_index(m.argmax(), m.shape)[1]), "top20_area_centroid_y": float(yy[m >= thr].mean() / SIZE),
            "top20_area_centroid_x": float(xx[m >= thr].mean() / SIZE)}


def pick_samples(seed=C.SEED, n_per_class=4, n_err=12):
    po, y = _test(ORIG, seed); pd_, _ = _test(DEIT, seed)
    ao, ad = po.argmax(1), pd_.argmax(1)
    rng = np.random.default_rng(seed)
    good = []
    for k in range(C.NUM_CLASSES):
        idx = np.where((y == k) & (ao == k) & (ad == k))[0]
        good += list(rng.choice(idx, min(n_per_class, len(idx)), replace=False))
    err = np.where(ao != y)[0]
    dist = np.abs(ao[err] - y[err])
    far = err[dist >= 2]; near = err[dist == 1]
    sel = list(rng.choice(far, min(len(far), n_err // 2), replace=False))
    sel += list(rng.choice(near, min(len(near), n_err - len(sel)), replace=False))
    return good, sel


# ----------------------------------------------------------------------------- çizim
def overlay(ax, g, m, title=None):
    ax.imshow(g, cmap="gray", vmin=0, vmax=1)
    if m is not None:
        ax.imshow(m, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.axis("off")
    if title: ax.set_title(title, fontsize=6.5)


def grid_figure(items, name, caption):
    cols = ["Görüntü", "Özgün: Grad-CAM", "Özgün: IG", "DeiT: Grad-CAM", "DeiT: rollout", "DeiT: IG"]
    fig, axes = plt.subplots(len(items), 6, figsize=(8.6, 1.45 * len(items)), squeeze=False)
    for r, it in enumerate(items):
        ms = [None, it["o_cam"], it["o_ig"], it["d_cam"], it["d_roll"], it["d_ig"]]
        for c in range(6):
            overlay(axes[r, c], it["gray"], ms[c], cols[c] if r == 0 else None)
        axes[r, 0].text(-0.05, 0.5, f"#{it['idx']}\nGT KL{it['y']}\nÖ:KL{it['po']} D:KL{it['pd']}", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=6)
    fig.tight_layout(w_pad=0.1, h_pad=0.3)
    S.save_fig(fig, C.FIG_DIR / f"{name}.png")
    return name, caption


# ----------------------------------------------------------------------------- ana akış
BORDER_W = 12                                       # kenar bandı genişliği (piksel)
BORDER_AREA = 1 - ((SIZE - 2 * BORDER_W) / SIZE) ** 2   # bandın görüntü alanındaki payı (~%20)


def border_mass(m):
    """Haritadaki toplam atfın dış kenar bandına düşen payı (düzgün dağılımda BORDER_AREA)."""
    m = np.clip(m.detach().cpu().numpy().astype(float), 0, None); m = m / (m.sum() + 1e-8)
    return float(1 - m[BORDER_W:-BORDER_W, BORDER_W:-BORDER_W].sum())


def deletion_and_border(mo, md, ds_o, ds_d, y, device, seed=C.SEED, n_per_class=12, caps=None):
    """Silme eğrisi GERÇEK sınıfla (proje belgesi: doğru sınıfın olasılığındaki düşüş): hem haritalar gerçek sınıf
    için üretilir hem de ölçülen olasılık gerçek sınıfındır. Dengeli n_per_class/sınıf. Ayrıca kenar-bandı payı."""
    caps = {} if caps is None else caps
    rng = np.random.default_rng(seed)
    cur = np.concatenate([rng.choice(np.where(y == k)[0], n_per_class, replace=False) for k in range(C.NUM_CLASSES)])
    fracs = [0, .05, .1, .2, .3, .4, .5, .6, .8, 1.0]
    names = ("Özgün Grad-CAM", "Özgün IG", "Özgün rastgele", "DeiT Grad-CAM", "DeiT rollout", "DeiT IG", "DeiT rastgele")
    res = {k: [] for k in names}; border = []
    for i in cur:
        c = int(y[i]); ct = torch.tensor([c], device=device)
        xo = ds_o[i][0][None].to(device); xd = ds_d[i][0][None].to(device)
        rr = torch.rand(SIZE, SIZE, device=device)
        maps = {"Özgün Grad-CAM": (mo, xo, gradcam(mo, xo, ct, "original")[0]), "Özgün IG": (mo, xo, integrated_gradients(mo, xo, c)[0]),
                "Özgün rastgele": (mo, xo, rr), "DeiT Grad-CAM": (md, xd, gradcam(md, xd, ct, "vit")[0]),
                "DeiT rollout": (md, xd, attention_rollout(md, xd)[0]), "DeiT IG": (md, xd, integrated_gradients(md, xd, c)[0]),
                "DeiT rastgele": (md, xd, rr)}
        for k, (mod, x, m) in maps.items():
            res[k].append(deletion_curve(mod, x, c, m, fracs))
            border.append({"idx": int(i), "true": c, "method": k, "border_mass": border_mass(m)})
    summ = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, mdl in zip(axes, ("Özgün", "DeiT")):
        for k, v in res.items():
            if not k.startswith(mdl): continue
            v = np.array(v); mu = v.mean(0); auc = float(_trapz(mu, fracs))
            summ.append({"yöntem": k, "deletion_AUC_düşük_iyi": auc, "p_gercek_sinif_baslangic": float(mu[0]), "n_görüntü": len(v)})
            ax.plot(np.array(fracs) * 100, mu, label=f"{k} (AUC {auc:.3f})", ls=":" if "rastgele" in k else "-")
        ax.set_xlabel("Silinen piksel oranı (%)"); ax.set_title(mdl, fontsize=9); ax.legend(fontsize=7)
    axes[0].set_ylabel("Gerçek sınıfın olasılığı")
    fig.tight_layout(); S.save_fig(fig, C.FIG_DIR / "fig10_silme_egrisi.png")
    caps["fig10_silme_egrisi"] = "Şekil 10. Silme eğrileri (60 dengeli test görüntüsü): önemli pikseller önce bulanıklaştırılır; hızlı düşüş = açıklama modelin kullandığı bölgeyi yakalıyor."
    pd.DataFrame(summ).to_csv(XAI / "deletion_auc.csv", index=False)
    pd.DataFrame({k: np.array(v).mean(0) for k, v in res.items()}, index=fracs).to_csv(XAI / "deletion_curves.csv")
    bd = pd.DataFrame(border); bd.to_csv(XAI / "border_mass.csv", index=False)
    bs = bd.groupby("method").border_mass.agg(["mean", "std"]).reset_index()
    bs["uniform_expected"] = BORDER_AREA; bs["ratio_to_uniform"] = bs["mean"] / BORDER_AREA
    bs.to_csv(XAI / "border_mass_summary.csv", index=False)
    return cur


def run_deletion(device=None, seed=C.SEED, n_curve_per_class=12):
    """Yalnızca silme eğrisi + kenar bandı (rastgeleleştirme ve örnek ızgaraları yeniden üretilmez; ~5-10 dk)."""
    S.apply_style(); XAI.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mo, meta_o = load_model(ORIG, seed, device); md, meta_d = load_model(DEIT, seed, device)
    _, y = _test(ORIG, seed)
    caps = {}
    deletion_and_border(mo, md, test_dataset(meta_o), test_dataset(meta_d), y, device, seed, n_curve_per_class, caps)
    cp = C.FIG_DIR / "captions.json"
    old = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}
    old.update(caps); cp.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
    return pd.read_csv(XAI / "deletion_auc.csv"), pd.read_csv(XAI / "border_mass_summary.csv")


def run(device=None, seed=C.SEED, n_curve_per_class=12):
    S.apply_style(); XAI.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mo, meta_o = load_model(ORIG, seed, device); md, meta_d = load_model(DEIT, seed, device)
    ds_o, ds_d = test_dataset(meta_o), test_dataset(meta_d)
    po, y = _test(ORIG, seed); pdp, _ = _test(DEIT, seed)
    good, bad = pick_samples(seed)

    def explain(i):
        xo = ds_o[i][0][None].to(device); xd = ds_d[i][0][None].to(device)
        co, cd = int(po[i].argmax()), int(pdp[i].argmax())
        it = {"idx": int(i), "y": int(y[i]), "po": co, "pd": cd, "gray": to_gray(xo, meta_o),
              "o_cam": gradcam(mo, xo, torch.tensor([co], device=device), "original")[0].cpu().numpy(),
              "d_cam": gradcam(md, xd, torch.tensor([cd], device=device), "vit")[0].cpu().numpy(),
              "o_ig": integrated_gradients(mo, xo, co)[0].cpu().numpy(), "d_ig": integrated_gradients(md, xd, cd)[0].cpu().numpy(),
              "d_roll": attention_rollout(md, xd)[0].cpu().numpy()}
        return it

    items_g, items_b, rows = [], [], []
    for group, lst, store in (("doğru", good, items_g), ("yanlış (özgün)", bad, items_b)):
        for i in lst:
            it = explain(i); store.append(it)
            for mdl, key in (("Özgün", "o_cam"), ("DeiT", "d_cam"), ("DeiT-rollout", "d_roll"), ("Özgün-IG", "o_ig"), ("DeiT-IG", "d_ig")):
                rows.append({"group": group, "idx": it["idx"], "true": it["y"], "pred_orig": it["po"], "pred_deit": it["pd"],
                             "map": mdl, "conf_orig": float(po[i].max()), "conf_deit": float(pdp[i].max()),
                             **map_stats(torch.tensor(it[key]))})
    pd.DataFrame(rows).to_csv(XAI / "xai_samples_stats.csv", index=False)
    caps = {}
    for nm, items, cap in (("fig08_xai_dogru_a", items_g[:10], "Şekil 8a. Doğru sınıflandırılan örneklerde Grad-CAM, rollout ve Integrated Gradients (1/2)."),
                           ("fig08_xai_dogru_b", items_g[10:], "Şekil 8b. Doğru sınıflandırılan örneklerde açıklamalar (2/2)."),
                           ("fig09_xai_hatali_a", items_b[:6], "Şekil 9a. Özgün modelin yanlış sınıflandırdığı örneklerde açıklamalar (uzak hatalar öncelikli)."),
                           ("fig09_xai_hatali_b", items_b[6:], "Şekil 9b. Yanlış sınıflandırılan örnekler (devam).")):
        if items:
            k, v = grid_figure(items, nm, cap); caps[k] = v

    cur = deletion_and_border(mo, md, ds_o, ds_d, y, device, seed, n_curve_per_class, caps)

    # rastgeleleştirme testi (her model için 20 görüntü; tahmin edilen sınıf)
    sub = cur[::3][:20]
    out = {}
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for mdl, mod, ds, p, key, dev_meta in (("Özgün", mo, ds_o, po, "original", meta_o), ("DeiT", md, ds_d, pdp, "vit", meta_d)):
        xs = torch.stack([ds[i][0] for i in sub]).to(device); cl = torch.tensor([int(p[i].argmax()) for i in sub], device=device)
        t = randomization_test(mod, key, xs, cl, device); t.insert(0, "model", mdl); out[mdl] = t
        ax.errorbar(range(len(t)), t.spearman_mean, yerr=t.spearman_std, marker="o", ms=4, capsize=2, label=mdl)
    ax.set_xticks(range(len(out["Özgün"])), ["orijinal"] + [f"{i}" for i in range(1, len(out["Özgün"]))], fontsize=8)
    ax.set_xlabel("Kademeli rastgeleleştirilen katman grubu sayısı (çıktıdan girdiye)"); ax.set_ylabel("Spearman benzerliği (Grad-CAM)")
    ax.legend(fontsize=8); fig.tight_layout(); S.save_fig(fig, C.FIG_DIR / "fig11_rastgelelestirme.png")
    caps["fig11_rastgelelestirme"] = "Şekil 11. Parametre rastgeleleştirme testi: ağırlıklar çıktıdan girdiye doğru sıfırlanınca harita eski haritaya benzemiyorsa açıklama modele duyarlıdır."
    pd.concat(out.values()).to_csv(XAI / "randomization_test.csv", index=False)
    cp = C.FIG_DIR / "captions.json"
    old = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}
    old.update(caps); cp.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"n_dogru": len(items_g), "n_hatali": len(items_b), "n_silme": len(cur)}
