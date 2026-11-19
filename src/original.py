"""Özgün model: KneeOrdinalNet  (ConvNeXt-Tiny omurgası + 4 bu veriye özgü bileşen).

Omurga: ConvNeXt-Tiny (ImageNet ağırlıklı, timm `convnext_tiny.fb_in1k`, 4 aşamalı özellik çıkarıcı).
Bileşenler (her biri bir anahtarla açılıp kapanır, ablasyon için):

 A  use_input_norm : görüntü başına kontrast standardizasyonu
      Veri özelliği: pozlama/parlaklık farkları (denetimde 136 koyu görüntü, geniş parlaklık dağılımı).
 B  use_multiscale : 3 derinlikten (aşama 2, 3, 4) ortalama+maks havuzlanmış özelliklerin birleştirilmesi
      Veri özelliği: osteofitler küçük/ince ayrıntı (sığ katman, maks havuzlama), eklem aralığı daralması geniş yapı
      (derin katman). Yalnızca son aşamaya bakmak ince bulguları kaybeder.
 C  use_row_attn   : eklem hattı satır dikkati (genişlik boyunca özetleyip satırları ağırlıklandırır)
      Veri özelliği: tanısal ipucu femur ile tibia arasındaki YATAY eklem hattında; hattın dikey konumu
      hastanın pozisyonuna göre değişir; yatay özetleme hattın yanal konumundan bağımsızdır.
 D  ordinal kayıp (OrdinalLoss, train.py'de loss='ordinal'): Gauss yumuşak etiket + beklenen-derece L1 terimi
      Veri özelliği: KL sıralıdır; hataların çoğu komşu derecededir, uzak derece hataları daha maliyetlidir.
"""
from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C

BACKBONE = "convnext_tiny.fb_in1k"


def per_image_standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """(B,C,H,W): her görüntüyü kendi ortalama/std'siyle standartlaştırır. Veri seti normalizasyonu
    kanal başına afin olduğundan sonuç, ham görüntünün standartlaştırılmasıyla aynıdır (parlaklık/kontrast
    değişimine değişmez)."""
    m = x.mean(dim=(2, 3), keepdim=True)
    s = x.std(dim=(2, 3), keepdim=True)
    return (x - m) / (s + eps)


class RowAttention(nn.Module):
    """Eklem hattı satır dikkati.

    Özellik haritası (B,C,H,W) -> her satır için genişlik boyunca ortalama ve maks özeti (B,2C,H) ->
    dikey eksende 1B evrişimler (komşu satır bağlamı) -> satır başına tek skor -> kapı 2*sigmoid(z).
    Son evrişim sıfırla başlatılır: eğitim başında kapı 1 (kimlik dönüşümü), önceden eğitilmiş omurga bozulmaz.
    """

    def __init__(self, channels: int, reduction: int = 8, kernel: int = 3):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.net = nn.Sequential(nn.Conv1d(2 * channels, mid, kernel, padding=kernel // 2), nn.GELU(),
                                 nn.Conv1d(mid, 1, kernel, padding=kernel // 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.last_gate: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.cat([x.mean(dim=3), x.amax(dim=3)], dim=1)       # (B,2C,H)
        g = 2.0 * torch.sigmoid(self.net(r))                        # (B,1,H)
        self.last_gate = g.detach()                                 # açıklanabilirlik için
        return x * g.unsqueeze(-1)


class KneeOrdinalNet(nn.Module):
    def __init__(self, num_classes: int = C.NUM_CLASSES, pretrained: bool = True, drop: float = 0.3,
                 use_input_norm: bool = True, use_multiscale: bool = True, use_row_attn: bool = True,
                 backbone: str = BACKBONE):
        super().__init__()
        import timm
        self.use_input_norm, self.use_multiscale, self.use_row_attn = use_input_norm, use_multiscale, use_row_attn
        self.out_indices = (1, 2, 3) if use_multiscale else (3,)
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                          out_indices=self.out_indices)       # önce omurga (parametre sırası)
        chs = self.backbone.feature_info.channels()
        self.row_attn = nn.ModuleList([RowAttention(c) for c in chs]) if use_row_attn else None
        mult = 2 if use_multiscale else 1                      # ortalama (+ maks) havuzlama
        self.norms = nn.ModuleList([nn.LayerNorm(c * mult) for c in chs])
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(sum(c * mult for c in chs), num_classes)
        self.pretrained_cfg = getattr(self.backbone, "pretrained_cfg", {})

    def new_parameters(self):
        """Omurga dışındaki (bu çalışmada eklenen) tüm parametreler."""
        bb = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in bb]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_input_norm:
            x = per_image_standardize(x)
        feats = self.backbone(x)
        pooled = []
        for i, f in enumerate(feats):
            if self.use_row_attn:
                f = self.row_attn[i](f)
            p = f.mean(dim=(2, 3))
            if self.use_multiscale:
                p = torch.cat([p, f.amax(dim=(2, 3))], dim=1)
            pooled.append(self.norms[i](p))
        return self.head(self.drop(torch.cat(pooled, dim=1)))


class OrdinalLoss(nn.Module):
    """Bileşen D: Gauss yumuşak etiketli çapraz entropi + beklenen derece üzerinde smooth-L1.

    q_k ∝ exp(-(k-y)^2 / (2σ^2)): doğru dereceye komşu derecelere kısmi olasılık verilir, uzak derecelere hemen
    hiç verilmez -> uzak hata daha çok cezalandırılır. L1 terimi E[k]=Σ k·p_k'yi gerçek dereceye çeker (MAE ile uyumlu).
    """

    def __init__(self, sigma: float = 0.75, lam: float = 0.3, num_classes: int = C.NUM_CLASSES):
        super().__init__()
        self.sigma, self.lam = sigma, lam
        self.register_buffer("k", torch.arange(num_classes, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        q = torch.exp(-((self.k[None, :] - y[:, None].float()) ** 2) / (2 * self.sigma ** 2))
        q = q / q.sum(dim=1, keepdim=True)
        logp = F.log_softmax(logits, dim=1)
        ce = -(q * logp).sum(dim=1).mean()
        expected = (logp.exp() * self.k[None, :]).sum(dim=1)
        return ce + self.lam * F.smooth_l1_loss(expected, y.float())


# --------------------------------------------------------------------------- varyantlar
# Tasarım günlüğü (sıralı sürümler) ve ablasyonlar. Her varyant: model anahtarları + kayıp.
VARIANTS: dict[str, dict] = {
    "Orig_v1":   dict(model=dict(use_input_norm=False, use_multiscale=True, use_row_attn=False), loss="ce+sampler"),
    "Orig_v2":   dict(model=dict(use_input_norm=False, use_multiscale=True, use_row_attn=True), loss="ce+sampler"),
    "Orig_v3":   dict(model=dict(use_input_norm=False, use_multiscale=True, use_row_attn=True), loss="ordinal+sampler"),
    "Orig_full": dict(model=dict(use_input_norm=True, use_multiscale=True, use_row_attn=True), loss="ordinal+sampler"),
    # ablasyonlar: tam modelden TEK bileşen çıkarılır / basitiyle değiştirilir
    "Orig_noMS":   dict(model=dict(use_input_norm=True, use_multiscale=False, use_row_attn=True), loss="ordinal+sampler"),
    "Orig_noRow":  dict(model=dict(use_input_norm=True, use_multiscale=True, use_row_attn=False), loss="ordinal+sampler"),
    "Orig_noOrd":  dict(model=dict(use_input_norm=True, use_multiscale=True, use_row_attn=True), loss="ce+sampler"),
    # kontrol: hiçbir özgün bileşen yok (son aşama + ortalama havuzlama + CE) = ConvNeXt-Tiny; özgün modelin Optuna
    # hiperparametreleriyle -> "fark tasarımdan mı, hiperparametrelerden mi?" sorusunu ayırır
    "Orig_base": dict(model=dict(use_input_norm=False, use_multiscale=False, use_row_attn=False), loss="ce+sampler"),
    # "kontrast standardizasyonu çıkarıldı" ablasyonu = Orig_v3 (aynı yapılandırma; ayrı koşu açılmaz)
}
DESIGN_LOG = ["Orig_v1", "Orig_v2", "Orig_v3", "Orig_full"]
ABLATIONS = {"full": "Orig_full", "- çok ölçekli birleştirme": "Orig_noMS", "- satır dikkati": "Orig_noRow",
             "- ordinal kayıp (yerine CE)": "Orig_noOrd", "- girişte kontrast standardizasyonu": "Orig_v3"}


def convnext_hparams() -> dict:
    """Özgün model için başlangıç hiperparametreleri: aynı omurganın (ConvNeXt-Tiny) Optuna'daki en iyisi."""
    import json
    return json.load(open(C.OUTPUT_DIR / "optuna" / "ConvNeXtTiny" / "search_summary.json"))["best_params"]


def original_hparams() -> dict:
    """Özgün model için Optuna'nın bulduğu en iyi hiperparametreler (ordinal_sigma/lambda dahil)."""
    import json
    return json.load(open(C.OUTPUT_DIR / "optuna" / "Original" / "search_summary.json"))["best_params"]


def train_variant(variant: str, seed: int = C.SEED, hp: dict | None = None, tuned: bool = False, **kw) -> dict:
    """tuned=False: tasarım günlüğü (ConvNeXt'in hiperparametreleri). tuned=True: özgün modelin Optuna sonucu,
    koşu adı `<variant>__opt`."""
    from .train import train_model
    spec = VARIANTS[variant]
    if hp is None:
        hp = original_hparams() if tuned else convnext_hparams()
    return train_model("Original", seed, loss=spec["loss"], name=variant + ("__opt" if tuned else ""),
                       model_kwargs=spec["model"], **hp, **kw)


# --------------------------------------------------------------------------- mimari özeti (diyagram için)
def architecture_summary(model: nn.Module, input_size: int = 224) -> pd.DataFrame:
    """Üst düzey bloklar için katman türü, çıktı boyutu ve parametre sayısı (mimari şema tablosu)."""
    rows, hooks = [], []

    def shape_of(o):
        o = o[0] if isinstance(o, (list, tuple)) else o
        return tuple(o.shape[1:])

    def add(name, mod):
        def hook(_m, _i, o):
            rows.append({"block": name, "type": type(mod).__name__, "output": shape_of(o),
                         "params": sum(p.numel() for p in mod.parameters())})
        hooks.append(mod.register_forward_hook(hook))

    for n, m in model.backbone.named_children():
        add(f"backbone.{n}", m)
    if model.row_attn is not None:
        for i, m in enumerate(model.row_attn):
            add(f"row_attn[{i}]", m)
    for i, m in enumerate(model.norms):
        add(f"pool+norm[{i}]", m)
    add("head (Linear)", model.head)
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.randn(2, 3, input_size, input_size))
    for h in hooks:
        h.remove()
    model.train(was)
    return pd.DataFrame(rows)
