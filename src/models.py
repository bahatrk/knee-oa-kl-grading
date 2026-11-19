"""Model fabrikası: sıfırdan 2 CNN + 7 transfer learning modeli + 1 ViT (timm).

Özgün ve hibrit modeller sonraki aşamada bu dosyaya eklenecektir.
Tüm modeller 3 kanallı 224x224 girdi alır (gri görüntü 3 kanala çoğaltılır).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C

# anahtar -> açıklama. Sıra = proje belgesindeki dağılım:
# 2 sıfırdan CNN, 7 transfer learning, 1 dikkat tabanlı (ViT)
ZOO: dict[str, dict] = {
    "SimpleCNN":     dict(family="CNN (sıfırdan)", kind="scratch"),
    "MiniResNet":    dict(family="CNN (sıfırdan, artık bağlantılı)", kind="scratch"),
    "ResNet50":      dict(family="CNN (transfer)", kind="timm", timm_name="resnet50.tv_in1k"),
    "DenseNet121":   dict(family="CNN (transfer)", kind="timm", timm_name="densenet121.tv_in1k"),
    "EfficientNetB0": dict(family="CNN (transfer)", kind="timm", timm_name="efficientnet_b0.ra_in1k"),
    "MobileNetV3L":  dict(family="CNN (transfer)", kind="timm", timm_name="mobilenetv3_large_100.ra_in1k"),
    "Xception":      dict(family="CNN (transfer)", kind="timm", timm_name="legacy_xception.tf_in1k"),
    "InceptionV3":   dict(family="CNN (transfer)", kind="timm", timm_name="inception_v3.tv_in1k"),
    "ConvNeXtTiny":  dict(family="CNN (transfer, modern)", kind="timm", timm_name="convnext_tiny.fb_in1k"),
    "DeiTSmall":     dict(family="Transformer (ViT, transfer)", kind="timm",
                          timm_name="deit_small_patch16_224.fb_in1k"),
    # Özgün model (src/original.py); karşılaştırma tablosundaki 10 hazır modelden ayrıdır
    "Original":      dict(family="Özgün (ConvNeXt-T + özgün bileşenler)", kind="original"),
}


class SimpleCNN(nn.Module):
    """4 evrişim bloğu (Conv-BN-ReLU-MaxPool) + global ortalama havuzlama + dropout + FC."""

    def __init__(self, num_classes=C.NUM_CLASSES, widths=(32, 64, 128, 256), drop=0.3):
        super().__init__()
        layers, cin = [], 3
        for w in widths:
            layers += [nn.Conv2d(cin, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            cin = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        return self.fc(self.drop(self.pool(self.features(x)).flatten(1)))


class _BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.short = (nn.Identity() if stride == 1 and cin == cout else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))

    def forward(self, x):
        y = self.act(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return self.act(y + self.short(x))


class MiniResNet(nn.Module):
    """Küçük ResNet: 3x3 stem + 4 aşama x 2 artık blok (32-64-128-256) + GAP + dropout + FC."""

    def __init__(self, num_classes=C.NUM_CLASSES, widths=(32, 64, 128, 256), drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 2, 1, bias=False), nn.BatchNorm2d(widths[0]),
                                  nn.ReLU(inplace=True))
        blocks, cin = [], widths[0]
        for i, w in enumerate(widths):
            blocks += [_BasicBlock(cin, w, 1 if i == 0 else 2), _BasicBlock(w, w, 1)]
            cin = w
        self.layers = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        return self.fc(self.drop(self.pool(self.layers(self.stem(x))).flatten(1)))


def build_model(key: str, pretrained: bool = True, drop: float = 0.3, **model_kwargs) -> nn.Module:
    spec = ZOO[key]
    if spec["kind"] == "original":
        from .original import KneeOrdinalNet
        return KneeOrdinalNet(pretrained=pretrained, drop=drop, **model_kwargs)
    if spec["kind"] == "scratch":
        return {"SimpleCNN": SimpleCNN, "MiniResNet": MiniResNet}[key](drop=drop)
    import timm
    return timm.create_model(spec["timm_name"], pretrained=pretrained, num_classes=C.NUM_CLASSES, drop_rate=drop)


def get_norm(key: str, model: nn.Module, scratch_stats: tuple[float, float] | None = None):
    """Transfer modellerde modelin kendi ön işlemesi; sıfırdan modellerde eğitim kümesi istatistiği."""
    if ZOO[key]["kind"] == "scratch":
        m, s = scratch_stats
        return (m,) * 3, (s,) * 3
    cfg = model.pretrained_cfg
    return tuple(cfg["mean"]), tuple(cfg["std"])


def head_parameters(model: nn.Module, key: str):
    if ZOO[key]["kind"] == "scratch":
        return list(model.fc.parameters())
    if ZOO[key]["kind"] == "original":
        return model.new_parameters()
    return list(model.get_classifier().parameters())


def set_trainable(model: nn.Module, key: str, stage: str, unfreeze_frac: float = 0.5) -> None:
    """stage='head': yalnızca sınıflandırıcı başı eğitilir (dondurulmuş öznitelik çıkarıcı).
    stage='finetune': parametre listesinin son `unfreeze_frac` kısmı + baş çözülür (seçili katmanlar).
    stage='all': hepsi (sıfırdan eğitim)."""
    params = list(model.parameters())
    if stage == "all":
        for p in params:
            p.requires_grad = True
        return
    head_ids = {id(p) for p in head_parameters(model, key)}
    if stage == "head":
        for p in params:
            p.requires_grad = id(p) in head_ids
        return
    n_unfreeze = int(round(len(params) * unfreeze_frac))
    for i, p in enumerate(params):
        p.requires_grad = (i >= len(params) - n_unfreeze) or (id(p) in head_ids)


def freeze_bn_stats(model: nn.Module) -> None:
    """Dondurulmuş BatchNorm katmanlarını eval modunda tutar (ImageNet istatistikleri bozulmasın)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            ps = list(m.parameters(recurse=False))
            if ps and not any(p.requires_grad for p in ps):
                m.eval()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
