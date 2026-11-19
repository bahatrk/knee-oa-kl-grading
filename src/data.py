"""Veri yükleme: sabit dosya listelerinden Dataset ve DataLoader üretimi.

Kurallar (proje belgesi)
- Görüntüler gri seviyedir; 3 kanala çoğaltılır.
- Transfer learning modellerinde her modelin KENDİ normalizasyonu (timm pretrained_cfg) kullanılır;
  sıfırdan eğitilen ağlarda eğitim kümesinden hesaplanan ortalama/std kullanılır.
- Augmentation YALNIZCA eğitim kümesinde. Dizler sol/sağ ayna simetrik olduğundan yatay çevirme
  KL derecesini değiştirmez (kullanılır); dikey çevirme kullanılmaz. Küçük döndürme/ölçekleme/öteleme
  ve parlaklık-kontrast oynatması eklem aralığı ve osteofit görünümünü bozmayacak kadar hafiftir.
- Sınıf dengesizliği: sınıf ağırlıklı kayıp için ağırlıklar yalnızca eğitim dağılımından hesaplanır.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T

from . import config as C
from .data_prep import get_dataset_root

# Veri boyutu 224x224'tür; tüm modeller aynı giriş boyutuyla (224) karşılaştırılır.
INPUT_SIZE = 224


def read_split(split: str, subset_fraction: float = 1.0, seed: int = C.SEED) -> pd.DataFrame:
    df = pd.read_csv(C.SPLITS_DIR / f"{split}.csv", dtype={"patient_id": str})
    if subset_fraction < 1.0:  # sınıf oranlarını koruyan sabit-seed alt küme (Optuna / veri verimliliği)
        idx = np.concatenate([df.index[df.label == k].to_series().sample(frac=subset_fraction, random_state=seed).values
                              for k in sorted(df.label.unique())])
        df = df.loc[np.sort(idx)]  # pandas 2 ve 3'te aynı davranır
    return df.reset_index(drop=True)


def class_counts(df: pd.DataFrame) -> np.ndarray:
    return np.bincount(df.label.values, minlength=C.NUM_CLASSES)


def class_weights(df: pd.DataFrame, mode: str = "balanced") -> torch.Tensor:
    """'balanced': n / (K * n_k). Yalnızca eğitim dağılımından hesaplanır."""
    cnt = class_counts(df).astype(np.float64)
    if mode == "sqrt":
        w = np.sqrt(cnt.sum() / (C.NUM_CLASSES * cnt))
    else:
        w = cnt.sum() / (C.NUM_CLASSES * cnt)
    return torch.tensor(w, dtype=torch.float32)


def train_mean_std(df: pd.DataFrame, root: Path, n: int = 1500, seed: int = C.SEED) -> tuple[float, float]:
    """Sıfırdan CNN'ler için: eğitim kümesinden (yalnızca eğitim!) piksel ortalama/std."""
    sub = df.sample(min(n, len(df)), random_state=seed)
    px = np.concatenate([np.asarray(Image.open(root / r).convert("L"), dtype=np.float32).ravel() / 255.0
                         for r in sub.relpath])
    return float(px.mean()), float(px.std())


def build_transforms(train: bool, mean, std, augment: bool = True):
    norm = T.Normalize(mean=mean, std=std)
    if train and augment:
        return T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.93, 1.07)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(), norm])
    return T.Compose([T.ToTensor(), norm])


class KneeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, transform, cache: bool = True):
        self.df, self.root, self.transform = df.reset_index(drop=True), root, transform
        self.labels = self.df.label.values.astype(np.int64)
        # 224x224 gri görüntüler küçüktür: tümü RAM'e alınır (8.260 x 50 KB ≈ 0.4 GB), disk okuması bitmez.
        self.cache = ([Image.open(root / r).convert("RGB").copy() for r in self.df.relpath] if cache else None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = self.cache[i] if self.cache is not None else Image.open(self.root / self.df.relpath[i]).convert("RGB")
        return self.transform(img), int(self.labels[i])


def make_loaders(mean, std, batch_size: int = 64, num_workers: int = 2, augment: bool = True,
                 subset_fraction: float = 1.0, sampler: str | None = None, splits=("train", "val"),
                 seed: int = C.SEED) -> dict[str, DataLoader]:
    """splits içinde 'test' yalnızca final değerlendirmede istenir."""
    root = get_dataset_root()
    out = {}
    g = torch.Generator()
    g.manual_seed(seed)
    for s in splits:
        df = read_split(s, subset_fraction if s == "train" else 1.0, seed)
        ds = KneeDataset(df, root, build_transforms(s == "train", mean, std, augment))
        if s == "train":
            smp = None
            if sampler == "weighted":  # ağırlıklı örnekleme (dengeleme seçeneklerinden biri)
                w = (1.0 / class_counts(df))[df.label.values]
                smp = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(df), generator=g)
            out[s] = DataLoader(ds, batch_size=batch_size, shuffle=smp is None, sampler=smp,
                                num_workers=num_workers, drop_last=True, generator=g,
                                persistent_workers=num_workers > 0)
        else:
            out[s] = DataLoader(ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers,
                                persistent_workers=num_workers > 0)
    return out
