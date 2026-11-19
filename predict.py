"""Tek bir diz röntgeni için KL derecesi tahmini (CPU).

Kullanım:  python predict.py goruntu.png [--temperature]
Çıktı: tahmin edilen derece, sınıf olasılıkları ve beklenen derece.
Not: Araştırma amaçlıdır; klinik karar aracı değildir.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models import build_model

HERE = Path(__file__).resolve().parent
DEPLOY = HERE / "deploy"


def load(deploy: Path = DEPLOY):
    meta = json.loads((deploy / "meta.json").read_text(encoding="utf-8"))
    model = build_model(meta["model_key"], pretrained=False, drop=meta.get("drop", 0.3), **meta.get("model_kwargs", {}))
    f = deploy / "model.pt"
    if not f.exists():  # depoda 100 MB sınırı nedeniyle float16 kopya taşınır; float32'ye çevrilir
        f = deploy / meta.get("fp16_file", "model_fp16.pt")
    sd = torch.load(f, map_location="cpu")
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
    return model.eval(), meta


def preprocess(path, meta):
    img = Image.open(path).convert("L").convert("RGB")
    size = meta.get("input_size", 224)
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(meta["mean"]).view(3, 1, 1); std = torch.tensor(meta["std"]).view(3, 1, 1)
    return ((x - mean) / std)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--temperature", action="store_true", help="doğrulamada öğrenilen sıcaklıkla kalibre et")
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model, meta = load()
    with torch.no_grad():
        logits = model(preprocess(a.image, meta))[0]
    p = torch.softmax(logits, 0).numpy()
    T = meta.get("temperature_val")
    if a.temperature and T:
        z = np.log(np.clip(p, 1e-12, 1)) / T; z -= z.max(); p = np.exp(z) / np.exp(z).sum()
    names = meta["class_names"]
    print(f"Tahmin: {names[int(p.argmax())]}  (güven {p.max():.2f})")
    print("Olasılıklar:", ", ".join(f"{n}={v:.3f}" for n, v in zip(names, p)))
    print(f"Beklenen derece: {float((p * np.arange(len(p))).sum()):.2f}")


if __name__ == "__main__":
    main()
