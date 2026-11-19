"""outputs klasörünü tek bir zip'e paketler (Kaggle'a veri seti olarak yüklemek için).

Varsayılan: model ağırlıkları (model.pt) HARİÇ -> küçük paket (birkaç MB).
  python tools/export_outputs.py --outputs <outputs klasörü> --out <zip yolu>
Ağırlıklar dahil (daha sonra, değerlendirme aşamasında gerekir):
  python tools/export_outputs.py --outputs ... --out ... --with-models
"""
import argparse
import os
import zipfile
from pathlib import Path


def export(outputs: Path, out: Path, with_models: bool = False) -> tuple[int, float]:
    n, total = 0, 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(outputs.rglob("*")):
            if not p.is_file() or p.suffix == ".db" and "optuna" in p.parts:
                continue  # Optuna veritabanları yerelde yeniden üretilir / ayrıca taşınır
            if p.name == "model.pt" and not with_models:
                continue
            if "figures" in p.parts and p.suffix == ".png" and not with_models:
                pass  # figürler küçük; dahil
            z.write(p, Path("outputs") / p.relative_to(outputs))
            n += 1
            total += p.stat().st_size
    return n, total / 2**20


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--with-models", action="store_true")
    a = ap.parse_args()
    n, mb = export(a.outputs, a.out, a.with_models)
    print(f"{n} dosya, {mb:.1f} MB (sıkıştırma öncesi) -> {a.out} ({os.path.getsize(a.out) / 2**20:.1f} MB)")
