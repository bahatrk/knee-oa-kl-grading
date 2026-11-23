"""Aşama 0-2: veri indirme, bütünlük özeti (SHA256), kneeKL224 klasörünün çıkarılması
ve resmi train/val/test dosya listelerinin oluşturulması.

Elle müdahale gerektirmez. Örnek kullanım (Colab):
    python -m src.data_prep --url "<Excel dosyasındaki doğrudan indirme bağlantısı>"
    python -m src.data_prep --zip /content/drive/MyDrive/KneeXrayData.zip

Notlar
- Resmi bölme AYNEN kullanılır (VS13 kullanım notu). Burada yeniden bölme yapılmaz.
- Görüntü olmayan / gereksiz dosyalar (Thumbs.db vb.) listeye alınmaz ve sayısı raporlanır.
- Dosya listeleri bir kez kaydedilir; sonraki tüm deneyler aynı listeleri okur.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config as C
from .utils import get_logger, load_json, save_json

log = get_logger("data_prep", C.LOG_DIR)

# OAI hasta kimlikleri 7 haneli sayılardır (ör. 9001695). Dosya adının başındaki
# sayı dizisi hasta kimliği, hemen ardından gelen L/R harfi diz tarafı kabul edilir.
# Desen tutmazsa kimlik boş bırakılır ve bu durum denetimde raporlanır.
PATIENT_RE = re.compile(r"^(?P<pid>\d{5,})[_\-]?(?P<side>[LRlr])?")


# --------------------------------------------------------------------------- indirme
def sha256sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> Path:
    """Yönlendirmeleri izleyerek (curl -L) ve kesintide kaldığı yerden devam ederek indirir."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        cmd = ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
               "-C", "-", "-o", str(dest), url]
        log.info("İndiriliyor: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:  # curl yoksa requests ile
        import requests
        with requests.get(url, stream=True, allow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for block in r.iter_content(1 << 22):
                    f.write(block)
    return dest


# --------------------------------------------------------------------------- çıkarma
def _is_target_member(name: str) -> bool:
    low = name.replace("\\", "/").lower()
    return f"{C.DATASET_SUBDIR.lower()}/" in low and "__macosx" not in low


def extract_target(zip_path: Path, out_dir: Path, depth: int = 0) -> int:
    """Arşivden yalnızca kneeKL224 altındaki dosyaları çıkarır. kneeKL224 doğrudan
    bulunamazsa iç içe ZIP'lere iner (en fazla 3 seviye). Çıkarılan dosya sayısını döndürür."""
    if depth > 3:
        return 0
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        targets = [m for m in names if _is_target_member(m) and not m.endswith("/")]
        if targets:
            log.info("%s içinde %d hedef dosya bulundu, çıkarılıyor...", zip_path.name, len(targets))
            for m in targets:
                zf.extract(m, out_dir)
            return len(targets)
        inner = [m for m in names if m.lower().endswith(".zip") and "__macosx" not in m.lower()]
        # kneeKL224 adını taşıyan iç ZIP'e öncelik ver
        inner.sort(key=lambda m: (C.DATASET_SUBDIR.lower() not in m.lower(), "kl" not in m.lower()))
        for m in inner:
            log.info("İç arşive iniliyor: %s", m)
            tmp_dir = C.RAW_DIR / f"_nested_{depth}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            inner_path = Path(zf.extract(m, tmp_dir))
            # İç ZIP'in adı kneeKL224.zip ise içerik kökte olabilir; o durumda hepsini aç.
            if Path(m).stem.lower() == C.DATASET_SUBDIR.lower():
                with zipfile.ZipFile(inner_path) as izf:
                    has_prefix = any(_is_target_member(x) for x in izf.namelist())
                    dest = out_dir if has_prefix else out_dir / C.DATASET_SUBDIR
                    izf.extractall(dest)
                    n = sum(1 for x in izf.namelist() if not x.endswith("/"))
            else:
                n = extract_target(inner_path, out_dir, depth + 1)
            inner_path.unlink(missing_ok=True)
            if n:
                break
    return n


def locate_dataset_root(base: Path) -> Path:
    """base altında train/val/test alt klasörleri olan kneeKL224 klasörünü bulur."""
    candidates = [p for p in base.rglob("*") if p.is_dir() and p.name.lower() == C.DATASET_SUBDIR.lower()]
    for c in candidates:
        subs = {d.name.lower() for d in c.iterdir() if d.is_dir()}
        if {"train", "test"} <= {C.SPLIT_ALIASES.get(s, s) for s in subs}:
            return c
    raise FileNotFoundError(f"{base} altında train/val/test içeren '{C.DATASET_SUBDIR}' klasörü bulunamadı. "
                            f"Bulunan adaylar: {candidates}")


# --------------------------------------------------------------------------- dosya listesi
def parse_patient(filename: str) -> tuple[str | None, str | None]:
    m = PATIENT_RE.match(Path(filename).stem)
    if not m:
        return None, None
    side = m.group("side")
    return m.group("pid"), side.upper() if side else None


def build_file_list(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resmi klasör yapısından (split/sınıf/dosya) liste üretir.
    Dönüş: (görüntü listesi, atlanan dosyalar listesi)."""
    rows, skipped = [], []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        split = C.SPLIT_ALIASES.get(split_dir.name.lower())
        if split is None:
            log.warning("Tanınmayan klasör atlandı: %s", split_dir)
            continue
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if not cls_dir.name.isdigit() or int(cls_dir.name) >= C.NUM_CLASSES:
                log.warning("Tanınmayan sınıf klasörü atlandı: %s", cls_dir)
                continue
            label = int(cls_dir.name)
            for f in sorted(cls_dir.iterdir()):
                if f.is_dir():
                    continue
                rel = f.relative_to(root).as_posix()
                if f.name.lower() in C.JUNK_FILES or f.suffix.lower() not in C.IMAGE_EXTS:
                    skipped.append({"split": split, "relpath": rel, "reason": "görüntü değil / gereksiz dosya"})
                    continue
                if f.stat().st_size == 0:
                    skipped.append({"split": split, "relpath": rel, "reason": "boş dosya (0 bayt)"})
                    continue
                pid, side = parse_patient(f.name)
                rows.append({"split": split, "label": label, "class_name": C.CLASS_NAMES[label],
                             "relpath": rel, "filename": f.name, "patient_id": pid, "side": side,
                             "bytes": f.stat().st_size})
    df = pd.DataFrame(rows)
    df["split"] = pd.Categorical(df["split"], C.SPLITS, ordered=True)
    df = df.sort_values(["split", "label", "filename"]).reset_index(drop=True)
    return df, pd.DataFrame(skipped, columns=["split", "relpath", "reason"])


def save_split_lists(df: pd.DataFrame, root: Path) -> None:
    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.SPLITS_DIR / "all_files.csv", index=False)
    for s in C.SPLITS:
        df[df.split == s][["relpath", "label", "class_name", "patient_id", "side"]].to_csv(
            C.SPLITS_DIR / f"{s}.csv", index=False)
    save_json({"dataset_root": str(root), "created": datetime.now().isoformat(timespec="seconds"),
               "note": "Resmi kneeKL224 bölmesi aynen kullanıldı; bu listeler tüm deneylerde sabittir."},
              C.SPLITS_DIR / "split_info.json")
    log.info("Dosya listeleri kaydedildi: %s", C.SPLITS_DIR)


def get_dataset_root() -> Path:
    """Sonraki aşamalar veri köküne buradan erişir."""
    info = C.SPLITS_DIR / "split_info.json"
    if info.exists():
        p = Path(load_json(info)["dataset_root"])
        if p.exists():
            return p
    return locate_dataset_root(C.EXTRACT_DIR)


def ensure_data(url: str | None = None) -> Path:
    """Notebook'ların tek veri çağrısı. Sırayla: (1) görüntüler oturum diskinde zaten açıksa hiçbir şey yapma;
    (2) Drive'daki arşiv kopyası varsa onu oturum diskine kopyalayıp aç; (3) yoksa bağlantıdan indir, arşivi
    Drive'a bir kez kaydet ve aç. Dosya listeleri (outputs/splits) yeniden üretilmez: bölme bir kez yapıldı."""
    info_p = C.AUDIT_DIR / "download_info.json"
    info = load_json(info_p) if info_p.exists() else {}
    if C.EXTRACT_DIR.exists() and any(C.EXTRACT_DIR.rglob(C.DATASET_SUBDIR)):
        return get_dataset_root()
    C.ensure_dirs()
    cached = C.ARCHIVE_CACHE / C.ARCHIVE_NAME
    local = C.RAW_DIR / C.ARCHIVE_NAME
    if cached.exists():
        log.info("Arşiv Drive önbelleğinden kopyalanıyor (%.1f GB)...", cached.stat().st_size / 1e9)
        shutil.copyfile(cached, local)
        if local.stat().st_size != (info.get("archive_bytes") or C.ARCHIVE_BYTES):
            raise RuntimeError("Önbellekteki arşivin boyutu kayıtla uyuşmuyor; dosyayı silip yeniden indirin.")
    else:
        url = url or info.get("download_url") or C.DATASET_URL
        download(url, local)
        digest = sha256sum(local)
        expected = info.get("sha256") or C.ARCHIVE_SHA256
        if digest != expected:
            raise ValueError(f"SHA256 uyuşmuyor! beklenen={expected} bulunan={digest}")
        try:
            C.ARCHIVE_CACHE.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, cached)
            log.info("Arşiv bir sonraki oturumlar için kaydedildi: %s", cached)
        except OSError as e:
            log.warning("Arşiv Drive'a kaydedilemedi (%s); bir sonraki oturumda yeniden indirilecek.", e)
    if extract_target(local, C.EXTRACT_DIR) == 0:
        raise RuntimeError("Arşivde kneeKL224 bulunamadı.")
    local.unlink(missing_ok=True)   # oturum diskinde yer aç
    return get_dataset_root()


# --------------------------------------------------------------------------- ana akış
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="VS13 veri indirme ve hazırlama")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--url", help="KneeXrayData.zip doğrudan indirme bağlantısı")
    src.add_argument("--zip", type=Path, help="Önceden indirilmiş arşivin yolu (ör. Drive)")
    ap.add_argument("--expected-sha256", default=None, help="Varsa beklenen SHA256 özeti")
    ap.add_argument("--skip-extract", action="store_true", help="Veri zaten açıksa yalnızca liste üret")
    args = ap.parse_args(argv)
    C.ensure_dirs()

    info: dict = {"dataset": C.DATASET_NAME, "page": C.DATASET_PAGE, "doi": C.DATASET_DOI,
                  "license": C.DATASET_LICENSE}
    if not args.skip_extract:
        if args.url:
            zip_path = C.RAW_DIR / C.ARCHIVE_NAME
            download(args.url, zip_path)
            info["download_url"] = args.url
        elif args.zip:
            zip_path = args.zip
        else:
            ap.error("--url veya --zip verilmeli (ya da --skip-extract)")
        log.info("SHA256 hesaplanıyor (birkaç dakika sürebilir)...")
        digest = sha256sum(zip_path)
        info.update({"archive": zip_path.name, "archive_bytes": zip_path.stat().st_size,
                     "sha256": digest, "downloaded_at": datetime.now().isoformat(timespec="seconds")})
        if args.expected_sha256 and digest.lower() != args.expected_sha256.lower():
            raise ValueError(f"SHA256 uyuşmuyor! beklenen={args.expected_sha256} bulunan={digest}")
        log.info("SHA256 = %s", digest)
        try:   # sonraki oturumlar internetten değil buradan kopyalar
            C.ARCHIVE_CACHE.mkdir(parents=True, exist_ok=True)
            if not (C.ARCHIVE_CACHE / C.ARCHIVE_NAME).exists():
                shutil.copyfile(zip_path, C.ARCHIVE_CACHE / C.ARCHIVE_NAME)
        except OSError as e:
            log.warning("Arşiv önbelleğe kaydedilemedi: %s", e)
        n = extract_target(zip_path, C.EXTRACT_DIR)
        if n == 0:
            raise RuntimeError("Arşivde kneeKL224 bulunamadı. Arşiv içeriğini kontrol edin.")
        info["extracted_files"] = n

    root = locate_dataset_root(C.EXTRACT_DIR)
    df, skipped = build_file_list(root)
    save_split_lists(df, root)
    C.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    skipped.to_csv(C.AUDIT_DIR / "skipped_files.csv", index=False)

    info.update({"dataset_root": str(root), "n_images": len(df), "n_skipped": len(skipped),
                 "split_counts": df.split.value_counts().reindex(C.SPLITS).astype(int).to_dict()})
    prev = C.AUDIT_DIR / "download_info.json"
    if args.skip_extract and prev.exists():
        info = {**load_json(prev), **info}
    save_json(info, prev)
    log.info("Görüntü sayısı: %d | atlanan: %d | bölmeler: %s",
             len(df), len(skipped), info["split_counts"])


if __name__ == "__main__":
    main()
