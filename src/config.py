"""Proje genelinde kullanılan sabitler ve yollar (VS13 - Diz osteoartriti KL derecesi).

Colab'da varsayılan düzen:
  - Kod ve çıktılar : /content/drive/MyDrive/YZProje   (kalıcı Drive)
  - Ham veri        : /content/data                    (oturum diski, Drive'a KOPYALANMAZ)
Yollar ortam değişkenleriyle değiştirilebilir: KNEE_DATA_ROOT, KNEE_OUTPUT_DIR.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- yollar
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("KNEE_DATA_ROOT", "/content/data"))
RAW_DIR = DATA_ROOT / "raw"            # indirilen arşiv
EXTRACT_DIR = DATA_ROOT / "extracted"  # yalnızca kneeKL224 açılır

# Ham ARŞİV (açılmamış zip) Drive'da bir kez saklanır; her oturumda yeniden indirmek yerine buradan kopyalanır.
# Açılmış görüntüler Drive'a konmaz (ödev belgesi: büyük veri oturum diskinde tutulur).
ARCHIVE_CACHE = Path(os.environ.get("KNEE_ARCHIVE_CACHE", "/content/drive/MyDrive/YZProje_cache"))

OUTPUT_DIR = Path(os.environ.get("KNEE_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
SPLITS_DIR = OUTPUT_DIR / "splits"     # sabit dosya listeleri (tüm deneylerde aynı)
AUDIT_DIR = OUTPUT_DIR / "audit"       # veri denetimi tabloları / json
LANG = os.environ.get("KNEE_LANG", "tr").lower()   # "en": İngilizce rapor şekilleri -> outputs/figures_en
FIG_DIR = OUTPUT_DIR / ("figures_en" if LANG == "en" else "figures")   # rapora girecek figürler (>=200 dpi)
LOG_DIR = OUTPUT_DIR / "logs"

# --------------------------------------------------------------------------- veri seti
DATASET_CODE = "VS13"
DATASET_NAME = "Knee Osteoarthritis Severity Grading Dataset"
DATASET_PAGE = "https://data.mendeley.com/datasets/56rmx5bjcr/1"
DATASET_DOI = "10.17632/56rmx5bjcr.1"
DATASET_LICENSE = "CC BY 4.0"
DATASET_SUBDIR = "kneeKL224"           # proje belgesine göre yalnızca bu klasör kullanılır
ARCHIVE_NAME = "KneeXrayData.zip"

NUM_CLASSES = 5
CLASS_NAMES = ["KL0", "KL1", "KL2", "KL3", "KL4"]
CLASS_DESCRIPTIONS_TR = (["Normal", "Doubtful", "Mild", "Moderate", "Severe"] if os.environ.get("KNEE_LANG", "tr").lower() == "en"
                         else ["Normal", "Şüpheli", "Hafif", "Orta", "Şiddetli"])
SPLITS = ["train", "val", "test"]
# Arşivdeki olası klasör adları -> standart ad
SPLIT_ALIASES = {"train": "train", "training": "train",
                 "val": "val", "valid": "val", "validation": "val",
                 "test": "test", "testing": "test"}

# Ödev belgesinde ve literatürde bildirilen sayılar; denetimde yeniden hesaplanıp karşılaştırılır.
EXPECTED_SPLIT_COUNTS = {"train": 5778, "val": 826, "test": 1656}
EXPECTED_TOTAL = 8260
# Literatürde (Chen ve ark., 2019) bildirilen sınıf dağılımı - yalnızca karşılaştırma içindir.
EXPECTED_CLASS_COUNTS = {
    "train": [2286, 1046, 1516, 757, 173],
    "val":   [328, 153, 212, 106, 27],
    "test":  [639, 296, 447, 223, 51],
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
JUNK_FILES = {"thumbs.db", ".ds_store", "desktop.ini"}

# --------------------------------------------------------------------------- deney
SEED = 42                      # bölme / alt küme seçimleri için sabit seed
SEEDS = [42, 123, 2026]        # çoklu seed tekrarları
IMBALANCE_RATIO_THRESHOLD = 3.0

# pHash yakın-kopya taraması
PHASH_HASH_SIZE = 8            # 64 bit
PHASH_THRESHOLDS = [0, 2, 4, 6, 8, 10]
PHASH_PRIMARY_THRESHOLD = 4    # raporda "yakın kopya" kabul edilen Hamming eşiği


def ensure_dirs() -> None:
    for d in (RAW_DIR, EXTRACT_DIR, OUTPUT_DIR, SPLITS_DIR, AUDIT_DIR, FIG_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
