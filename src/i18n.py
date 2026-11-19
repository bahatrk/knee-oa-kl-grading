"""Şekil metinleri için TR -> EN çeviri katmanı.

KNEE_LANG=en ortam değişkeniyle etkinleşir. matplotlib Text.set_text çağrıları yakalanır ve bilinen Türkçe
ifadeler İngilizcesiyle değiştirilir; böylece şekil kodu (ve veri anahtarları) değişmeden İngilizce şekil üretilir.
Uzun ifadeler önce değiştirilir (en uzun eşleşme önceliklidir).
"""
from __future__ import annotations

import os

LANG = os.environ.get("KNEE_LANG", "tr").lower()

_PAIRS = {
    # akış şeması
    "Veri seti\n(8.260 görüntü)": "Dataset\n(8,260 images)",
    "Denetim:\nsızıntı, kopya,\nsınıf dengesizliği": "Audit: leakage,\nduplicates,\nclass imbalance",
    "Ön işleme\n224×224": "Preprocessing\n224×224",
    "HOG+SVM\nve 10 hazır model": "HOG+SVM and\n10 standard\nmodels",
    "Optuna arama\n(≥20 deneme)": "Optuna search\n(≥20 trials)",
    "Özgün model:\ntasarım günlüğü,\nablasyon": "Original model:\ndesign log,\nablation",
    "Hibrit\nbirleştirme": "Hybrid\nfusion",
    "Test (tek sefer):\nmetrik, GA, kalibrasyon": "Test (once):\nmetrics, CI,\ncalibration",
    "XAI ve\nzorlu görevler": "XAI and\nchallenge tasks",
    "Tüm seçimler (model, epoch, eşik, ağırlık) yalnızca doğrulama macro F1 ile yapıldı; test kümesi son aşamada bir kez kullanıldı.":
        "All selections (model, epoch, threshold, weights) used validation macro F1 only; the test set was used once at the end.",
    # mimari
    "Giriş\n3×224×224": "Input\n3×224×224",
    "A: görüntü başına\nstandardizasyon": "A: per-image\nstandardization",
    "ConvNeXt-Tiny omurga\n(ImageNet)": "ConvNeXt-Tiny backbone\n(ImageNet)",
    "aşama 2": "stage 2", "aşama 3": "stage 3", "aşama 4": "stage 4",
    "C: satır dikkati": "C: row attention",
    "B: ort.+maks.\nhavuz + LayerNorm": "B: avg+max\npool + LayerNorm",
    "Birleştir\n+ Dropout\n+ Linear": "Concatenate\n+ Dropout\n+ Linear",
    "Çıktı: 5 sınıf (KL0–KL4)\nD: ordinal kayıp (eğitimde)": "Output: 5 classes (KL0–KL4)\nD: ordinal loss (training)",
    "Turuncu: bu çalışmada eklenen bileşenler (A, B, C, D). Mavi: hazır ağırlıklar / standart katmanlar.":
        "Orange: components added in this work (A, B, C, D). Blue: pretrained weights / standard layers.",
    " par.": " params",
    # eğriler / matrisler
    "Doğruluk": "Accuracy", "Kayıp": "Loss",
    "Eğitim kümesinden rastgele örnekler (seed=42), her satır bir KL derecesi":
        "Random training samples (seed=42), one KL grade per row",
    "Eğitim": "Training", "Doğrulama": "Validation",
    "en iyi epoch": "best epoch",
    "erken durdurma (ep.": "early stopping (ep.",
    "epoch bütçesi bitti (ep.": "epoch budget reached (ep.",
    "Gerçek": "True", "Tahmin": "Predicted",
    "Yanlış pozitif oranı": "False positive rate", "Doğru pozitif oranı": "True positive rate",
    "Duyarlılık (recall)": "Recall", "Kesinlik (precision)": "Precision",
    "Sınıf (çizgi: aşağıda)": "Class (line style below)",
    "Sınıf (tüm bölmeler)": "Class (all splits)",
    "Özgün model": "Original model",
    "Ortalama güven": "Mean confidence",
    "sıcaklık (T=": "temperature (T=",
    # ablasyon / verimlilik
    "Ablasyon (parametre sayısı çubuk içinde)": "Ablation (parameter count at bar start)",
    "Test macro F1 (ort. ± std; noktalar: seed'ler)": "Test macro F1 (mean ± SD; dots: seeds)",
    "Başarım - gecikme": "Performance vs. latency",
    "CPU gecikmesi (ms, batch 1, medyan)": "CPU latency (ms, batch 1, median)",
    "Tam model": "Full model", "- çok ölçekli": "- multi-scale", "- satır dikkati": "- row attention",
    "- ordinal kayıp": "- ordinal loss", "- standardizasyon": "- standardization",
    "Kontrol (düz ConvNeXt)": "Control (plain ConvNeXt)",
    "DeiT-Small (en iyi mevcut)": "DeiT-Small (best existing)",
    "Hibrit": "Hybrid", "Özgün": "Original",
    # xai
    "Görüntü": "Image",
    "Özgün rastgele": "Original random", "DeiT rastgele": "DeiT random",
    "yanlış (özgün)": "wrong (original)",
    "\nÖ:KL": "\nOrig:KL", " D:KL": " DeiT:KL",
    " rastgele": " random",
    "Silinen piksel oranı (%)": "Fraction of pixels deleted (%)",
    "Tahmin edilen sınıfın olasılığı": "Probability of predicted class",
    "Kademeli rastgeleleştirilen katman grubu sayısı (çıktıdan girdiye)": "Layer groups randomized cumulatively (output to input)",
    "Spearman benzerliği (Grad-CAM)": "Spearman similarity (Grad-CAM)",
    # zorlu görevler
    "Gauss bulanıklığı": "Gaussian blur", "Gauss gürültüsü": "Gaussian noise", "JPEG sıkıştırma": "JPEG compression",
    "Kontrast kaybı": "Contrast loss", "Şiddet": "Severity",
    "Hata oranı (kabul edilenlerde)": "Error rate (accepted cases)", "Kapsam": "Coverage",
    "Test kapsamı (eşik doğrulamadan)": "Test coverage (threshold from validation)",
    "Kabul edilenlerde macro F1": "Macro F1 (accepted cases)",
    " (sıcaklık)": " (temperature)",
    # denetim şekilleri
    "(a) Sınıf başına görüntü sayısı": "(a) Images per class", "Görüntü sayısı": "Number of images",
    "(b) Bölme içi sınıf oranı": "(b) Class share within split", "Bölme içindeki oran (%)": "Share within split (%)",
    "kneeKL224 sınıf dağılımı — eğitimde en büyük/en küçük sınıf oranı": "kneeKL224 class distribution — train largest/smallest class ratio",
    "Ortalama piksel yoğunluğu (0–255)": "Mean pixel intensity (0–255)", "Yoğunluk": "Density",
    "(a) Parlaklık dağılımı": "(a) Brightness distribution", "(b) Kontrast dağılımı": "(b) Contrast distribution",
    "Test kümesi": "Test set",
    "maks. olasılık": "max. probability",
    "sıcaklık": "temperature",
    "Şimdiye dek en iyi": "Best so far", "Tamamlanan": "Completed", "Budanan": "Pruned",
    "Deneme no": "Trial", ": deneme geçmişi": ": trial history", "(b) Hiperparametre önemi": "(b) Hyperparameter importance",
    "Önem (fANOVA)": "Importance (fANOVA)",
    # augmentation şekli (figures_en)
    "Orijinal": "Original", "Artırılmış": "Augmented",
    "Test": "Test",
}
_EXACT = {"ham": "uncalibrated", "orijinal": "trained"}
_ORDERED = sorted(_PAIRS.items(), key=lambda kv: -len(kv[0]))


def tr2en(s):
    if not isinstance(s, str) or not s:
        return s
    if s in _EXACT:
        return _EXACT[s]
    for k, v in _ORDERED:
        if k in s:
            s = s.replace(k, v)
    return s


_installed = False


def install() -> None:
    """matplotlib metin nesnelerine çeviri kancası ekler (yalnızca KNEE_LANG=en iken)."""
    global _installed
    if LANG != "en" or _installed:
        return
    from matplotlib import text as mtext
    orig = mtext.Text.set_text

    def set_text(self, s):
        return orig(self, tr2en(s) if isinstance(s, str) else s)

    mtext.Text.set_text = set_text
    _installed = True
