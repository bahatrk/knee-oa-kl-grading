# VS13 · Diz Röntgenlerinde Osteoartrit Şiddeti (KL 0–4) Sınıflandırma

MBM6 453 Yapay Zekâya Giriş dersi projesi · Baha Türk · 220709071

Tüm sayılar `results.csv` ve `outputs/final/` içindeki CSV dosyalarından gelir; rapor bu dosyalardan üretilmiştir.

## Veri seti
- **Ad:** Knee Osteoarthritis Severity Grading Dataset (Chen, 2018), OAI kaynaklı
- **Sayfa:** https://data.mendeley.com/datasets/56rmx5bjcr/1 · **DOI:** 10.17632/56rmx5bjcr.1 · **Lisans:** CC BY 4.0
- **Kullanılan kısım:** `KneeXrayData.zip` → `kneeKL224` (8.260 görüntü; resmi bölme 5.778 / 826 / 1.656)
- Ham veri bu teslime dahil değildir. İndirme bağlantısı, komutu ve arşivin SHA256 özeti aşağıdadır.
- **Doğrudan indirme bağlantısı:** `https://data.mendeley.com/public-files/datasets/56rmx5bjcr/files/205a9a5d-5de5-4f06-8f63-e430ed23ca44/file_downloaded`
- **Arşiv:** `KneeXrayData.zip` (7.171.571.746 bayt)
- **Arşiv SHA256:** `3b4b93b3a3e2e2f059684f5d2b8bbd05aa7776d0199365fc6db51b93dc667505`

### İndirme ve hazırlama komutu
```bash
pip install -r requirements.txt
export KNEE_DATA_ROOT=./data            # ham veri ve çıkarılan görüntüler
export KNEE_OUTPUT_DIR=./outputs
python -m src.data_prep --url "https://data.mendeley.com/public-files/datasets/56rmx5bjcr/files/205a9a5d-5de5-4f06-8f63-e430ed23ca44/file_downloaded" \
                        --expected-sha256 3b4b93b3a3e2e2f059684f5d2b8bbd05aa7776d0199365fc6db51b93dc667505
```
Komut arşivi indirir, SHA256 özetini doğrular (uyuşmazsa durur), yalnızca `kneeKL224` klasörünü açar ve resmi bölmeden
sabit dosya listelerini (`outputs/splits/`) üretir. Bölme yeniden yapılmaz; resmi 5.778 / 826 / 1.656 ayrımı aynen kullanılır.

> Bu model eğitim ve araştırma amaçlıdır; klinik karar destek aracı **değildir**.

## Klasör yapısı (teslim)
```
220709071_BahaTurk_YZProje/
├── README.md
├── requirements.txt                   # sürümleri sabitlenmiş ortam (ayrıntı: docs/ENVIRONMENT.md)
├── 220709071_BahaTurk_YZProje.ipynb   # tek, baştan sona çalışan notebook
├── predict.py                         # tek görüntü, CPU, eğitim verisi gerekmez
├── results.csv                        # model ve seed başına bir satır (doğrulama + test metrikleri, gecikme)
├── YZProje_Rapor_220709071.docx / .pdf
├── src/                               # modüler kaynak kod (veri, modeller, eğitim, değerlendirme, XAI)
├── deploy/                            # model_fp16.pt + meta.json (sınıf adları, ön işleme, sıcaklık, SHA256)
├── outputs/                           # denetim, bölme listeleri, nihai tablolar, şekiller, XAI çıktıları
├── notebooks/asamalar/                # 01–15 aşama notebook'ları
├── tools/                             # sonuç genişletme, teslim paketi, notebook üreticileri
└── docs/
```
Aşağıdaki `src/` listesi modüllerin görevlerini gösterir:
```
src/
├── config.py        # yollar, sınıflar, seed'ler, sabitler        ├── models.py     # 2 sıfırdan CNN + 7 transfer + ViT + özgün model
├── utils.py         # seed, loglama, ortam bilgisi                 ├── original.py   # KneeOrdinalNet tasarım günlüğü ve varyantlar
├── data_prep.py     # indirme, SHA256, kneeKL224, dosya listeleri ├── train.py      # ortak eğitim protokolü, dengeleme
├── data.py          # Dataset/DataLoader, augmentation (yalnızca eğitim) ├── optimize.py   # Optuna araması
├── audit.py, dup_verify.py  # veri denetimi, kopya doğrulama       ├── seeds.py, ablation.py, hybrid.py
├── baseline.py      # HOG + Linear SVM                             ├── final_eval.py # test (tek sefer), CI, kalibrasyon
├── metrics.py, results.py                                          ├── xai.py, challenges.py, export.py
└── plot_style.py, figures_final.py, figures_en.py, i18n.py
```

## Sonuçları yeniden üretme (en kısa yol)
Tek çalıştırılabilir notebook: **`220709071_BahaTurk_YZProje.ipynb`** (Google Colab, T4 GPU önerilir).
1. Teslim klasörünü Google Drive'a `MyDrive/YZProje` olarak yükleyin ve notebook'u Colab'da açın.
2. Hücreleri baştan sona çalıştırın. Varsayılan `RETRAIN = False` modunda notebook veriyi indirip denetler, HOG+SVM'i ve
   kayıtlı sonuçlardan tabloları ve şekilleri üretir, `predict.py` ile örnek bir görüntüyü tahmin eder.
3. `RETRAIN = True` yapılırsa tüm eğitim, Optuna, ablasyon, hibrit ve değerlendirme aşamaları sırayla yeniden çalışır
   (kayıtlı eğitim süresi toplamı 7.2 saat, T4 GPU; Colab oturum sınırı nedeniyle aşamalara bölünmüş notebook'lar `notebooks/asamalar/` içindedir).
Aşama notebook'ları (01 → 15) deneylerin gerçekte çalıştırıldığı sıradır.

## Komut satırı kullanımı
```bash
pip install -r requirements.txt
export KNEE_DATA_ROOT=/content/data
python -m src.data_prep --url "<KneeXrayData.zip bağlantısı>"   # indir + çıkar + dosya listeleri
python -m src.audit                                              # veri denetimi
python -m src.dup_verify                                         # yakın-kopya doğrulama
python -m src.baseline                                           # HOG+SVM (seçim yalnızca val)
python -m src.train --models EfficientNetB0 ResNet50 ...        # model eğitimi (validation ile seçim)
python -c "from src import optimize as O; O.run_search('ConvNeXtTiny'); O.retrain_best('ConvNeXtTiny')"   # Optuna
```

### Test değerlendirmesi, XAI ve tahmin
```bash
python -c "from src import final_eval as F; F.predict_all(); F.analyze(); F.cpu_latency(); F.cpu_latency_extra()"   # test: TEK SEFER
python -c "from src import xai; xai.run()"                         # Grad-CAM, IG, rollout, rastgeleleştirme, silme eğrisi
python -c "from src import challenges as H; H.robustness(); H.selective()"   # zorlu görevler
python -c "from src import export; export.export(); export.export_fp16()"   # deploy/ paketi
python predict.py ornek_goruntu.png            # tek görüntü, CPU (deploy/ gerekir)
python predict.py ornek_goruntu.png --temperature   # doğrulamada öğrenilen sıcaklıkla kalibre
```
Not: Eğitim ve deneyler Colab'da `notebooks/asamalar/` sırasıyla (01 → 15) çalıştırıldı; aynı seed'lerle (42, 123, 2026) GPU kaynaklı küçük sapmalar olabilir.

## Dağıtım paketi
`deploy/model_fp16.pt` (özgün model KneeOrdinalNet, ağırlıklar float16; yüklenirken float32'ye çevrilir, GitHub 100 MB sınırı nedeniyle)
ve `deploy/meta.json` (ön işleme ortalaması/standart sapması, sıcaklık, SHA256, float16 kontrolü). Girdi: 224×224 diz röntgeni.
Çıktı: KL0–KL4 olasılıkları. **Araştırma amaçlıdır, klinik karar aracı değildir.**

## Son aşamalar
- `11_nihai_test_degerlendirme.ipynb`: test kümesi (tek sefer), güven aralıkları, kalibrasyon
- `12_hog_svm_test_ve_sekiller.ipynb`: HOG+SVM test, rapor şekilleri, `deploy/` paketi
- `13_xai_ve_zorlu_gorevler.ipynb`: Grad-CAM/IG/rollout, bozulma dayanıklılığı, seçici tahmin
- `14_ingilizce_rapor_sekilleri.ipynb`: rapor şekillerinin İngilizce sürümü (`KNEE_LANG=en`, çıktı `outputs/figures_en/`; model eğitilmez)
- `15_gecikme_ve_teslim_paketi.ipynb`: ablasyon/kontrol CPU gecikmesi, float16 dağıtım paketi, teslim klasörü
- Tek görüntü tahmini (CPU): `python predict.py goruntu.png`
