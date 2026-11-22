# VS13 · Diz Röntgenlerinde Osteoartrit Şiddeti (KL 0–4) Sınıflandırma

MBM6 453 Yapay Zekâya Giriş dersi projesi · Baha Türk · 220709071

Tüm sayılar `results.csv` ve `outputs/final/` içindeki CSV dosyalarından gelir; rapor bu dosyalardan üretilmiştir.

## Veri seti
- **Ad:** Knee Osteoarthritis Severity Grading Dataset (Chen, 2018), OAI kaynaklı
- **Sayfa:** https://data.mendeley.com/datasets/56rmx5bjcr/1 · **DOI:** 10.17632/56rmx5bjcr.1 · **Lisans:** CC BY 4.0
- **Kullanılan kısım:** `KneeXrayData.zip` → `kneeKL224` (8.260 görüntü; resmi bölme 5.778 / 826 / 1.656)
- Ham veri bu teslime dahil değildir. İndirme bağlantısı, komutu ve arşivin SHA256 özeti aşağıdadır.
- **Arşiv SHA256:** `{{SHA256}}`

> Bu model eğitim ve araştırma amaçlıdır; klinik karar destek aracı **değildir**.

## Klasör yapısı
```
YZProje/
├── src/                 # modüler kaynak kod
│   ├── config.py        # yollar, sınıflar, seed'ler, sabitler
│   ├── utils.py         # seed, loglama, ortam bilgisi
│   ├── plot_style.py    # ortak figür stili (>=200 dpi)
│   ├── data_prep.py     # indirme, SHA256, kneeKL224 çıkarma, sabit dosya listeleri
│   ├── audit.py         # veri denetimi
│   ├── dup_verify.py    # pHash adaylarının SSIM/NCC ile doğrulanması
│   ├── metrics.py       # tüm zorunlu metrikler (QWK, MAE dahil)
│   ├── results.py       # results.csv yönetimi
│   ├── baseline.py      # HOG + Linear SVM
│   ├── data.py          # Dataset / DataLoader (sabit listeler, augmentation yalnızca eğitimde)
│   ├── models.py        # 2 sıfırdan CNN + 7 transfer + ViT
│   ├── train.py         # ortak eğitim protokolü, dengeleme karşılaştırması
│   ├── optimize.py      # Optuna hiperparametre araması (en iyi 3 model)
│   └── seeds.py         # en iyi 3 model için seed 42/123/2026 ve özet
├── notebooks/           # Colab notebook'ları (aşama sırasıyla)
├── outputs/             # splits/, audit/, figures/, logs/, env/ (otomatik oluşur)
├── requirements.txt
└── README.md
```

## Kurulum (Google Colab)
1. Bu klasörü Google Drive'a `MyDrive/YZProje` olarak yükleyin.
2. `notebooks/01_veri_hazirlik_ve_denetim.ipynb` dosyasını Colab'da açın.
3. `DATA_URL` değişkenine VS13 indirme bağlantısını yapıştırıp tüm hücreleri çalıştırın.


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
Not: Eğitim ve deneyler Colab'da `notebooks/` sırasıyla (01 → 15) çalıştırıldı; aynı seed'lerle (42, 123, 2026) GPU kaynaklı küçük sapmalar olabilir.

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
