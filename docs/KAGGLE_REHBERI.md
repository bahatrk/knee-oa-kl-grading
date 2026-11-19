# Kaggle'da çalıştırma rehberi

Neden: Colab ücretsiz GPU kotası dolduğunda Kaggle'ın haftalık 30 saatlik ücretsiz GPU kotası kullanılabilir.
Notebook'lar (03, 04, ...) hem Colab'da hem Kaggle'da aynen çalışır; kurulum hücresi platformu kendisi algılar.

## Bir kerelik hazırlık
1. kaggle.com'a kayıt ol (Google ile giriş yeterli) → Settings → **Phone Verification** (GPU ve internet için zorunlu).
2. **Veri seti 1 — `yzproje-code`:** güncelleme zip'ini (ör. `YZProje_guncelleme_4.zip`) Datasets → New Dataset ile yükle.
   Kaggle zip'i otomatik açar. Her yeni güncellemede aynı veri setine *New Version* ile yükle.
3. **Veri seti 2 — `yzproje-outputs`:** Colab'da (GPU gerekmez, CPU yeterli) `tools/export_outputs.py` ile
   `outputs_small.zip` üret, indir ve Datasets → New Dataset ile yükle (model ağırlıkları hariç, birkaç MB).

## Her çalıştırmada
1. Kaggle → Code → New Notebook → File → **Import Notebook** → proje klasöründeki `.ipynb` dosyasını seç
   (ör. `notebooks/04_hiperparametre_optimizasyonu.ipynb`).
2. Sağ panel → **Add Input** → `yzproje-code` ve `yzproje-outputs` veri setlerini ekle (yalnızca bu ikisi).
3. Sağ panel → **Session options**: Accelerator = GPU (T4 x2 veya P100), **Internet = On**.
4. Veri hücresindeki `DATA_URL` satırını doldur (Mendeley bağlantısı). Veri her oturumda yeniden indirilir (7 GB, birkaç dakika).
5. Çalıştır. Uzun işler için **Save Version → Save & Run All (Commit)** tarayıcıyı kapatsan da arka planda çalışır.

## Sonuçları alma
- Notebook'un **Output** sekmesinde `YZProje/outputs/` altındaki dosyalar görünür (`results.csv`, `optuna/*/search_summary.json`, ...).
  Gerekenleri indirip sohbete yükle.
- Bir sonraki oturumda yeni sonuçların olması için çıktıyı yeni veri seti sürümü olarak `yzproje-outputs`'a ekle
  (Output sekmesindeki *New Dataset* / Datasets sayfasındaki *New Version* ile).
