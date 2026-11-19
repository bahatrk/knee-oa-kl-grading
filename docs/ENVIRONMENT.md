# Çalıştırma ortamları

Tüm deneyler Google Colab'da çalıştırıldı (Linux 6.6.159+, glibc 2.39, Python 3.13.18). Aşağıdaki sürümler, her oturumda
kaydedilen `pip freeze` / `env_info.json` dosyalarından alınmıştır.

| Oturum | Donanım | torch | torchvision | Not |
|---|---|---|---|---|
| Veri denetimi, HOG+SVM (3 Kas 2026) | CPU, 2 vCPU | 2.11.0+cpu | 0.26.0+cpu | `env_info.json` |
| Model eğitimi, Optuna, ablasyon, değerlendirme | NVIDIA T4 (16 GB) | 2.11.0+cu130 | 0.26.0+cu130 | `pip_freeze_train.txt` |
| Ek CPU gecikme ölçümü, fp16 paketi (19 Kas 2026) | CPU, tek iş parçacığı | 2.14.1+cpu | 0.29.1+cpu | `pip_freeze_final.txt` |

Üç oturumda ortak sürümler: numpy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, scikit-image 0.25.2, scipy 1.16.3,
matplotlib 3.10.0, Pillow 11.3.0, joblib 1.6.0, timm 1.0.30, ImageHash 4.3.2.
Colab'da hazır gelmeyen paketler (optuna 5.0.0, grad-cam 1.5.7, captum 0.9.0) ilgili notebook'larda `pip install` ile sabit sürümden kurulur.

Gecikme tabloları farklı oturumlarda ölçüldüğü için yalnızca aynı tablo içinde karşılaştırılmalıdır.
