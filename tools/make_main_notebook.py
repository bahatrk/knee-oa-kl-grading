"""Tek, baştan sona çalışan teslim notebook'unu üretir: notebooks/220709071_BahaTurk_YZProje.ipynb
  python tools/make_main_notebook.py
Notebook temiz (çıktısız) üretilir; teslimden önce Colab'da baştan sona çalıştırılıp çıktılarıyla kaydedilir."""
import json
from pathlib import Path

NAME = "220709071_BahaTurk_YZProje"
cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)})


md("""# VS13 · Diz röntgeninde KL derecelendirme: tek notebook
MBM6 453 Yapay Zekâya Giriş · Baha Türk · 220709071

Bu notebook projeyi baştan sona çalıştırır: veri indirme ve denetim, klasik taban, kayıtlı sonuçlardan ana tablolar ve şekiller, tek görüntü tahmini.
Uzun süren eğitim aşamaları `RETRAIN = True` ile açılır (varsayılan kapalıdır; kayıtlı sonuçlar `results.csv` ve `outputs/` içindedir).
Test kümesi yalnızca nihai değerlendirmede, bir kez kullanılır; model ve hiperparametre seçimi yalnızca doğrulama makro F1 ile yapılmıştır.
Araştırma amaçlıdır, klinik karar aracı değildir.""")

md("## 0. Ayarlar ve ortam")
code("""RETRAIN = False   # True: tüm eğitim, Optuna, ablasyon, hibrit ve nihai test aşamaları sıfırdan çalışır (GPU gerekir, saatler sürer)

import os, sys, subprocess
os.environ['KNEE_LANG'] = 'en'   # rapor şekilleri İngilizce: outputs/figures_en
IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT = next((p for p in ('/content/drive/MyDrive/YZProje', '/content/drive/MyDrive/YZProje/YZProje')
                    if os.path.isfile(p + '/src/config.py')), None)
    assert PROJECT, "Drive'da YZProje/src/config.py bulunamadı: teslim klasörünü MyDrive/YZProje olarak yerleştirin."
    os.environ.setdefault('KNEE_DATA_ROOT', '/content/data')
else:
    PROJECT = os.getcwd()
    assert os.path.isfile(PROJECT + '/src/config.py'), 'Notebook proje klasöründen açılmalı.'
    os.environ.setdefault('KNEE_DATA_ROOT', PROJECT + '/data')
os.chdir(PROJECT); sys.path.insert(0, PROJECT)
os.environ['KNEE_OUTPUT_DIR'] = PROJECT + '/outputs'
print('Proje klasörü:', PROJECT, '| RETRAIN =', RETRAIN)""")
code("""# Colab'da hazır gelmeyen paketler (torch ve torchvision Colab'dakiyle kullanılır; tüm sürümler requirements.txt'te sabittir)
!pip install -q imagehash==4.3.2 tabulate==0.9.0 timm==1.0.30 optuna==5.0.0 grad-cam==1.5.7 captum==0.9.0 scikit-image==0.25.2
import torch
print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'yok (CPU)')
if RETRAIN:
    assert torch.cuda.is_available(), 'RETRAIN için GPU gerekir (Colab: Çalışma zamanı > T4 GPU).'
from src.utils import env_info, save_json, set_seed
from src import config as C
set_seed(C.SEED)
save_json(env_info(), C.OUTPUT_DIR / 'env' / 'env_info.json')""")

md("## 1. Veri: indirme, SHA256 doğrulaması, resmi bölme\n`ensure_data()` arşivi README'deki bağlantıdan indirir, SHA256 özetini doğrular ve yalnızca `kneeKL224` klasörünü açar. Görüntüler zaten açıksa hiçbir şey yapmaz.")
code("""from src import data_prep
root = data_prep.ensure_data()
print('Veri hazır:', root)
if not (C.SPLITS_DIR / 'split_info.json').exists():      # dosya listeleri yoksa resmi bölmeden üret
    data_prep.main(['--skip-extract'])
import json
print(json.dumps(json.load(open(C.SPLITS_DIR / 'split_info.json')), indent=1)[:600])""")

md("## 2. Veri denetimi ve yakın kopya doğrulaması")
code("""from src import audit, dup_verify
from IPython.display import Markdown, Image, display
audit.main([])
display(Markdown(open(C.AUDIT_DIR / 'audit_summary.md', encoding='utf-8').read()))""")
code("""dup_verify.main([])
for f in sorted(C.FIG_DIR.glob('audit_*.png')):
    print(f.name); display(Image(filename=str(f), width=800))""")

md("## 3. Klasik taban: HOG + doğrusal SVM (seçim yalnızca doğrulama kümesiyle)")
code("""from src import baseline
import pandas as pd
if RETRAIN:
    baseline.main([])
    g = pd.read_csv(C.OUTPUT_DIR / 'models/baseline_hog_svm/grid_search_val.csv')
    display(g[['ppc','C','class_weight','n_features','val_macro_f1','val_balanced_accuracy','val_qwk']].sort_values('val_macro_f1', ascending=False).head(5).round(4))
else:   # kayıtlı sonuç: results.csv içindeki HOG+SVM satırı ve tek seferlik test özeti
    r0 = pd.read_csv(PROJECT + '/results.csv' if os.path.isfile(PROJECT + '/results.csv') else C.OUTPUT_DIR / 'results.csv')
    cols = [c for c in ['model_name','param_count','val_macro_f1','val_balanced_accuracy','val_qwk','test_macro_f1','test_qwk'] if c in r0.columns]
    display(r0[r0.model_name.str.contains('HOG')][cols].round(4))""")

md("""## 4. Eğitim aşamaları (yalnızca `RETRAIN = True`)
Sıra: ortak kayıp seçimi ve 10 ağın eğitimi, Optuna (en iyi 3 ağ), 3 seed, özgün model (tasarım günlüğü, Optuna, ablasyonlar, kontrol), hibrit füzyon, nihai test değerlendirmesi, XAI ve zorlu görevler.
Aşamalar `notebooks/asamalar/` içindeki notebook'larla aynı çağrıları yapar. Colab oturum sınırı nedeniyle uzun aşamalarda o notebook'ları tek tek çalıştırmak daha güvenlidir.""")
code("""if RETRAIN:
    from src.train import balancing_comparison, get_chosen_loss, train_model
    from src.models import ZOO
    if not (C.OUTPUT_DIR / 'balancing_choice.json').exists():
        balancing_comparison('EfficientNetB0')
    LOSS = get_chosen_loss(); print('Ortak kayıp:', LOSS)
    ORDER = ['EfficientNetB0', 'MobileNetV3L', 'SimpleCNN', 'MiniResNet', 'ResNet50', 'DenseNet121',
             'ConvNeXtTiny', 'DeiTSmall', 'Xception', 'InceptionV3']
    assert set(ORDER) == set(ZOO) - {'Original'}
    for k in ORDER:
        train_model(k, 42, loss=LOSS)
else:
    print('RETRAIN kapalı: bu aşama atlandı; sonuçlar results.csv ve outputs/ içinden okunur.')""")
code("""if RETRAIN:
    from src import optimize as O, seeds as SD
    CANDIDATES = O.top_candidates(3); print('Optuna adayları:', CANDIDATES)
    for key in CANDIDATES:
        O.run_search(key, n_trials=25, subset_fraction=0.4)
        O.retrain_best(key)
    SD.run_seeds(CANDIDATES, seeds=(123, 2026)); SD.summarize(CANDIDATES)""")
code("""if RETRAIN:
    from src import original as OR, ablation as A
    for v in OR.DESIGN_LOG:
        OR.train_variant(v, seed=42)
    warm = {k: v for k, v in OR.convnext_hparams().items() if k != 'label_smoothing'}
    warm.update(ordinal_sigma=0.75, ordinal_lambda=0.3)
    spec = OR.VARIANTS['Orig_full']
    O.run_search('Original', n_trials=25, subset_fraction=0.4, loss=spec['loss'], model_kwargs=spec['model'], warm_start=warm)
    O.retrain_best('Original', seed=42, name='Orig_full__opt', loss=spec['loss'], model_kwargs=spec['model'])
    A.run_ablations(seeds=(42, 123, 2026)); A.summarize(seeds=(42, 123, 2026))
    A.run_control(seeds=(42, 123, 2026)); A.control_summary(seeds=(42, 123, 2026))""")
code("""if RETRAIN:
    from src import hybrid as H, final_eval as F
    summ = H.run_all(seeds=(42, 123, 2026)); print(H.choose_final(summ))
    print(json.dumps(F.freeze(), indent=2, ensure_ascii=False))      # test öncesi seçimlerin dondurulması
    F.predict_all(include_appendix=True)                             # test kümesi: TEK SEFER
    F.analyze(include_appendix=True)
    F.cpu_latency(); F.cpu_latency_extra()
    !python -m src.baseline --final-test
    F.baseline_summary()""")
code("""if RETRAIN:
    from src import xai, challenges as CH, export, figures_en
    xai.run(); CH.robustness(); CH.selective(); CH.captions()
    print(figures_en.run_all())
    print(export.export()); print(export.export_fp16())""")
code("""if RETRAIN and not (C.OUTPUT_DIR / 'results_final.csv').exists():
    # results.csv'yi tamamlar: hibrit satırları ve CPU gecikme sütunları (yeni eğitim yapılmaz)
    !python tools/extend_results.py --results outputs/results.csv --hybrid-val outputs/hybrid/hybrid_val_runs.csv \\
        --test-runs outputs/final/test_runs_all.csv --cpu-a outputs/final/cpu_latency.csv \\
        --cpu-b outputs/final/cpu_latency_extra.csv --out outputs/results_final.csv""")

md("## 5. Sonuçlar (kayıtlı çıktılardan)\n`results.csv`: model ve seed başına bir satır, doğrulama ve test metrikleri, parametre sayısı, eğitim süresi ve gecikme.")
code("""res_path = next(p for p in (str(C.OUTPUT_DIR / 'results_final.csv'), PROJECT + '/results.csv', str(C.OUTPUT_DIR / 'results.csv')) if os.path.isfile(p))
res = pd.read_csv(res_path)
print(res_path, '| satır:', len(res), '| model:', res.model_name.nunique())
need = ['model_name','model_family','seed','input_size','param_count','epochs','batch_size','optimizer','learning_rate','train_time_s']
print('eksik zorunlu sütunlar:', [c for c in need if c not in res.columns])
display(res.head(3))""")
code("""from pathlib import Path
pd.set_option('display.width', 250, 'display.max_columns', 40)
FINAL = C.OUTPUT_DIR / 'final'
def show(path, n=None, cols=None):
    p = Path(path)
    if not p.exists():
        print('bulunamadı:', p); return
    d = pd.read_csv(p)
    display((d[cols] if cols else d).round(4) if n is None else (d[cols] if cols else d).head(n).round(4))
print('Ana karşılaştırma (test kümesi, seed ortalaması)'); show(FINAL / 'main_table_test_raw.csv')""")
code("""print('Test metrikleri için %95 güven aralıkları (bootstrap, 1000)'); show(FINAL / 'bootstrap_ci.csv')
print('Kalibrasyon (ECE 15 kutu, Brier, sıcaklık ölçekleme)'); show(FINAL / 'calibration.csv')
print('Eşleştirilmiş bootstrap farkları'); show(FINAL / 'paired_bootstrap.csv')""")
code("""print('Ablasyon (3 seed)'); show(C.OUTPUT_DIR / 'ablation_summary.csv')
print('Sınıf bazında test metrikleri'); show(FINAL / 'per_class_test.csv', n=20)
print('CPU gecikmesi'); show(FINAL / 'cpu_latency_extra.csv')""")

md("## 6. Şekiller (`outputs/figures_en`)")
code("""figs = sorted(C.FIG_DIR.glob('*.png'))
print(len(figs), 'şekil')
for p in figs:
    print(p.name); display(Image(filename=str(p), width=700))""")

md("## 7. Tek görüntü tahmini (CPU, eğitim verisi gerekmez)\n`predict.py` yalnızca `deploy/model_fp16.pt` ve `deploy/meta.json` dosyalarını kullanır.")
code("""from src.data import read_split
r = read_split('test').iloc[0]
img = data_prep.get_dataset_root() / r.relpath
print('görüntü:', img.name, '| gerçek derece:', r.label)
!python predict.py "{img}" """)
code("""# Kendi görüntünüz için:  !python predict.py /yol/goruntu.png        (isteğe bağlı: --temperature)""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / f"{NAME}.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("yazıldı:", out, len(cells), "hücre")
