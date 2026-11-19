"""07_ozgun_model_optuna.ipynb ve 08_ozgun_model_ablasyon.ipynb dosyalarını üretir (geliştirici aracı)."""
from pathlib import Path

import nbformat as nbf

from nb_common import DATA_CELL, SETUP_CELL

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
PIP = ("!pip install -q imagehash==4.3.2 tabulate==0.9.0 timm==1.0.30 optuna==5.0.0\n"
       "import torch\n"
       "assert torch.cuda.is_available(), 'GPU yok! Çalışma zamanı > T4 GPU'\n"
       "print(torch.__version__, torch.cuda.get_device_name(0))")
META = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
ROOT = Path(__file__).resolve().parents[1] / "notebooks"


def write(cells, name):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = META
    nbf.write(nb, ROOT / name)
    print(ROOT / name)


write([
    md("# 07 · Özgün model: Optuna hiperparametre araması\n\n"
       "**Önce GPU açın** (T4). Özgün model (Orig_full: tüm bileşenler) için 25 deneme.\n\n"
       "- Arama uzayı: diğer modellerle aynı + özgün kaybın `ordinal_sigma` ve `ordinal_lambda` parametreleri "
       "(ordinal kayıp etiket yumuşatma kullanmadığından `label_smoothing` aranmaz)\n"
       "- İlk deneme: aynı omurganın (ConvNeXt-Tiny) en iyi parametreleri (adil başlangıç noktası)\n"
       "- Aynı bütçe: eğitim kümesinin %40'ı, 3+10 epoch, MedianPruner; amaç validation macro F1\n"
       "- Süre yaklaşık 1-1,5 saat. Oturum kopsa aynı notebook'u yeniden çalıştırın, kaldığı yerden devam eder\n"
       "- Sonunda en iyi parametrelerle tam eğitim kümesinde seed 42 ile yeniden eğitilir (`Orig_full__opt`)\n"
       "- Test kümesi yüklenmez"),
    code(SETUP_CELL), code(PIP), code(DATA_CELL),
    md("## Arama"),
    code("from src import optimize as O, original as OR\n"
         "warm = {k: v for k, v in OR.convnext_hparams().items() if k != 'label_smoothing'}\n"
         "warm.update(ordinal_sigma=0.75, ordinal_lambda=0.3)\n"
         "spec = OR.VARIANTS['Orig_full']\n"
         "study = O.run_search('Original', n_trials=25, subset_fraction=0.4, loss=spec['loss'],\n"
         "                     model_kwargs=spec['model'], warm_start=warm)\n"
         "print(study.best_params, study.best_value)"),
    md("## En iyi parametrelerle tam eğitim (seed 42)"),
    code("r = O.retrain_best('Original', seed=42, name='Orig_full__opt', loss=spec['loss'], model_kwargs=spec['model'])\n"
         "r['val_metrics']"),
    md("`outputs/results.csv`, `outputs/optuna/Original/search_summary.json` ve `outputs/optuna/Original/trials.csv` "
       "dosyalarını paylaşın."),
], "07_ozgun_model_optuna.ipynb")

write([
    md("# 08 · Özgün model: ablasyonlar (3 seed) ve McNemar\n\n"
       "**Önce GPU açın** (T4). 5 varyant × 3 seed (42, 123, 2026), hepsi özgün modelin Optuna hiperparametreleriyle:\n\n"
       "| Varyant | Ne çıkarıldı |\n|---|---|\n"
       "| full | hiçbir şey |\n| - çok ölçekli birleştirme | yalnızca son aşama özelliği |\n"
       "| - satır dikkati | RowAttention yok |\n| - ordinal kayıp | yerine CE+sampler |\n"
       "| - girişte kontrast standardizasyonu | per-image standardizasyon yok |\n\n"
       "- 15 koşu (`Orig_full__opt` seed 42 zaten var), toplam 2-2,5 saat. Biten koşular atlanır, "
       "oturum kopsa aynı notebook'u yeniden çalıştırın\n"
       "- Yalnızca **validation**. McNemar: tam model ile en iyi ablasyon, seed başına"),
    code(SETUP_CELL), code(PIP), code(DATA_CELL),
    md("## Koşular"),
    code("from src import ablation as A\n"
         "A.run_ablations(seeds=(42, 123, 2026))"),
    md("## Özet"),
    code("import json\n"
         "df = A.summarize(seeds=(42, 123, 2026))\n"
         "show = df[['variant','n_seeds','param_count','val_macro_f1_mean','val_macro_f1_std','delta_f1_vs_full',\n"
         "           'val_qwk_mean','val_mae_grade_mean','val_distant_error_rate_mean','val_accuracy_mean']]\n"
         "display(show.round(4))\n"
         "print(json.dumps(json.load(open(C.OUTPUT_DIR / 'ablation_mcnemar.json')), indent=1, ensure_ascii=False))"),
    md("`outputs/results.csv`, `outputs/ablation_summary.csv` ve `outputs/ablation_mcnemar.json` dosyalarını paylaşın."),
], "08_ozgun_model_ablasyon.ipynb")


write([
    md("# 09 · Kontrol deneyi ve tüm McNemar testleri\n\n"
       "**Önce GPU açın** (T4). Soru: özgün modelin ConvNeXt'e göre kazancı tasarımdan mı, yoksa yeni hiperparametrelerden mi geliyor?\n\n"
       "- Kontrol: hiçbir özgün bileşeni olmayan ConvNeXt-Tiny, **aynı** Optuna hiperparametreleriyle (3 seed, yaklaşık 35 dakika)\n"
       "- Ardından tam model ile tüm ablasyonlar ve kontrol arasında seed başına McNemar testleri hesaplanır (yeni eğitim gerektirmez)\n"
       "- Yalnızca **validation**"),
    code(SETUP_CELL), code(PIP), code(DATA_CELL),
    code("from src import ablation as A\n"
         "A.run_control(seeds=(42, 123, 2026))"),
    code("import pandas as pd, json\n"
         "mc = A.control_summary(seeds=(42, 123, 2026))\n"
         "display(mc.round(4))\n"
         "print(json.dumps(json.load(open(C.OUTPUT_DIR / 'control_summary.json')), indent=1))"),
    md("`outputs/results.csv`, `outputs/mcnemar_all.csv` ve `outputs/control_summary.json` dosyalarını paylaşın."),
], "09_kontrol_deneyi.ipynb")

write([
    md("# 10 · Hibrit model: füzyon yöntemleri ve üye ablasyonu\n\n"
       "**GPU gerekmez** (kayıtlı doğrulama olasılıkları birleştirilir, birkaç dakika). Üyeler: Orig (özgün), DeiT, ConvNeXt, ResNet50 "
       "(`__opt` modelleri, her seed için kendi seed'iyle). Ağırlıklar ve meta-öğrenici macro F1 ölçütüyle seçilir (proje kuralı).\n\n"
       "- 4 füzyon yöntemi: `mean` (eşit ağırlık), `weighted` (ağırlıklar doğrulamada), `stack` (lojistik regresyon meta-öğrenici), "
       "`mean_ord` (ortalama + sıralı karar kuralı)\n"
       "- 11 üye kümesi (2, 3 ve 4 üyeli tüm kombinasyonlar) = üye ablasyonu\n"
       "- Öğrenilen yöntemlerde doğrulama sonuçları 5 katlı çapraz doğrulama (katman dışı) ile hesaplanır, iyimser olmaz\n"
       "- Yalnızca **validation**; test kullanılmaz"),
    code(SETUP_CELL),
    code("from src import hybrid as H\n"
         "import pandas as pd\n"
         "summ = H.run_all(seeds=(42, 123, 2026))\n"
         "show = ['method','members','macro_f1_mean','macro_f1_std','qwk_mean','mae_grade_mean','distant_error_rate_mean','accuracy_mean']\n"
         "display(summ[show].head(15).round(4))"),
    md("## Üye ablasyonu (yalnızca parametresiz `mean` yöntemi, yöntem etkisi karışmasın diye)"),
    code("display(summ[summ.method == 'mean'].sort_values('macro_f1_mean', ascending=False)[show].round(4))\n"
         "print(H.choose_final(summ))"),
    md("`outputs/hybrid_val_summary.csv`, `outputs/hybrid_val_runs.csv` ve `outputs/hybrid_choice.json` dosyalarını paylaşın."),
], "10_hibrit_model.ipynb")
