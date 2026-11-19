"""04_hiperparametre_optimizasyonu.ipynb dosyasını üretir (geliştirici aracı)."""
from pathlib import Path

import nbformat as nbf

from nb_common import DATA_CELL, SETUP_CELL

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 04 · Hiperparametre optimizasyonu (Optuna) — en güçlü 3 model\n\n"
       "**Önce GPU açın.** Colab: `Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU`. Kaggle: sağ panel → *Session options* → *Accelerator* → GPU ve *Internet: On*. Bu notebook her iki platformda da çalışır.\n\n"
       "- Adaylar: `results.csv`'de validation macro F1'e göre en iyi 3 model (otomatik seçilir)\n"
       "- Her model için **25 deneme**; eğitim kümesinin %40'lık sabit-seed alt kümesi ve kısaltılmış epoch bütçesi; MedianPruner ile budama\n"
       "- Amaç: validation macro F1. **Test kümesi yüklenmez.**\n"
       "- Bittikten sonra en iyi parametrelerle model **tam eğitim kümesinde** yeniden eğitilir (`<model>__opt`)\n"
       "- Oturum kopsa bile aynı notebook'u yeniden çalıştırmanız yeterli: biten denemeler Drive'dan okunur, kalınan yerden devam edilir\n\n"
       "⏱ Model başına yaklaşık 40–60 dk (toplam 2–3 saat). İstediğiniz modeli ayrı oturumda da çalıştırabilirsiniz."),
    code(SETUP_CELL),
    code("!pip install -q imagehash==4.3.2 tabulate==0.9.0 timm==1.0.30 optuna==5.0.0\n"
         "import torch\n"
         "assert torch.cuda.is_available(), 'GPU yok! Colab: Çalışma zamanı > Çalışma zamanı türünü değiştir > T4 GPU | Kaggle: sağ panel Session options > Accelerator > GPU'\n"
         "print(torch.__version__, torch.cuda.get_device_name(0))"),
    code(DATA_CELL),
    md("## Adayların seçimi"),
    code("from src import optimize as O\n"
         "import pandas as pd\n"
         "CANDIDATES = O.top_candidates(3)\n"
         "print('Optuna adayları (val macro F1 sırasıyla):', CANDIDATES)\n"
         "r = pd.read_csv(C.OUTPUT_DIR / 'results.csv')\n"
         "display(r[r.model_name.isin(CANDIDATES)][['model_name','val_macro_f1','val_balanced_accuracy','val_qwk']].round(4))"),
    md("## Arama + yeniden eğitim\n\nHer aday için sırayla çalışır."),
    code("from IPython.display import Image, display\n"
         "for key in CANDIDATES:\n"
         "    O.run_search(key, n_trials=25, subset_fraction=0.4)\n"
         "    O.retrain_best(key)\n"
         "    display(Image(filename=str(C.FIG_DIR / f'optuna_{key}.png'), width=900))"),
    md("## Özet: ayarsız ve Optuna ile ayarlı (validation)"),
    code("import json\n"
         "r = pd.read_csv(C.OUTPUT_DIR / 'results.csv')\n"
         "rows = []\n"
         "for k in CANDIDATES:\n"
         "    a = r[r.model_name == k].iloc[0]; b = r[r.model_name == f'{k}__opt'].iloc[0]\n"
         "    s = json.load(open(C.OUTPUT_DIR / 'optuna' / k / 'search_summary.json'))\n"
         "    rows.append({'model': k, 'F1_ayarsiz': a.val_macro_f1, 'F1_optuna': b.val_macro_f1,\n"
         "                 'fark': b.val_macro_f1 - a.val_macro_f1, 'QWK_ayarsiz': a.val_qwk, 'QWK_optuna': b.val_qwk,\n"
         "                 'denemeler': s['trial_states']})\n"
         "display(pd.DataFrame(rows).round(4))\n"
         "for k in CANDIDATES:\n"
         "    print(k, json.load(open(C.OUTPUT_DIR / 'optuna' / k / 'search_summary.json'))['best_params'])"),
    md("`outputs/results.csv` ve `outputs/optuna/` klasöründeki `search_summary.json` dosyalarını paylaşın."),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
               "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "04_hiperparametre_optimizasyonu.ipynb"
nbf.write(nb, out)
print(out)
