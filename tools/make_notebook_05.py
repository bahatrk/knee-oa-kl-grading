"""05_cok_seed.ipynb dosyasını üretir (geliştirici aracı)."""
from pathlib import Path

import nbformat as nbf

from nb_common import DATA_CELL, SETUP_CELL

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 05 · Çoklu seed: en iyi 3 modelin seed 123 ve 2026 tekrarları\n\n"
       "**Önce GPU açın** (Colab: T4 GPU). Optuna'da bulunan en iyi parametrelerle ConvNeXt-Tiny, DeiT-Small ve ResNet50 "
       "iki yeni seed ile (123, 2026) tam eğitim kümesinde eğitilir; seed 42 zaten vardır.\n\n"
       "- Veri bölmesi sabit; yalnızca ağırlık başlatma, veri sırası ve augmentation rastgeleliği değişir\n"
       "- 6 koşu, toplam yaklaşık 50 dakika. Biten koşular atlanır, oturum kopsa aynı notebook'u yeniden çalıştırın\n"
       "- Özet yalnızca **validation** kullanır. Test kümesi yüklenmez\n"
       "- \"En iyi mevcut model\" kuralı: 3 seed ortalama validation macro F1'i en yüksek model"),
    code(SETUP_CELL),
    code("!pip install -q imagehash==4.3.2 tabulate==0.9.0 timm==1.0.30 optuna==5.0.0\n"
         "import torch\n"
         "assert torch.cuda.is_available(), 'GPU yok! Colab: Çalışma zamanı > Çalışma zamanı türünü değiştir > T4 GPU | Kaggle: sağ panel Session options > Accelerator > GPU'\n"
         "print(torch.__version__, torch.cuda.get_device_name(0))"),
    code(DATA_CELL),
    md("## Seed 123 ve 2026 koşuları"),
    code("from src import optimize as O, seeds as SD\n"
         "import pandas as pd\n"
         "KEYS = O.top_candidates(3); print('Modeller:', KEYS)\n"
         "SD.run_seeds(KEYS, seeds=(123, 2026))"),
    md("## Özet (validation, 3 seed ortalaması ± std)"),
    code("import json\n"
         "df = SD.summarize(KEYS)\n"
         "show = df[['model','n_seeds','val_macro_f1_mean','val_macro_f1_std','val_balanced_accuracy_mean','val_qwk_mean',\n"
         "           'val_qwk_std','val_mae_grade_mean','val_accuracy_mean']]\n"
         "display(show.round(4))\n"
         "print(json.load(open(C.OUTPUT_DIR / 'best_existing_model.json'))['best_existing_model'])"),
    md("`outputs/results.csv`, `outputs/seed_summary.csv` ve `outputs/best_existing_model.json` dosyalarını paylaşın."),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
               "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "05_cok_seed.ipynb"
nbf.write(nb, out)
print(out)
