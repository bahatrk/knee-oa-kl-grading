"""06_ozgun_model_tasarim_gunlugu.ipynb dosyasını üretir (geliştirici aracı)."""
from pathlib import Path

import nbformat as nbf

from nb_common import DATA_CELL, SETUP_CELL

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 06 · Özgün model: KneeOrdinalNet tasarım günlüğü\n\n"
       "**Önce GPU açın** (T4). ConvNeXt-Tiny omurgası + 4 veriye özgü bileşen. Sürümler sırayla eklenir:\n\n"
       "| Sürüm | İçerik |\n|---|---|\n"
       "| Orig_v1 | çok ölçekli ortalama+maks havuzlama (B), CE+sampler |\n"
       "| Orig_v2 | v1 + eklem hattı satır dikkati (C) |\n"
       "| Orig_v3 | v2 + ordinal kayıp (D) |\n"
       "| Orig_full | v3 + girişte kontrast standardizasyonu (A) |\n\n"
       "- Hiperparametreler: ConvNeXt-Tiny'nin Optuna'daki en iyisi (adil kıyas, aynı omurga)\n"
       "- Seed 42, 4 koşu, yaklaşık 35-40 dakika; biten koşular atlanır\n"
       "- Yalnızca **validation** kullanılır, test yüklenmez"),
    code(SETUP_CELL),
    code("!pip install -q imagehash==4.3.2 tabulate==0.9.0 timm==1.0.30 optuna==5.0.0\n"
         "import torch\n"
         "assert torch.cuda.is_available(), 'GPU yok! Çalışma zamanı > T4 GPU'\n"
         "print(torch.__version__, torch.cuda.get_device_name(0))"),
    code(DATA_CELL),
    md("## Mimari özeti (blok şeması için)"),
    code("from src import original as OR\n"
         "from src.models import build_model, count_params\n"
         "m = build_model('Original', pretrained=False)\n"
         "arch = OR.architecture_summary(m)\n"
         "display(arch)\n"
         "arch.to_csv(C.OUTPUT_DIR / 'original_architecture.csv', index=False)\n"
         "print('toplam parametre:', sum(p.numel() for p in m.parameters()), '| eklenen:', sum(p.numel() for p in m.new_parameters()))\n"
         "del m"),
    md("## Tasarım günlüğü koşuları"),
    code("import pandas as pd\n"
         "print('hiperparametreler:', OR.convnext_hparams())\n"
         "for v in OR.DESIGN_LOG:\n"
         "    OR.train_variant(v, seed=42)\n"),
    md("## Sürümlerin karşılaştırması (validation)"),
    code("res = pd.read_csv(C.OUTPUT_DIR / 'results.csv')\n"
         "cols = [c for c in ['model_name','seed','val_macro_f1','val_balanced_accuracy','val_qwk','val_mae_grade','val_accuracy',\n"
         "                     'val_distant_error_rate','param_count','epochs'] if c in res.columns]\n"
         "log = res[res.model_name.isin(OR.DESIGN_LOG + ['ConvNeXtTiny__opt'])][cols]\n"
         "display(log.round(4))\n"
         "log.to_csv(C.OUTPUT_DIR / 'original_design_log.csv', index=False)"),
    md("`outputs/results.csv`, `outputs/original_design_log.csv` ve `outputs/original_architecture.csv` dosyalarını paylaşın."),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
               "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "06_ozgun_model_tasarim_gunlugu.ipynb"
nbf.write(nb, out)
print(out)
