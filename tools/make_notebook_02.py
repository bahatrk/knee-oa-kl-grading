"""02_kopya_dogrulama_ve_baseline.ipynb dosyasını üretir (geliştirici aracı)."""
from pathlib import Path

import nbformat as nbf

from nb_common import DATA_CELL, SETUP_CELL

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 02 · Yakın-kopya doğrulama ve klasik baseline (HOG + Linear SVM)\n\n"
       "1. pHash adaylarını SSIM ve korelasyonla piksel düzeyinde doğrular\n"
       "2. HOG + doğrusal SVM baseline modelini eğitir; hiperparametreler **yalnızca doğrulama** kümesinde macro F1 ile seçilir\n\n"
       "Test kümesine bu notebook'ta dokunulmaz. GPU gerekmez (yaklaşık 15–25 dk)."),
    code(SETUP_CELL),
    code("!pip install -q imagehash==4.3.2 tabulate==0.9.0"),
    code(DATA_CELL),
    md("## 1. pHash adaylarının doğrulanması"),
    code("from src import dup_verify\ndup_verify.main([])\n"
         "from IPython.display import Image, display\n"
         "display(Image(filename=str(C.FIG_DIR / 'audit_duplicate_verification.png'), width=600))"),
    md("## 2. HOG + Linear SVM"),
    code("from src import baseline\nbaseline.main([])"),
    code("import pandas as pd\n"
         "g = pd.read_csv(C.OUTPUT_DIR / 'models/baseline_hog_svm/grid_search_val.csv')\n"
         "cols = ['ppc','C','class_weight','n_features','val_macro_f1','val_balanced_accuracy','val_qwk','val_accuracy']\n"
         "display(g[cols].sort_values('val_macro_f1', ascending=False).round(4))\n"
         "display(pd.read_csv(C.OUTPUT_DIR / 'models/baseline_hog_svm/val_per_class.csv').round(3))\n"
         "display(pd.read_csv(C.OUTPUT_DIR / 'results.csv').T)"),
    md("Çıktıları (`outputs/audit/duplicate_verification.json`, `outputs/models/baseline_hog_svm/grid_search_val.csv`, "
       "`outputs/results.csv`) değerlendirme için paylaşın."),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "colab": {"provenance": []}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "02_kopya_dogrulama_ve_baseline.ipynb"
nbf.write(nb, out)
print(out)
