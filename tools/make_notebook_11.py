"""11_nihai_test_degerlendirme.ipynb (geliştirici aracı)."""
from pathlib import Path

import nbformat as nbf

from nb_common import DATA_CELL, SETUP_CELL

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
PIP = ("!pip install -q imagehash==4.3.2 tabulate==0.9.0 timm==1.0.30\n"
       "import torch\nprint(torch.__version__, torch.cuda.is_available())")
META = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
ROOT = Path(__file__).resolve().parents[1] / "notebooks"

cells = [
    md("# 11 · Nihai test değerlendirmesi\n\n"
       "**Bu not defteri test kümesini kullanır. Tüm seçimler (en iyi model, hibrit, ablasyon hedefi) bundan önce "
       "doğrulama kümesinde yapılıp dondurulmuştur.** Her model test kümesinde yalnızca bir kez tahmin eder; "
       "olasılıklar saklanır, sonraki tüm analizler bunlardan hesaplanır.\n\n"
       "- Önce GPU açın (T4). Süre yaklaşık 20-30 dk (test: 1.656 görüntü × ~40 koşu)\n"
       "- Oturum kopsa yeniden çalıştırın; tahmini yapılmış koşular atlanır\n"
       "- Çıktılar: `outputs/final/`"),
    code(SETUP_CELL), code(PIP), code(DATA_CELL),
    md("## 1 · Seçimleri dondur ve kaydet"),
    code("from src import final_eval as F\nimport json\nprint(json.dumps(F.freeze(), indent=2, ensure_ascii=False))"),
    md("## 2 · Test olasılıkları (her koşu bir kez)"),
    code("F.predict_all(include_appendix=True)"),
    md("## 3 · Metrikler, güven aralıkları, kalibrasyon, eşleştirilmiş testler"),
    code("info = F.analyze(include_appendix=True)\nprint(info)"),
    code("import pandas as pd\n"
         "pd.set_option('display.width', 250, 'display.max_columns', 40)\n"
         "T = pd.read_csv(F.OUT / 'main_table_test_raw.csv')\n"
         "cols = ['model','n_seeds'] + [f'{k}_mean' for k in ['accuracy','balanced_accuracy','macro_f1','qwk','mcc','roc_auc_macro','mae_grade','ece15']]\n"
         "T[cols].round(4)"),
    code("print(open(F.OUT / 'mcnemar_test.json').read())\n"
         "pd.read_csv(F.OUT / 'paired_bootstrap.csv').round(4)"),
    md("## 4 · CPU gecikmesi (batch 1)"),
    code("L = F.cpu_latency(); L.round(2)"),
    md("## 5 · Çıktıları indir\nAşağıdaki hücre `outputs/final` içeriğini tek zip yapar; Drive'a kopyalanır."),
    code("import shutil\n"
         "shutil.make_archive('/content/final_outputs', 'zip', str(F.OUT))\n"
         "shutil.copy('/content/final_outputs.zip', str(F.OUT.parent / 'final_outputs.zip'))\n"
         "print('hazır: outputs/final_outputs.zip')"),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = META
nbf.write(nb, ROOT / "11_nihai_test_degerlendirme.ipynb")
print("ok")
