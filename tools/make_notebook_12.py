"""12_hog_svm_test_ve_sekiller.ipynb ve 13_xai_ve_zorlu_gorevler.ipynb (geliştirici aracı)."""
from pathlib import Path

import nbformat as nbf

from nb_common import DATA_CELL, SETUP_CELL

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
PIP = ("!pip install -q imagehash==4.3.2 tabulate==0.9.0 timm==1.0.30 scikit-image\n"
       "import torch\nprint(torch.__version__, torch.cuda.is_available())")
META = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
ROOT = Path(__file__).resolve().parents[1] / "notebooks"


def write(cells, name):
    nb = nbf.v4.new_notebook(cells=cells); nb.metadata = META
    nbf.write(nb, ROOT / name); print(name)


ZIP = ("import shutil\n"
       "from src import config as C\n"
       "shutil.make_archive('/content/{n}', 'zip', str({d}))\n"
       "shutil.copy('/content/{n}.zip', str(C.OUTPUT_DIR / '{n}.zip'))\n"
       "print('hazır: outputs/{n}.zip')")

write([
    md("# 12 · HOG+SVM test, rapor şekilleri, dağıtım paketi\n\n"
       "Notebook 11 çalışmış olmalı (`outputs/final/` dolu). Önce GPU açın (HOG özniteliği CPU'da çıkar, yaklaşık 2-3 dk).\n\n"
       "1. HOG+SVM taban çizgisi test kümesinde **tek sefer** değerlendirilir\n"
       "2. Rapor şekilleri (Şekil 1-7): akış şeması, mimari, eğitim eğrileri, karışıklık matrisleri, ROC/PR, güvenilirlik, verimlilik\n"
       "3. Dağıtım paketi (`deploy/`): özgün model + `predict.py` denemesi"),
    code(SETUP_CELL), code(PIP), code(DATA_CELL),
    md("## 1 · HOG+SVM test (tek sefer)"),
    code("!python -m src.baseline --final-test\n"
         "from src import final_eval as F\n"
         "F.baseline_summary().T"),
    md("## 2 · Şekiller"),
    code("from src import figures_final as FF\nprint(FF.run_all())"),
    code("from IPython.display import Image, display\n"
         "from src import config as C\n"
         "for f in sorted(C.FIG_DIR.glob('fig0[1-7]*.png')):\n"
         "    print(f.name); display(Image(str(f), width=800))"),
    md("## 3 · Dağıtım paketi ve predict.py denemesi"),
    code("from src import export\nprint(export.export())"),
    code("from src.data import read_split\n"
         "from src.data_prep import get_dataset_root\n"
         "r = read_split('val').iloc[0]\n"
         "!python predict.py \"{get_dataset_root() / r.relpath}\"\n"
         "print('gerçek derece:', r.label)"),
    md("## 4 · Çıktıları kaydet"),
    code(ZIP.format(n="sekiller_ve_deploy", d="C.FIG_DIR")),
], "12_hog_svm_test_ve_sekiller.ipynb")

write([
    md("# 13 · Açıklanabilirlik (XAI) ve zorlu görevler\n\n"
       "Önce GPU açın (T4). Notebook 11 çalışmış olmalı. Süre yaklaşık 25-40 dk.\n\n"
       "- **XAI:** Grad-CAM + Integrated Gradients (iki model), DeiT için dikkat yayılımı; 20 doğru + 12 yanlış örnek; "
       "rastgeleleştirme testi; silme eğrisi\n"
       "- **Zorlu görev 1:** bozulma dayanıklılığı (4 bozulma × 5 şiddet; özgün, DeiT, kontrol)\n"
       "- **Zorlu görev 2:** seçici tahmin (risk-kapsam; eşikler yalnızca doğrulamadan)"),
    code(SETUP_CELL), code(PIP), code(DATA_CELL),
    md("## 1 · XAI"),
    code("from src import xai\nprint(xai.run())"),
    code("import pandas as pd\n"
         "pd.set_option('display.width', 220)\n"
         "print(pd.read_csv(xai.XAI / 'deletion_auc.csv').round(4))\n"
         "print(pd.read_csv(xai.XAI / 'randomization_test.csv').round(3))"),
    md("## 2 · Bozulma dayanıklılığı"),
    code("from src import challenges as H\nH.robustness().round(4)"),
    md("## 3 · Seçici tahmin"),
    code("df = H.selective(); H.captions()\n"
         "df[df.güven == 'maks. olasılık'][['model','olasılık','hedef_kapsam(val)','test_kapsam','macro_f1','qwk','mae_grade','AURC']].round(4)"),
    code("from IPython.display import Image, display\nfrom src import config as C\n"
         "for f in ['fig10_silme_egrisi','fig11_rastgelelestirme','fig12_bozulma_dayaniklilik','fig13_secici_tahmin','fig08_xai_dogru_a','fig09_xai_hatali_a']:\n"
         "    display(Image(str(C.FIG_DIR / f'{f}.png'), width=800))"),
    md("## 4 · Çıktıları kaydet"),
    code("import shutil\n"
         "from src import config as C\n"
         "from src import xai, challenges as H\n"
         "for d in (xai.XAI, H.CH): print(d, len(list(d.glob('*'))))\n"
         "shutil.make_archive('/content/xai_ve_gorevler', 'zip', str(xai.OUT))\n"
         "shutil.copy('/content/xai_ve_gorevler.zip', str(C.OUTPUT_DIR / 'xai_ve_gorevler.zip'))\n"
         "shutil.make_archive('/content/sekiller_xai', 'zip', str(C.FIG_DIR))\n"
         "shutil.copy('/content/sekiller_xai.zip', str(C.OUTPUT_DIR / 'sekiller_xai.zip'))\n"
         "print('hazır')"),
], "13_xai_ve_zorlu_gorevler.ipynb")
