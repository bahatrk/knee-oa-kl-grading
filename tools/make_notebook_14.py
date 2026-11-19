"""14_ingilizce_rapor_sekilleri.ipynb (geliştirici aracı): rapor için İngilizce şekil seti."""
from pathlib import Path

import nbformat as nbf

from nb_common import DATA_CELL, SETUP_CELL

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
PIP = ("!pip install -q imagehash==4.3.2 tabulate==0.9.0 timm==1.0.30 scikit-image\n"
       "import torch\nprint(torch.__version__, torch.cuda.is_available())")
META = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
ROOT = Path(__file__).resolve().parents[1] / "notebooks"

cells = [
    md("# 14 · Rapor için İngilizce şekil seti\n\n"
       "Rapor İngilizce yazıldığı için tüm şekil metinleri İngilizce olarak yeniden üretilir (`outputs/figures_en/`). "
       "**Hiçbir model yeniden eğitilmez ve test kümesinde yeni bir seçim yapılmaz**; yalnızca kayıtlı sonuçlardan ve "
       "eğitilmiş modellerden şekil çizilir. GPU'yu (T4) açın. Süre yaklaşık 40-60 dk (bölüm 2 ve 3 en uzun olanlardır).\n\n"
       "1. Denetim şekilleri + veri artırma (augmentation) şeması + Şekil 1-7\n"
       "2. XAI şekilleri (Şekil 8-11)\n"
       "3. Zorlu görev şekilleri (Şekil 12-13)\n"
       "4. Zip'leri kaydet"),
    code("import os\nos.environ['KNEE_LANG'] = 'en'   # ÖNEMLİ: diğer hücrelerden önce"),
    code(SETUP_CELL), code(PIP), code(DATA_CELL),
    md("## 1 · Denetim, veri artırma ve Şekil 1-7"),
    code("from src import figures_en as E\nprint(E.run_all())"),
    md("## 2 · XAI (Şekil 8-11)"),
    code("from src import xai\nprint(xai.run())"),
    md("## 3 · Zorlu görevler (Şekil 12-13)"),
    code("from src import challenges as H\nH.robustness().round(4)"),
    code("df = H.selective(); H.captions()\nprint(len(df))"),
    md("## 4 · Göz kontrolü ve kayıt"),
    code("from IPython.display import Image, display\nfrom src import config as C\n"
         "for p in sorted(C.FIG_DIR.glob('*.png')):\n"
         "    print(p.name); display(Image(str(p), width=700))"),
    code("import shutil\nfrom src import config as C, xai, challenges as H\n"
         "shutil.make_archive('/content/sekiller_en', 'zip', str(C.FIG_DIR))\n"
         "shutil.copy('/content/sekiller_en.zip', str(C.OUTPUT_DIR / 'sekiller_en.zip'))\n"
         "shutil.make_archive('/content/xai_ve_gorevler_v2', 'zip', str(xai.OUT))\n"
         "shutil.copy('/content/xai_ve_gorevler_v2.zip', str(C.OUTPUT_DIR / 'xai_ve_gorevler_v2.zip'))\n"
         "print('hazır: outputs/sekiller_en.zip ve outputs/xai_ve_gorevler_v2.zip')"),
]
nb = nbf.v4.new_notebook(cells=cells); nb.metadata = META
nbf.write(nb, ROOT / "14_ingilizce_rapor_sekilleri.ipynb"); print("14_ingilizce_rapor_sekilleri.ipynb")
