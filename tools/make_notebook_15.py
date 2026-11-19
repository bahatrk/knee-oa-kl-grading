"""15_gecikme_ve_teslim_paketi.ipynb (geliştirici aracı)."""
from pathlib import Path

import nbformat as nbf

from nb_common import DATA_CELL, SETUP_CELL

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
PIP = ("!pip install -q imagehash==4.3.2 tabulate==0.9.0 timm==1.0.30 scikit-image\n"
       "import torch, platform\nprint(torch.__version__, platform.python_version(), 'GPU' if torch.cuda.is_available() else 'CPU')")
META = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "colab": {"provenance": []}}
ROOT = Path(__file__).resolve().parents[1] / "notebooks"

cells = [
    md("# 15 · Ek CPU gecikme ölçümü ve teslim paketi\n\n"
       "**Çalışma zamanı: CPU** (GPU gerekmez; Çalışma zamanı → Çalışma zamanı türünü değiştir → CPU). Süre yaklaşık 10-15 dk. "
       "Notebook 11-12 çalışmış olmalı (model ağırlıkları `outputs/runs/` içinde).\n\n"
       "1. Ablasyon varyantları ve kontrol modeli için CPU gecikmesi (batch 1). Karşılaştırılabilirlik için tam model, "
       "ConvNeXt-Tiny ve DeiT-Small aynı oturumda yeniden ölçülür.\n"
       "2. Dağıtım paketi: `deploy/` içinde float16 ağırlık kopyası (GitHub 100 MB sınırı için) ve float32 ile doğruluk kontrolü\n"
       "3. `predict.py` denemesi (CPU)\n"
       "4. Ortam kaydı (`pip freeze`, sürümler) ve çıktıların zip'lenmesi"),
    code(SETUP_CELL), code(PIP), code(DATA_CELL),
    md("## 1 · CPU gecikme (ablasyon + kontrol)"),
    code("import os, platform, subprocess\n"
         "print(platform.processor() or platform.machine(), '| CPU sayısı:', os.cpu_count())\n"
         "print(subprocess.run(['bash','-c','lscpu | grep -E \"Model name|Thread|Core\"'], capture_output=True, text=True).stdout)\n"
         "from src import final_eval as F\n"
         "df = F.cpu_latency_extra()\n"
         "df[['model','params','latency_ms_median','latency_ms_p95','round_medians']].round(1)"),
    md("## 2 · Dağıtım paketi (float16 kopya + doğruluk kontrolü)"),
    code("from src import export\nprint(export.export())          # model.pt + meta.json (float32)\n"
         "print(export.export_fp16())      # model_fp16.pt ve val üzerinde karşılaştırma"),
    md("## 3 · predict.py denemesi"),
    code("from src.data import read_split\n"
         "from src.data_prep import get_dataset_root\n"
         "r = read_split('val').iloc[3]\n"
         "!python predict.py \"{get_dataset_root() / r.relpath}\"\n"
         "print('gerçek derece:', r.label)"),
    md("## 4 · Ortam kaydı ve çıktıları kaydet"),
    code("import shutil, json, subprocess\n"
         "from src import config as C\n"
         "(C.OUTPUT_DIR / 'env').mkdir(parents=True, exist_ok=True)\n"
         "(C.OUTPUT_DIR / 'env' / 'pip_freeze_final.txt').write_text(subprocess.run(['pip','freeze'], capture_output=True, text=True).stdout)\n"
         "shutil.copytree(C.PROJECT_ROOT / 'deploy', '/content/deploy_out', dirs_exist_ok=True)\n"
         "shutil.make_archive('/content/gecikme_ve_deploy', 'zip', '/content/deploy_out')\n"
         "import pandas as pd\n"
         "z = C.OUTPUT_DIR / 'final'\n"
         "shutil.copy(z / 'cpu_latency_extra.csv', '/content/cpu_latency_extra.csv')\n"
         "print('hazır: /content/gecikme_ve_deploy.zip ve /content/cpu_latency_extra.csv -> ikisini de indirip bana gönderin')\n"
         "from google.colab import files\n"
         "files.download('/content/cpu_latency_extra.csv'); files.download('/content/gecikme_ve_deploy.zip')"),
    md("## 5 · Teslim klasörü (en son, rapor son hâlini aldıktan sonra)\n\n"
       "Rapor dosyalarını (`YZProje_Rapor_*.docx/.pdf`) Drive'da `YZProje/report/` klasörüne koyun; sonra bu hücreyi çalıştırın. "
       "Çıktı: `22070090071_BahaTurk_YZProje.zip` (README'deki SHA256 otomatik doldurulur)."),
    code("!python tools/make_submission.py --project {C.PROJECT_ROOT} --out /content\n"
         "from google.colab import files\n"
         "files.download('/content/22070090071_BahaTurk_YZProje.zip')"),
]
nb = nbf.v4.new_notebook(cells=cells); nb.metadata = META
nbf.write(nb, ROOT / "15_gecikme_ve_teslim_paketi.ipynb"); print("15 ok")
