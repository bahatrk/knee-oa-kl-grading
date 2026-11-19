"""01_veri_hazirlik_ve_denetim.ipynb dosyasını üretir (geliştirici aracı)."""
from pathlib import Path

import nbformat as nbf

from nb_common import SETUP_CELL

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("# 01 · Veri hazırlama ve veri denetimi (VS13 – kneeKL224)\n\n"
       "Bu notebook şunları yapar:\n"
       "1. Google Drive'ı bağlar ve proje klasörüne geçer\n"
       "2. Gerekli ek kütüphaneleri kurar, ortam bilgisini kaydeder\n"
       "3. `KneeXrayData.zip` arşivini indirir, SHA256 özetini hesaplar, **yalnızca `kneeKL224`** klasörünü oturum diskine açar\n"
       "4. Resmi train/val/test dosya listelerini oluşturur (bölme değiştirilmez)\n"
       "5. Veri denetimini çalıştırır: bütünlük, sınıf dağılımı, boyutlar, aykırı görüntüler, pHash yakın-kopya taraması, hasta çakışması\n\n"
       "**Çalıştırma:** `Çalışma zamanı → Tümünü çalıştır`. Bu aşama GPU gerektirmez."),
    md("## 1. Drive ve proje klasörü"),
    code(SETUP_CELL),
    md("## 2. Kurulum ve ortam bilgisi"),
    code("!pip install -q imagehash==4.3.2 tabulate==0.9.0\n"
         "!mkdir -p outputs/env && pip freeze > outputs/env/pip_freeze.txt\n\n"
         "from src.utils import env_info, save_json, set_seed\n"
         "from src import config as C\n"
         "set_seed(C.SEED)\n"
         "info = env_info(); save_json(info, C.OUTPUT_DIR / 'env' / 'env_info.json'); info"),
    md("## 3. İndirme ve çıkarma\n\n"
       "`DATA_URL` değişkenine **YZ_2026_Proje_Veri_Seti_Atamalari.xlsx** dosyasındaki VS13 doğrudan indirme bağlantısını yapıştırın. "
       "Mendeley bağlantıları yönlendirme içerdiği için indirme `curl -L` ile yapılır ve kesilirse kaldığı yerden devam eder.\n\n"
       "Arşivi daha önce Drive'a indirdiyseniz `ZIP_PATH` değişkenini doldurun, `DATA_URL` boş kalsın.\n\n"
       "⏱ 7,17 GB indirme + SHA256 + çıkarma yaklaşık 10–20 dakika sürer. Colab oturumu kapanınca `/content/data` silinir; "
       "yeni oturumda bu hücre tekrar çalıştırılır (dosya listeleri Drive'da kalır)."),
    code("DATA_URL = ''      # ör. 'https://data.mendeley.com/public-files/datasets/56rmx5bjcr/files/.../file_downloaded'\n"
         "ZIP_PATH = ''      # ör. '/content/drive/MyDrive/KneeXrayData.zip'\n\n"
         "from pathlib import Path\n"
         "from src import data_prep\n"
         "if Path(C.EXTRACT_DIR).exists() and any(Path(C.EXTRACT_DIR).rglob('kneeKL224')):\n"
         "    print('Veri zaten açılmış, yalnızca dosya listesi yenileniyor.')\n"
         "    data_prep.main(['--skip-extract'])\n"
         "elif DATA_URL:\n"
         "    data_prep.main(['--url', DATA_URL])\n"
         "elif ZIP_PATH:\n"
         "    data_prep.main(['--zip', ZIP_PATH])\n"
         "else:\n"
         "    raise ValueError('DATA_URL veya ZIP_PATH doldurulmalı')"),
    code("# Ham arşiv artık gerekmiyorsa disk açmak için silinebilir (SHA256 özeti download_info.json'da kayıtlı)\n"
         "# !rm -f /content/data/raw/KneeXrayData.zip\n"
         "!df -h . | tail -1\n"
         "print(open(C.AUDIT_DIR / 'download_info.json').read())"),
    md("## 4. Veri denetimi"),
    code("from src import audit\n"
         "audit.main([])"),
    code("from IPython.display import Markdown, Image, display\n"
         "display(Markdown(open(C.AUDIT_DIR / 'audit_summary.md', encoding='utf-8').read()))"),
    md("## 5. Figürler"),
    code("for f in sorted(C.FIG_DIR.glob('audit_*.png')):\n"
         "    print(f.name); display(Image(filename=str(f), width=900))"),
    md("## 6. Sonraki adım\n\n"
       "`outputs/audit/audit_summary.md`, `audit_summary.json` ve `outputs/figures/audit_*.png` dosyalarını "
       "değerlendirme için paylaşın. Dosya listeleri `outputs/splits/` altında sabitlendi; sonraki tüm deneyler bu listeleri kullanır."),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
               "accelerator": "None", "colab": {"provenance": []}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "01_veri_hazirlik_ve_denetim.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print(out)
