"""03_model_karsilastirma.ipynb dosyasını üretir (geliştirici aracı)."""
from pathlib import Path

import nbformat as nbf

from nb_common import DATA_CELL, SETUP_CELL

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 03 · Model karşılaştırma (10 model: 2 sıfırdan CNN, 7 transfer, 1 ViT)\n\n"
       "**Önce GPU açın.** Colab: `Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU`. Kaggle: sağ panel → *Session options* → *Accelerator* → GPU ve *Internet: On*. Bu notebook her iki platformda da çalışır.\n\n"
       "Akış:\n"
       "1. Kurulum, GPU kontrolü, veri\n"
       "2. Hızlı deneme (≈5 dk): indirme ve eğitim döngüsü sorunsuz mu?\n"
       "3. **Dengeleme karşılaştırması** (yalnızca validation): ağırlıksız CE, ağırlıklı CE, focal, ağırlıklı örnekleme → kazanan tüm modellerde kullanılır\n"
       "4. 10 modelin eğitimi (seed 42). **Biten koşular atlanır**: oturum kopsa bile aynı notebook'u yeniden çalıştırmanız yeterli\n"
       "5. Özet tablo\n\n"
       "Test kümesi bu notebook'ta **hiç yüklenmez**."),
    code(SETUP_CELL),
    code("!pip install -q imagehash==4.3.2 tabulate==0.9.0 timm==1.0.30\n"
         "import torch\n"
         "assert torch.cuda.is_available(), 'GPU yok! Colab: Çalışma zamanı > Çalışma zamanı türünü değiştir > T4 GPU | Kaggle: sağ panel Session options > Accelerator > GPU'\n"
         "print(torch.__version__, torch.cuda.get_device_name(0))\n"
         "!mkdir -p outputs/env && pip freeze > outputs/env/pip_freeze_train.txt"),
    code(DATA_CELL),
    md("## Doğrulanan yakın-kopya çiftleri (görsel)\n\n4 çiftin neye benzediğini görmek için (yaklaşık 1 dk)."),
    code("from src import dup_verify\n"
         "from IPython.display import Image, display\n"
         "dup_verify.main([])\n"
         "display(Image(filename=str(C.FIG_DIR / 'audit_confirmed_duplicates.png'), width=900))"),
    md("## Hızlı deneme\n\nGerçek ImageNet ağırlıklarının inmesini ve eğitim döngüsünü küçük alt kümede (1 epoch) dener."),
    code("from src.train import train_model\n"
         "for k in ['EfficientNetB0', 'SimpleCNN']:\n"
         "    train_model(k, 42, name=f'{k}__quick', subset_fraction=0.1, max_epochs_override=1, force=True)"),
    md("## Dengeleme karşılaştırması (yalnızca validation)\n\n"
       "EfficientNet-B0 ile 4 seçenek (her biri ≈10–15 dk). Kazanan `outputs/balancing_choice.json` dosyasına yazılır."),
    code("from src.train import balancing_comparison, get_chosen_loss\n"
         "import json, pandas as pd\n"
         "if not (C.OUTPUT_DIR / 'balancing_choice.json').exists():\n"
         "    balancing_comparison('EfficientNetB0')\n"
         "ch = json.load(open(C.OUTPUT_DIR / 'balancing_choice.json'))\n"
         "display(pd.DataFrame(ch['results']).T.round(4)); print('Seçilen kayıp:', ch['chosen_loss'])"),
    md("## 10 modelin eğitimi\n\nSıra hızlıdan yavaşa. Her koşu bitince Drive'a yazılır."),
    code("from src.models import ZOO\n"
         "ORDER = ['EfficientNetB0', 'MobileNetV3L', 'SimpleCNN', 'MiniResNet', 'ResNet50', 'DenseNet121',\n"
         "         'ConvNeXtTiny', 'DeiTSmall', 'Xception', 'InceptionV3']\n"
         "assert set(ORDER) == set(ZOO) - {'Original'}\n"
         "LOSS = get_chosen_loss(); print('Ortak kayıp:', LOSS)\n"
         "for k in ORDER:\n"
         "    train_model(k, 42, loss=LOSS)"),
    md("## Özet (validation)"),
    code("r = pd.read_csv(C.OUTPUT_DIR / 'results.csv')\n"
         "r = r[~r.model_name.str.contains('__')]\n"
         "cols = ['model_name','param_count','epochs','train_time_s','latency_ms_median','val_accuracy',\n"
         "        'val_balanced_accuracy','val_macro_f1','val_qwk','val_mae_grade','val_roc_auc_macro']\n"
         "display(r[cols].sort_values('val_macro_f1', ascending=False).round(4))"),
    md("`outputs/results.csv` ve `outputs/balancing_choice.json` dosyalarını, ayrıca sonuç tablosunu paylaşın."),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
               "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "03_model_karsilastirma.ipynb"
nbf.write(nb, out)
print(out)
