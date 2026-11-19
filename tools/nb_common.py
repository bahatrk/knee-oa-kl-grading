"""Notebook'larda ortak kullanılan hücreler."""

SETUP_CELL = '''import os, sys, glob, shutil, zipfile
ON_KAGGLE = os.path.exists('/kaggle/working')

if ON_KAGGLE:
    # ---- Kaggle: kod ve önceki çıktılar /kaggle/input altındaki veri setlerinden gelir (salt okunur) ----
    hits = glob.glob('/kaggle/input/**/src/config.py', recursive=True)
    assert hits, 'Kod veri seti bulunamadı: sağ panelden Add Input ile yzproje-code veri setini ekleyin.'
    code_root = os.path.dirname(os.path.dirname(hits[0]))
    PROJECT = '/kaggle/working/YZProje'
    shutil.copytree(code_root, PROJECT, dirs_exist_ok=True)          # kod her oturumda tazelenir
    os.makedirs(PROJECT + '/outputs', exist_ok=True)
    out_hits = glob.glob('/kaggle/input/**/splits/all_files.csv', recursive=True)
    assert out_hits, 'Çıktı veri seti bulunamadı: Add Input ile yzproje-outputs veri setini ekleyin.'
    out_src = os.path.dirname(os.path.dirname(out_hits[0]))
    # yalnızca çalışma klasöründe OLMAYAN dosyaları kopyala (oturum içi yeni sonuçlar ezilmesin)
    shutil.copytree(out_src, PROJECT + '/outputs', dirs_exist_ok=True,
                    copy_function=lambda s, d: None if os.path.exists(d) else shutil.copy2(s, d))
    os.makedirs('/kaggle/temp/data', exist_ok=True)
    os.environ['KNEE_DATA_ROOT'] = '/kaggle/temp/data'                # ham veri: geçici disk (çıktıya sayılmaz)
else:
    # ---- Colab: Drive ----
    from google.colab import drive
    drive.mount('/content/drive')
    # Projeyi Drive'da bul (iç içe klasör olsa bile); zip açılmamışsa aç
    hits = glob.glob('/content/drive/MyDrive/**/src/config.py', recursive=True)
    if not hits:
        zips = glob.glob('/content/drive/MyDrive/**/YZProje*.zip', recursive=True)
        if zips:
            zipfile.ZipFile(zips[0]).extractall('/content/drive/MyDrive/')
            hits = glob.glob('/content/drive/MyDrive/**/src/config.py', recursive=True)
    if not hits:
        raise FileNotFoundError('Drive içinde src/config.py bulunamadı.')
    roots = [os.path.dirname(os.path.dirname(h)) for h in hits]
    # Birden fazla kopya varsa, veri listeleri (outputs/splits) olan klasör tercih edilir
    with_out = [r for r in roots if os.path.exists(f'{r}/outputs/splits/all_files.csv')]
    PROJECT = sorted(with_out or roots, key=len)[0]
    os.environ['KNEE_DATA_ROOT'] = '/content/data'

os.chdir(PROJECT); sys.path.insert(0, PROJECT)
os.environ['KNEE_OUTPUT_DIR'] = f'{PROJECT}/outputs'
print('Platform:', 'Kaggle' if ON_KAGGLE else 'Colab', '| Proje klasörü:', PROJECT)
print(sorted(os.listdir(PROJECT + '/outputs')) if os.path.exists(PROJECT + '/outputs') else 'outputs yok (ilk çalıştırma)')'''

DATA_CELL = '''# Veri oturum diskinde yoksa yeniden indir (Colab oturumu kapanınca /content/data silinir)
DATA_URL = ''   # boş bırakabilirsiniz: 01 notebook'unda kullandığınız bağlantı Drive'daki kayıttan okunur
ZIP_PATH = ''
from pathlib import Path
from src import config as C, data_prep
if not (DATA_URL or ZIP_PATH):
    _info = Path(C.AUDIT_DIR) / 'download_info.json'
    if _info.exists():
        import json
        DATA_URL = json.load(open(_info)).get('download_url', '')
        if DATA_URL: print('Veri bağlantısı kayıttan okundu.')
if any(Path(C.EXTRACT_DIR).rglob('kneeKL224')) if Path(C.EXTRACT_DIR).exists() else False:
    print('Veri hazır:', data_prep.get_dataset_root())
else:
    if not (DATA_URL or ZIP_PATH):
        raise ValueError('Veri bu oturumda yok: DATA_URL veya ZIP_PATH doldurun.')
    # Sabit dosya listeleri Drive'da duruyor; yalnızca görüntüler yeniden açılır.
    data_prep.main(['--url', DATA_URL] if DATA_URL else ['--zip', ZIP_PATH])
    print('Veri hazır:', data_prep.get_dataset_root())'''
