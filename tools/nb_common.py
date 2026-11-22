"""Notebook'larda ortak kullanılan hücreler."""

SETUP_CELL = '''import os, sys
from google.colab import drive
drive.mount('/content/drive')
# Kod ve çıktılar Drive'da tek klasörde durur; ham veri oturum diskinde (/content/data) tutulur.
PROJECT = next((p for p in ('/content/drive/MyDrive/YZProje', '/content/drive/MyDrive/YZProje/YZProje')
                if os.path.isfile(p + '/src/config.py')), None)
assert PROJECT, "Drive'da YZProje/src/config.py bulunamadı: proje klasörünü MyDrive/YZProje olarak yerleştirin."
os.chdir(PROJECT); sys.path.insert(0, PROJECT)
os.environ['KNEE_OUTPUT_DIR'] = PROJECT + '/outputs'
os.environ['KNEE_DATA_ROOT'] = '/content/data'
print('Proje klasörü:', PROJECT)'''

DATA_CELL = '''# Görüntüler oturum diskinde yoksa: önce Drive'daki arşiv kopyası, o da yoksa bağlantıdan indirip arşivi Drive'a kaydeder.
from src import config as C, data_prep
print('Veri hazır:', data_prep.ensure_data())'''
