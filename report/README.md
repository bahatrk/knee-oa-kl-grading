# Rapor üreticisi

`prep.py`: sonuç CSV'lerinden (`results.csv`, `outputs/final/*.csv`) tablo verisini (`data.json`) üretir.
`build.js`: `data.json` ve şekil PNG'lerinden İngilizce makale biçimindeki Word raporunu üretir (`npm i docx`; PDF için LibreOffice).
Rapordaki sayılar elle yazılmaz; CSV'lerden gelir. `.docx/.pdf` çıktıları depoya girmez.
