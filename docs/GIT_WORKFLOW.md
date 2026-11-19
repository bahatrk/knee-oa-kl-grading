# Git iş akışı

Tek geliştiricili bir ders projesi için **GitHub Flow**: `main` her zaman çalışır ve teslim edilebilir durumdadır;
her iş kısa ömürlü bir dalda yapılır, Pull Request ile `main`'e girer.

## Dal adları: `<kategori>/<kısa-açıklama>`

| Kategori | Ne için | Örnek |
|---|---|---|
| `feat/` | Yeni analiz, şekil, tablo, kod yeteneği | `feat/figures-compliance` |
| `fix/` | Hata veya şartnameye aykırılık düzeltmesi | `fix/report-methods-text` |
| `docs/` | README, rapor metni, kaynakça | `docs/references-and-attributions` |
| `chore/` | Paketleme, ortam, repo bakımı | `chore/submission-packaging` |
| `exp/` | Deney (sonuç `main`'e yalnızca doğrulama sonrası girer) | `exp/threshold-adjustment` |

## Commit mesajları: Conventional Commits

`<tür>(<kapsam>): <özet>` — türler `feat`, `fix`, `docs`, `chore`, `refactor`, `test`.
Özet emir kipinde ve 72 karakteri geçmez; gerekçe gövdede yazılır. Bir commit tek bir mantıksal değişiklik içerir.

## Akış

1. `git switch main && git pull` → `git switch -c <kategori>/<ad>`
2. Küçük, anlamlı commit'ler.
3. Pull Request aç (şablon: `.github/pull_request_template.md`), `Closes #<issue>` ile bağla, kendi kendine gözden geçir, **squash değil merge commit** ile birleştir (dal geçmişi görünür kalsın).
4. Birleşen dal silinir.
5. Dönüm noktaları etiketlenir: `v0.1-deneyler-tamam`, `v0.2-rapor-v1`, `v1.0-teslim`.

## Depoya girmeyenler

Ham veri, `outputs/`, model ağırlıkları (`*.pt`), `.npy/.npz` önbellekleri, rapor ikilileri (`.docx/.pdf`) ve zip'ler `.gitignore` ile dışarıda tutulur.
Ağırlıklar ve rapor teslim zip'inde, istenirse GitHub Release eki olarak paylaşılır. Gizli anahtar veya kişisel bağlantı jetonu commit edilmez.

## Bilinen iş paketleri ve dalları

| Dal | Kapsam |
|---|---|
| `fix/student-number` | Öğrenci numarası düzeltmesi (atama dosyasındaki 220709071) |
| `fix/report-methods-text` | Normalizasyon ve "12 ağ" ifadeleri, sürüm/donanım bilgisi, gerekçeler |
| `feat/figures-compliance` | Aynı epoch aralığı, ROC/PR macro-ortalama, ayrıntılı mimari diyagramı, şekil başlıkları |
| `feat/xai-true-class-deletion` | Silme eğrisinin doğru sınıfla yeniden hesaplanması, kenar-bandı analizi |
| `feat/report-clinical-ablation-tables` | Klinik hata tanımı ve duyarlılık/özgüllük, şartname biçiminde ablasyon ve hibrit ablasyon tabloları, tüm modeller için ek tablo/şekil, güven aralığı tablosu |
| `docs/references-and-attributions` | Kaynakça doğrulaması, SE/CBAM/yumuşak etiket ve yazılım atıfları, literatür tablosuna "güçlü yön" |
| `feat/results-csv-hybrid-rows` | `results.csv`: hibrit satırları, CPU gecikme sütunları |
| `feat/main-notebook` | Başından sonuna çalışan ana notebook (kayıtlı çıktılardan) |
| `chore/submission-packaging` | README (indirme komutu, SHA256), `make_submission.py`, teslim klasörü |
