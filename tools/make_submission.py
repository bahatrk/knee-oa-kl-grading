"""Teslim klasörünü kurar: 220709071_BahaTurk_YZProje (ve zip).
  python tools/make_submission.py --project <YZProje klasörü> --out /content
Ham veri ve büyük/yeniden üretilebilir dosyalar dahil edilmez; float32 model.pt dahil edilmez (fp16 kopya girer)."""
import argparse, json, shutil, zipfile
from pathlib import Path

NAME = "220709071_BahaTurk_YZProje"


def main(project: Path, out: Path):
    dst = out / NAME
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    ig = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")
    for d in ("src", "notebooks", "tools", "docs"):
        if (project / d).exists():
            shutil.copytree(project / d, dst / d, ignore=ig)
    mn = project / "notebooks" / f"{NAME}.ipynb"
    if mn.exists():
        shutil.copy(mn, dst / mn.name)
        shutil.rmtree(dst / "notebooks", ignore_errors=True)
        shutil.copytree(project / "notebooks", dst / "notebooks" / "asamalar", ignore=shutil.ignore_patterns(
            "__pycache__", ".ipynb_checkpoints", mn.name))
    for f in ("predict.py", "requirements.txt", "README.md"):
        shutil.copy(project / f, dst / f)
    # deploy: yalnızca fp16 + meta
    dep = project / "deploy"
    (dst / "deploy").mkdir()
    for f in ("model_fp16.pt", "meta.json"):
        shutil.copy(dep / f, dst / "deploy" / f)
    o = project / "outputs"
    # results.csv (kök) + sonuç dosyaları
    shutil.copy(o / "results.csv", dst / "results.csv")
    for sub in ("final", "audit", "splits", "env", "figures", "figures_en"):
        if (o / sub).exists():
            shutil.copytree(o / sub, dst / "outputs" / sub, ignore=shutil.ignore_patterns("*.npy", "*.npz", "__pycache__"))
    for f in ("best_existing_model.json", "balancing_choice.json", "hybrid_choice.json", "ablation_mcnemar.json",
              "ablation_summary.csv", "control_summary.json", "original_design_log.csv", "mcnemar_all.csv", "seed_summary.csv"):
        if (o / f).exists():
            (dst / "outputs").mkdir(exist_ok=True); shutil.copy(o / f, dst / "outputs" / f)
    if (o / "xai").exists():
        shutil.copytree(o / "xai", dst / "outputs" / "xai", ignore=shutil.ignore_patterns("*.npy"), dirs_exist_ok=True)
    # README: SHA256
    sha = "?"
    info = o / "audit" / "download_info.json"
    if info.exists():
        sha = json.loads(info.read_text(encoding="utf-8")).get("sha256", "?")
    r = dst / "README.md"; r.write_text(r.read_text(encoding="utf-8").replace("{{SHA256}}", sha), encoding="utf-8")
    # rapor varsa
    for rep in (project / "report").glob("YZProje_Rapor*") if (project / "report").exists() else []:
        shutil.copy(rep, dst / rep.name)
    # zorunlu dosyalar var mı? (eksik olanlar uyarı olarak yazılır)
    need = ["README.md", "requirements.txt", "predict.py", "results.csv", "deploy/model_fp16.pt", "deploy/meta.json",
            f"{NAME}.ipynb", "src/config.py"]
    missing = [f for f in need if not (dst / f).exists()]
    if not list(dst.glob("*.docx")) or not list(dst.glob("*.pdf")):
        missing.append("rapor .docx/.pdf (Drive'da report/ klasörüne YZProje_Rapor_220709071.docx/.pdf koyun)")
    if missing:
        print("UYARI - eksik:", missing)
    z = out / f"{NAME}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(dst.rglob("*")):
            if p.is_file():
                zf.write(p, Path(NAME) / p.relative_to(dst))
    big = [(p.relative_to(dst), p.stat().st_size / 2**20) for p in dst.rglob("*") if p.is_file() and p.stat().st_size > 50 * 2**20]
    print("hazır:", z, f"{z.stat().st_size / 2**20:.1f} MB", "| 50 MB üstü dosyalar:", big, "| SHA256:", sha)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--project", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); main(a.project, a.out)
