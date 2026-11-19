"""Dağıtım paketi: en iyi (özgün) model + meta + SHA256. `deploy/` klasörü predict.py ile birlikte teslim edilir."""
from __future__ import annotations

import shutil

from . import config as C
from .final_eval import RUNS, _sha256
from .utils import load_json, save_json

DEPLOY = C.PROJECT_ROOT / "deploy"


def export(name: str = "Orig_full__opt", seed: int = C.SEED) -> dict:
    src = RUNS / name / f"seed{seed}"
    DEPLOY.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "model.pt", DEPLOY / "model.pt")
    meta = load_json(src / "meta.json")
    cal = None
    try:  # sıcaklık: yalnızca doğrulamadan (isteğe bağlı kullanım)
        import numpy as np
        from .final_eval import fit_temperature, _val
        vp, vy = _val(name, seed); cal = fit_temperature(vp, vy)
    except Exception:
        pass
    meta.update({"run_name": name, "seed": seed, "temperature_val": cal, "sha256_model": _sha256(DEPLOY / "model.pt"),
                 "note": "Girdi: 224x224 gri/RGB diz kapsülü görüntü; çıktı: KL0-KL4 olasılıkları. Araştırma amaçlıdır, klinik karar aracı değildir."})
    save_json(meta, DEPLOY / "meta.json")
    return meta


def export_fp16() -> dict:
    """GitHub 100 MB dosya sınırı için ağırlıkları float16 olarak da saklar (model.pt 108 MB). Yüklerken float32'ye çevrilir.
    Doğrulama setinde float32 ile float16 tahminleri karşılaştırılır; sonuç meta.json içine yazılır."""
    import numpy as np
    import torch
    from .data import make_loaders
    from .train import predict
    from .models import build_model
    meta = load_json(DEPLOY / "meta.json")
    sd = torch.load(DEPLOY / "model.pt", map_location="cpu")
    sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
    torch.save(sd16, DEPLOY / "model_fp16.pt")
    m32 = build_model(meta["model_key"], pretrained=False, drop=meta.get("drop", 0.3), **meta.get("model_kwargs", {}))
    m32.load_state_dict(sd); m32.eval()
    m16 = build_model(meta["model_key"], pretrained=False, drop=meta.get("drop", 0.3), **meta.get("model_kwargs", {}))
    m16.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd16.items()}); m16.eval()
    dev = torch.device("cpu")
    ld = make_loaders(tuple(meta["mean"]), tuple(meta["std"]), 64, 2, augment=False, splits=("val",), seed=C.SEED)["val"]
    l32, y = predict(m32, ld, dev, amp=False); l16, _ = predict(m16, ld, dev, amp=False)
    p32 = torch.softmax(l32, 1).numpy(); p16 = torch.softmax(l16, 1).numpy()
    chk = {"n_val": int(len(y)), "argmax_agreement": float((p32.argmax(1) == p16.argmax(1)).mean()),
           "max_abs_prob_diff": float(np.abs(p32 - p16).max())}
    meta.update({"fp16_file": "model_fp16.pt", "sha256_model_fp16": _sha256(DEPLOY / "model_fp16.pt"), "fp16_check_val": chk})
    save_json(meta, DEPLOY / "meta.json")
    return chk
