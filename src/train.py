"""Ortak eğitim protokolü — tüm modeller aynı döngüyü kullanır.

Protokol
- Kayıp: varsayılan sınıf ağırlıklı çapraz entropi (+ label smoothing). Dengesizlik oranı 13 (>=3) olduğu
  için dengeleme zorunludur; 'ce' / 'wce' / 'focal' / 'ce+sampler' seçenekleri validation'da karşılaştırılır.
- Transfer learning: önce dondurulmuş gövde (yalnızca baş), sonra seçili katmanların ince ayarı.
- Düzenlileştirme (>=2): dropout, weight decay (AdamW), label smoothing, augmentation, cosine LR.
- Model seçimi ve erken durdurma: validation macro F1; en iyi doğrulama ağırlıkları geri yüklenir.
- TEST KÜMESİ BURADA HİÇ YÜKLENMEZ.
- Kaydetme: her (model, seed) koşusu bittiğinde Drive'a yazılır; yarım kalan koşu baştan başlar.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from . import config as C
from .data import class_weights, make_loaders, read_split, train_mean_std, INPUT_SIZE
from .data_prep import get_dataset_root
from .metrics import compute_metrics, confusion, per_class_report
from .original import OrdinalLoss
from .models import (ZOO, build_model, count_params, freeze_bn_stats, get_norm, set_trainable)
from .results import upsert_result
from .utils import env_info, get_logger, load_json, save_json, set_seed

log = get_logger("train", C.LOG_DIR)
RUNS = C.OUTPUT_DIR / "runs"

DEFAULTS = dict(batch_size=64, drop=0.3, weight_decay=1e-2, label_smoothing=0.1, loss="wce",
                lr_head=1e-3, lr_ft=1e-4, lr_scratch=1e-3, head_epochs=4, ft_epochs=20,
                scratch_epochs=40, patience=6, unfreeze_frac=0.5, num_workers=2, min_delta=1e-4,
                ordinal_sigma=0.75, ordinal_lambda=0.3)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma, self.ls = gamma, label_smoothing

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, reduction="none", label_smoothing=self.ls)
        pt = torch.exp(-F.cross_entropy(logits, y, reduction="none"))
        return ((1 - pt) ** self.gamma * ce).mean()


def make_criterion(loss: str, train_df: pd.DataFrame, ls: float, device, cfg: dict | None = None) -> nn.Module:
    if loss.startswith("ordinal"):
        cfg = cfg or DEFAULTS
        return OrdinalLoss(cfg["ordinal_sigma"], cfg["ordinal_lambda"]).to(device)
    if loss in ("ce", "ce+sampler"):
        return nn.CrossEntropyLoss(label_smoothing=ls)
    if loss == "wce":
        return nn.CrossEntropyLoss(weight=class_weights(train_df).to(device), label_smoothing=ls)
    if loss == "focal":
        return FocalLoss(2.0, ls)
    raise ValueError(loss)


@torch.no_grad()
def predict(model: nn.Module, loader, device, amp: bool = True):
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits.append(model(x).float().cpu())
        ys.append(y)
    logits = torch.cat(logits)
    return logits, torch.cat(ys).numpy()


def measure_latency(model: nn.Module, device, n: int = 110, warmup: int = 10) -> dict:
    """Batch=1; ilk 10 çalıştırma ölçüm dışı; medyan ve 95. yüzdelik (ms)."""
    model.eval()
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    times = []
    with torch.no_grad():
        for i in range(n):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
    dev = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    return {"latency_ms_median": float(np.median(times)), "latency_ms_p95": float(np.percentile(times, 95)),
            "latency_device": dev}


def run_dir(name: str, seed: int) -> Path:
    return RUNS / name / f"seed{seed}"


def train_model(key: str, seed: int = C.SEED, loss: str | None = None, name: str | None = None,
                pretrained: bool = True, subset_fraction: float = 1.0, force: bool = False,
                max_epochs_override: int | None = None, trial_mode: bool = False, epoch_callback=None,
                model_kwargs: dict | None = None, **overrides) -> dict:
    """Bir (model, seed) koşusu. Biten koşular atlanır (force=False)."""
    cfg = {**DEFAULTS, **overrides}
    if loss:
        cfg["loss"] = loss
    name = name or key
    out = run_dir(name, seed)
    if not trial_mode:  # trial_mode (Optuna denemesi): diske hiçbir şey yazılmaz
        if (out / "done.json").exists() and not force:
            log.info("[%s | seed %d] zaten tamamlanmış, atlanıyor", name, seed)
            return load_json(out / "done.json")
        out.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = ZOO[key]
    scratch = spec["kind"] == "scratch"
    model = build_model(key, pretrained=pretrained and not scratch, drop=cfg["drop"], **(model_kwargs or {}))

    root = get_dataset_root()
    train_df = read_split("train", subset_fraction)
    stats = train_mean_std(read_split("train"), root) if scratch else None  # yalnızca eğitim kümesi
    mean, std = get_norm(key, model, stats)
    loaders = make_loaders(mean, std, cfg["batch_size"], cfg["num_workers"], subset_fraction=subset_fraction,
                           sampler="weighted" if cfg["loss"].endswith("+sampler") else None,
                           splits=("train", "val"), seed=seed)
    model.to(device)
    crit = make_criterion(cfg["loss"], train_df, cfg["label_smoothing"], device, cfg)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # aşamalar: (ad, epoch sayısı, lr, trainable aşaması)
    n_ep = lambda e: min(e, max_epochs_override) if max_epochs_override else e  # noqa: E731
    stages = ([("scratch", n_ep(cfg["scratch_epochs"]), cfg["lr_scratch"], "all")] if scratch else
              [("head", n_ep(cfg["head_epochs"]), cfg["lr_head"], "head"),
               ("finetune", n_ep(cfg["ft_epochs"]), cfg["lr_ft"], "finetune")])

    history, best = [], {"f1": -1.0, "state": None, "epoch": -1, "logits": None}
    t_start, epoch_global, bad = time.perf_counter(), 0, 0
    log.info("[%s | seed %d] başlıyor | loss=%s | cihaz=%s | eğitim=%d", name, seed, cfg["loss"], device,
             len(train_df))
    stop = False
    for stage_name, n_epochs, lr, trainable in stages:
        if n_epochs <= 0:
            continue
        set_trainable(model, key, trainable, cfg["unfreeze_frac"])
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["weight_decay"])
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)
                 if stage_name != "head" else None)
        for ep in range(n_epochs):
            model.train()
            if stage_name in ("head", "finetune"):
                freeze_bn_stats(model)
            tl, tc, tn = 0.0, 0, 0
            for x, y in loaders["train"]:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    out_ = model(x)
                    l = crit(out_.float(), y)
                scaler.scale(l).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(opt)
                scaler.update()
                tl += l.item() * len(y)
                tc += (out_.argmax(1) == y).sum().item()
                tn += len(y)
            if sched:
                sched.step()
            logits, yv = predict(model, loaders["val"], device, use_amp)
            pv = logits.argmax(1).numpy()
            vloss = F.cross_entropy(logits, torch.from_numpy(yv)).item()
            vf1 = f1_score(yv, pv, average="macro", labels=list(range(C.NUM_CLASSES)), zero_division=0)
            rec = {"epoch": epoch_global + 1, "stage": stage_name, "lr": opt.param_groups[0]["lr"],
                   "train_loss": tl / tn, "train_acc": tc / tn, "val_loss": vloss,
                   "val_acc": accuracy_score(yv, pv), "val_macro_f1": vf1}
            history.append(rec)
            improved = vf1 > best["f1"] + cfg["min_delta"]
            if improved:
                best.update(f1=vf1, epoch=epoch_global + 1, state=copy.deepcopy(model.state_dict()), logits=logits)
                bad = 0
            elif stage_name != "head":  # erken durdurma sayacı yalnızca ince ayar / sıfırdan aşamada
                bad += 1
            log.info("  [%s] ep %2d (%s) train_loss=%.3f train_acc=%.3f | val_loss=%.3f val_acc=%.3f "
                     "val_macroF1=%.4f%s", name, epoch_global + 1, stage_name, rec["train_loss"],
                     rec["train_acc"], vloss, rec["val_acc"], vf1, "  *" if improved else "")
            epoch_global += 1
            if epoch_callback is not None:  # ör. Optuna budama: callback TrialPruned fırlatabilir
                epoch_callback(epoch_global, vf1)
            if stage_name != "head" and bad >= cfg["patience"]:
                log.info("  erken durdurma: %d epoch iyileşme yok", cfg["patience"])
                stop = True
                break
        if stop and stage_name != "head":
            break
    train_time = time.perf_counter() - t_start
    if trial_mode:
        return {"val_macro_f1": float(best["f1"]), "best_epoch": best["epoch"], "epochs": epoch_global,
                "train_time_s": train_time}

    # en iyi doğrulama ağırlıklarını geri yükle ve kaydet
    model.load_state_dict(best["state"])
    probs = torch.softmax(best["logits"], 1).numpy()
    _, yv = predict(model, loaders["val"], device, use_amp)
    vm = compute_metrics(yv, probs.argmax(1), probs)
    torch.save(model.state_dict(), out / "model.pt")
    np.save(out / "val_probs.npy", probs)
    np.save(out / "val_labels.npy", yv)
    np.save(out / "val_confusion.npy", confusion(yv, probs.argmax(1)))
    per_class_report(yv, probs.argmax(1)).to_csv(out / "val_per_class.csv", index=False)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    meta = {"model_key": key, "timm_name": spec.get("timm_name"), "class_names": C.CLASS_NAMES,
            "mean": list(mean), "std": list(std), "input_size": INPUT_SIZE, "drop": cfg["drop"],
            "best_epoch": best["epoch"], "total_epochs": epoch_global, "model_kwargs": model_kwargs or {},
            "loss": cfg["loss"]}
    save_json(meta, out / "meta.json")

    lat = measure_latency(model, device)
    row = {"model_name": name, "model_family": spec["family"], "seed": seed,
           "input_size": f"{INPUT_SIZE}x{INPUT_SIZE}x3", "param_count": count_params(model),
           "epochs": epoch_global, "batch_size": cfg["batch_size"], "optimizer": "AdamW",
           "learning_rate": cfg["lr_scratch"] if scratch else f"{cfg['lr_head']}(head)/{cfg['lr_ft']}(ft)",
           "loss": cfg["loss"] + ("" if cfg["loss"].startswith("ordinal") else f"+LS{cfg['label_smoothing']}"), "train_time_s": round(train_time, 1),
           "model_size_mb": round((out / "model.pt").stat().st_size / 2**20, 2), **lat,
           "hyperparams": {k: cfg[k] for k in ("drop", "weight_decay", "label_smoothing", "lr_head", "lr_ft",
                                              "lr_scratch", "head_epochs", "ft_epochs", "patience",
                                              "unfreeze_frac", "ordinal_sigma", "ordinal_lambda")} | {"subset_fraction": subset_fraction,
                                                                   "best_epoch": best["epoch"]},
           "notes": "Seçim val macro F1; test final aşamada"}
    upsert_result(row, vm)
    done = {"name": name, "seed": seed, "best_epoch": best["epoch"], "epochs": epoch_global,
            "train_time_s": train_time, "val_metrics": vm, "device": str(device)}
    save_json(done, out / "done.json")
    log.info("[%s | seed %d] bitti | en iyi ep %d | val macroF1=%.4f bAcc=%.4f QWK=%.4f | %.0f sn",
             name, seed, best["epoch"], vm["macro_f1"], vm["balanced_accuracy"], vm["qwk"], train_time)
    return done


BALANCE_FILE = C.OUTPUT_DIR / "balancing_choice.json"
BALANCE_OPTIONS = ["ce", "wce", "focal", "ce+sampler"]


def get_chosen_loss() -> str:
    """Dengeleme karşılaştırması yapıldıysa validation'da kazanan kayıp; yoksa 'wce'."""
    return load_json(BALANCE_FILE)["chosen_loss"] if BALANCE_FILE.exists() else "wce"


def balancing_comparison(key: str = "EfficientNetB0", seed: int = C.SEED, **kw) -> dict:
    """Sınıf dengesizliği (oran 13 >= 3) için dengeli/dengesiz karşılaştırması — YALNIZCA validation.
    Kazanan (val macro F1) tüm modeller için ortak kayıp olarak kaydedilir."""
    rows = {}
    for loss in BALANCE_OPTIONS:
        d = train_model(key, seed, loss=loss, name=f"{key}__bal_{loss}", **kw)
        m = d["val_metrics"]
        rows[loss] = {"val_macro_f1": m["macro_f1"], "val_balanced_accuracy": m["balanced_accuracy"],
                      "val_accuracy": m["accuracy"], "val_qwk": m["qwk"]}
    # Proje kuralı: oran >= 3 ise en az bir dengeleme yöntemi UYGULANIR; 'ce' yalnızca karşılaştırma referansıdır.
    best = max((k for k in rows if k != "ce"), key=lambda k: rows[k]["val_macro_f1"])
    out = {"model": key, "seed": seed, "criterion": "val macro F1 (yalnızca dengeleme yöntemleri arasında)",
           "results": rows, "chosen_loss": best,
           "unbalanced_better_than_chosen": rows["ce"]["val_macro_f1"] > rows[best]["val_macro_f1"]}
    save_json(out, BALANCE_FILE)
    log.info("Dengeleme karşılaştırması: kazanan = %s | %s", best, {k: round(v["val_macro_f1"], 4)
                                                                  for k, v in rows.items()})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[k for k in ZOO if k != "Original"])
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--loss", default=None, choices=[None, "ce", "wce", "focal", "ce+sampler"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true", help="Hızlı deneme: alt küme ve 1-2 epoch")
    args = ap.parse_args(argv)
    C.ensure_dirs()
    save_json(env_info(), C.OUTPUT_DIR / "env" / "env_info_train.json")
    for k in args.models:
        kw = dict(subset_fraction=0.1, max_epochs_override=1, name=f"{k}__quick") if args.quick else {}
        train_model(k, args.seed, loss=args.loss or get_chosen_loss(), force=args.force or args.quick, **kw)


if __name__ == "__main__":
    main()
