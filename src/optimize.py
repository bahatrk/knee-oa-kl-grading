"""Aşama 5: Hiperparametre optimizasyonu (Optuna).

Protokol (proje belgesi)
- Arama YALNIZCA eğitim + doğrulama verisinde yapılır; test kümesi yüklenmez.
- Amaç fonksiyonu: validation macro F1 (denemenin en iyi epoch'u).
- Büyük veri için: eğitim kümesinin sabit-seed'li, sınıf oranlarını koruyan alt kümesi ve azaltılmış epoch bütçesi;
  Optuna MedianPruner ile budama.
- En az 20 deneme (burada 25; budananlar dahil sayılır, tamamlanan/budanan sayısı raporlanır).
- Sonunda en iyi parametrelerle model TAM eğitim kümesinde yeniden eğitilir (`<model>__opt`).
- Colab kopmalarına karşı: Optuna veritabanı yerelde tutulur, her denemeden sonra Drive'a kopyalanır;
  notebook yeniden çalıştırılınca kalınan yerden devam eder.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from . import config as C
from . import plot_style as S
from .models import ZOO
from .train import get_chosen_loss, train_model
from .utils import get_logger, save_json

log = get_logger("optimize", C.LOG_DIR)
OPT_DIR = C.OUTPUT_DIR / "optuna"
LOCAL_DB = Path("/tmp/optuna_dbs")

# Deneme başına azaltılmış bütçe (yalnızca arama için); nihai yeniden eğitim varsayılan bütçeyle yapılır.
SEARCH_BUDGET = dict(head_epochs=3, ft_epochs=10, scratch_epochs=15, patience=4)

SEARCH_SPACE_DOC = {
    "transfer": {"lr_head": "log-uniform [3e-4, 3e-3]", "lr_ft": "log-uniform [2e-5, 5e-4]",
                 "weight_decay": "log-uniform [1e-4, 1e-1]", "label_smoothing": "uniform [0.0, 0.15]",
                 "drop": "uniform [0.1, 0.5]", "unfreeze_frac": "{0.25, 0.5, 0.75, 1.0}",
                 "batch_size": "{32, 64}"},
    "scratch": {"lr_scratch": "log-uniform [3e-4, 3e-3]", "weight_decay": "log-uniform [1e-4, 1e-1]",
                "label_smoothing": "uniform [0.0, 0.15]", "drop": "uniform [0.1, 0.5]",
                "batch_size": "{32, 64}"},
}


def suggest(trial: optuna.Trial, key: str, loss: str = "") -> dict:
    p = {"weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
         "drop": trial.suggest_float("drop", 0.1, 0.5),
         "batch_size": trial.suggest_categorical("batch_size", [32, 64])}
    if not loss.startswith("ordinal"):  # ordinal kayıp etiket yumuşatmayı kullanmaz -> ölü parametre aranmaz
        p["label_smoothing"] = trial.suggest_float("label_smoothing", 0.0, 0.15)
    if ZOO[key]["kind"] == "original":  # özgün kaybın kendi tasarım hiperparametreleri
        p["ordinal_sigma"] = trial.suggest_float("ordinal_sigma", 0.5, 1.25)
        p["ordinal_lambda"] = trial.suggest_float("ordinal_lambda", 0.0, 0.6)
    if ZOO[key]["kind"] == "scratch":
        p["lr_scratch"] = trial.suggest_float("lr_scratch", 3e-4, 3e-3, log=True)
    else:
        p["lr_head"] = trial.suggest_float("lr_head", 3e-4, 3e-3, log=True)
        p["lr_ft"] = trial.suggest_float("lr_ft", 2e-5, 5e-4, log=True)
        p["unfreeze_frac"] = trial.suggest_categorical("unfreeze_frac", [0.25, 0.5, 0.75, 1.0])
    return p


def top_candidates(k: int = 3, exclude=()) -> list[str]:
    """results.csv'den validation macro F1'e göre en güçlü k model (ana koşular, seed 42)."""
    r = pd.read_csv(C.OUTPUT_DIR / "results.csv")
    r = r[(r.seed == C.SEED) & r.model_name.isin(list(ZOO)) & ~r.model_name.isin(list(exclude))]
    return r.sort_values("val_macro_f1", ascending=False).model_name.head(k).tolist()


def _paths(key: str) -> tuple[Path, Path]:
    LOCAL_DB.mkdir(parents=True, exist_ok=True)
    (OPT_DIR / key).mkdir(parents=True, exist_ok=True)
    return LOCAL_DB / f"{key}.db", OPT_DIR / key / "study.db"


def run_search(key: str, n_trials: int = 25, subset_fraction: float = 0.4, seed: int = C.SEED,
               loss: str | None = None, model_kwargs: dict | None = None, warm_start: dict | None = None,
               **budget_over) -> optuna.Study:
    """warm_start: ilk deneme olarak denenecek parametreler (ör. aynı omurganın en iyisi; adil başlangıç noktası)."""
    loss = loss or get_chosen_loss()
    local, drive = _paths(key)
    if drive.exists() and not local.exists():
        shutil.copy2(drive, local)  # kopma sonrası devam
    budget = {**SEARCH_BUDGET, **budget_over}
    study = optuna.create_study(
        study_name=key, direction="maximize", storage=f"sqlite:///{local}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4))

    if warm_start and not study.trials:
        study.enqueue_trial(warm_start)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, key, loss)

        def cb(epoch: int, f1: float):
            trial.report(f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        r = train_model(key, seed, loss=loss, name=f"{key}__trial", subset_fraction=subset_fraction,
                        trial_mode=True, epoch_callback=cb, model_kwargs=model_kwargs, **budget, **params)
        trial.set_user_attr("best_epoch", r["best_epoch"])
        trial.set_user_attr("train_time_s", round(r["train_time_s"], 1))
        return r["val_macro_f1"]

    def sync(study_, trial_):
        shutil.copy2(local, drive)

    done = len([t for t in study.trials if t.state.is_finished()])
    log.info("[%s] Optuna: %d/%d deneme tamam, %d kaldı", key, done, n_trials, max(n_trials - done, 0))
    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done, callbacks=[sync], catch=(RuntimeError,))
    shutil.copy2(local, drive)
    export_study(study, key, subset_fraction, budget, loss, n_trials)
    return study


def export_study(study: optuna.Study, key: str, subset_fraction: float, budget: dict, loss: str, n_trials: int):
    d = OPT_DIR / key
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    df.to_csv(d / "trials.csv", index=False)
    states = df.state.value_counts().to_dict()
    save_json({"model": key, "objective": "validation macro F1 (denemenin en iyi epoch'u)",
               "sampler": "TPE (seed 42)", "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=4)",
               "n_trials": n_trials, "trial_states": states, "train_subset_fraction": subset_fraction,
               "trial_budget": budget, "loss": loss,
               "search_space": {k: v for k, v in SEARCH_SPACE_DOC["scratch" if ZOO[key]["kind"] == "scratch" else "transfer"].items()
                                if not (k == "label_smoothing" and loss.startswith("ordinal"))}
               | ({"ordinal_sigma": "uniform [0.5, 1.25]", "ordinal_lambda": "uniform [0.0, 0.6]"}
                  if ZOO[key]["kind"] == "original" else {}),
               "best_value": study.best_value, "best_params": study.best_params,
               "best_trial": study.best_trial.number}, d / "search_summary.json")
    plot_study(study, key)


def plot_study(study: optuna.Study, key: str) -> None:
    import matplotlib.pyplot as plt
    S.apply_style()
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.scatter([t.number for t in done], [t.value for t in done], s=28, color=S.SPLIT_COLORS["train"],
               marker="o", label=f"Tamamlanan ({len(done)})", zorder=3)
    if pruned:
        ax.scatter([t.number for t in pruned], [t.intermediate_values[max(t.intermediate_values)]
                                                 for t in pruned], s=34, color=S.SPLIT_COLORS["val"],
                   marker="x", label=f"Budanan ({len(pruned)})", zorder=3)
    best = np.maximum.accumulate([t.value if t.value is not None else -1 for t in study.trials])
    ax.step([t.number for t in study.trials], best, where="post", color=S.INK, lw=1.8, label="Şimdiye dek en iyi")
    ax.set_xlabel("Deneme no"); ax.set_ylabel("Validation macro F1")
    ax.set_title(f"(a) {key}: deneme geçmişi", loc="left"); ax.legend(loc="lower right")
    ax = axes[1]
    try:
        imp = optuna.importance.get_param_importances(study)
        names, vals = list(imp)[::-1], list(imp.values())[::-1]
        ax.barh(names, vals, color=S.SPLIT_COLORS["train"])
        ax.set_xlabel("Önem (fANOVA)")
    except Exception as e:  # az deneme / sabit parametre
        ax.text(0.5, 0.5, f"önem hesaplanamadı\n{type(e).__name__}", ha="center", transform=ax.transAxes)
    ax.set_title("(b) Hiperparametre önemi", loc="left"); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    S.save_fig(fig, C.FIG_DIR / f"optuna_{key}.png")


def retrain_best(key: str, seed: int = C.SEED, loss: str | None = None, name: str | None = None, **kw) -> dict:
    """En iyi parametrelerle TAM eğitim kümesinde ve tam epoch bütçesiyle yeniden eğitim."""
    import json
    best = json.load(open(OPT_DIR / key / "search_summary.json"))["best_params"]
    log.info("[%s] en iyi parametreler: %s", key, best)
    return train_model(key, seed, loss=loss or get_chosen_loss(), name=name or f"{key}__opt", **best, **kw)
