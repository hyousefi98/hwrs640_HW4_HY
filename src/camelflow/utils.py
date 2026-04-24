from __future__ import annotations
import json, os, random
from pathlib import Path
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nse(obs, sim):
    obs, sim = np.asarray(obs, dtype=np.float64), np.asarray(sim, dtype=np.float64)
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 2:
        return float("nan")
    obs, sim = obs[m], sim[m]
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1 - np.sum((obs - sim) ** 2) / denom)


def kge(obs, sim):
    obs, sim = np.asarray(obs, dtype=np.float64), np.asarray(sim, dtype=np.float64)
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 2:
        return float("nan")
    obs, sim = obs[m], sim[m]
    if obs.std() == 0 or sim.std() == 0:
        return float("nan")
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean() if obs.mean() != 0 else float("nan")
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def rmse(obs, sim):
    obs, sim = np.asarray(obs, dtype=np.float64), np.asarray(sim, dtype=np.float64)
    m = np.isfinite(obs) & np.isfinite(sim)
    return float(np.sqrt(np.mean((obs[m] - sim[m]) ** 2))) if m.any() else float("nan")


def pearson_r(obs, sim):
    obs, sim = np.asarray(obs, dtype=np.float64), np.asarray(sim, dtype=np.float64)
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 2:
        return float("nan")
    return float(np.corrcoef(obs[m], sim[m])[0, 1])


def peak_bias(obs, sim, q=0.98):
    obs, sim = np.asarray(obs, dtype=np.float64), np.asarray(sim, dtype=np.float64)
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 2:
        return float("nan")
    o, s = obs[m], sim[m]
    thr = np.quantile(o, q)
    idx = o >= thr
    if idx.sum() == 0:
        return float("nan")
    return float((s[idx].mean() - o[idx].mean()) / o[idx].mean())


def save_checkpoint(path, model, normalizer, cfg, history=None, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_kwargs": getattr(model, "kwargs", {}),
            "normalizer": normalizer.state_dict(),
            "cfg": cfg,
            "history": history or {},
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location or get_device(), weights_only=False)


def dump_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
