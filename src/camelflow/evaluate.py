from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .data import build_loaders
from .models import build_model
from .utils import get_device, kge, load_checkpoint, nse, pearson_r, peak_bias, rmse, dump_json


def _infer(model, loader, device):
    model.eval()
    preds, ys, basins, ts = [], [], [], []
    with torch.no_grad():
        for x_dyn, x_static, y, b, t in loader:
            x_dyn = x_dyn.to(device, non_blocking=True)
            x_static = x_static.to(device, non_blocking=True)
            yhat = model(x_dyn, x_static).cpu().numpy()
            preds.append(yhat)
            ys.append(y.numpy())
            basins.append(b.numpy())
            ts.append(t.numpy())
    return (np.concatenate(preds), np.concatenate(ys),
            np.concatenate(basins), np.concatenate(ts))


def evaluate(checkpoint_path: str, out_dir: str = "outputs", split: str = "test"):
    device = get_device()
    ck = load_checkpoint(checkpoint_path, map_location=device)
    cfg = ck["cfg"]
    print(f"[eval] loaded {checkpoint_path} | cfg seq_len={cfg['seq_len']} hidden={cfg['hidden']}")

    loaders = build_loaders(
        seq_len=cfg["seq_len"],
        batch_size=cfg.get("batch_size", 256),
        num_workers=cfg.get("num_workers", 0),
        local_dir=cfg.get("local_dir"),
    )
    model = build_model(cfg.get("model", "lstm"), **ck["model_kwargs"]).to(device)
    model.load_state_dict(ck["model_state"])

    preds_n, ys_n, b_idx, t_idx = _infer(model, loaders[split], device)
    nz = loaders["normalizer"]
    preds = nz.inverse_y(preds_n)
    ys = nz.inverse_y(ys_n)

    basins = loaders["basins"]
    per_basin = defaultdict(lambda: {"y": [], "yhat": [], "t": []})
    for p, y, b, t in zip(preds, ys, b_idx, t_idx):
        per_basin[basins[int(b)]]["y"].append(y)
        per_basin[basins[int(b)]]["yhat"].append(p)
        per_basin[basins[int(b)]]["t"].append(int(t))

    rows = []
    for bid, d in per_basin.items():
        y = np.array(d["y"]); yhat = np.array(d["yhat"])
        rows.append({
            "basin_id": bid,
            "n": len(y),
            "nse": nse(y, yhat),
            "kge": kge(y, yhat),
            "rmse": rmse(y, yhat),
            "r": pearson_r(y, yhat),
            "peak_bias": peak_bias(y, yhat),
        })
    df = pd.DataFrame(rows).sort_values("nse", ascending=False).reset_index(drop=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{split}_metrics.csv", index=False)
    summary = {
        "split": split,
        "n_basins": len(df),
        "median_nse": float(df["nse"].median()),
        "mean_nse": float(df["nse"].mean()),
        "median_kge": float(df["kge"].median()),
        "median_rmse": float(df["rmse"].median()),
    }
    dump_json(out / f"{split}_summary.json", summary)

    print(f"[eval] basins={len(df)} | median NSE={summary['median_nse']:.3f} | "
          f"median KGE={summary['median_kge']:.3f} | median RMSE={summary['median_rmse']:.3f}")
    print(f"[eval] top 3:\n{df.head(3).to_string(index=False)}")
    print(f"[eval] bottom 3:\n{df.tail(3).to_string(index=False)}")

    np.savez(
        out / f"{split}_predictions.npz",
        preds=preds, obs=ys,
        basin_idx=b_idx, time_idx=t_idx,
        basins=np.array(basins, dtype=object),
        times=pd.to_datetime(loaders["ds"]["time"].values).to_numpy(),
    )
    return {"metrics": df, "summary": summary, "checkpoint": checkpoint_path}
