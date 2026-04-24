from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from .data import build_loaders
from .models import build_model
from .utils import get_device, nse, save_checkpoint, set_seed, dump_json


class BasinNSELoss(nn.Module):
    def __init__(self, std_per_basin: np.ndarray, eps: float = 0.1):
        super().__init__()
        self.register_buffer("s", torch.tensor(std_per_basin, dtype=torch.float32))
        self.eps = eps

    def forward(self, yhat, y, b):
        w = 1.0 / (self.s[b] + self.eps) ** 2
        return (w * (yhat - y) ** 2).mean()


class MSELossWrap(nn.Module):
    def forward(self, yhat, y, b):
        return ((yhat - y) ** 2).mean()


def _run_epoch(model, loader, loss_fn, optim, device, train: bool):
    model.train(train)
    tot, n = 0.0, 0
    preds_n, ys_n = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x_dyn, x_static, y, b, _t in loader:
            x_dyn = x_dyn.to(device, non_blocking=True)
            x_static = x_static.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            b = b.to(device, non_blocking=True)
            yhat = model(x_dyn, x_static)
            loss = loss_fn(yhat, y, b)
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            tot += loss.item() * y.size(0)
            n += y.size(0)
            preds_n.append(yhat.detach().cpu().numpy())
            ys_n.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds_n) if preds_n else np.array([])
    ys = np.concatenate(ys_n) if ys_n else np.array([])
    return tot / max(n, 1), preds, ys


def train_model(cfg: dict):
    set_seed(cfg.get("seed", 42))
    device = get_device()
    print(f"[train] device={device}")

    loaders = build_loaders(
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 0),
        local_dir=cfg.get("local_dir"),
    )
    print(f"[train] train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)} "
          f"test={len(loaders['test'].dataset)} n_dyn={loaders['n_dyn']} n_static={loaders['n_static']}")

    model = build_model(
        cfg.get("model", "lstm"),
        n_dyn=loaders["n_dyn"],
        n_static=loaders["n_static"],
        hidden=cfg["hidden"],
        num_layers=cfg.get("num_layers", 1),
        dropout=cfg.get("dropout", 0.4),
    ).to(device)
    print(f"[train] params={sum(p.numel() for p in model.parameters()):,}")

    optim = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0)
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=3)

    loss_name = cfg.get("loss", "nse").lower()
    if loss_name == "nse":
        loss_fn = BasinNSELoss(loaders["y_std_per_basin"]).to(device)
        print(f"[train] loss=basin-NSE (eps=0.1)")
    elif loss_name == "mse":
        loss_fn = MSELossWrap().to(device)
        print(f"[train] loss=MSE")
    else:
        raise ValueError(f"unknown loss: {loss_name}")

    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "best.pt"

    history = {"train_loss": [], "val_loss": [], "val_nse": [], "lr": [], "epoch_time_s": []}
    best_nse = -np.inf
    best_epoch = 0
    bad_epochs = 0
    patience = int(cfg.get("patience", 8))
    min_delta = float(cfg.get("min_delta", 1e-4))

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, _, _ = _run_epoch(model, loaders["train"], loss_fn, optim, device, train=True)
        va_loss, va_pred, va_y = _run_epoch(model, loaders["val"], loss_fn, optim, device, train=False)
        nz = loaders["normalizer"]
        va_pred_raw = nz.inverse_y(va_pred)
        va_y_raw = nz.inverse_y(va_y)
        va_nse = nse(va_y_raw, va_pred_raw)
        sched.step(va_loss)
        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_nse"].append(va_nse)
        history["lr"].append(optim.param_groups[0]["lr"])
        history["epoch_time_s"].append(dt)

        improved = va_nse > best_nse + min_delta
        print(f"[train] ep {epoch:03d} | train {tr_loss:.4f} | val {va_loss:.4f} | "
              f"val_nse {va_nse:.3f} | lr {optim.param_groups[0]['lr']:.2e} | {dt:.1f}s"
              f"{' *best*' if improved else f'  (no-improve {bad_epochs + 1}/{patience})'}")

        if improved:
            best_nse = va_nse
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(ckpt_path, model, loaders["normalizer"], cfg,
                            history={"best_val_nse": best_nse, "epoch": epoch})
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[train] early stop: no val_nse improvement for {patience} epochs "
                      f"(best {best_nse:.3f} at ep {best_epoch})")
                break

    save_checkpoint(out / "last.pt", model, loaders["normalizer"], cfg,
                    history={"best_val_nse": best_nse, "best_epoch": best_epoch, "stopped_at": epoch})
    dump_json(out / "history.json", history)
    print(f"[train] done. best val NSE = {best_nse:.3f} @ ep {best_epoch} -> {ckpt_path}")
    return {"history": history, "best_val_nse": best_nse, "best_epoch": best_epoch,
            "stopped_at": epoch, "checkpoint": str(ckpt_path)}
