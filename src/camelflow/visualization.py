from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data import TARGET, load_raw, _basin_ids


_STATES_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
_STATES_CACHE = None


def _ensure(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def _load_states():
    global _STATES_CACHE
    if _STATES_CACHE is None:
        import geopandas as gpd
        _STATES_CACHE = gpd.read_file(_STATES_URL)
    return _STATES_CACHE


def _conus_ax(ax):
    try:
        states = _load_states()
        states.boundary.plot(ax=ax, linewidth=0.5, color="gray")
        states.dissolve().boundary.plot(ax=ax, linewidth=1.0, color="black")
    except Exception as e:
        print(f"[viz] state overlay failed: {e}")
    ax.set_xlim(-125, -65); ax.set_ylim(24, 50)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    return ax


def _plot_attr_map(attrs, col, title, out_path, cmap="viridis"):
    if col not in attrs.columns:
        print(f"[viz] missing attr column {col!r}, skipping")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    _conus_ax(ax)
    sc = ax.scatter(attrs["lon"], attrs["lat"], c=attrs[col],
                    cmap=cmap, s=70, edgecolor="k", linewidth=0.5)
    plt.colorbar(sc, ax=ax, label=col)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_data_summary(out_dir: str = "outputs/figs/summary", n_basins_ts: int = 6):
    out = _ensure(out_dir)
    ds, attrs = load_raw()
    basins = _basin_ids(ds)

    fig, ax = plt.subplots(figsize=(12, 4))
    for bid in basins[:n_basins_ts]:
        q = ds[TARGET].sel(basin=bid).values
        ax.plot(ds["time"].values, q, lw=0.6, label=bid)
    ax.set_ylabel("qobs (mm/day)"); ax.set_xlabel("date")
    ax.set_title(f"Observed streamflow — {n_basins_ts} basins"); ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(out / "qobs_timeseries.png", dpi=140); plt.close(fig)

    q_all = ds[TARGET].values.ravel()
    q_all = q_all[np.isfinite(q_all) & (q_all >= 0)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log1p(q_all), bins=80, color="steelblue", edgecolor="white")
    ax.set_xlabel("log1p(qobs)"); ax.set_ylabel("count")
    ax.set_title("Streamflow distribution (log1p, all basins)")
    fig.tight_layout(); fig.savefig(out / "qobs_histogram.png", dpi=140); plt.close(fig)

    bid = basins[0]
    sub = ds.sel(basin=bid, time=slice("2000-10-01", "2002-09-30"))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].bar(sub["time"].values, sub["prcp"].values, width=1.0, color="slategray")
    axes[0].set_ylabel("prcp (mm/day)"); axes[0].set_title(f"Basin {bid} — forcing and streamflow (WY2001-2002)")
    axes[1].plot(sub["time"].values, sub[TARGET].values, color="navy", lw=0.9)
    axes[1].set_ylabel("qobs (mm/day)"); axes[1].set_xlabel("date")
    fig.tight_layout(); fig.savefig(out / "precip_vs_qobs.png", dpi=140); plt.close(fig)

    _plot_attr_map(attrs, "q_mean", "Basin mean streamflow (q_mean)", out / "map_q_mean.png", cmap="viridis")
    _plot_attr_map(attrs, "aridity", "Basin aridity (PET/P)", out / "map_aridity.png", cmap="YlOrBr")

    fig, ax = plt.subplots(figsize=(6, 5))
    if {"aridity", "runoff_ratio"}.issubset(attrs.columns):
        ax.scatter(attrs["aridity"], attrs["runoff_ratio"], s=30, alpha=0.8)
        ax.set_xlabel("aridity"); ax.set_ylabel("runoff ratio")
        ax.set_title("Static attributes — aridity vs runoff ratio")
    fig.tight_layout(); fig.savefig(out / "static_scatter.png", dpi=140); plt.close(fig)

    attrs.describe().to_csv(out / "attributes_summary.csv")
    print(f"[viz] wrote summary figures to {out}")


def plot_training(history: dict | str, out_dir: str = "outputs/figs/train"):
    if isinstance(history, (str, Path)):
        with open(history) as f:
            history = json.load(f)
    out = _ensure(out_dir)
    ep = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, history["train_loss"], label="train")
    ax.plot(ep, history["val_loss"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (standardized log-flow)")
    ax.set_title("Training and validation loss"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "loss_curves.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, history["val_nse"], color="darkgreen")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation NSE")
    ax.set_title("Validation NSE vs epoch")
    fig.tight_layout(); fig.savefig(out / "val_nse.png", dpi=140); plt.close(fig)
    print(f"[viz] wrote training figures to {out}")


def plot_predictions(pred_npz: str, metrics_csv: str, out_dir: str = "outputs/figs/pred"):
    out = _ensure(out_dir)
    data = np.load(pred_npz, allow_pickle=True)
    df = pd.read_csv(metrics_csv, dtype={"basin_id": str})
    basins = list(map(str, data["basins"]))
    times = pd.to_datetime(data["times"])

    best = df.iloc[0]["basin_id"]
    worst = df.iloc[-1]["basin_id"]
    med = df.iloc[(df["nse"] - df["nse"].median()).abs().idxmin()]["basin_id"]

    for tag, bid in [("best", str(best)), ("median", str(med)), ("worst", str(worst))]:
        b = basins.index(bid)
        mask = data["basin_idx"] == b
        t_i = data["time_idx"][mask]
        y = data["obs"][mask]
        yhat = data["preds"][mask]
        order = np.argsort(t_i)
        t_i, y, yhat = t_i[order], y[order], yhat[order]
        t = times[t_i]

        nse_val = df[df["basin_id"] == bid]["nse"].iloc[0]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, y, label="observed", color="black", lw=0.9)
        ax.plot(t, yhat, label="predicted", color="tab:red", lw=0.9, alpha=0.8)
        ax.set_ylabel("streamflow (mm/day)"); ax.set_xlabel("date")
        ax.set_title(f"{tag.title()} basin {bid} — test hydrograph (NSE={nse_val:.2f})")
        ax.legend()
        fig.tight_layout(); fig.savefig(out / f"hydrograph_{tag}_{bid}.png", dpi=140); plt.close(fig)

    y_all = data["obs"]; yhat_all = data["preds"]
    m = np.isfinite(y_all) & np.isfinite(yhat_all)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_all[m], yhat_all[m], s=2, alpha=0.15, color="navy")
    lim = max(np.quantile(y_all[m], 0.999), np.quantile(yhat_all[m], 0.999))
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("observed (mm/day)"); ax.set_ylabel("predicted (mm/day)")
    ax.set_title(f"Parity — test (n={m.sum():,})")
    fig.tight_layout(); fig.savefig(out / "parity.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["nse"].clip(-1, 1), bins=30, color="steelblue", edgecolor="white")
    ax.axvline(df["nse"].median(), color="red", ls="--", label=f"median={df['nse'].median():.2f}")
    ax.set_xlabel("NSE (clipped to [-1, 1])"); ax.set_ylabel("number of basins")
    ax.set_title("Per-basin test NSE"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "nse_hist.png", dpi=140); plt.close(fig)

    print(f"[viz] wrote prediction figures to {out}")


def plot_flow_duration(pred_npz: str, metrics_csv: str, out_dir: str = "outputs/figs/pred"):
    out = _ensure(out_dir)
    data = np.load(pred_npz, allow_pickle=True)
    df = pd.read_csv(metrics_csv, dtype={"basin_id": str})
    basins = list(map(str, data["basins"]))

    picks = [
        ("best",   df.iloc[0]["basin_id"]),
        ("median", df.iloc[(df["nse"] - df["nse"].median()).abs().idxmin()]["basin_id"]),
        ("worst",  df.iloc[-1]["basin_id"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (tag, bid) in zip(axes, picks):
        bid = str(bid)
        if bid not in basins:
            continue
        b = basins.index(bid)
        m = data["basin_idx"] == b
        y = np.sort(data["obs"][m])[::-1]
        yhat = np.sort(data["preds"][m])[::-1]
        exc = np.arange(1, len(y) + 1) / (len(y) + 1) * 100
        ax.plot(exc, np.clip(y, 1e-3, None), label="obs", color="black", lw=1.0)
        ax.plot(exc, np.clip(yhat, 1e-3, None), label="pred", color="tab:red", lw=1.0, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("exceedance probability (%)")
        nse_val = df[df["basin_id"] == bid]["nse"].iloc[0]
        ax.set_title(f"{tag.title()} — {bid} (NSE={nse_val:.2f})")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("streamflow (mm/day, log)")
    fig.suptitle("Flow duration curves — observed vs predicted")
    fig.tight_layout(); fig.savefig(out / "fdc.png", dpi=140); plt.close(fig)


def plot_metrics_box(metrics_csv: str, out_dir: str = "outputs/figs/pred"):
    out = _ensure(out_dir)
    df = pd.read_csv(metrics_csv, dtype={"basin_id": str})
    cols = ["nse", "kge", "r"]
    labels = ["NSE", "KGE", "Pearson r"]
    data = [df[c].clip(-1, 1).values for c in cols]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55,
                    medianprops=dict(color="crimson", lw=2))
    for patch, c in zip(bp["boxes"], ["#aec7e8", "#98df8a", "#ffbb78"]):
        patch.set_facecolor(c)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("metric value (clipped to [-1, 1])")
    ax.set_title(f"Per-basin test skill distribution (n={len(df)} basins)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "metrics_boxplot.png", dpi=140); plt.close(fig)


def plot_skill_vs_attrs(metrics_csv: str, out_dir: str = "outputs/figs/pred"):
    out = _ensure(out_dir)
    df = pd.read_csv(metrics_csv, dtype={"basin_id": str}).set_index("basin_id")
    _, attrs = load_raw()
    merged = df.join(attrs, how="inner")

    panels = [
        ("aridity", "aridity (PET/P)"),
        ("area_km2", "catchment area (km²)"),
        ("mean_prcp", "mean annual precip (mm/day)"),
        ("baseflow_index", "baseflow index"),
        ("frac_snow", "snow fraction"),
        ("elev_mean", "mean elevation (m)"),
    ]
    panels = [(c, lab) for c, lab in panels if c in merged.columns]
    ncol = 3
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))
    axes = np.array(axes).ravel()
    nse_clip = merged["nse"].clip(-1, 1)
    for ax, (c, lab) in zip(axes, panels):
        x = merged[c].values
        logx = c in {"area_km2"}
        if logx:
            ax.set_xscale("log")
        sc = ax.scatter(x, nse_clip, c=nse_clip, cmap="RdYlGn",
                        vmin=-1, vmax=1, s=45, edgecolor="k", linewidth=0.4)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel(lab); ax.set_ylabel("test NSE")
        if merged[c].notna().sum() >= 2 and not logx:
            r = np.corrcoef(merged[c].values, merged["nse"].clip(-1, 1).values)[0, 1]
            ax.set_title(f"NSE vs {c} (Pearson r = {r:+.2f})")
        else:
            ax.set_title(f"NSE vs {c}")
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.suptitle("Test skill vs basin attributes")
    fig.tight_layout(); fig.savefig(out / "skill_vs_attrs.png", dpi=140); plt.close(fig)


def plot_nse_map(metrics_csv: str, out_dir: str = "outputs/figs/pred"):
    out = _ensure(out_dir)
    df = pd.read_csv(metrics_csv, dtype={"basin_id": str}).set_index("basin_id")

    _, attrs = load_raw()
    attrs.index = attrs.index.astype(str)
    merged = df.join(attrs[["lat", "lon"]], how="inner")

    fig, ax = plt.subplots(figsize=(12, 6))
    _conus_ax(ax)
    sc = ax.scatter(merged["lon"], merged["lat"], c=merged["nse"].clip(-1, 1),
                    cmap="RdYlGn", vmin=-1, vmax=1, s=70, edgecolor="k", linewidth=0.5)
    plt.colorbar(sc, ax=ax, label="Test NSE (clipped)")
    ax.set_title("Test NSE by basin")
    fig.tight_layout(); fig.savefig(out / "nse_map.png", dpi=140); plt.close(fig)
    print(f"[viz] wrote NSE map to {out}")
