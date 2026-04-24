from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader


DYN_VARS = ["prcp", "tmax", "tmin", "srad", "vp"]
TARGET = "qobs"
STATIC_FEATURES = [
    "elev_mean", "slope_mean", "area_km2",
    "mean_prcp", "mean_pet", "aridity", "frac_snow",
    "q_mean", "runoff_ratio", "hfd_mean", "baseflow_index",
    "soil_depth_pelletier", "frac_forest", "lai_max",
]


def _basin_ids(ds: xr.Dataset) -> list[str]:
    return [str(b) for b in ds["basin"].values.tolist()]


def load_raw(local_dir: str | None = None) -> tuple[xr.Dataset, pd.DataFrame]:
    from minicamels import MiniCamels
    ds = MiniCamels(local_data_dir=local_dir) if local_dir else MiniCamels()
    data = ds.load_all()
    attrs = ds.attributes()
    if "basin_id" in attrs.columns:
        attrs = attrs.set_index("basin_id")
    attrs.index = attrs.index.map(str)
    return data, attrs


def wy_bounds(start_wy: int, end_wy: int) -> tuple[str, str]:
    return f"{start_wy - 1}-10-01", f"{end_wy}-09-30"


def split_by_water_year(
    ds: xr.Dataset,
    train=(1981, 2000),
    val=(2001, 2005),
    test=(2006, 2010),
) -> dict[str, xr.Dataset]:
    return {
        "train": ds.sel(time=slice(*wy_bounds(*train))),
        "val": ds.sel(time=slice(*wy_bounds(*val))),
        "test": ds.sel(time=slice(*wy_bounds(*test))),
    }


@dataclass
class Normalizer:
    dyn_mean: np.ndarray
    dyn_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, train_ds: xr.Dataset, train_static: np.ndarray) -> "Normalizer":
        dyn = np.stack([train_ds[v].values for v in DYN_VARS], axis=-1)  # (B, T, F)
        dyn_flat = dyn.reshape(-1, dyn.shape[-1])
        dyn_mean = np.nanmean(dyn_flat, axis=0)
        dyn_std = np.nanstd(dyn_flat, axis=0) + 1e-6

        y = np.log1p(np.clip(train_ds[TARGET].values, a_min=0, a_max=None))
        y_mean = float(np.nanmean(y))
        y_std = float(np.nanstd(y) + 1e-6)

        s_mean = np.nanmean(train_static, axis=0)
        s_std = np.nanstd(train_static, axis=0) + 1e-6
        return cls(dyn_mean, dyn_std, s_mean, s_std, y_mean, y_std)

    def transform_dyn(self, x: np.ndarray) -> np.ndarray:
        return (x - self.dyn_mean) / self.dyn_std

    def transform_static(self, s: np.ndarray) -> np.ndarray:
        return (s - self.static_mean) / self.static_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (np.log1p(np.clip(y, 0, None)) - self.y_mean) / self.y_std

    def inverse_y(self, y_std: np.ndarray) -> np.ndarray:
        return np.expm1(y_std * self.y_std + self.y_mean)

    def state_dict(self):
        return {
            "dyn_mean": self.dyn_mean.tolist(),
            "dyn_std": self.dyn_std.tolist(),
            "static_mean": self.static_mean.tolist(),
            "static_std": self.static_std.tolist(),
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }

    @classmethod
    def from_state(cls, s):
        return cls(
            np.array(s["dyn_mean"]), np.array(s["dyn_std"]),
            np.array(s["static_mean"]), np.array(s["static_std"]),
            s["y_mean"], s["y_std"],
        )


def _static_matrix(attrs: pd.DataFrame, basin_ids: list[str]) -> np.ndarray:
    cols = [c for c in STATIC_FEATURES if c in attrs.columns]
    missing = set(STATIC_FEATURES) - set(cols)
    if missing:
        print(f"[data] missing static columns, skipping: {sorted(missing)}")
    df = attrs.loc[basin_ids, cols].copy()
    df = df.fillna(df.median(numeric_only=True))
    return df.values.astype(np.float32), cols


class BasinSequenceDataset(Dataset):
    def __init__(
        self,
        ds_full: xr.Dataset,
        static_arr: np.ndarray,
        normalizer: Normalizer,
        split_dates: tuple[str, str],
        seq_len: int,
    ):
        self.seq_len = seq_len
        basins = _basin_ids(ds_full)
        times = pd.to_datetime(ds_full["time"].values)

        dyn = np.stack([ds_full[v].values for v in DYN_VARS], axis=-1).astype(np.float32)  # (B, T, F)
        y = ds_full[TARGET].values.astype(np.float32)  # (B, T)

        dyn = normalizer.transform_dyn(dyn)
        static = normalizer.transform_static(static_arr).astype(np.float32)
        y_n = normalizer.transform_y(y).astype(np.float32)

        self.dyn = torch.from_numpy(dyn)
        self.static = torch.from_numpy(static)
        self.y = torch.from_numpy(y_n)
        self.y_raw = torch.from_numpy(y)
        self.basins = basins
        self.times = times

        start = pd.Timestamp(split_dates[0])
        end = pd.Timestamp(split_dates[1])
        in_split = (times >= start) & (times <= end)

        valid_ends = []
        for b in range(len(basins)):
            for t in np.where(in_split)[0]:
                if t < seq_len - 1:
                    continue
                w_start, w_end = t - seq_len + 1, t + 1
                if np.isnan(dyn[b, w_start:w_end]).any():
                    continue
                if np.isnan(y[b, t]):
                    continue
                valid_ends.append((b, int(t)))
        self.index = np.array(valid_ends, dtype=np.int64)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        b, t = self.index[i]
        x_dyn = self.dyn[b, t - self.seq_len + 1 : t + 1]
        x_static = self.static[b]
        y = self.y[b, t]
        return x_dyn, x_static, y, b, t


def build_loaders(
    seq_len: int = 365,
    batch_size: int = 256,
    num_workers: int = 0,
    local_dir: str | None = None,
    train_wy=(1981, 2000),
    val_wy=(2001, 2005),
    test_wy=(2006, 2010),
):
    ds, attrs = load_raw(local_dir)
    basins = _basin_ids(ds)
    static_arr, static_cols = _static_matrix(attrs, basins)

    train_ds = ds.sel(time=slice(*wy_bounds(*train_wy)))
    normalizer = Normalizer.fit(train_ds, static_arr)

    y_log_train = np.log1p(np.clip(train_ds[TARGET].values, 0, None))
    y_std_per_basin = (
        (y_log_train - normalizer.y_mean) / normalizer.y_std
    )
    y_std_per_basin = np.nanstd(y_std_per_basin, axis=1).astype(np.float32)

    def make(split_dates):
        return BasinSequenceDataset(ds, static_arr, normalizer, split_dates, seq_len)

    train = make(wy_bounds(*train_wy))
    val = make(wy_bounds(*val_wy))
    test = make(wy_bounds(*test_wy))

    dl = lambda d, shuffle: DataLoader(
        d, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )
    return {
        "train": dl(train, True),
        "val": dl(val, False),
        "test": dl(test, False),
        "normalizer": normalizer,
        "static_cols": static_cols,
        "basins": basins,
        "attrs": attrs,
        "ds": ds,
        "n_dyn": len(DYN_VARS),
        "n_static": static_arr.shape[1],
        "y_std_per_basin": y_std_per_basin,
    }
