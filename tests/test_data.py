import numpy as np
import pandas as pd
import xarray as xr
import pytest

from camelflow.data import Normalizer, BasinSequenceDataset, wy_bounds, DYN_VARS, TARGET


def _fake_ds(n_basins=3, n_days=400):
    rng = np.random.default_rng(0)
    time = pd.date_range("1999-10-01", periods=n_days, freq="D")
    basins = [f"b{i:02d}" for i in range(n_basins)]
    data = {v: (("basin", "time"), rng.normal(size=(n_basins, n_days)).astype("float32"))
            for v in DYN_VARS}
    data[TARGET] = (("basin", "time"), np.abs(rng.normal(size=(n_basins, n_days))).astype("float32"))
    return xr.Dataset(data, coords={"basin": basins, "time": time})


def test_wy_bounds():
    assert wy_bounds(1981, 1981) == ("1980-10-01", "1981-09-30")
    assert wy_bounds(1981, 2000) == ("1980-10-01", "2000-09-30")


def test_dataset_shape():
    ds = _fake_ds()
    static = np.random.default_rng(0).normal(size=(3, 4)).astype("float32")
    nz = Normalizer.fit(ds, static)

    d = BasinSequenceDataset(ds, static, nz, ("1999-10-01", "2000-12-31"), seq_len=30)
    assert len(d) > 0
    x_dyn, x_static, y, b, t = d[0]
    assert x_dyn.shape == (30, 5)
    assert x_static.shape == (4,)
    assert y.ndim == 0


def test_no_time_leak():
    ds = _fake_ds(n_days=800)
    static = np.zeros((3, 2), dtype="float32")
    nz = Normalizer.fit(ds.sel(time=slice("1999-10-01", "2000-09-30")), static)

    train = BasinSequenceDataset(ds, static, nz, wy_bounds(2000, 2000), seq_len=30)
    test = BasinSequenceDataset(ds, static, nz, wy_bounds(2001, 2001), seq_len=30)

    train_end_dates = {pd.Timestamp(ds["time"].values[t]) for _, t in train.index}
    test_end_dates = {pd.Timestamp(ds["time"].values[t]) for _, t in test.index}
    assert train_end_dates.isdisjoint(test_end_dates)
