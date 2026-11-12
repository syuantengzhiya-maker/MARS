"""Feature engineering shared by training (Colab) and serving (Render).
Do NOT change without retraining the model."""
import numpy as np
import pandas as pd

N_BANDS = 8
LAGS = [1, 2, 3, 5, 7]
BAND_COLS = [f"b{b}" for b in range(N_BANDS)]


def soy_to_ls(soy):
    """Sol-of-year to Ls (approximate, via Kepler's equation)."""
    e, P = 0.0934, 668.6
    nu0 = np.deg2rad(0 - 251.0)
    E0 = 2*np.arctan2(np.sqrt(1-e)*np.sin(nu0/2), np.sqrt(1+e)*np.cos(nu0/2))
    M = E0 - e*np.sin(E0) + 2*np.pi*np.asarray(soy, float)/P
    E = M.copy()
    for _ in range(30):
        E = E - (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
    nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    return (np.rad2deg(nu) + 251) % 360


def year_features(g, ls, band_lat, thr, horizon):
    """g: DataFrame indexed by sol (consecutive ints), columns b0..b7 = dust fraction.
    ls: array of Ls values (same length as g). Returns one row per (sol, band)."""
    B = g[BAND_COLS]
    glob_ = B.mean(axis=1)
    south = B[BAND_COLS[:N_BANDS // 2]].mean(axis=1)
    north = B[BAND_COLS[N_BANDS // 2:]].mean(axis=1)
    idx = np.arange(len(g))
    frames = []
    for b in range(N_BANDS):
        s = g[f"b{b}"]
        d = pd.DataFrame({"sol": g.index.values, "band": b, "band_lat": band_lat[b], "ls": ls,
                          "ls_sin": np.sin(np.deg2rad(ls)), "ls_cos": np.cos(np.deg2rad(ls)),
                          "cur": s.values, "global": glob_.values,
                          "south_mean": south.values, "north_mean": north.values})
        for L in LAGS:
            d[f"lag{L}"] = s.shift(L).values
        d["mean7"]   = s.rolling(7, min_periods=3).mean().values
        d["max7"]    = s.rolling(7, min_periods=3).max().values
        d["trend3"]  = (s - s.shift(3)).values
        d["growth1"] = (s - s.shift(1)).values
        d["ratio3"]  = (s / (s.shift(3) + 1e-4)).values
        d["global_lag3"]   = glob_.shift(3).values
        d["global_trend3"] = (glob_ - glob_.shift(3)).values

        active = (s > thr).astype(float).where(s.notna())
        d["active_days7"]  = active.rolling(7,  min_periods=3).sum().values
        d["active_days14"] = active.rolling(14, min_periods=5).sum().values
        last = pd.Series(np.where(active == 1, idx, np.nan)).ffill()
        d["days_since"] = (idx - last).values

        for nb, name in [(b-1, "south"), (b+1, "north")]:
            if 0 <= nb < N_BANDS:
                d[f"nb_{name}"]      = g[f"b{nb}"].values
                d[f"nb_{name}_lag3"] = g[f"b{nb}"].shift(3).values
            else:
                d[f"nb_{name}"] = np.nan
                d[f"nb_{name}_lag3"] = np.nan

        # Label (only meaningful for historical data): storm in next `horizon` sols?
        fut = pd.concat([s.shift(-h) for h in range(1, horizon + 1)], axis=1)
        fm = fut.max(axis=1, skipna=True).values
        d["y"] = np.where(np.isnan(fm) | s.isna().values, np.nan, (fm > thr).astype(float))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def build_table(raw, band_lat, thr, horizon):
    """raw: DataFrame with columns MY, soy, b0..b7 (the band_dust.csv from Colab)."""
    out = []
    for my, g in raw.groupby("MY"):
        g = g.set_index("soy").reindex(range(0, 670))
        ls = soy_to_ls(g.index.values)
        f = year_features(g[BAND_COLS], ls, band_lat, thr, horizon)
        f["MY"] = int(my)
        f["soy"] = f["sol"]
        out.append(f)
    return pd.concat(out, ignore_index=True)
