import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from features import BAND_COLS, N_BANDS, build_table, year_features

BASE = Path(__file__).parent
META = json.loads((BASE / "meta.json").read_text())
FEATS     = META["features"]
DECISION  = META["threshold"]
STORM_THR = META["storm_threshold"]
HORIZON   = META["horizon_sols"]
BAND_LAT  = META["band_lat"]
TRAIN_MY  = set(META["trained_on_MY"])

booster = xgb.Booster()
booster.load_model(str(BASE / "model.json"))


def score(df):
    return booster.predict(xgb.DMatrix(df[FEATS].astype(float)))


# Precompute features + risk for every sol of MY28-32 at startup
RAW = pd.read_csv(BASE / "band_dust.csv")
TABLE = build_table(RAW, BAND_LAT, STORM_THR, HORIZON)
TABLE["risk"] = score(TABLE)

INFO = {
    "horizon_sols": HORIZON,
    "storm_definition": f"dust covers more than {STORM_THR*100:.0f}% of the latitude band",
    "decision_threshold": DECISION,
    "note": "risk is a model score (0-1), not a calibrated probability",
}

app = FastAPI(title="Mars Dust Storm Risk API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _num(x):
    return None if pd.isna(x) else float(x)


def make_payload(rows, with_actual=False):
    bands, out = [], {}
    for _, r in rows.sort_values("band").iterrows():
        b = int(r["band"])
        observed = not pd.isna(r["cur"])
        risk = float(r["risk"]) if observed else None
        item = {"band": b, "lat": round(float(BAND_LAT[b]), 1), "dust_now": _num(r["cur"]),
                "risk": risk, "storm_likely": (risk >= DECISION) if observed else None}
        if with_actual:
            item["actual_storm"] = None if pd.isna(r["y"]) else bool(r["y"])
        bands.append(item)
        out[f"risk_b{b}"] = risk          # flat keys are easy to read in GDevelop
        out[f"dust_b{b}"] = item["dust_now"]
    risks = [x["risk"] for x in bands if x["risk"] is not None]
    out["max_risk"] = max(risks) if risks else None
    out["any_storm_likely"] = any(x["storm_likely"] for x in bands if x["storm_likely"] is not None)
    out["bands"] = bands
    return out


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", **INFO}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/replay/info")
def replay_info():
    obs = TABLE[TABLE["cur"].notna()]
    return {int(my): {"first_soy": int(g["soy"].min()), "last_soy": int(g["soy"].max()),
                      "observed_sols": int(g["soy"].nunique()), "in_sample": int(my) in TRAIN_MY}
            for my, g in obs.groupby("MY")}


@app.get("/replay/sols")
def replay_sols(my: int = Query(32)):
    g = TABLE[(TABLE["MY"] == my) & TABLE["cur"].notna()]
    if g.empty:
        raise HTTPException(404, "Unknown Mars Year. Valid: 28-32.")
    return {"MY": my, "sols": sorted(int(x) for x in g["soy"].unique())}


@app.get("/forecast")
def forecast(soy: int = Query(..., ge=0, le=669), my: int = Query(32)):
    """Replay a real sol from MARCI data: risk for the next HORIZON sols vs what actually happened."""
    rows = TABLE[(TABLE["MY"] == my) & (TABLE["soy"] == soy)]
    if rows.empty or rows["cur"].isna().all():
        raise HTTPException(404, "No MARCI observation for that sol. See /replay/sols.")
    return {**INFO, "mode": "replay", "MY": my, "soy": soy,
            "ls": round(float(rows["ls"].iloc[0]), 1),
            "in_sample": my in TRAIN_MY,          # MY32 = held-out
            **make_payload(rows, with_actual=True)}


class HistoryIn(BaseModel):
    ls: float = Field(..., ge=0, le=360, description="Solar longitude (Ls) of the most recent sol")
    history: List[List[Optional[float]]] = Field(
        ..., min_length=8, max_length=60,
        description="Oldest -> newest. Each row = dust fraction (0-1) for the 8 latitude bands "
                    "(south -> north). Use null for unobserved. 14+ sols recommended.")


@app.post("/predict")
def predict(req: HistoryIn):
    """Custom scenario: send your own recent dust history."""
    for row in req.history:
        if len(row) != N_BANDS:
            raise HTTPException(422, f"Each history row needs exactly {N_BANDS} values.")
    g = pd.DataFrame(req.history, columns=BAND_COLS).astype(float)
    f = year_features(g, np.full(len(g), req.ls), BAND_LAT, STORM_THR, HORIZON)
    last = f[f["sol"] == len(g) - 1].copy()
    if last["cur"].isna().all():
        raise HTTPException(422, "The most recent sol has no observed bands.")
    last["risk"] = score(last)
    return {**INFO, "mode": "custom", "ls": req.ls, **make_payload(last)}
