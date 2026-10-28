"""Aegis — web app for Project Sentinel. Serves the real trained model (artifacts/sentinel_model.joblib)."""
import io, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
app = FastAPI(title="Aegis — Project Sentinel")

MODEL_KEYS = {"model", "features", "threshold"}


def load_artifacts():
    try:
        artifact = joblib.load(BASE / "artifacts" / "sentinel_model.joblib")
        if not isinstance(artifact, dict):
            raise ValueError("Artifact root must be a dictionary.")
        missing = sorted(MODEL_KEYS - set(artifact.keys()))
        if missing:
            raise ValueError(f"Artifact missing required keys: {', '.join(missing)}")
        card = json.loads((BASE / "artifacts" / "model_card.json").read_text())
        return artifact, card, None
    except Exception as e:  # surfaced in the UI, never replaced by fake results
        return None, None, f"{type(e).__name__}: {e}"


ART, CARD, LOAD_ERROR = load_artifacts()

LD_KM = 384_400
# Ranges seen in the training data (slightly widened). Inputs outside them are rejected.
FEATURES = [
    dict(name="absolute_magnitude", label="Absolute magnitude (H)", unit="", min=9, max=34, default=20,
         help="Brightness of the object. Lower = bigger. H≈22 is roughly 140 m across."),
    dict(name="relative_velocity", label="Relative velocity", unit="km/h", min=100, max=250000, default=60000,
         help="Speed of the object relative to Earth during the flyby."),
    dict(name="miss_distance", label="Miss distance", unit="km", min=1000, max=80_000_000, default=20_000_000,
         help="How close the object passes Earth. The Moon is about 384,400 km away."),
]
NAMES = [f["name"] for f in FEATURES]

def featurize(df: pd.DataFrame) -> pd.DataFrame:
    # Same transformations as training. Diameter uses the albedo (0.05) behind the dataset's est_diameter_max.
    d = 1329 / np.sqrt(0.05) * 10 ** (-df.absolute_magnitude / 5)
    out = pd.DataFrame({"absolute_magnitude": df.absolute_magnitude,
                        "log_velocity": np.log10(df.relative_velocity),
                        "log_miss_distance": np.log10(df.miss_distance),
                        "log_size_dist_ratio": np.log10(d / df.miss_distance)})
    return out[ART["features"]]

def score(df):
    if ART is None:
        raise HTTPException(503, f"Model could not be loaded. Check artifacts/ folder. ({LOAD_ERROR})")
    df = df.copy()
    missing = [c for c in ["absolute_magnitude", "relative_velocity", "miss_distance"] if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing required prediction columns: {', '.join(missing)}")
    p = ART["model"].predict_proba(featurize(df.reindex(columns=["absolute_magnitude", "relative_velocity", "miss_distance"])))[:, 1]
    if not np.all(np.isfinite(p)):
        raise HTTPException(500, "Model returned invalid output.")
    return p

def label(p): return "Hazardous" if p >= ART["threshold"] else "Non-hazardous"
def diam_m(H): return float(1329 / np.sqrt(0.05) * 10 ** (-H / 5) * 1000)

class Asteroid(BaseModel):
    absolute_magnitude: float = Field(ge=9, le=34)
    relative_velocity: float = Field(ge=100, le=250000)
    miss_distance: float = Field(ge=1000, le=80_000_000)

class Sens(Asteroid):
    feature: str

@app.get("/api/health")
def health():
    return dict(status="ok", ready=ART is not None, artifact_path=str((BASE / 'artifacts' / 'sentinel_model.joblib')), error=LOAD_ERROR)


@app.get("/api/info")
def info():
    imp = {}
    if ART is not None:
        m = ART["model"]; imp = dict(zip(ART["features"], map(float, m.feature_importances_)))
    return dict(ready=ART is not None, error=LOAD_ERROR, features=FEATURES, threshold=ART and ART["threshold"],
                metrics=(CARD["test_metrics"][1] if CARD else None), importances=imp)

@app.post("/api/predict")
def predict(a: Asteroid):
    p = float(score(pd.DataFrame([a.model_dump()]))[0])
    return dict(label=label(p), score=p, diameter_m=diam_m(a.absolute_magnitude),
                miss_ld=a.miss_distance / LD_KM, threshold=ART["threshold"])

@app.post("/api/sensitivity")
def sensitivity(s: Sens):
    if s.feature not in NAMES: raise HTTPException(400, f"Unsupported feature: {s.feature}")
    grid = {"absolute_magnitude": np.linspace(14, 30, 80), "relative_velocity": np.geomspace(1000, 240000, 80),
            "miss_distance": np.geomspace(1e5, 7.5e7, 80)}[s.feature]
    df = pd.DataFrame([s.model_dump()] * len(grid)); df[s.feature] = grid
    return dict(x=grid.tolist(), y=score(df).tolist(), threshold=ART["threshold"])

@app.get("/api/template")
def template():
    csv = "absolute_magnitude,relative_velocity,miss_distance\n17.5,90000,5000000\n20,60000,20000000\n26,40000,50000000\n"
    return Response(csv, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=aegis_template.csv"})

@app.post("/api/batch")
async def batch(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw.strip(): raise HTTPException(400, "The file is empty.")
    try: df = pd.read_csv(io.BytesIO(raw))
    except Exception: raise HTTPException(400, "Could not read this file as CSV.")
    df.columns = [c.strip().lower() for c in df.columns]
    miss = [c for c in NAMES if c not in df.columns]
    if miss: raise HTTPException(400, f"Missing columns: {', '.join(miss)}. Download the template to see the format.")
    if len(df) == 0: raise HTTPException(400, "The file has headers but no rows.")
    if len(df) > 20000: raise HTTPException(400, "Too many rows (max 20,000).")
    num = df[NAMES].apply(pd.to_numeric, errors="coerce")
    ok = num.notna().all(axis=1)
    for f in FEATURES: ok &= num[f["name"]].between(f["min"], f["max"])
    out = df.copy(); out["classification"] = "Invalid row"; out["hazard_score"] = np.nan
    if ok.any():
        p = score(num[ok])
        out.loc[ok, "hazard_score"] = p.round(4)
        out.loc[ok, "classification"] = [label(x) for x in p]
    out = out.replace({np.nan: None})
    vc = out["classification"].value_counts()
    return dict(summary=dict(total=len(out), hazardous=int(vc.get("Hazardous", 0)),
                             non_hazardous=int(vc.get("Non-hazardous", 0)), invalid=int(vc.get("Invalid row", 0))),
                rows=out.to_dict("records"))

app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
