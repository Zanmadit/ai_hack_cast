from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import json

DATA_DIR = "data"
MODEL_DIR = "models"
FEATURES = [
    ("all_ai_models.csv", "Citations"),
    ("all_ai_models.csv", "Parameters"),
    ("all_ai_models.csv", "Training compute (FLOP)"),
    ("frontier_ai_models.csv", "Parameters"),
    ("frontier_ai_models.csv", "Training compute cost (2023 USD)"),
    ("frontier_ai_models.csv", "Training compute (FLOP)"),
    ("gpu_clusters.csv", "Calculated Cost"),
    ("gpu_clusters.csv", "Calculated Power Capacity (MW)"),
    ("gpu_clusters.csv", "Hardware Cost"),
    ("gpu_clusters.csv", "% of largest cluster when first operational"),
    ("gpu_clusters.csv", "Power Capacity (MW)"),
    ("gpu_clusters.csv", "Rank when first operational"),
    ("large_scale_ai_models.csv", "Citations"),
    ("large_scale_ai_models.csv", "Parameters"),
    ("large_scale_ai_models.csv", "Training compute (FLOP)"),
    ("ml_hardware.csv", "Max performance"),
    ("ml_hardware.csv", "TDP (W)"),
    ("notable_ai_models.csv", "Citations"),
    ("notable_ai_models.csv", "Parameters"),
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для локальной разработки
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Вспомогательные функции -----------------
def safe_column_match(df: pd.DataFrame, target: str):
    import re
    norm = lambda s: re.sub(r"[\W_]+", "", s.lower())
    tgt = norm(target)
    for c in df.columns:
        if norm(c) == tgt:
            return c
    for c in df.columns:
        if tgt in norm(c) or norm(c) in tgt:
            return c
    return None

def detect_date_column(df: pd.DataFrame):
    for c in df.columns:
        if any(k in c.lower() for k in ("date", "year", "published", "operational")):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() > 5:
                return c
    if "year" in df.columns:
        return "year"
    return None

def aggregate_by_year(path: str, metric_col: str, date_col: str, min_year=2010):
    df = pd.read_csv(path, low_memory=False)
    metric_actual = safe_column_match(df, metric_col)
    if metric_actual is None:
        raise KeyError(f"Metric {metric_col} not found in {path}")
    if date_col == "year":
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["year"] = df[date_col].dt.year
    df = df.dropna(subset=["year", metric_actual])
    df["year"] = df["year"].astype(int)
    df = df[df["year"] >= min_year]
    df["value"] = pd.to_numeric(df[metric_actual].astype(str).str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=["value"])
    agg = df.groupby("year")["value"].median().reset_index().sort_values("year")
    return agg

# ----------------- Endpoint для графиков -----------------
@app.get("/graphs")
def get_graphs():
    response = []
    for file, metric in FEATURES:
        path = os.path.join(DATA_DIR, file)
        if not os.path.exists(path):
            continue

        try:
            df = aggregate_by_year(path, metric, detect_date_column(pd.read_csv(path)))
            years = df["year"]
            values = df["value"]

            # === График истории ===
            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(x=years, y=values, mode="lines+markers", name="History"))
            fig_history.update_layout(
                title=f"{metric} — {file}",
                xaxis_title="Year",
                yaxis_title=metric,
                template="plotly_white"
            )

            fig_forecast = go.Figure(fig_history)  # копия
            # === Пробуем добавить предсказания ===
            model_path = os.path.join(MODEL_DIR, f"{file.replace('.csv','')}_{metric.replace(' ','_')}.keras")
            scaler_path = os.path.join(MODEL_DIR, f"{file.replace('.csv','')}_{metric.replace(' ','_')}_scaler.pkl")

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = tf.keras.models.load_model(model_path)
                scaler = joblib.load(scaler_path)

                window = 3
                vals_log = np.log1p(values).values.reshape(-1, 1)
                scaled = scaler.transform(vals_log).ravel()
                sim_buffer = list(scaled[-window:])
                preds_scaled = []
                n_future = 5
                for _ in range(n_future):
                    x_in = np.array(sim_buffer[-window:]).reshape(1, window, 1)
                    p = model.predict(x_in, verbose=0).ravel()[0]
                    preds_scaled.append(p)
                    sim_buffer.append(p)

                preds_vals = np.expm1(scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel())
                future_years = np.arange(years.iloc[-1] + 1, years.iloc[-1] + 1 + n_future)

                fig_forecast.add_trace(go.Scatter(
                    x=future_years,
                    y=preds_vals,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(dash="dot", color="orange")
                    
                ))

            response.append({
                "file": file,
                "metric": metric,
                "hist_graph": json.dumps(fig_history, cls=PlotlyJSONEncoder),
                "forecast_graph": json.dumps(fig_forecast, cls=PlotlyJSONEncoder)
            })

        except Exception as e:
            print(f"[SKIP] {file}::{metric} — {e}")
            continue

    return response

