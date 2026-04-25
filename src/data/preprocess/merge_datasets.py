import json
import os
import pandas as pd
import yaml

params = yaml.safe_load(open("params.yaml"))["preprocess"]

ENERGY_PATH = params["energy_path"]
WEATHER_HISTORY_PATH = params["weather_history_path"]
WEATHER_FORECAST_PATH = params["weather_forecast_path"]
OUTPUT_PATH = params["output_path"]


def ensure_directory(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# ENERGY → DataFrame
# ─────────────────────────────────────────────
def preprocess_energy(data: list) -> pd.DataFrame:
    df = pd.DataFrame(data)

    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.rename(columns={"Value": "energy_demand"})

    return df[["Date", "energy_demand"]]


# ─────────────────────────────────────────────
# WEATHER → DataFrame (generic)
# ─────────────────────────────────────────────
def preprocess_weather(data: dict, is_forecast: bool) -> pd.DataFrame:
    daily = data["daily"]

    df = pd.DataFrame({
        "Date": pd.to_datetime(daily["time"]).date,
        "temp_max": daily["temperature_2m_max"],
        "temp_min": daily["temperature_2m_min"],
        "daylight_duration": daily.get("daylight_duration", [None] * len(daily["time"]))
    })

    df["is_forecast"] = is_forecast

    return df


# ─────────────────────────────────────────────
# MERGE LOGIC
# ─────────────────────────────────────────────
def merge_all():
    ensure_directory(OUTPUT_PATH)

    # loading raw data
    energy_raw = load_json(ENERGY_PATH)
    weather_history_data = load_json(WEATHER_HISTORY_PATH)
    weather_forecast_data = load_json(WEATHER_FORECAST_PATH)

    # preprocess
    energy_df = preprocess_energy(energy_raw)
    weather_history_df = preprocess_weather(weather_history_data, is_forecast=False)
    weather_forecast_df = preprocess_weather(weather_forecast_data, is_forecast=True)

    # remove forecast rows that overlap with history dates
    overlap_dates = set(weather_history_df["Date"])
    weather_forecast_df = weather_forecast_df[~weather_forecast_df["Date"].isin(overlap_dates)]

    # merge
    weather_df = pd.concat([weather_history_df, weather_forecast_df], ignore_index=True)

    final_df = pd.merge(energy_df, weather_df, on="Date", how="outer")

    # SORTING BY DATE
    final_df = final_df.sort_values(by="Date")

    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"final dataset saved to {OUTPUT_PATH}")
    print(f"Shape: {final_df.shape}")


if __name__ == "__main__":
    merge_all()
