from src.data.utils.weather_utils import (
    ensure_directory,
    save_data,
    fetch_json
)

FILE_PATH = "data/raw/weather/forecast/weather_forecast.json"
BASE_URL = "https://api.open-meteo.com/v1/forecast"

LAT = 46.05
LON = 14.50


def build_url():
    return (
        f"{BASE_URL}"
        f"?latitude={LAT}&longitude={LON}"
        f"&daily=temperature_2m_max,temperature_2m_min,daylight_duration"
        f"&timezone=auto"
        f"&forecast_days=16"
    )


def main():
    ensure_directory(FILE_PATH)

    url = build_url()
    print("Fetching forecast data...")

    data = fetch_json(url)

    forecast_data_len = len(data["daily"]["time"])
    print(f"Saving data: {forecast_data_len} entries.")

    # overwrite file every time
    save_data(FILE_PATH, data)

    print(f"Forecast saved to {FILE_PATH}")


if __name__ == "__main__":
    main()
