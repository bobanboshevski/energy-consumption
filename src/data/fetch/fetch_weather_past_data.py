# import os
# import json
# from datetime import datetime
# import requests
#
# FILE_PATH = "data/raw/weather/history/weather_history.json"
# BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
#
# LAT = 46.05
# LON = 14.50
#
#
# # ─────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────
# def ensure_directory(path: str):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#
#
# def load_existing_data(path: str):
#     if os.path.exists(path):
#         print(f"File {path} exists. Loading data...")
#         with open(path, "r") as file:
#             return json.load(file)
#     else:
#         print(f"File {path} does not exist. Starting fresh...")
#         return None  # []
#
#
# def get_from_date(existing_data: dict) -> str:
#     if existing_data:
#         last_date_str = existing_data["daily"]["time"][-1]
#         return last_date_str
#     else:
#         return "2020-01-01"
#
#
# def get_today_date() -> str:
#     return datetime.now().strftime("%Y-%m-%d")
#
#
# def save_data(path: str, data: dict):
#     with open(path, "w") as file:
#         json.dump(data, file, indent=4)
#
#
# # ─────────────────────────────────────────────
# # Fetch
# # ─────────────────────────────────────────────
#
# def fetch_weather_data(from_date: str, to_date: str) -> dict:
#     url = (
#         f"{BASE_URL}"
#         f"?latitude={LAT}&longitude={LON}"
#         f"&start_date={from_date}&end_date={to_date}"
#         f"&daily=temperature_2m_max,temperature_2m_min"
#         f"&timezone=auto"
#     )
#     print(f"Fetching weather data for {from_date} to {to_date}...")
#
#     response_json = requests.get(url)
#     response_json.raise_for_status()
#
#     response_json = response_json.json()
#     response_len = len(response_json["daily"]["time"])
#
#     print(f"Length of response_json: {response_len}")
#
#     return response_json
#
#
# # ─────────────────────────────────────────────
# # Deduplication
# # ─────────────────────────────────────────────
# # we can actually avoid this checking by adding +1 when getting new data.
# def merge_weather_data(existing_data: dict, api_response: dict) -> dict:
#     if not existing_data:
#         return api_response
#
#     existing_dates = set(existing_data["daily"]["time"])
#     new_dates = api_response["daily"]["time"]
#
#     # finding indices of new dates only!
#     indices_to_add = [i for i, d in enumerate(new_dates) if d not in existing_dates]
#
#     print(f"New records to add: {len(indices_to_add)}")
#
#     # append values for each key in daily
#     for key in existing_data["daily"]:
#         if key == "time":
#             continue
#
#         existing_data["daily"][key].extend(
#             [api_response["daily"][key][i] for i in indices_to_add]
#         )
#
#     existing_data["daily"]["time"].extend(
#         [new_dates[i] for i in indices_to_add]
#     )
#
#     return existing_data
#
#
# # ─────────────────────────────────────────────
# # Main pipeline
# # ─────────────────────────────────────────────
# def fetch_weather_history():
#     try:
#         ensure_directory(FILE_PATH)
#
#         existing_data = load_existing_data(FILE_PATH)
#         from_date = get_from_date(existing_data)
#         to_date = get_today_date()
#
#         api_response = fetch_weather_data(from_date, to_date)
#
#         updated_data = merge_weather_data(existing_data, api_response)
#
#         save_data(FILE_PATH, updated_data)
#
#         print(f"Weather history for {from_date} to {to_date} saved to {FILE_PATH}...")
#     except requests.RequestException as e:
#         print(f"Error fetch history weather data: {e}")
#
#
# if __name__ == "__main__":
#     fetch_weather_history()


from datetime import datetime

from src.data.utils.weather_utils import ensure_directory, load_existing_data, fetch_json, save_data

FILE_PATH = "data/raw/weather/history/weather_history.json"
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

LAT = 46.05
LON = 14.50


def get_from_date(existing_data):
    if existing_data:
        return existing_data["daily"]["time"][-1]
    return "2020-01-01"


def get_today():
    return datetime.now().strftime("%Y-%m-%d")


def build_url(from_date, to_date):
    return (
        f"{BASE_URL}"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={from_date}&end_date={to_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,daylight_duration"
        f"&timezone=auto"
    )


def merge(existing_data, new_data):
    if not existing_data:
        return new_data

    print(f"Existing data: {len(existing_data['daily']['time'])}")

    existing_dates = set(existing_data["daily"]["time"])
    new_dates = new_data["daily"]["time"]

    print(f"Merging {len(new_dates)} new dates")

    indices = [i for i, d in enumerate(new_dates) if d not in existing_dates]

    for key in existing_data["daily"]:
        if key == "time":
            continue

        existing_data["daily"][key].extend(
            [new_data["daily"][key][i] for i in indices]
        )

    existing_data["daily"]["time"].extend(
        [new_dates[i] for i in indices]
    )

    return existing_data


def main():
    ensure_directory(FILE_PATH)

    existing = load_existing_data(FILE_PATH)
    from_date = get_from_date(existing)
    to_date = get_today()

    url = build_url(from_date, to_date)
    print(f"Fetching history: {from_date} → {to_date}")

    new_data = fetch_json(url)
    updated = merge(existing, new_data)

    save_data(FILE_PATH, updated)
    print("History updated.")


if __name__ == "__main__":
    main()
