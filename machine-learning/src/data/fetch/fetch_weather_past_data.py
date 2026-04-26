from datetime import datetime
from src.data.utils.weather_utils import ensure_directory, load_existing_data, fetch_json, save_data
import yaml

params = yaml.safe_load(open("params.yaml"))["fetch"]["weather_history"]

FILE_PATH = params["file_path"]
BASE_URL = params["url"]
LAT = params["lat"]
LON = params["lon"]
INITIAL_DATE = params["initial_date"]


def get_from_date(existing_data):
    if existing_data:
        return existing_data["daily"]["time"][-1]
    return INITIAL_DATE


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
