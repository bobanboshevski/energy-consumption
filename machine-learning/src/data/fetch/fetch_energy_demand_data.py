import os
import json
from datetime import datetime
import requests
import yaml

params = yaml.safe_load(open("params.yaml"))["fetch"]["energy_demand"]

FILE_PATH = params["file_path"]
BASE_URL = params["url"]
INITIAL_DATE = params["initial_date"]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def ensure_directory(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def get_today_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def load_existing_data(path: str):
    if os.path.exists(path):
        print(f"File {path} exists. Loading data...")
        with open(path, "r") as file:
            return json.load(file)
    else:
        print(f"File {path} does not exist. Starting fresh...")
        return []


def get_from_date(existing_data: list) -> str:
    if existing_data:
        last_date_str = existing_data[-1]["Date"]
        last_date = datetime.fromisoformat(last_date_str.replace("Z", ""))
        return last_date.strftime("%Y-%m-%d")
    else:
        return INITIAL_DATE


def fetch_data(from_date: str, to_date: str) -> list:
    url = f"{BASE_URL}?from={from_date}&to={to_date}&precision=day"
    print(f"Fetching data from {from_date} to {to_date}...")

    response = requests.get(url)
    response.raise_for_status()

    return response.json()


def remove_duplicates(existing_data: list, new_data: list) -> list:
    if not existing_data:
        return new_data

    existing_dates = {item["Date"] for item in existing_data}
    return [item for item in new_data if item["Date"] not in existing_dates]


def save_data(path: str, data: list):
    with open(path, "w") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


# ─────────────────────────────────────────────
# Main pipeline function
# ─────────────────────────────────────────────

def fetch_energy_demand_data():
    try:
        ensure_directory(FILE_PATH)

        existing_data = load_existing_data(FILE_PATH)
        from_date = get_from_date(existing_data)
        to_date = get_today_date()

        new_data = fetch_data(from_date, to_date)
        new_data = remove_duplicates(existing_data, new_data)

        updated_data = existing_data + new_data

        save_data(FILE_PATH, updated_data)

        print(f"Added {len(new_data)} new records.")
        print(f"Total records: {len(updated_data)}")
        print(f"Saved to {FILE_PATH} at {datetime.now()}")

    except requests.RequestException as e:
        print(f"Error fetching data: {e}")


# ─────────────────────────────────────────────

if __name__ == "__main__":
    fetch_energy_demand_data()
