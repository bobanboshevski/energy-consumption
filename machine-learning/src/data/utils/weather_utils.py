import os
import json
import requests


def ensure_directory(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_existing_data(path: str):
    if os.path.exists(path):
        print(f"File {path} exists. Loading data...")
        with open(path, "r") as file:
            return json.load(file)
    else:
        print(f"File {path} does not exist.")
        return None


def save_data(path: str, data: dict):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def fetch_json(url: str) -> dict:
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
