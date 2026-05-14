### This works with 127.0.0.1 and localhost:
uv run uvicorn app.main:app --reload --port 8000 --host 0.0.0.0

uv run uvicorn app.main:app --reload --port 8002 --host 0.0.0.0