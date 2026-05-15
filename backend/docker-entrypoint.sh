#!/bin/sh
set -e

echo "=== Energy Forecast Backend ==="

if [ ! -f "/app/active_model.json" ]; then
    echo "→ Initialising active_model.json with defaults..."
    cat > /app/active_model.json <<'EOF'
{
  "multivariate": {
    "active_version": "latest",
    "loaded_version": null
  },
  "univariate": {
    "active_version": "latest",
    "loaded_version": null
  }
}
EOF
fi

echo "→ Starting FastAPI server..."
exec /opt/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8002 --workers 1