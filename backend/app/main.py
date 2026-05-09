from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import predictions, monitoring, models, univariate

app = FastAPI(title="Energy Demand API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3002", "http://localhost:3000"],  # todo: i should add this in .env file
    # todo: all frontend domain in origins when deploying in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(predictions.router)
app.include_router(monitoring.router)
app.include_router(models.router)
app.include_router(univariate.router)


@app.get("/health")
def health():
    return {"status": "ok"}
