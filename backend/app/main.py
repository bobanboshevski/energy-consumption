from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import predictions, monitoring, models, univariate, explainability

app = FastAPI(title="Energy Demand API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3002", "http://localhost:3000", "http://172.20.10.8:3000",
                   'https://harsh-election-esteemed.ngrok-free.dev'],
    # todo: i should add this in .env file
    # todo: all frontend domain in origins when deploying in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(predictions.router)
app.include_router(monitoring.router)
app.include_router(models.router)
app.include_router(univariate.router)
app.include_router(explainability.router)


@app.get("/health")
def health():
    return {"status": "ok"}
