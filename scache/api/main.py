import logging
import time

from fastapi import FastAPI

from scache.api.metrics import metrics_endpoint
from scache.api.routes import router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="scache")
app.include_router(router)


@app.get("/")
def main():
    local_time = time.localtime()
    formatted = time.strftime("%Y-%m-%d %H:%M:%S", local_time).split()
    return {
        "title": "scache",
        "description": "scache is a semantic cache for LLM APIs",
        "time": formatted[1],
    }


@app.get("/health")
def health():
    local_time = time.localtime()
    formatted = time.strftime("%Y-%m-%d %H:%M:%S", local_time).split()
    return {"status": "ok", "time": formatted[1]}


@app.get("/metrics")
def metrics():
    return metrics_endpoint()
