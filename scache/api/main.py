import logging
import time

from fastapi import FastAPI

from api import routes

logging.basicConfig(level=logging.INFO)

app = FastAPI(title = "scache")
app.include_router()

@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}
