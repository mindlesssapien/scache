import logging
import time

from scache.api.routes import router
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)

app = FastAPI(title = "scache")
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}
