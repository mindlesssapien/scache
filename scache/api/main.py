import logging
import time

from fastapi import FastAPI

from scache.api.routes import router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title = "scache")
app.include_router(router)

local_time = time.localtime()
formatted = time.strftime("%Y-%m-%d %H:%M:%S", local_time).split()

@app.get("/")
def main():
    return {
            "title": "scache",
            "Description": "Scache is a semantic cache for LLm API's",
            "time": formatted[1]
            }

@app.get("/health")
def health():
    return {"status": "ok", "time": formatted[1]}
