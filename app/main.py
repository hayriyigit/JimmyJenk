import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app import engine
from app.backends import BackendError, reachable
from app.registry import MODELS, backend_for, get_model
from app.schemas import SystemOneRequest, SystemOneResponse

app = FastAPI(title="Open System-One")


@app.post("/v1/systemone", response_model=SystemOneResponse)
async def systemone(req: SystemOneRequest):
    try:
        cfg = get_model(req.model)
    except KeyError as e:
        raise HTTPException(404, str(e))
    try:
        return await engine.run(req, cfg)
    except engine.UnsupportedQuestion as e:
        raise HTTPException(422, str(e))
    except BackendError as e:
        raise HTTPException(502, str(e))


@app.get("/v1/models")
async def models():
    urls = sorted({cfg.url for cfg in MODELS.values()})
    online = dict(zip(urls, await asyncio.gather(*map(reachable, urls))))
    return [cfg.model_dump() | {"capabilities": vars(backend_for(cfg).capabilities), "url": cfg.url, "online": online[cfg.url]}
            for cfg in MODELS.values()]


@app.get("/")
def playground():
    return FileResponse(Path(__file__).parent / "playground.html")
