from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from app.config import CORS_ORIGINS
from app.routes.catalog import router as catalog_router
from app.routes.health import router as health_router
from app.routes.jobs import router as jobs_router
from app.routes.library import router as library_router

app = FastAPI(title="Spectrogram Viewer API")

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(jobs_router)
app.include_router(library_router)
app.include_router(catalog_router)
