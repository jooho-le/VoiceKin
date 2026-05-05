import logging
import sys

from fastapi import FastAPI

from app.api.routes.voice import router as voice_router
from app.core.config import get_settings


def configure_logging() -> None:
    """Configure simple stdout logging for local development and servers."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


configure_logging()
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="VoiceKin AI model based speaker verification API server.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint for load balancers and simple server checks."""

    return {"status": "ok"}


app.include_router(voice_router, prefix=settings.api_v1_prefix)
