import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main.routes.routes import router, set_agent_service
from main.service.agent_service import AgentService

# Load environment variables from .env file
load_dotenv()

from main.logs.logging import configure_logging, get_logger

# Configure logging after loading environment-based logging settings.
configure_logging()

logger = get_logger(__name__)

agent_service = AgentService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup"""
    logger.info("=" * 60)
    logger.info("Initializing MedLearn AI Assistant Agent...")
    logger.info("=" * 60)
    try:
        await agent_service.initialize()
        set_agent_service(agent_service)
        logger.info("✓ Agent ready for medical related analysis and query...")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"✗ Agent init failed: {e}")
        logger.exception("Agent initialization exception trace:")
    yield
    logger.info("=" * 60)
    logger.info("Shutting down MedLearn AI Assistant...")
    logger.info("=" * 60)


app = FastAPI(
    title="MedLearn AI Assistant Agent",
    description="input user query for any medical related questions and analyse and answer from knowledge base",
    version="1.0.0",
    lifespan=lifespan
)

# Browser clients must be explicitly trusted because authenticated requests
# include an Authorization header. Override this comma-separated list in
# deployed environments; wildcard origins are intentionally not used.
cors_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)
logger.info("CORS enabled for origins: %s", cors_origins)

app.include_router(router)
logger.info(f"FastAPI app initialized with version {app.version}")


@app.get("/")
async def root():
    """Root endpoint providing service information."""
    logger.debug("Root endpoint called")
    return {
        "service": "Medical related query analysis...",
        "status": "running",
        "endpoints": {
            "analyze": "/api/reviews/analyze",
            "docs": "/docs"
        }
    }
