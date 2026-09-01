import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from main.routes.routes import router, set_agent_service
from main.service.agent_service import AgentService
from main.logs.logging import configure_logging, get_logger

# Configure logging on startup
configure_logging()

# Load environment variables from .env file
load_dotenv()

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
