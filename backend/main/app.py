import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from main.routes.routes import router, set_agent_service
from main.services.agent_service import AgentService

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

agent_service = AgentService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup"""
    logger.info("Initializing MedLearn AI Assistant Agent...")
    try:
        await agent_service.initialize()
        set_agent_service(agent_service)
        logger.info("✓ Agent ready for medcal related analysis and query...")
    except Exception as e:
        logger.error(f"✗ Agent init failed: {e}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="MedLearn AI Assistant Agent",
    description="input user query for any medical related questions and analyse and answer from knowledge base",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "Medical related query anlaysis...",
        "status": "running",
        "endpoints": {
            "analyze": "/api/reviews/analyze",
            "docs": "/docs"
        }
    }
