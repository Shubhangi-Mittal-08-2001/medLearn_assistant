import logging
import re
from typing import Dict, Any, Optional
from main.service.hybrid_query_service import HybridQueryService, HybridQueryResult
from llama_index.core.workflow import Context
from main.logs.logging import get_logger

logger = get_logger(__name__)


class AgentService:
    """Application façade for the hybrid SQL/vector/BM25/MCP agent."""
    
    def __init__(self):
        self.hybrid_service = HybridQueryService()
        self.agent = None
        self.tools = []
        self._initialized = False
        self._mcp_connected = False
        self._context = None
        logger.debug("AgentService instance created")
    
    async def initialize(self):
        """Initialize MCP connection and agent"""
        if self._initialized:
            logger.info("Agent already initialized")
            return
        
        logger.info("Initializing hybrid query agent service...")
        await self.hybrid_service.initialize()
        self.agent = self.hybrid_service.agent
        self.tools = self.hybrid_service.tools
        self._initialized = True
        self._mcp_connected = not any(e.name == "huggingface_mcp" for e in self.hybrid_service.events)
        logger.info("Hybrid agent service initialized with %d available tools", len(self.tools))
    
    def is_initialized(self) -> bool:
        """Check if agent is initialized"""
        return self._initialized and self.agent is not None
    
    def is_mcp_connected(self) -> bool:
        """Check if MCP server is connected"""
        return self._mcp_connected
    
    def get_tools_count(self) -> int:
        """Get number of loaded tools"""
        return len(self.tools)

    async def execute_query(self, question: str) -> HybridQueryResult:
        """Execute a medical query and retain routing and per-tool failure metadata."""
        return await self.hybrid_service.execute(question)
    
    async def analyze_sentiment(self, text: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Analyze sentiment using MCP workflow.
        
        Args:
            text: Text to analyze
            verbose: Show detailed agent reasoning
            
        Returns:
            Dict with sentiment, confidence, model_used, and success status
        """
        if not self.is_initialized():
            raise RuntimeError("Agent service not initialized")
        
        prompt = f"""
        Analyze the sentiment of this text: "{text}"
        
        Task:
        1. Use 'model_search' MCP tool to find sentiment-analysis models sorted by downloads
        2. Select the MOST DOWNLOADED model
        3. Use 'hf_inference' tool to analyze the text
        4. If model fails, try the next most downloaded model (max 3 attempts)
        
        Return ONLY in this exact format:
        Sentiment: [POSITIVE/NEGATIVE/NEUTRAL/etc]
        Confidence: [score between 0 and 1]
        Model: [model_id]
        
        Important: Be concise, return only the required information.
        """
        
        try:
            # Execute agent using 0.14.10 workflow API
            # Use the stored context (or create new one if needed)
            ctx = self._context if self._context else Context(self.agent)
            
            # Run the agent (returns a handler)
            handler = self.agent.run(prompt, ctx=ctx)
            
            # Await the handler to get the response
            response = await handler
            response_text = str(response)
            
            # Parse response
            result = self._parse_response(response_text)
            
            if result:
                result["success"] = True
                return result
            else:
                # Fallback parsing
                return {
                    "sentiment": "UNKNOWN",
                    "confidence": 0.0,
                    "model_used": "parsing_failed",
                    "success": False,
                    "error": "Failed to parse agent response",
                    "raw_response": response_text
                }
                
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {str(e)}")
            return {
                "sentiment": "ERROR",
                "confidence": 0.0,
                "model_used": "N/A",
                "success": False,
                "error": str(e)
            }
    
    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse agent response to extract sentiment, confidence, and model"""
        try:
            # Try to extract structured information
            sentiment_match = re.search(r'Sentiment[:\s]+([A-Z]+)', response, re.IGNORECASE)
            confidence_match = re.search(r'Confidence[:\s]+([\d.]+)', response, re.IGNORECASE)
            model_match = re.search(r'Model[:\s]+([^\s\n]+)', response, re.IGNORECASE)
            
            # Alternative patterns
            if not sentiment_match:
                sentiment_match = re.search(r'label[:\s]+["\']?([a-z]+)["\']?', response, re.IGNORECASE)
            
            if not confidence_match:
                confidence_match = re.search(r'score[:\s]+([\d.]+)', response, re.IGNORECASE)
            
            if sentiment_match:
                sentiment = sentiment_match.group(1).upper()
                confidence = float(confidence_match.group(1)) if confidence_match else 0.0
                model_used = model_match.group(1) if model_match else "unknown"
                
                return {
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "model_used": model_used
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return None
