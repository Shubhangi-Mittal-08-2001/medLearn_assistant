"""
Langfuse Callback Handler Module

This module provides a centralized Langfuse callback handler for tracing
LangChain chain executions in the Langfuse dashboard.
"""
import os
from typing import Optional
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()


def _is_langfuse_enabled() -> bool:
    """Return whether Langfuse callback tracing should be active."""
    raw = str(os.getenv("LANGFUSE_ENABLED", "true")).strip().lower()
    return raw in {"1", "true", "yes", "on"}

## Task 2: Add Observability with Langfuse Callback Handler (Chat API)
class LangfuseCallbackManager:
    """
    Manages Langfuse callback handler instances for observability.
    
    This class provides a centralized way to create and configure
    Langfuse callback handlers for tracing LangChain executions.
    """
    
    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Initialize Langfuse callback manager.
        
        Args:
            public_key: Langfuse public key (defaults to LANGFUSE_PUBLIC_KEY env var)
            secret_key: Langfuse secret key (defaults to LANGFUSE_SECRET_KEY env var)
            host: Langfuse host URL (defaults to LANGFUSE_HOST env var or https://cloud.langfuse.com)
           
        """
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        # Support both LANGFUSE_HOST and LANGFUSE_BASE_URL (Langfuse uses BASE_URL in some versions)
        self.host = host or os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"
        self.session_id = session_id
        self.user_id = user_id
        
      
    
    def get_callback_handler(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        trace_name: Optional[str] = None,
    ) -> Optional[CallbackHandler]:
        """
        Create a Langfuse callback handler instance.
        
        In Langfuse 3.x, CallbackHandler accepts public_key, secret_key, and host directly.
        Session ID, user ID, and trace name can be set via metadata in the chain config.
        
        Args:
            session_id: Session ID for grouping traces (overrides instance default)
            user_id: User ID for user tracking (overrides instance default)
            trace_name: Name for the trace in Langfuse dashboard
            
        Returns:
            CallbackHandler instance configured with Langfuse credentials, or None if credentials are not provided
        """
        if not _is_langfuse_enabled():
            return None

        if not self.public_key or not self.secret_key:
            # Return None if credentials are not provided (optional observability)
            return None
        
        # Use provided session_id/user_id or fall back to instance defaults
        effective_session_id = session_id or self.session_id
        effective_user_id = user_id or self.user_id
        
        # Ensure environment variables are set for all Langfuse credentials
        # Langfuse CallbackHandler and internal clients read these from environment automatically
        # IMPORTANT: Set these BEFORE creating the CallbackHandler
        # Langfuse SDK may use either LANGFUSE_HOST or LANGFUSE_BASE_URL
        os.environ["LANGFUSE_PUBLIC_KEY"] = self.public_key
        os.environ["LANGFUSE_SECRET_KEY"] = self.secret_key
        os.environ["LANGFUSE_HOST"] = self.host
        os.environ["LANGFUSE_BASE_URL"] = self.host  # Set both for compatibility
        
        # Create Langfuse client explicitly to ensure it's initialized with correct credentials
        # This helps verify credentials are correct before creating the CallbackHandler
        try:
            langfuse_client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host
            )
            
            # Test authentication (silently)
            try:
                langfuse_client.auth_check()
            except Exception:
                # Continue anyway - the client might still work for tracing
                pass
        except Exception:
            langfuse_client = None
        
        # CallbackHandler signatures vary by langfuse version.
        # Try the most specific constructor first and degrade gracefully.
        constructor_attempts = [
            {
                "public_key": self.public_key,
                "secret_key": self.secret_key,
                "host": self.host,
                "session_id": effective_session_id,
                "user_id": effective_user_id,
                "trace_name": trace_name,
            },
            {
                "public_key": self.public_key,
                "secret_key": self.secret_key,
                "host": self.host,
            },
            {
                "public_key": self.public_key,
            },
            {},
        ]

        handler = None
        for attempt in constructor_attempts:
            kwargs = {key: value for key, value in attempt.items() if value is not None}
            try:
                handler = CallbackHandler(**kwargs)
                break
            except TypeError:
                continue
            except Exception:
                continue

        if handler is None:
            return None
        
        # If we created an explicit client, try to use it
        # Note: CallbackHandler might create its own client, but we can verify they match
        if langfuse_client and hasattr(handler, 'client'):
            # The handler's client should use the same credentials from environment
            # Store our explicit client reference for potential use
            handler._explicit_langfuse_client = langfuse_client
        
        # Verify the handler's client is properly configured
        if hasattr(handler, 'client'):
            # Store reference to client for easier access
            handler._langfuse_client = handler.client
        
        return handler
    
    def flush(self):
        """
        Flush any pending traces to Langfuse.
        
        This should be called during application shutdown to ensure
        all traces are properly exported.
        """
        # Note: CallbackHandler automatically flushes, but we can add
        # explicit flush logic here if needed in the future
        pass


# Global instance for easy access
_langfuse_manager: Optional[LangfuseCallbackManager] = None


def get_langfuse_manager() -> LangfuseCallbackManager:
    """
    Get or create the global Langfuse callback manager instance.
    
    Returns:
        LangfuseCallbackManager instance
    """
    global _langfuse_manager
    if _langfuse_manager is None:
        _langfuse_manager = LangfuseCallbackManager()
    return _langfuse_manager


__all__ = ["LangfuseCallbackManager", "get_langfuse_manager"]

