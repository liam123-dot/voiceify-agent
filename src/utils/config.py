"""
Configuration management for the LiveKit agent.

Handles loading environment variables and validating required configuration.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


@dataclass
class Config:
    """
    Application configuration loaded from environment variables.
    
    Attributes:
        api_url: Base URL for the Voiceify API
        livekit_url: LiveKit server WebSocket URL
        livekit_api_key: LiveKit API key
        livekit_api_secret: LiveKit API secret
        openai_api_key: OpenAI API key (optional)
        deepgram_api_key: Deepgram API key (optional)
        elevenlabs_api_key: ElevenLabs API key (optional)
        supabase_project_ref: Supabase project reference
        supabase_s3_access_key: Supabase S3 access key for recordings
        supabase_s3_secret_key: Supabase S3 secret key for recordings
        supabase_s3_region: Supabase S3 region
    """
    
    # Required configuration
    api_url: str
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    
    # Optional AI provider keys (at least one STT, LLM, TTS needed)
    openai_api_key: Optional[str] = None
    deepgram_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    
    # Recording configuration
    supabase_project_ref: Optional[str] = None
    supabase_s3_access_key: Optional[str] = None
    supabase_s3_secret_key: Optional[str] = None
    supabase_s3_region: Optional[str] = None
    
    # Telemetry configuration (Langfuse)
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: Optional[str] = None
    
    @classmethod
    def from_env(cls, env_file: str = ".env") -> "Config":
        """
        Load configuration from environment variables.
        
        Args:
            env_file: Path to .env file (default: ".env")
            
        Returns:
            Config instance with loaded values
            
        Raises:
            ValueError: If required environment variables are missing
        """
        # Load environment variables from file
        load_dotenv(env_file)
        
        # Validate required variables
        required_vars = [
            "API_URL",
            "LIVEKIT_URL",
            "LIVEKIT_API_KEY",
            "LIVEKIT_API_SECRET",
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )
        
        return cls(
            api_url=os.getenv("API_URL"),
            livekit_url=os.getenv("LIVEKIT_URL"),
            livekit_api_key=os.getenv("LIVEKIT_API_KEY"),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            deepgram_api_key=os.getenv("DEEPGRAM_API_KEY"),
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
            supabase_project_ref=os.getenv("SUPABASE_PROJECT_REF"),
            supabase_s3_access_key=os.getenv("SUPABASE_S3_ACCESS_KEY"),
            supabase_s3_secret_key=os.getenv("SUPABASE_S3_SECRET_KEY"),
            supabase_s3_region=os.getenv("SUPABASE_S3_REGION"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            langfuse_host=os.getenv("LANGFUSE_HOST"),
        )
    
    def log_config(self, logger) -> None:
        """
        Log configuration values (masking sensitive data).
        
        Args:
            logger: Logger instance to use for output
        """
        logger.info("Configuration loaded:")
        logger.info(f"  API_URL: {self.api_url}")
        logger.info(f"  LIVEKIT_URL: {self.livekit_url}")
        logger.info(f"  OPENAI_API_KEY: {'SET' if self.openai_api_key else 'NOT SET'}")
        logger.info(f"  DEEPGRAM_API_KEY: {'SET' if self.deepgram_api_key else 'NOT SET'}")
        logger.info(f"  ELEVENLABS_API_KEY: {'SET' if self.elevenlabs_api_key else 'NOT SET'}")
        logger.info(f"  LANGFUSE: {'ENABLED' if self.langfuse_host and self.langfuse_public_key and self.langfuse_secret_key else 'DISABLED'}")

