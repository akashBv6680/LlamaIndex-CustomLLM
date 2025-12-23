"""Configuration module for LlamaIndex Custom LLM application with Gemini API."""
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Settings:
    """Application settings and configuration."""

    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Model Configuration
    MODEL_NAME: str = "gemini-pro"
    EMBEDDING_MODEL: str = "models/embedding-001"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 500
    TOP_P: float = 0.95
    TOP_K: int = 40

    # Application Settings
    APP_NAME: str = "LlamaIndex Custom LLM with Gemini"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 20
    RETRIEVAL_TOP_K: int = 3
    
    # Gemini-specific settings
    SAFETY_SETTINGS: list = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
    ]


settings = Settings()
