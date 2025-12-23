"""Configuration module for LlamaIndex Custom LLM application."""
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Settings:
    """Application settings and configuration."""

    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "")

    # Model Configuration
    MODEL_NAME: str = "gpt-3.5-turbo"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 500

    # Vector Store
    PINECONE_INDEX_NAME: str = "llamaindex-custom-llm"
    VECTOR_STORE_DIMENSION: int = 1536  # Ada embedding dimension

    # Application Settings
    APP_NAME: str = "LlamaIndex Custom LLM"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 20
    TOP_K: int = 3


settings = Settings()
