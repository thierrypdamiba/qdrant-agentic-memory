"""
Shared configuration for the workshop.

Embeddings: FastEmbed (local, free)
LLM: OpenAI (optional, for advanced features)
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Qdrant Cloud settings
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# OpenAI settings (optional - only for LLM features)
_openai_key = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_KEY = _openai_key if _openai_key and not _openai_key.startswith("<") else None
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Embedding settings (FastEmbed - runs locally, FREE)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Collection names for the workshop
MEMORIES_COLLECTION = "agent_memories"
EPISODES_COLLECTION = "agent_episodes"
