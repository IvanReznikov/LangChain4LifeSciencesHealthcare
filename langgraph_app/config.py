"""
LangGraph LifeScienceBench — LLM Configuration.
Uses environment variables with sensible defaults for local development.
Auto-detects DeepSeek if DEEPSEEK_API_KEY is set in .env or environment.
"""

import os
from typing import Optional

# ── Load .env file (if python-dotenv is installed) ─────────────
try:
    from dotenv import load_dotenv
    _ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(_ENV_FILE):
        load_dotenv(_ENV_FILE)
except ImportError:
    pass  # python-dotenv not installed — env vars must be set manually


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Dual-mode secret resolver: env var first, fallback to default."""
    return os.environ.get(name, default)


# ── DeepSeek (primary, auto-detected from .env) ────────────────
DEEPSEEK_API_KEY = get_secret("DEEPSEEK_API_KEY", "")
DEEPSEEK_PRO_MODEL = get_secret("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro")
DEEPSEEK_FLASH_MODEL = get_secret("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash")
DEEPSEEK_BASE_URL = get_secret("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ── LLM provider ──────────────────────────────────────────────
# Auto-detect: DeepSeek if key present, else fall back to configured provider
if DEEPSEEK_API_KEY:
    LLM_PROVIDER = "deepseek"
else:
    LLM_PROVIDER = get_secret("LC4LSH_LLM_PROVIDER", "openai")  # openai | groq | anthropic | ollama | deepseek

# OpenAI / compatible
OPENAI_API_KEY = get_secret("LC4LSH_OPENAI_API_KEY") or get_secret("OPENAI_API_KEY", "")
OPENAI_BASE_URL = get_secret("LC4LSH_OPENAI_BASE_URL") or get_secret("OPENAI_BASE_URL", "")
MODEL_ID = get_secret("LC4LSH_MODEL_ID", "gpt-4o-mini")

# Groq
GROQ_API_KEY = get_secret("LC4LSH_GROQ_API_KEY") or get_secret("GROQ_API_KEY", "")

# Anthropic
ANTHROPIC_API_KEY = get_secret("LC4LSH_ANTHROPIC_API_KEY") or get_secret("ANTHROPIC_API_KEY", "")

# Ollama (local)
OLLAMA_BASE_URL = get_secret("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = get_secret("LC4LSH_OLLAMA_MODEL", "llama3.1:8b")

# ── LangSmith (optional) ──────────────────────────────────────
LANGSMITH_API_KEY = get_secret("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = get_secret("LANGCHAIN_PROJECT", "lc4lsh-langgraph-app")
LANGSMITH_ENDPOINT = get_secret("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ── Other settings ────────────────────────────────────────────
MAX_RETRIEVAL_CHUNKS = int(get_secret("LC4LSH_MAX_CHUNKS", "8"))
SEED = 42
TEMPERATURE = 0.0


def _configure_langsmith():
    """Enable LangSmith tracing if API key is available."""
    if LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
        os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT


_configure_langsmith()
