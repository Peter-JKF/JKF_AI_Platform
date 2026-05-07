"""Shared Flask extension singletons – import from here, not from app.py."""
import os
from flask_sqlalchemy import SQLAlchemy
from openai import OpenAI

db = SQLAlchemy()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# Optional: OpenRouter for non-OpenAI models (e.g. Claude)
_openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
openrouter_client = OpenAI(
    api_key=_openrouter_key or "none",
    base_url="https://openrouter.ai/api/v1",
) if _openrouter_key else None
