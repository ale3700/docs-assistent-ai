"""Configuração central do projeto, lida de variáveis de ambiente (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # Dados / persistência
    docs_dir: Path = field(default_factory=lambda: Path(_env("DOCS_DIR", str(ROOT / "data" / "docs"))))
    persist_dir: Path = field(default_factory=lambda: Path(_env("PERSIST_DIR", str(ROOT / ".chroma"))))
    collection: str = _env("COLLECTION", "portfolio")

    # Chunking
    chunk_size: int = int(_env("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "120"))

    # Embeddings: "huggingface" (local, gratuito) ou "openai"
    embedding_provider: str = _env("EMBEDDING_PROVIDER", "huggingface")
    embedding_model: str = _env(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # LLM: "ollama" (local) ou "openai"
    llm_provider: str = _env("LLM_PROVIDER", "ollama")
    llm_model: str = _env("LLM_MODEL", "llama3.1")
    temperature: float = float(_env("LLM_TEMPERATURE", "0.1"))

    # Recuperação
    top_k: int = int(_env("TOP_K", "4"))
    hybrid: bool = _env("HYBRID", "true").lower() == "true"


settings = Settings()
