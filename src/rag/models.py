"""Fábricas de embeddings e LLM, com troca de provedor via configuração."""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings, settings as default_settings


def get_embeddings(cfg: Settings = default_settings) -> Embeddings:
    """Embeddings transformam texto em vetores; textos semanticamente
    próximos ficam próximos no espaço vetorial."""
    if cfg.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=cfg.embedding_model)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=cfg.embedding_model,
        encode_kwargs={"normalize_embeddings": True},  # cosseno ~ produto interno
    )


def get_llm(cfg: Settings = default_settings) -> BaseChatModel:
    if cfg.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=cfg.llm_model, temperature=cfg.temperature)

    from langchain_ollama import ChatOllama

    return ChatOllama(model=cfg.llm_model, temperature=cfg.temperature)
