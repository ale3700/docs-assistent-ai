"""Recuperação: busca vetorial (MMR) + BM25, combinadas por Reciprocal Rank Fusion."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from .config import Settings, settings as default_settings
from .ingest import get_vectorstore


def reciprocal_rank_fusion(
    rankings: list[list[Document]], k: int = 60, top_n: int | None = None
) -> list[Document]:
    """RRF: score(d) = soma(1 / (k + posição_i(d))) sobre cada ranking.
    Não precisa normalizar scores de sistemas diferentes (cosseno vs BM25)."""
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for ranking in rankings:
        for pos, doc in enumerate(ranking, start=1):
            key = doc.page_content
            by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + pos)
    ordered = sorted(scores, key=scores.get, reverse=True)
    docs = [by_key[key] for key in ordered]
    return docs[:top_n] if top_n else docs


class HybridRetriever(BaseRetriever):
    """Semântica (embeddings) acha sinônimos; lexical (BM25) acha termos
    exatos como nomes de libs e siglas. Juntas cobrem mais casos."""

    vector_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    top_k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        sem = self.vector_retriever.invoke(query)
        lex = self.bm25_retriever.invoke(query)
        return reciprocal_rank_fusion([sem, lex], top_n=self.top_k)


def build_retriever(cfg: Settings = default_settings) -> BaseRetriever:
    store = get_vectorstore(cfg)
    fetch = max(cfg.top_k * 3, 10)
    vector = store.as_retriever(
        search_type="mmr",  # diversifica os resultados, evita chunks quase iguais
        search_kwargs={"k": cfg.top_k * 2, "fetch_k": fetch},
    )
    if not cfg.hybrid:
        return store.as_retriever(search_kwargs={"k": cfg.top_k})

    from langchain_community.retrievers import BM25Retriever

    data = store.get(include=["documents", "metadatas"])
    corpus = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]
    if not corpus:
        raise RuntimeError("Índice vazio. Rode `python -m rag.ingest` primeiro.")
    bm25 = BM25Retriever.from_documents(corpus, k=cfg.top_k * 2)
    return HybridRetriever(vector_retriever=vector, bm25_retriever=bm25, top_k=cfg.top_k)
