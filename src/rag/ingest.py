"""Ingestão: carrega documentos, divide em chunks e indexa no Chroma."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings, settings as default_settings
from .models import get_embeddings


def load_documents(docs_dir: Path) -> list[Document]:
    """Carrega .md, .txt e .pdf de uma pasta (recursivo)."""
    docs: list[Document] = []
    for pattern in ("**/*.md", "**/*.txt"):
        loader = DirectoryLoader(
            str(docs_dir),
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        docs.extend(loader.load())
    for pdf in docs_dir.glob("**/*.pdf"):
        docs.extend(PyPDFLoader(str(pdf)).load())
    return docs


def split_documents(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Chunking: pedaços pequenos o bastante para um embedding focado e
    grandes o bastante para manter contexto. O overlap evita cortar ideias."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["source"] = Path(c.metadata.get("source", "?")).name
    return chunks


def chunk_id(chunk: Document) -> str:
    """ID determinístico: reindexar o mesmo texto não duplica vetores."""
    raw = f"{chunk.metadata.get('source')}|{chunk.metadata.get('start_index')}|{chunk.page_content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def get_vectorstore(cfg: Settings = default_settings) -> Chroma:
    return Chroma(
        collection_name=cfg.collection,
        embedding_function=get_embeddings(cfg),
        persist_directory=str(cfg.persist_dir),
    )


def ingest(cfg: Settings = default_settings, reset: bool = False) -> int:
    docs = load_documents(cfg.docs_dir)
    if not docs:
        raise SystemExit(f"Nenhum documento encontrado em {cfg.docs_dir}")
    chunks = split_documents(docs, cfg.chunk_size, cfg.chunk_overlap)

    store = get_vectorstore(cfg)
    if reset:
        store.reset_collection()
    # upsert por ID determinístico
    store.add_documents(chunks, ids=[chunk_id(c) for c in chunks])
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa documentos no Chroma")
    parser.add_argument("--reset", action="store_true", help="limpa a coleção antes")
    args = parser.parse_args()
    n = ingest(reset=args.reset)
    print(f"✓ {n} chunks indexados em {default_settings.persist_dir}")


if __name__ == "__main__":
    main()
