"""Testes que não exigem modelos, rede ou chaves de API."""
from langchain_core.documents import Document

from rag.ingest import chunk_id, split_documents
from rag.retrieval import reciprocal_rank_fusion


def test_split_respects_size_and_keeps_source():
    text = "## Título\n\n" + ("Frase de exemplo sobre RAG. " * 100)
    docs = [Document(page_content=text, metadata={"source": "/x/y/doc.md"})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert all(c.metadata["source"] == "doc.md" for c in chunks)


def test_chunk_id_is_deterministic():
    a = Document(page_content="olá", metadata={"source": "a.md", "start_index": 0})
    b = Document(page_content="olá", metadata={"source": "a.md", "start_index": 0})
    c = Document(page_content="olá!", metadata={"source": "a.md", "start_index": 0})
    assert chunk_id(a) == chunk_id(b)
    assert chunk_id(a) != chunk_id(c)


def test_rrf_prefers_docs_present_in_both_rankings():
    d = lambda t: Document(page_content=t)
    semantic = [d("A"), d("B"), d("C")]
    lexical = [d("C"), d("B"), d("D")]
    fused = reciprocal_rank_fusion([semantic, lexical])
    order = [x.page_content for x in fused]
    assert set(order[:2]) == {"B", "C"}  # aparecem nas duas listas
    assert order[-1] in {"A", "D"}


def test_rrf_top_n():
    d = lambda t: Document(page_content=t)
    fused = reciprocal_rank_fusion([[d("A"), d("B")], [d("C")]], top_n=2)
    assert len(fused) == 2
