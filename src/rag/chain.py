"""Cadeia RAG em LCEL: pergunta -> recuperação -> prompt -> LLM -> resposta + fontes."""
from __future__ import annotations

from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from .config import Settings, settings as default_settings
from .models import get_llm
from .retrieval import build_retriever

SYSTEM_PROMPT = """Você é um assistente que responde perguntas usando SOMENTE o contexto abaixo.
Regras:
- Se a resposta não estiver no contexto, diga que não encontrou essa informação.
- Responda no mesmo idioma da pergunta, de forma objetiva.
- Cite as fontes entre colchetes, por exemplo [projetos.md].

Contexto:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
    )


def build_chain(cfg: Settings = default_settings) -> Runnable:
    """Retorna um Runnable que recebe uma string (pergunta) e devolve
    {"answer": str, "sources": list[str]}."""
    retriever = build_retriever(cfg)
    llm = get_llm(cfg)

    answer_chain = (
        {
            "context": itemgetter("docs") | RunnableLambda(format_docs),
            "question": itemgetter("question"),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return (
        {"docs": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(answer=answer_chain)
        | RunnableLambda(
            lambda x: {
                "answer": x["answer"],
                "sources": sorted({d.metadata.get("source", "?") for d in x["docs"]}),
            }
        )
    )
