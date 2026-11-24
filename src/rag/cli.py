"""CLI interativa: python -m rag.cli "sua pergunta"  (ou sem argumento para chat)."""
from __future__ import annotations

import sys

from .chain import build_chain


def ask(chain, question: str) -> None:
    result = chain.invoke(question)
    print(f"\n{result['answer']}\n")
    print("Fontes:", ", ".join(result["sources"]) or "—")


def main() -> None:
    chain = build_chain()
    if len(sys.argv) > 1:
        ask(chain, " ".join(sys.argv[1:]))
        return
    print("assistent-ai — digite 'sair' para encerrar.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"sair", "exit", "quit"}:
            break
        if q:
            ask(chain, q)


if __name__ == "__main__":
    main()
