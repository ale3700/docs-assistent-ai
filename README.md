# assistent-ai

Pipeline de **Retrieval-Augmented Generation** feito do zero com LangChain, embeddings e banco vetorial (Chroma). Ele responde perguntas sobre os seus próprios documentos (currículo, projetos, artigos) citando as fontes.

## Como funciona

```
 Documentos (.md/.txt/.pdf)
        │  load + split (chunks com overlap)
        ▼
   Embeddings ──► Chroma (banco vetorial persistente)
                         │
 Pergunta ──► Retriever híbrido ◄─┘
              ├─ busca semântica (MMR)
              └─ BM25 (lexical)
                    │  Reciprocal Rank Fusion
                    ▼
        Prompt com contexto ──► LLM ──► Resposta + fontes
```

## Conceitos demonstrados

| Conceito | Onde |
|---|---|
| Chunking com `RecursiveCharacterTextSplitter` e overlap | `src/rag/ingest.py` |
| Embeddings (HuggingFace local ou OpenAI) | `src/rag/models.py` |
| Banco vetorial persistente com IDs determinísticos (upsert) | `src/rag/ingest.py` |
| Busca por MMR para diversificar resultados | `src/rag/retrieval.py` |
| Busca híbrida (vetorial + BM25) com RRF | `src/rag/retrieval.py` |
| Cadeia LCEL com prompt ancorado no contexto e fontes | `src/rag/chain.py` |
| CLI e API REST (FastAPI) | `src/rag/cli.py`, `src/rag/api.py` |
| Testes unitários sem rede | `tests/` |

## Início rápido

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# LLM local (padrão): instale o Ollama e baixe um modelo
ollama pull llama3.1

# 1) indexar os documentos de data/docs
PYTHONPATH=src python -m rag.ingest --reset

# 2) perguntar
PYTHONPATH=src python -m rag.cli "Quais projetos usam FastAPI?"

# ou subir a API
PYTHONPATH=src uvicorn rag.api:app --reload
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "Quais são minhas habilidades em IA?"}'
```

Para usar OpenAI, edite o `.env` (veja `.env.example`) e defina `OPENAI_API_KEY`. **Atenção:** se trocar o modelo de embeddings, reindexe com `--reset`, pois vetores de modelos diferentes não são comparáveis.

## Usando seus dados

Troque os arquivos em `data/docs/` pelos seus (currículo, README dos projetos, posts) e rode a ingestão de novo.

## Testes

```bash
pytest
```

## Decisões de design

- **Embeddings locais por padrão**: custo zero e funciona offline; o modelo multilíngue atende bem português.
- **Busca híbrida**: embeddings pegam sinônimos, BM25 pega termos exatos (nomes de bibliotecas, siglas). RRF combina os rankings sem precisar normalizar scores.
- **Prompt restritivo**: o modelo só responde com o contexto recuperado e admite quando não sabe, reduzindo alucinação.
- **IDs por hash**: reindexar não duplica chunks.

## Ideias de evolução

- Reranker cross-encoder após a recuperação
- Avaliação automática (RAGAS) com um conjunto de perguntas e respostas esperadas
- Histórico de conversa com reescrita de pergunta
- Streaming de resposta na API
- Dockerfile e CI com GitHub Actions

## Licença

MIT
