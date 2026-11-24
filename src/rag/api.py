"""API REST: uvicorn rag.api:app --reload"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .chain import build_chain

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["chain"] = build_chain()
    yield
    _state.clear()


app = FastAPI(title="assistent-ai", version="1.0.0", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Answer(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(body: Question) -> Answer:
    try:
        return Answer(**_state["chain"].invoke(body.question))
    except Exception as exc:  # erro do LLM/vetorstore vira 502 legível
        raise HTTPException(status_code=502, detail=str(exc)) from exc
