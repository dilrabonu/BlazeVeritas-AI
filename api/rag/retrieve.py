from typing import Dict, Any, List, Tuple
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from .index import get_vectordb

SYSTEM_PROMPT = """
You are **BlazeVeritas Wildfire  Copilot** — a disciplined incident support assistant.

RULES:
- Be concise and operational.
- Use retrieved documents + tools (weather/topography) when available.
- Always provide short **cited sources** at the end (file paths or URLs).
- If information is missing, say so and propose validation steps.

Return your answer in clear markdown sections.
"""

PLAN_INSTRUCTIONS = """
Write an **operational action plan** for the given context.

Sections:
1) Situation (1-3 lines)
2) Confidence (0-1) & Key Uncertainties
3) Immediate Actions (0-30 min)
4) Next Steps (2-6 hr)
5) Key Risks (wind, fuel, slope)
6) Public Comms (short paragraph)
7) Sources (bulleted list of titles/paths)

Context:
- lat: {lat}
- lon: {lon}
- model_label: {label}
- prob: {prob}
- uncertainty: {uncertainty}
- Objective: {objective}
- Tools: {tools}
- Retrieved docs: {docs}
"""

QA_INSTRUCTIONS = """
Answer the user question **succinctly** using retrieved docs & tools.

Question: {objective}
Context:
- lat: {lat}
- lon: {lon}
- model_label: {label}
- prob: {prob}
- uncertainty: {uncertainty}
- Tools: {tools}
- Retrieved docs: {docs}

Always end with a short **Sources** section (bulleted).
"""

def get_weather(lat: float, lon: float):
    return {"temp_c": 29.0, "wind_kph": 18.0, "wind_dir": "NW", "humidity": 22}

def get_ndvi(lat: float, lon: float):
    return {"ndvi": 0.21, "dryness": "medium"}

def get_topography(lat: float, lon: float):
    return {"slope_deg": 12, "aspect": "SW"}

def _build_retriever() -> Tuple[object, Chroma]:
    """
    Returns a retriever that never crashes:
    - If no docs in store → return dense-only retriever.
    - If docs exist       → return Ensemble(dense + BM25).
    """
    vectordb: Chroma = get_vectordb()
    dense = vectordb.as_retriever(search_kwargs={"k": 4})

    
    docs_list = []
    try:
        payload = getattr(vectordb, "_collection").get(include=["metadatas", "documents"])
        docs = payload.get("documents", []) or []
        metas = payload.get("metadatas", []) or []
        docs_list = list(zip(docs, metas))
    except Exception:
        docs_list = []

    if not docs_list:
        return dense, vectordb

    corpus = [Document(page_content=t, metadata=m) for t, m in docs_list]
    if not corpus:
        return dense, vectordb

    keyword = BM25Retriever.from_documents(corpus)
    keyword.k = 4
    ens = EnsembleRetriever(retrievers=[dense, keyword], weights=[0.6, 0.4])
    return ens, vectordb

def _format_docs(docs: List[Document]) -> str:
    formatted = []
    for d in docs:
        src = d.metadata.get("source", "?")
        formatted.append(f"[SOURCE] {src}\n{d.page_content[:1200]}")
    return "\n\n".join(formatted)

def _extract_sources(docs: List[Document]):
    out = []
    for d in docs:
        src = d.metadata.get("source", "")
        title = d.metadata.get("title") or src.split("/")[-1]
        out.append({"title": title, "source": src, "url": src})
    # unique in order
    seen, uniq = set(), []
    for s in out:
        key = s["source"]
        if key not in seen:
            uniq.append(s); seen.add(key)
    return uniq

def generate_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Required for ChatOpenAI & embeddings
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or your environment.")

    lat = payload.get("lat"); lon = payload.get("lon")
    label = payload.get("label", "unknown")
    prob = payload.get("prob", 0.0)
    uncertainty = payload.get("uncertainty", 0.0)
    objective = payload.get("objectives", "Rapid triage and recommended actions")

    # Optional UI hints
    model_name = payload.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(payload.get("temperature", 0.2))

    retriever, _ = _build_retriever()
    # LangChain deprecation: invoke() instead of get_relevant_documents()
    query_text = objective if isinstance(objective, str) else ""
    docs = retriever.invoke(query_text)
    docs_md = _format_docs(docs)

    tools = {}
    if lat is not None and lon is not None:
        tools["weather"] = get_weather(lat, lon)
        tools["ndvi"] = get_ndvi(lat, lon)
        tools["topography"] = get_topography(lat, lon)

    # Decide mode: plan vs Q&A
    is_question = isinstance(objective, str) and ("?" in objective)
    template = QA_INSTRUCTIONS if is_question else PLAN_INSTRUCTIONS

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", template),
    ])

    chain = (
        {
            "docs": lambda _: docs_md,
            "lat": lambda _: lat,
            "lon": lambda _: lon,
            "label": lambda _: label,
            "prob": lambda _: prob,
            "uncertainty": lambda _: uncertainty,
            "objective": lambda _: objective,
            "tools": lambda _: tools,
        }
        | prompt
        | ChatOpenAI(model=model_name, temperature=temperature)
        | StrOutputParser()
    )

    plan_text = chain.invoke({})
    sources = _extract_sources(docs)

    return {
        "plan": plan_text,            
        "sources": sources,
        "used_provider": "openai",
        "usage": {"prompt": 0, "completion": 0},
    }
