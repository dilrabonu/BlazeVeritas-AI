from pathlib import Path
from typing import Dict, Any, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader

import os

CHROMA_DIR = Path(".chroma")
CHROMA_DIR.mkdir(exist_ok=True)

splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)

def _infer_geo_meta(path: Path) -> Dict[str, Any]:
    parts = path.parts
    region = None
    for p in parts:
        if len(p) <= 8 and ("_" in p or p.isupper()):
            region = p
            break
    return {"region": region} if region else {}

def _embeddings():
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or your environment.")
    return OpenAIEmbeddings()  # uses env key

def build_or_update_index(docs_dir: str = "docs") -> Chroma:
    docs_path = Path(docs_dir); docs_path.mkdir(parents=True, exist_ok=True)
    raw_docs: List = []

    for p in docs_path.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        meta = {"source": str(p)}
        meta.update(_infer_geo_meta(p))
        if p.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(p))
            loaded = loader.load()
            for d in loaded:
                d.metadata.update(meta)
            raw_docs.extend(loaded)
        else:
            loader = TextLoader(str(p), encoding="utf-8")
            loaded = loader.load()
            for d in loaded:
                d.metadata.update(meta)
            raw_docs.extend(loaded)

    chunks = splitter.split_documents(raw_docs)

    vectordb = Chroma(
        collection_name="blazeveritas_docs",
        embedding_function=_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )

    if chunks:
        vectordb.add_documents(chunks)
        vectordb.persist()

    return vectordb

def get_vectordb() -> Chroma:
    return Chroma(
        collection_name="blazeveritas_docs",
        embedding_function=_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
