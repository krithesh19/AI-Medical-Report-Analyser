# ===========================================
# file: vectorstore.py
# Pinecone v3 setup and vectorstore helpers
# ===========================================
import os
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from embeddings import get_embeddings

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


DEFAULT_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBED_DIM = 384  # Google text-embedding-004

def _pc() -> Pinecone:
    if not PINECONE_API_KEY:
        raise RuntimeError("Missing PINECONE_API_KEY")
    return Pinecone(api_key=PINECONE_API_KEY)
# Check if indexes are present.. if not create them
def ensure_indexes(general_index_name: str, patient_index_name: str, region: str = DEFAULT_REGION):
    pc = _pc()
    existing = {i["name"] for i in pc.list_indexes().indexes}

    for name in [general_index_name, patient_index_name]:
        if name not in existing:
            pc.create_index(
                name=name,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=region or DEFAULT_REGION),
            )

# get the index contents
def get_vectorstore(index_name: str) -> PineconeVectorStore:
    # LangChain VectorStore wrapper (auto-connects by name)
    return PineconeVectorStore(
        index_name=index_name,
        embedding=get_embeddings(),
        # pinecone_api_key pulled from env automatically
    )

def delete_patient_session_vectors(patient_index_name: str, session_id: str):
    pc = _pc()
    idx = pc.Index(patient_index_name)
    # Delete by metadata filter: all chunks for this session
    # Note: Deletion is async; immediate count is not returned.
    idx.delete(filter={"session_id": {"$eq": session_id}})

def drop_patient_index(patient_index_name: str):
    pc = _pc()
    pc.delete_index(patient_index_name)
