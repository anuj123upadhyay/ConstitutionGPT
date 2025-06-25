from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import HUGGINGFACE_MODEL_NAME

def initialize_embeddings():
    """Initialize HuggingFace embeddings model"""
    embeddings = HuggingFaceEmbeddings(
        model_name=HUGGINGFACE_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

def embed_documents(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of documents"""
    try:
        embeddings = initialize_embeddings()
        embeddings_list = embeddings.embed_documents(texts)
        return embeddings_list
    except Exception as e:
        raise Exception(f"Error generating embeddings: {str(e)}")

def embed_query(query: str) -> List[float]:
    """Generate embedding for a single query"""
    try:
        embeddings = initialize_embeddings()
        embedding = embeddings.embed_query(query)
        return embedding
    except Exception as e:
        raise Exception(f"Error generating query embedding: {str(e)}")
