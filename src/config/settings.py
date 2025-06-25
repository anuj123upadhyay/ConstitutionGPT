import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "constitution-index"

# GitHub Model Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_MODEL_ENDPOINT = os.getenv("GITHUB_MODEL_ENDPOINT")

# HuggingFace Configuration
HUGGINGFACE_MODEL_NAME = os.getenv("HUGGINGFACE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


EMBEDDING_DIMENSION=384
# Text Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K_RESULTS = 5
