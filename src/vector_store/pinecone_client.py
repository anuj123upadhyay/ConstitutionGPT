from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
from langchain.schema import Document
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION
from embeddings.embedding_service import embed_documents, embed_query

def initialize_pinecone():
    """Initialize Pinecone client"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc

def create_index_if_not_exists(pc, index_name: str):
    """Create Pinecone index if it doesn't exist"""
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,  # Dynamic dimension based on model
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"Created new Pinecone index: {index_name} with {EMBEDDING_DIMENSION} dimensions")
    else:
        print(f"Using existing Pinecone index: {index_name}")

def get_pinecone_index():
    """Get Pinecone index instance"""
    pc = initialize_pinecone()
    create_index_if_not_exists(pc, PINECONE_INDEX_NAME)
    index = pc.Index(PINECONE_INDEX_NAME)
    return index

def store_documents(documents: List[Document]) -> bool:
    """Store documents in Pinecone vector database"""
    try:
        index = get_pinecone_index()
        
        # Extract text content
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings
        embeddings = embed_documents(texts)
        
        # Prepare data for upsert
        vectors_to_upsert = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vector_data = {
                "id": f"chunk_{i}",
                "values": embedding,
                "metadata": {
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "chunk_id": doc.metadata.get("chunk_id", i)
                }
            }
            vectors_to_upsert.append(vector_data)
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
        
        print(f"Successfully stored {len(documents)} documents in Pinecone")
        return True
        
    except Exception as e:
        print(f"Error storing documents: {str(e)}")
        return False

def similarity_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for similar documents"""
    try:
        index = get_pinecone_index()
        
        # Generate query embedding
        query_embedding = embed_query(query)
        
        # Search in Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results['matches']:
            formatted_results.append({
                'text': match['metadata']['text'],
                'score': match['score'],
                'source': match['metadata'].get('source', ''),
                'chunk_id': match['metadata'].get('chunk_id', '')
            })
        
        return formatted_results
        
    except Exception as e:
        print(f"Error during similarity search: {str(e)}")
        return []
