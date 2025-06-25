from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_store.pinecone_client import similarity_search
from llm.github_model_client import generate_response
from config.settings import TOP_K_RESULTS

def prepare_context(documents: List[Dict[str, Any]]) -> str:
    """Prepare context from retrieved documents"""
    context_parts = []
    for doc in documents:
        context_parts.append(doc["text"])
    
    # Join all contexts with separators
    context = "\n\n---\n\n".join(context_parts)
    
    # Limit context length if needed (adjust based on your model's limits)
    max_context_length = 4000  # Adjust based on your model
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    
    return context

def query_rag(question: str) -> Dict[str, Any]:
    """Main RAG query function"""
    try:
        # Step 1: Retrieve relevant documents
        relevant_docs = similarity_search(
            query=question,
            top_k=TOP_K_RESULTS
        )
        
        if not relevant_docs:
            return {
                "answer": "I couldn't find relevant information in the Constitution to answer your question.",
                "sources": [],
                "error": None
            }
        
        # Step 2: Prepare context from retrieved documents
        context = prepare_context(relevant_docs)
        
        # Step 3: Generate answer using LLM
        answer = generate_response(question, context)
        
        # Step 4: Prepare response
        response = {
            "answer": answer,
            "sources": [
                {
                    "text": doc["text"][:200] + "...",
                    "score": doc["score"],
                    "chunk_id": doc["chunk_id"]
                }
                for doc in relevant_docs
            ],
            "error": None
        }
        
        return response
        
    except Exception as e:
        return {
            "answer": "An error occurred while processing your question.",
            "sources": [],
            "error": str(e)
        }
