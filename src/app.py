import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.constitution_processor import process_constitution
from vector_store.pinecone_client import store_documents
from retrieval.rag_pipeline import query_rag
from utils.helpers import validate_environment_variables, print_validation_results
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

def initialize_rag_system():
    """Initialize the RAG system by processing and storing documents"""
    if 'rag_initialized' not in st.session_state:
        try:
            with st.spinner("Initializing RAG system..."):
                # Validate environment variables
                validation_results = validate_environment_variables()
                if not all(validation_results.values()):
                    st.error("Missing required environment variables!")
                    print_validation_results(validation_results)
                    return False
                
                # Check if constitution file exists (support both PDF and TXT)
                constitution_paths = [
                    "src/documents/COI.pdf"
                ]
                
                constitution_path = None
                for path in constitution_paths:
                    if os.path.exists(path):
                        constitution_path = path
                        break
                
                if constitution_path is None:
                    st.error("Constitution file not found!")
                    st.info("Please add the Constitution of India file (PDF or TXT) to the documents folder with one of these names:")
                    for path in constitution_paths:
                        st.info(f"- {path}")
                    return False
                
                # Process constitution document
                documents = process_constitution(
                    constitution_path,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                
                # Store in vector database
                success = store_documents(documents)
                
                if success:
                    st.session_state.rag_initialized = True
                    st.success("RAG system initialized successfully!")
                    return True
                else:
                    st.error("Failed to initialize vector store")
                    return False
                    
        except Exception as e:
            st.error(f"Error initializing RAG system: {str(e)}")
            return False
    
    return True

def main():
    st.title("🏛️ Constitution of India RAG Assistant")
    st.markdown("Ask questions about the Constitution of India and get accurate, context-based answers.")
    
    # Initialize RAG system
    if not initialize_rag_system():
        st.stop()
    
    # Create the query interface
    st.markdown("### Ask your question:")
    user_question = st.text_input(
        "Enter your question about the Constitution of India:",
        placeholder="e.g., What are the fundamental rights mentioned in the Constitution?"
    )
    
    if st.button("Get Answer", type="primary"):
        if user_question.strip():
            with st.spinner("Searching and generating answer..."):
                # Get response from RAG pipeline
                response = query_rag(user_question)

                print(response)  # Debugging output
                
                if response["error"]:
                    st.error(f"Error: {response['error']}")
                else:
                    # Display answer
                    st.markdown("### 📝 Answer:")
                    st.markdown(response["answer"])
                    
                    # Display sources
                    if response["sources"]:
                        st.markdown("### 📚 Sources:")
                        for i, source in enumerate(response["sources"], 1):
                            with st.expander(f"Source {i} (Relevance: {source['score']:.3f})"):
                                st.text(source["text"])
        else:
            st.warning("Please enter a question.")
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### About")
        st.info(
            "This RAG (Retrieval-Augmented Generation) system helps you find information "
            "from the Constitution of India using advanced AI techniques."
        )
        
        st.markdown("### How it works:")
        st.markdown("""
        1. **Document Processing**: Constitution text is split into chunks
        2. **Embedding**: Text chunks are converted to vector embeddings using HuggingFace
        3. **Vector Storage**: Embeddings stored in Pinecone database
        4. **Retrieval**: Similar chunks retrieved for your query
        5. **Generation**: AI model generates answer based on retrieved context
        """)
        
        st.markdown("### Features:")
        st.markdown("""
        - 🔍 Semantic search through Constitution
        - 🤖 AI-powered answer generation
        - 📊 Source attribution and relevance scores
        - ⚡ Fast vector-based retrieval
        - 🔧 HuggingFace embeddings (no OpenAI API required)
        """)

if __name__ == "__main__":
    main()
