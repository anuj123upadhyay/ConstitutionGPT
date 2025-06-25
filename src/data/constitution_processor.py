import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import PyPDF2
from pathlib import Path

def load_constitution_text(file_path: str) -> str:
    """Load the constitution text from file (supports both TXT and PDF)"""
    try:
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return load_pdf_text(file_path)
        elif file_extension == '.txt':
            return load_txt_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Only PDF and TXT files are supported.")
            
    except FileNotFoundError:
        raise FileNotFoundError(f"Constitution file not found at {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

def load_pdf_text(file_path: str) -> str:
    """Load text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        
        if not text.strip():
            raise Exception("No text could be extracted from the PDF file")
            
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF file: {str(e)}")

def load_txt_text(file_path: str) -> str:
    """Load text from TXT file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def preprocess_text(text: str) -> str:
    """Clean and preprocess the text"""
    # Remove extra whitespaces and normalize
    text = ' '.join(text.split())
    
    # Add specific preprocessing for constitution text
    # Remove common PDF artifacts
    text = text.replace('\x0c', ' ')  # Form feed character
    text = text.replace('\xa0', ' ')  # Non-breaking space
    
    # Handle common constitution formatting
    # You can add more specific preprocessing here based on your PDF structure
    
    return text

def split_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split text into chunks and create Document objects"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    preprocessed_text = preprocess_text(text)
    
    # Split text into chunks
    chunks = text_splitter.split_text(preprocessed_text)
    
    # Create Document objects with metadata
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": "Constitution of India",
                "chunk_id": i,
                "chunk_size": len(chunk)
            }
        )
        documents.append(doc)
    
    return documents

def process_constitution(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Main function to process the constitution document"""
    text = load_constitution_text(file_path)
    documents = split_into_chunks(text, chunk_size, chunk_overlap)
    
    print(f"Processed constitution into {len(documents)} chunks")
    return documents
