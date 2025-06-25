# Constitution of India RAG System

A Retrieval-Augmented Generation (RAG) system for querying the Constitution of India using LangChain, Pinecone, HuggingFace embeddings, and GitHub Marketplace models.

## Key Features

- 🔍 Semantic search through Constitution text
- 🤖 AI-powered answer generation using GitHub Marketplace models
- 📊 Source attribution with relevance scores
- ⚡ Fast vector-based retrieval using Pinecone
- 🎯 Context-aware responses
- 🔧 HuggingFace embeddings (no OpenAI API required)
- 📄 Support for both PDF and TXT file formats

## Setup

### Prerequisites

- Python 3.8 or higher
- Pinecone account and API key
- GitHub token with access to marketplace models

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd COI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Add the Constitution document:
   - Place the Constitution of India file (PDF or TXT format) in the `documents/` folder
   - Supported filenames:
     - `documents/COI.pdf`
     - `documents/constitution_of_india.pdf`
     - `documents/constitution_of_india.txt`

5. Run setup validation:
```bash
python setup.py
```

### Running the Application

```bash
streamlit run src/app.py
```

## Usage

1. The app will automatically process the Constitution document on first run
2. Enter your question in the text input
3. Click "Get Answer" to receive AI-generated responses with sources

## Project Structure

```
COI/
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── config/
│   │   └── settings.py        # Configuration settings
│   ├── data/
│   │   └── constitution_processor.py  # Document processing
│   ├── embeddings/
│   │   └── embedding_service.py       # HuggingFace embedding generation
│   ├── vector_store/
│   │   └── pinecone_client.py         # Pinecone operations
│   ├── llm/
│   │   └── github_model_client.py     # GitHub model client
│   ├── retrieval/
│   │   └── rag_pipeline.py            # RAG pipeline
│   └── utils/
│       └── helpers.py                 # Utility functions
├── documents/
│   └── COI.pdf                        # Constitution document (PDF or TXT supported)
├── requirements.txt
├── .env.example
├── setup.py
└── README.md
```

## Configuration

Edit `src/config/settings.py` to modify:
- Chunk size and overlap for document processing
- Number of retrieved documents
- HuggingFace model parameters

## Environment Variables

Required environment variables in `.env`:
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Your Pinecone environment (default: gcp-starter)
- `GITHUB_TOKEN`: Your GitHub token for marketplace models
- `GITHUB_MODEL_ENDPOINT`: Your GitHub marketplace model endpoint
- `HUGGINGFACE_MODEL_NAME`: HuggingFace model name (default: sentence-transformers/all-MiniLM-L6-v2)

## Key Differences from Class-based Version

- All functionality implemented as functions instead of classes
- Uses HuggingFace embeddings instead of OpenAI
- No OpenAI API key required
- Simplified imports and module structure
- Function-based approach for better modularity

## Troubleshooting

1. **Import errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
2. **Environment variables**: Ensure all required variables are set in `.env`
3. **Constitution file**: Place the constitution file (PDF or TXT) in the `documents/` folder with supported filename
4. **Pinecone connection**: Verify your Pinecone API key and environment

## License

MIT License
