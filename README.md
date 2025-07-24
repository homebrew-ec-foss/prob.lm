# Prob.lm

A low-resource RAG-based assistant that answers academic questions using student-provided materials like notes and PDFs.It supports PDF documents and provides an interactive CLI for document management and question-answering.

## Mentees:

- Atharv Sawarkar (@kazabiteboltiz)
- Anshul Banda (@AnshulBanda)
- Shashank A (@ShadowMarty)
- S S Adhithya Sriram (@SS-AdhithyaSriram)
- Ayushi Mittal (@mittal-ayushi)


## Mentors:

- Saijyoti Panda
- Pranavjeet Naidu
- Pranav Hemanth
- Yashmitha Shailesh
- Anshul Paruchuri

## Features

- **Document Management**: Upload and manage study materials (PDFs, text files)
- **Interactive CLI**: Command-line interface for document interaction
- **Document Processing**: Text extraction and chunking for efficient retrieval
- **Vector Search**: Semantic search capabilities using embeddings

## Installation

1. **Clone the repository**
   ```bash
   git clone <repo-link>
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   Windows:
     ```
     venv/Scripts/activate
     ```
   macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
## Usage

1. **Start the application**
   ```bash
   python main.py
   ```

2. **Main Menu Options**
   - **Upload Document(s)**: Add study materials to the system
   - **List Uploaded Documents**: View all uploaded documents
   - **Process Documents & Start Q&A**: Initialize the RAG pipeline and start asking questions
   - **Exit**: Close the application

3. **Using the Q&A System**
   - After processing documents, type your questions about the content
   - Type 'back' to return to the main menu

## Project Structure

- **main.py**: The core application script containing the RAG pipeline, document processing, and interactive CLI interface.
- **requirements.txt**: Lists all Python package dependencies required to run the application.
- **rag_documents/**: Directory that stores all uploaded documents and their metadata.
  - **uploaded_docs.json**: JSON file containing metadata about all uploaded documents.
  - **doc_*.{pdf,txt}**: User-uploaded document files.
- **chroma_db_problm/**: Directory where Chroma DB stores the vector embeddings for document retrieval.

## Configuration

Edit the following variables in `main.py` to customize the application based on your requirements:

```python
# Model Configuration
LOCAL_MODEL_NAME = "llama3.2:1b"  # Local Ollama model name

# Embedding and Reranking
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Database
DB_PATH = "chroma_db_problm"  # Directory for vector store
```

## Dependencies

The RAG Pipeline requires the following Python packages (Installed via `pip install -r requirements.txt`):

- `langchain`: Core LangChain components
- `langchain-text-splitters`: Text splitting utilities for document processing
- `langchain-chroma`: Chroma vector store integration
- `langchain-huggingface`: HuggingFace models integration
- `langchain-ollama`: Local LLM integration via Ollama
- `langchain-google-genai`: Google Gemini API integration
- `langchain-community`: Community-maintained LangChain components
- `huggingface-hub`: Access to Hugging Face models and datasets
- `chromadb`: Vector database for document storage and retrieval
- `sentence-transformers`: Embedding models for text
- `rank-bm25`: BM25 algorithm for keyword-based retrieval
- `pypdf`: PDF text extraction
- `python-dotenv`: Environment variable management
- `rich`: Beautiful terminal formatting and user interface

## How It Works

1. **Document Processing**
   - Loads and processes PDFs and text documents
   - Splits content into manageable chunks with overlap
   - Generates embeddings for semantic search

2. **Retrieval System**
   - Implements hybrid search combining:
     - BM25 for keyword-based retrieval
     - Vector similarity for semantic search
   - Uses Reciprocal Rank Fusion (RRF) to combine results
   - Applies cross-encoder reranking for improved relevance

3. **Question Answering**
   - Processes natural language questions
   - Retrieves relevant document chunks
   - Generates accurate, context-aware answers

## Model Options

### Local LLM - Ollama
- Uses Ollama for local LLM inference 
- No API keys required
- Recommended model for most devices: `llama3.2:1b` (install via `ollama pull llama3.2:1b`)

### Hardware Requirements

#### Minimum (Basic Usage)
- **CPU**: 4 cores (Intel i5/Ryzen 5 or better)
- **RAM**: 8GB (16GB recommended for better performance)
- **Storage**: 5GB free space (SSD recommended for faster processing)
- **GPU**: Not required, but recommended for faster embeddings

#### Recommended (For Local LLM Usage)
- **CPU**: 8+ cores (Intel i7/Ryzen 7 or better)
- **RAM**: 16GB+ (32GB recommended for larger models)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for local LLM acceleration)
- **Storage**: 20GB+ free SSD space

### Software Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux
- **Python**: 3.8+
- **Ollama**: Required for local LLM support ([Download Ollama](https://ollama.com/))
- **Git**: For version control and cloning the repository ([Download Git](https://git-scm.com/downloads))

### Recommended Python Environment
- Use Python 3.10 or later for best compatibility
- Always use a virtual environment (included in setup instructions)
- Ensure you have the latest version of pip: `python -m pip install --upgrade pip`

## Notes

- For best results with local models, ensure you have sufficient system resources
- First document processing might take more time to generate embeddings
- The app maintains a local vector store for faster subsequent queries


