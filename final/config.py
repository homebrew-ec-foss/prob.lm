# /problm_rag_app/config.py
"""
Centralized configuration for the RAG application.
Includes model names, paths, feature toggles, and hardware detection.
"""
import torch
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# ==============================================================================
# CONSOLE & ENVIRONMENT
# ==============================================================================
console = Console()

# ==============================================================================
# GPU DETECTION & SETUP
# ==============================================================================
def detect_gpu_setup():
    """Detects and prints GPU information."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        console.print(Panel(
            f"[bold green]GPU Detected![/bold green]\n"
            f"Device: {device_name}\n"
            f"Memory: {memory_gb:.1f} GB",
            title="GPU Configuration",
            border_style="green"
        ))
        return {'device': 'cuda', 'name': device_name}
    else:
        console.print("[yellow]No GPU detected. Using CPU instead.[/yellow]")
        return {'device': 'cpu', 'name': 'cpu'}

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
GPU_CONFIG = detect_gpu_setup()
MODEL_KWARGS = {'device': GPU_CONFIG['device']}

# --- Application Toggles ---
USE_API_LLM = False              # True for Groq API, False for local Ollama
USE_SEMANTIC_CHUNKING = False    # True for semantic chunking, False for faster recursive chunking

# --- Model Names ---
LOCAL_MODEL_NAME = "llama3.2:1b" # Local model to use with Ollama
API_MODEL_NAME = "gemma2-9b-it"  # API model to use with Groq
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Path Configuration ---
DB_PATH = "chroma_db_problm"
STORAGE_DIR = Path("rag_documents")
CACHE_DIR = Path("problm_cache")
BM25_CACHE_FILE = CACHE_DIR / "bm25_retriever.pkl"

# --- Create necessary directories ---
STORAGE_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# --- Dependency Availability Flags ---
# These flags prevent crashes if optional libraries are not installed.
try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False

try:
    import fitz
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
print("PDF_SUPPORT:", PDF_SUPPORT)

try:
    import groq
    GROQ_API_AVAILABLE = True
except ImportError:
    GROQ_API_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False