"""
Prob.lm - RAG Based Study Assistant with GPU/CUDA Support
"""

# Base imports
import os
import json
import sys
from pathlib import Path
from datetime import datetime

# Rich - User-friendly frontend
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table, box

# GPU Detection and Management
import torch

# Initialize Rich Console
console = Console()

# GPU Detection and Setup
def detect_gpu_setup():
    """Detect and configure GPU usage for the pipeline."""
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': None,
        'device': 'cpu',
        'memory_gb': 0
    }
    
    if gpu_info['cuda_available']:
        gpu_info['device'] = 'cuda'
        gpu_info['device_name'] = torch.cuda.get_device_name(0)
        gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Set memory allocation strategy for better performance
        torch.cuda.empty_cache()
        
        console.print(Panel(
            f"[bold green]🚀 GPU Detected![/bold green]\n"
            f"Device: {gpu_info['device_name']}\n"
            f"Memory: {gpu_info['memory_gb']:.1f} GB\n"
            f"CUDA Version: {torch.version.cuda}",
            title="GPU Configuration",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "[yellow]⚠️  No GPU detected. Using CPU mode.\n"
            "For better performance, install CUDA-compatible PyTorch:\n"
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121[/yellow]",
            title="CPU Mode",
            border_style="yellow"
        ))
    
    return gpu_info

# Initialize GPU configuration
GPU_CONFIG = detect_gpu_setup()

try:
    import pypdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    console.print("[yellow]Warning: pypdf not installed. PDF support disabled. Install with: pip install pypdf[/yellow]")

# Document Loading and splitting
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma Vector Store and GPU-Enhanced Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#semantic chunking imports
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document


# Retrievers
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Prompt Chains (Prompt Engineering)
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# LLM Import with GPU support
from langchain_ollama import OllamaLLM

# GPU-Optimized Model Configurations
LOCAL_MODEL_NAME = "gemma3:4b"  # Ensure this model is available in ollama

# GPU-Enhanced Embedding Configuration
if GPU_CONFIG['cuda_available']:
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Use GPU-optimized models for better performance
    MODEL_KWARGS = {'device': 'cuda'}
    ENCODE_KWARGS = {'normalize_embeddings': True, 'batch_size': 32}  # Larger batch for GPU
else:
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MODEL_KWARGS = {'device': 'cpu'}
    ENCODE_KWARGS = {'normalize_embeddings': True, 'batch_size': 8}   # Smaller batch for CPU

# Database Path
DB_PATH = "chroma_db_problm"

#===================================================================================================================

# GPU Memory Management Utilities
class GPUMemoryManager:
    """Manages GPU memory for optimal performance."""
    
    @staticmethod
    def clear_cache():
        """Clear GPU cache to free memory."""
        if GPU_CONFIG['cuda_available']:
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_usage():
        """Get current GPU memory usage."""
        if GPU_CONFIG['cuda_available']:
            allocated = torch.cuda.memory_allocated() / (1024**3)
            cached = torch.cuda.memory_reserved() / (1024**3)
            return {'allocated': allocated, 'cached': cached}
        return {'allocated': 0, 'cached': 0}
    
    @staticmethod
    def optimize_batch_size(total_docs, base_batch_size=32):
        """Dynamically adjust batch size based on available GPU memory."""
        if not GPU_CONFIG['cuda_available']:
            return min(8, total_docs)  # CPU fallback
        
        memory_gb = GPU_CONFIG['memory_gb']
        if memory_gb >= 8:
            return min(base_batch_size, total_docs)
        elif memory_gb >= 4:
            return min(16, total_docs)
        else:
            return min(8, total_docs)

#===================================================================================================================

# Enhanced DocumentUploader with GPU optimization hints
class DocumentUploader:
    def __init__(self, storage_dir: str = "rag_documents"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.docs_file = self.storage_dir / "uploaded_docs.json"
        self.documents = self._load_documents()
        self.gpu_manager = GPUMemoryManager()

    def _load_documents(self) -> dict:
        """Load list of uploaded documents."""
        if self.docs_file.exists():
            try:
                with open(self.docs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for doc_id, doc_info in data.items():
                        if 'is_indexed' not in doc_info:
                            doc_info['is_indexed'] = False
                    return data
            except (json.JSONDecodeError, IOError):
                console.print("[yellow]Warning: Could not load documents file. Starting fresh.[/yellow]")
        return {}

    def _save_documents(self):
        """Save document list to file."""
        try:
            with open(self.docs_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2, default=str)
        except IOError as e:
            console.print(f"[bold red]Error saving documents: {e}[/bold red]")

    def upload_document(self, file_path: str, title: str = None) -> bool:
        """Upload a document and prepare its metadata for RAG pipeline."""
        source_path = Path(file_path)
        if not source_path.exists():
            console.print(f"[bold red]Error: File not found: {file_path}[/bold red]")
            return False

        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{source_path.stem}"
        file_extension = source_path.suffix.lower()
        stored_path = self.storage_dir / f"{doc_id}{file_extension}"

        try:
            import shutil
            shutil.copy2(source_path, stored_path)

            self.documents[doc_id] = {
                "title": title or source_path.name,
                "original_path": str(source_path),
                "stored_file": str(stored_path),
                "file_type": file_extension,
                "size": source_path.stat().st_size,
                "upload_date": datetime.now().isoformat(),
                "is_indexed": False
            }
            self._save_documents()
            
            # Show GPU optimization hint for large documents
            size_mb = source_path.stat().st_size / (1024*1024)
            gpu_hint = ""
            if size_mb > 10 and GPU_CONFIG['cuda_available']:
                gpu_hint = f"\n       [dim]💡 Large document detected ({size_mb:.1f}MB) - GPU acceleration will help with processing[/dim]"
            
            console.print(Panel(f"[green]✓ Document uploaded: [bold]{doc_id}[/bold]\n"
                                f"       Title: {self.documents[doc_id]['title']}{gpu_hint}",
                                title="Upload Success", border_style="green"))
            return True
        except Exception as e:
            console.print(Panel(f"[bold red]Error uploading document: {e}[/bold red]",
                                title="Upload Failed", border_style="red"))
            return False

    def list_documents(self):
        """List all uploaded documents with GPU memory status."""
        if not self.documents:
            console.print("[yellow]No documents uploaded yet.[/yellow]")
            return

        table = Table(title="Uploaded Documents", border_style="blue", header_style="bold", box=box.SQUARE)
        table.add_column("ID", style="cyan", header_style="bold cyan", no_wrap=True)
        table.add_column("Title", style="magenta", header_style="bold magenta")
        table.add_column("Type", style="green", header_style="bold green")
        table.add_column("Size", style="yellow", header_style="bold yellow")
        table.add_column("Upload Date", style="dim", header_style="bold dim")
        table.add_column("Indexed?", style="white", header_style="bold white")

        for doc_id, doc_info in self.documents.items():
            title = doc_info['title']
            if len(title) > 38:
                title = title[:35] + "..."
            size = self._format_size(doc_info['size'])
            date = doc_info['upload_date'][:10]
            indexed_status = "[green]Yes[/green]" if doc_info.get('is_indexed', False) else "[red]No[/red]"
            table.add_row(doc_id, title, doc_info['file_type'], size, date, indexed_status)

        console.print(table)
        console.print(f" [bold green]Total documents: {len(self.documents)}[/bold green]")
        
        # Show GPU memory status if available
        if GPU_CONFIG['cuda_available']:
            memory = self.gpu_manager.get_memory_usage()
            console.print(f" [dim]GPU Memory: {memory['allocated']:.2f}GB allocated, {memory['cached']:.2f}GB cached[/dim]")

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"

    def get_unindexed_document_paths(self) -> list:
        """Returns a list of 'stored_file' paths for documents not yet indexed."""
        return [doc_info['stored_file'] for doc_id, doc_info in self.documents.items() if not doc_info.get('is_indexed', False)]

    def get_all_indexed_document_paths(self) -> list:
        """Returns a list of 'stored_file' paths for all documents that have been indexed."""
        return [doc_info['stored_file'] for doc_id, doc_info in self.documents.items() if doc_info.get('is_indexed', False)]

    def mark_as_indexed(self, stored_file_path: str):
        """Marks a document as indexed in the document tracking."""
        for doc_id, doc_info in self.documents.items():
            if doc_info['stored_file'] == stored_file_path:
                doc_info['is_indexed'] = True
                self._save_documents()
                return

#==============================================================================================

# GPU-Enhanced Core RAG Pipeline
def initialize_llm():
    """Initializes the LLM with GPU support if available."""
    if GPU_CONFIG['cuda_available']:
        console.print(f"[bold green]🚀 Using Local Model with GPU acceleration: {LOCAL_MODEL_NAME}[/bold green]")
        # Ollama can utilize GPU automatically if properly configured
        return OllamaLLM(model=LOCAL_MODEL_NAME, temperature=0.1)
    else:
        console.print(f"[bold yellow]Using Local Model (CPU): {LOCAL_MODEL_NAME}[/bold yellow]")
        return OllamaLLM(model=LOCAL_MODEL_NAME, temperature=0.1)

def process_documents_for_rag(file_paths):
    """
    Loads and splits documents into semantic chunks with GPU-optimized processing.
    """
    all_docs = []
    processed_file_paths = []

    if not file_paths:
        return [], []

    # GPU memory optimization: clear cache before processing
    if GPU_CONFIG['cuda_available']:
        GPUMemoryManager.clear_cache()

    for file_path in file_paths:
        target_path = Path(file_path)

        if target_path.is_dir():
            console.print(f"[yellow]Skipping directory path: {file_path}. Please upload individual files.[/yellow]")
            continue

        file_extension = target_path.suffix.lower()
        loader = None

        if file_extension == ".pdf":
            if not PDF_SUPPORT:
                console.print(f"[yellow]Skipping PDF {file_path}: pypdf not installed.[/yellow]")
                continue
            loader = PyPDFLoader(str(target_path))
        elif file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']:
            loader = TextLoader(str(target_path), encoding='utf-8')
        else:
            console.print(f"[yellow]Skipping unsupported file type: {file_path}[/yellow]")
            continue

        try:
            loaded_docs = loader.load()
            all_docs.extend(loaded_docs)
            processed_file_paths.append(file_path)
        except Exception as e:
            console.print(f"[bold red]Error loading {file_path} with {loader.__class__.__name__}: {e}[/bold red]")

    if not all_docs:
        return [], []

    # GPU-optimized semantic chunking setup
    with console.status("[bold cyan]🧠 Setting up GPU-enhanced semantic chunking...[/bold cyan]", spinner="dots"):
        # Initialize embeddings for semantic chunking (same as your existing embeddings)
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=MODEL_KWARGS,
            encode_kwargs=ENCODE_KWARGS
        )
        
        # Create semantic chunker with GPU optimization
        semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",  # Options: "percentile", "standard_deviation", "interquartile"
            breakpoint_threshold_amount=95,  # Adjust this value (80-99) to control chunk sensitivity
            number_of_chunks=None,  # Let it determine naturally
            buffer_size=1  # Number of sentences to group together for comparison
        )

    # Process documents with semantic chunking
    console.print("[cyan]🧠 Applying GPU-accelerated semantic chunking...[/cyan]")
    
    try:
        # Clear GPU cache before semantic processing
        if GPU_CONFIG['cuda_available']:
            GPUMemoryManager.clear_cache()
        
        split_docs = semantic_chunker.split_documents(all_docs)
        
        # Optional: Filter out very small or very large chunks for better performance
        filtered_docs = []
        for doc in split_docs:
            chunk_length = len(doc.page_content)
            if 100 <= chunk_length <= 3000:  # Adjust these thresholds as needed
                filtered_docs.append(doc)
            elif chunk_length < 100:
                console.print(f"[dim]Skipping very small chunk ({chunk_length} chars)[/dim]")
            elif chunk_length > 3000:
                # Split large semantic chunks further if needed
                large_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=200
                )
                sub_chunks = large_splitter.split_documents([doc])
                filtered_docs.extend(sub_chunks)
                console.print(f"[dim]Split large semantic chunk ({chunk_length} chars) into {len(sub_chunks)} sub-chunks[/dim]")
        
        console.print(f"[green]✅ Created {len(filtered_docs)} semantic chunks from {len(all_docs)} documents[/green]")
        return filtered_docs, processed_file_paths
        
    except Exception as e:
        console.print(f"[bold red]Error in semantic chunking: {e}[/bold red]")
        console.print("[yellow]Falling back to recursive character splitting...[/yellow]")
        
        # Fallback to your original recursive chunking
        chunk_size = 1200 if GPU_CONFIG['cuda_available'] else 1000
        chunk_overlap = 200 if GPU_CONFIG['cuda_available'] else 150
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        split_docs = text_splitter.split_documents(all_docs)
        return split_docs, processed_file_paths

def create_gpu_enhanced_retriever(new_split_docs, newly_processed_file_paths, uploader):
    """
    Creates a GPU-enhanced retriever with optimized Hybrid Search and Re-ranker.
    """
    # GPU-optimized embeddings setup
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=MODEL_KWARGS,
        encode_kwargs=ENCODE_KWARGS
    )
    
    vector_store = None
    db_path_obj = Path(DB_PATH)

    with console.status("[bold cyan]🚀 Setting up GPU-enhanced retriever...[/bold cyan]", spinner="dots"):
        # Clear GPU cache before vector operations
        if GPU_CONFIG['cuda_available']:
            GPUMemoryManager.clear_cache()
        
        # Initialize or Load Chroma Vector Store
        if db_path_obj.exists() and any(db_path_obj.iterdir()):
            console.print("[cyan]✅ Chroma DB Initialized (loading existing)[/cyan]")
            vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

            if new_split_docs:
                try:
                    # GPU optimization: process in optimized batches
                    batch_size = GPUMemoryManager.optimize_batch_size(len(new_split_docs))
                    console.print(f"[dim]Processing {len(new_split_docs)} chunks in batches of {batch_size}[/dim]")
                    
                    # Process documents in batches for better GPU memory management
                    for i in range(0, len(new_split_docs), batch_size):
                        batch = new_split_docs[i:i+batch_size]
                        vector_store.add_documents(batch)
                        if GPU_CONFIG['cuda_available'] and i % (batch_size * 4) == 0:
                            GPUMemoryManager.clear_cache()  # Periodic cleanup
                    
                    console.print("[green]✅ Added new document chunks to Chroma DB with GPU acceleration.[/green]")
                    
                    for path in newly_processed_file_paths:
                        uploader.mark_as_indexed(path)
                except Exception as e:
                    console.print(f"[bold red]Error adding new documents to existing Chroma DB: {e}[/bold red]")
                    console.print("[yellow]Consider deleting the 'chroma_db_problm' directory to rebuild if issues persist.[/yellow]")
            else:
                console.print("[cyan]No new documents to add to existing Chroma DB.[/cyan]")
        else:
            if not new_split_docs:
                console.print("[bold red]Cannot create new Chroma DB: No documents to process initially.[/bold red]")
                return None
            
            console.print("[cyan]✅ Creating new Chroma DB with GPU acceleration...[/cyan]")
            
            # GPU optimization for initial vector store creation
            batch_size = GPUMemoryManager.optimize_batch_size(len(new_split_docs))
            
            if len(new_split_docs) > batch_size:
                # Create initial store with first batch
                vector_store = Chroma.from_documents(
                    documents=new_split_docs[:batch_size],
                    embedding=embeddings,
                    persist_directory=DB_PATH
                )
                
                # Add remaining documents in batches
                for i in range(batch_size, len(new_split_docs), batch_size):
                    batch = new_split_docs[i:i+batch_size]
                    vector_store.add_documents(batch)
                    if GPU_CONFIG['cuda_available']:
                        GPUMemoryManager.clear_cache()
            else:
                vector_store = Chroma.from_documents(
                    documents=new_split_docs,
                    embedding=embeddings,
                    persist_directory=DB_PATH
                )
            
            for path in newly_processed_file_paths:
                uploader.mark_as_indexed(path)

    if vector_store is None:
        console.print("[bold red]Failed to initialize or load Chroma DB. Cannot create retriever.[/bold red]")
        return None

    # Prepare ALL indexed documents for BM25 retriever
    all_indexed_file_paths = uploader.get_all_indexed_document_paths()
    with console.status("[bold cyan]Preparing BM25 retriever with GPU optimization...[/bold cyan]", spinner="dots"):
        all_chunks_for_bm25, _ = process_documents_for_rag(all_indexed_file_paths)

    if not all_chunks_for_bm25:
        console.print("[bold red]No indexed documents available for BM25 retriever. Using vector search only.[/bold red]")
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
        
        # GPU-enhanced cross-encoder
        cross_encoder = HuggingFaceCrossEncoder(
            model_name=RERANKER_MODEL_NAME,
            model_kwargs=MODEL_KWARGS
        )
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=vector_retriever
        )
        console.print("[cyan]✅ GPU-Enhanced Vector Retriever (BM25 not initialized).[/cyan]")
        return compression_retriever

    # Create optimized retrievers
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    bm25_retriever = BM25Retriever.from_documents(all_chunks_for_bm25)
    bm25_retriever.k = 15
    console.print("[cyan]✅ BM25 Retriever initialized with all indexed documents.[/cyan]")

    # Create Ensemble Retriever with optimized weights for GPU processing
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],  # Slightly favor vector search for GPU optimization
        search_type="rrf"
    )
    console.print("[cyan]✅ GPU-Optimized Hybrid Retriever (BM25 + Vector with RRF) created.[/cyan]")

    # GPU-Enhanced Cross-Encoder Re-ranker
    cross_encoder = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL_NAME,
        model_kwargs=MODEL_KWARGS
    )
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )
    
    device_info = f" (GPU: {GPU_CONFIG['device_name']})" if GPU_CONFIG['cuda_available'] else " (CPU)"
    console.print(f"[cyan]✅ GPU-Enhanced Cross-Encoder Re-ranker initialized{device_info}.[/cyan]")
    
    return compression_retriever

def create_document_chain(llm):
    """Creates the LangChain document combining chain with GPU-optimized prompt."""
    qa_prompt = ChatPromptTemplate.from_template(
        """You are an expert assistant powered by advanced GPU-accelerated retrieval. Your goal is to provide a comprehensive and accurate answer based *only* on the provided context.
        Synthesize the information from all relevant document chunks to form a cohesive answer.
        Do not add any information that is not present in the context.
        If the context does not contain the answer, state that clearly.
        Structure your answers clearly with proper formatting and bullet points when appropriate.

        Context:
        {context}

        Question: {input}

        Detailed Answer:
        """
    )
    return create_stuff_documents_chain(llm, qa_prompt)

def handle_user_query_gpu(query, retriever, document_chain):
    """Handles user queries with GPU acceleration and memory management."""
    with console.status("[bold cyan]🚀 GPU-accelerated thinking...[/bold cyan]", spinner="dots"):
        console.print("[cyan]Performing GPU-enhanced hybrid search and re-ranking...[/cyan]")

        # Clear GPU cache before retrieval for optimal performance
        if GPU_CONFIG['cuda_available']:
            GPUMemoryManager.clear_cache()

        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            console.print(Panel("[bold yellow]No relevant documents found for your query in the loaded context. Therefore, I cannot provide an answer based on the provided documents.[/bold yellow]",
                                 title="No Context Available", border_style="yellow"))
            return

        # GPU memory optimization during LLM inference
        try:
            answer = document_chain.invoke({"context": retrieved_docs, "input": query})
            
            # Display enhanced answer with GPU indicator
            gpu_indicator = "🚀 GPU-Accelerated" if GPU_CONFIG['cuda_available'] else "💻 CPU-Powered"
            console.print(Panel(Markdown(answer), title=f"💡 Answer ({gpu_indicator})", border_style="blue"))

            # Clear cache after processing
            if GPU_CONFIG['cuda_available']:
                GPUMemoryManager.clear_cache()

        except Exception as e:
            console.print(f"[bold red]Error during GPU-accelerated processing: {e}[/bold red]")
            if GPU_CONFIG['cuda_available']:
                console.print("[yellow]Trying to clear GPU cache and retry...[/yellow]")
                GPUMemoryManager.clear_cache()

        # Format and display sources
        if retrieved_docs:
            console.print(Panel(format_sources(retrieved_docs), title="📚 Sources", border_style="yellow"))

def format_sources(docs):
    """Formats source documents for display, removing duplicates."""
    unique_sources = {}

    for doc in docs:
        source_path = doc.metadata.get('source', 'Unknown')
        source_name = os.path.basename(source_path)
        page_num = doc.metadata.get('page', -1)

        source_info = f"📄 {source_name}"
        if page_num != -1:
            source_info += f" (Page: {page_num + 1})"

        key = (source_name, page_num)
        unique_sources[key] = source_info

    if not unique_sources:
        return "No specific sources found in the retrieved context."

    return "\n".join(source_text for key, source_text in sorted(unique_sources.items()))

#==============================================================================================================

# GPU-Enhanced Frontend with Performance Monitoring
def main():
    console.print(Panel(
        f"[bold magenta]✨ Prob.lm - GPU-Enhanced RAG Study Assistant ✨[/bold magenta]\n"
        f"[dim]Powered by Hybrid Search + {GPU_CONFIG['device'].upper()} Acceleration[/dim]",
        subtitle=f"[default]Device: {GPU_CONFIG.get('device_name', 'CPU')}[/default]",
        expand=False,
        width=80
    ))

    uploader = DocumentUploader()
    llm = initialize_llm()

    retriever = None
    document_chain = None

    while True:
        try:
            console.print("\n[bold]Main Menu:[/bold]")
            console.print("[green]1. Upload Document(s)[/green]")
            console.print("[cyan]2. List Uploaded Documents[/cyan]")
            console.print("[blue]3. Process Documents & Start Q&A (GPU-Enhanced RAG Pipeline)[/blue]")
            if GPU_CONFIG['cuda_available']:
                console.print("[magenta]4. GPU Status & Memory Info[/magenta]")
                console.print("[red]5. Exit[/red]")
                choices = ["1", "2", "3", "4", "5"]
            else:
                console.print("[red]4. Exit[/red]")
                choices = ["1", "2", "3", "4"]

            choice = Prompt.ask("Choose an option", choices=choices)

            if choice == "1":
                file_path = Prompt.ask(
                    "Enter the full path to the document file (e.g., /path/to/my_doc.pdf)"
                )
                if not Path(file_path).exists():
                    console.print(f"[bold red]Error: File not found at {file_path}[/bold red]")
                    continue
                title = Prompt.ask(
                    "Enter a custom title for the document (optional)",
                    default=None
                )
                uploader.upload_document(file_path, title)

            elif choice == "2":
                uploader.list_documents()

            elif choice == "3":
                unindexed_file_paths = uploader.get_unindexed_document_paths()

                new_split_docs = []
                newly_processed_file_paths = []

                if unindexed_file_paths:
                    with console.status("[bold cyan]🚀 GPU-accelerated document processing...[/bold cyan]", spinner="dots"):
                        new_split_docs, newly_processed_file_paths = process_documents_for_rag(unindexed_file_paths)
                    if new_split_docs:
                        unique_sources_new = len({doc.metadata.get('source', '') for doc in new_split_docs if hasattr(doc, 'metadata')})
                        device_info = f" using {GPU_CONFIG['device'].upper()}" if GPU_CONFIG['cuda_available'] else ""
                        console.print(f"[green]✅ Processed {unique_sources_new} new document(s) and created {len(new_split_docs)} chunks{device_info}.[/green]")
                    else:
                        console.print("[yellow]No supported new documents found to process for initial Chroma indexing.[/yellow]")
                else:
                    console.print("[cyan]No new documents to process. Loading existing indexed data.[/cyan]")

                retriever = create_gpu_enhanced_retriever(new_split_docs, newly_processed_file_paths, uploader)

                if not retriever:
                    console.print("[bold red]Failed to create GPU-enhanced retriever. Exiting Q&A mode.[/bold red]")
                    continue

                document_chain = create_document_chain(llm)
                if not document_chain:
                    console.print("[bold red]Failed to create RAG document chain. Exiting Q&A mode.[/bold red]")
                    continue

                performance_indicator = "GPU-Accelerated" if GPU_CONFIG['cuda_available'] else "CPU-Powered"
                console.print(f"\n[bold green]🚀 {performance_indicator} RAG Pipeline is ready! You can now ask questions.[/bold green]")
                console.print("[italic]Type 'back' to return to the main menu.[/italic]")

                while True:
                    try:
                        query = Prompt.ask("[bold cyan]Ask a question (or 'back') [/bold cyan]")
                        if query.lower() == 'back':
                            break
                        if not query.strip():
                            continue

                        handle_user_query_gpu(query, retriever, document_chain)

                    except KeyboardInterrupt:
                        console.print("\n[yellow]Returning to main menu...[/yellow]")
                        break
                    except Exception as e:
                        console.print(f"[bold red]An error occurred during query processing: {e}[/bold red]")
                        if GPU_CONFIG['cuda_available']:
                            console.print("[yellow]Clearing GPU cache and continuing...[/yellow]")
                            GPUMemoryManager.clear_cache()

            elif choice == "4" and GPU_CONFIG['cuda_available']:
                # GPU Status and Memory Information
                memory_info = GPUMemoryManager.get_memory_usage()
                
                gpu_status_table = Table(title="GPU Status & Performance", border_style="green", box=box.ROUNDED)
                gpu_status_table.add_column("Metric", style="cyan", header_style="bold cyan")
                gpu_status_table.add_column("Value", style="green", header_style="bold green")
                
                gpu_status_table.add_row("Device Name", GPU_CONFIG['device_name'])
                gpu_status_table.add_row("Total Memory", f"{GPU_CONFIG['memory_gb']:.1f} GB")
                gpu_status_table.add_row("Allocated Memory", f"{memory_info['allocated']:.2f} GB")
                gpu_status_table.add_row("Cached Memory", f"{memory_info['cached']:.2f} GB")
                gpu_status_table.add_row("Free Memory", f"{GPU_CONFIG['memory_gb'] - memory_info['cached']:.2f} GB")
                gpu_status_table.add_row("CUDA Version", str(torch.version.cuda))
                gpu_status_table.add_row("PyTorch Version", torch.__version__)
                
                console.print(gpu_status_table)
                
                # Memory cleanup option
                cleanup_choice = Prompt.ask(
                    "\n[yellow]Clear GPU cache to free memory?[/yellow]", 
                    choices=["y", "n"], 
                    default="n"
                )
                if cleanup_choice.lower() == "y":
                    GPUMemoryManager.clear_cache()
                    new_memory = GPUMemoryManager.get_memory_usage()
                    console.print(f"[green]✅ GPU cache cleared! Memory freed: {memory_info['cached'] - new_memory['cached']:.2f} GB[/green]")

            elif (choice == "5" and GPU_CONFIG['cuda_available']) or (choice == "4" and not GPU_CONFIG['cuda_available']):
                console.print("\n[bold magenta]Thank you! Hope you had a productive GPU-accelerated study session![/bold magenta]")
                if GPU_CONFIG['cuda_available']:
                    console.print("[dim]Clearing GPU cache before exit...[/dim]")
                    GPUMemoryManager.clear_cache()
                break

        except KeyboardInterrupt:
            console.print("\n[bold magenta]Exiting... Cleaning up GPU resources... Thank You! [/bold magenta]")
            if GPU_CONFIG['cuda_available']:
                GPUMemoryManager.clear_cache()
            sys.exit(0)
        except Exception as e:
            console.print(f"[bold red]Unexpected error: {e}[/bold red]")
            if GPU_CONFIG['cuda_available']:
                console.print("[yellow]Clearing GPU cache and continuing...[/yellow]")
                GPUMemoryManager.clear_cache()

if __name__ == "__main__":
    main()