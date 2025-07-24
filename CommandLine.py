"""
Prob.lm - RAG Based Study Assistant
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

# Initialize Rich Console
console = Console()

try:
    import pypdf # Changed from PyPDF2 to pypdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    console.print("[yellow]Warning: pypdf not installed. PDF support disabled. Install with: pip install pypdf[/yellow]")
    
# Document Loading and splitting
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma Vector Store and Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Retrievers
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Prompt Chains (Prompt Engineering)
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# LLM Import
from langchain_ollama import OllamaLLM

# Local model name
LOCAL_MODEL_NAME = "llama3.2:1b" # Ensure this model is available in ollama (ollama pull)

# Embedding and Reranking Models
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Database Path
DB_PATH = "chroma_db_problm"

#===================================================================================================================
"""
RAG Document Upload CLI
Upload documents and extract text for RAG pipeline.
"""

# Front-End and Preprocessing
class DocumentUploader:
    def __init__(self, storage_dir: str = "rag_documents"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.docs_file = self.storage_dir / "uploaded_docs.json"
        self.documents = self._load_documents()

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
        """Upload a document and prepare its metadata for RAG pipeline (no text file created)."""
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
            console.print(Panel(f"[green]✓ Document uploaded: [bold]{doc_id}[/bold]\n"
                                f"       Title: {self.documents[doc_id]['title']}",title="Upload Success", border_style="green"))
            return True
        except Exception as e:
            console.print(Panel(f"[bold red]Error uploading document: {e}[/bold red]",
                                title="Upload Failed", border_style="red"))
            return False

    def list_documents(self):
        """List all uploaded documents."""
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

# Core RAG Pipeline
def initialize_llm():
    """Initializes the LLM based on the configuration."""
    console.print(f"[bold yellow]Using Local Model: {LOCAL_MODEL_NAME}[/bold yellow]")
    return OllamaLLM(model=LOCAL_MODEL_NAME, temperature=0.1)

def process_documents_for_rag(file_paths):
    """
    Loads and splits documents into chunks.
    """
    all_docs = []   # Stores all loaded document objects
    processed_file_paths = []   # Tracks paths of successfully processed files

    # Early return if no file paths provided
    if not file_paths:
        return [], []

    # Process each file in the input list
    for file_path in file_paths:
        target_path = Path(file_path)

        # Skip directories and notify user
        if target_path.is_dir():
            console.print(f"[yellow]Skipping directory path: {file_path}. Please upload individual files.[/yellow]")
            continue

        # Determine file type and initialize appropriate loader
        file_extension = target_path.suffix.lower()
        loader = None
        
        # Initialize appropriate document loader based on file extension
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
            # Load and process the document
            loaded_docs = loader.load()
            all_docs.extend(loaded_docs)
            processed_file_paths.append(file_path)
        except Exception as e:
            # Log errors but continue processing other files
            console.print(f"[bold red]Error loading {file_path} with {loader.__class__.__name__}: {e}[/bold red]")

    # Return empty results if no documents were successfully loaded
    if not all_docs:
        return [], []

    # Split documents into smaller chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # Target size of each chunk in characters
        chunk_overlap=150       # Overlap between chunks to maintain context
    )
    split_docs = text_splitter.split_documents(all_docs)

    return split_docs, processed_file_paths

def create_retriever(new_split_docs, newly_processed_file_paths, uploader):
    """
    Creates a powerful retriever with Hybrid Search and a Re-ranker.
    It will load existing Chroma DB and add new documents, and rebuild BM25 for all indexed documents.
    """
    # Setting up the brain for understanding text (embeddings)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = None

    db_path_obj = Path(DB_PATH)

    with console.status("[bold cyan]Setting up retriever...[/bold cyan]", spinner="dots"):
        # 1. Initialize or Load Chroma Vector Store
        # Checking if we already have a vector database (Chroma DB)
        if db_path_obj.exists() and any(db_path_obj.iterdir()):
            console.print("[cyan]✅ Chroma DB Initialized (loading existing)[/cyan]")
            vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

            if new_split_docs: # Only add documents if there are new ones to add
                try:
                    vector_store.add_documents(new_split_docs)
                    console.print("[green]✅ Added new document chunks to Chroma DB.[/green]")
                    # Mark newly processed documents as indexed
                    for path in newly_processed_file_paths:
                        uploader.mark_as_indexed(path)
                except Exception as e:
                    console.print(f"[bold red]Error adding new documents to existing Chroma DB: {e}[/bold red]")
                    console.print("[yellow]Consider deleting the 'chroma_db_problm' directory to rebuild if issues persist.[/yellow]")
            else:
                console.print("[cyan]No new documents to add to existing Chroma DB.[/cyan]")
        else:
            if not new_split_docs: # If no existing DB and no new docs, can't proceed
                console.print("[bold red]Cannot create new Chroma DB: No documents to process initially.[/bold red]")
                return None
            console.print("[cyan]✅ Creating new Chroma DB...[/cyan]")
            # If no DB, creating a new one from scratch with the new docs
            vector_store = Chroma.from_documents(documents=new_split_docs, embedding=embeddings, persist_directory=DB_PATH)
            # Mark all newly added documents as indexed
            for path in newly_processed_file_paths:
                uploader.mark_as_indexed(path)

    if vector_store is None:
        console.print("[bold red]Failed to initialize or load Chroma DB. Cannot create retriever.[/bold red]")
        return None

    # 2. Prepare ALL indexed documents for BM25 retriever (always rebuild BM25 with full corpus)
    # Getting ALL the documents that have been indexed so far to build a good search index
    all_indexed_file_paths = uploader.get_all_indexed_document_paths()
    with console.status("[bold cyan]Preparing all indexed documents for BM25 retriever...[/bold cyan]", spinner="dots"):
        all_chunks_for_bm25, _ = process_documents_for_rag(all_indexed_file_paths) # Re-use processing logic for all docs

    if not all_chunks_for_bm25:
        console.print("[bold red]No indexed documents available for BM25 retriever. Hybrid search may be limited to vector search.[/bold red]")
        # Fallback to just vector retriever if no docs available for BM25
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
        cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME)
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=vector_retriever
        )
        console.print("[cyan]✅ Vector Retriever only (BM25 not initialized).[/cyan]")
        console.print("[cyan]✅ Cross-Encoder Re-ranker initialized.[/cyan]")
        return compression_retriever

    # 3. Create Vector Retriever
    # This retriever finds documents based on their meaning (embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 15})

    # 4. Create BM25 Retriever from all chunks
    # Finds documents based on keyword matching 
    bm25_retriever = BM25Retriever.from_documents(all_chunks_for_bm25)
    bm25_retriever.k = 15
    console.print("[cyan]✅ BM25 Retriever initialized with all indexed documents.[/cyan]")


    # 5. Create Ensemble Retriever
    # Combining both the smart search (vector) and the keyword search (BM25) for better results!
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5], # Giving equal importance to both
        search_type="rrf" # combine results using rrf
    )
    console.print("[cyan]✅ Hybrid Retriever (BM25 + Vector with RRF) created.[/cyan]")

    # 6. Create Cross-Encoder Re-ranker
    # It takes the top results from the hybrid search and re-ranks them to find the most relevant ones
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=5) # Only keeping the top 5 after re-ranking

    # 7. Create Contextual Compression Retriever
    # This wraps everything up, making sure we get only the most relevant and compressed info
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    console.print("[cyan]✅ Cross-Encoder Re-ranker initialized.[/cyan]")
    return compression_retriever

def create_document_chain(llm):
    """Creates the LangChain document combining chain."""
    # This is the prompt that tells the LLM how to answer based on the context
    qa_prompt = ChatPromptTemplate.from_template(
        """You are an expert assistant. Your goal is to provide a comprehensive and accurate answer based *only* on the provided context.
        Synthesize the information from all relevant document chunks to form a cohesive answer.
        Do not add any information that is not present in the context.
        If the context does not contain the answer, state that clearly.
        Try to give the answers in a proper structured format point wise.

        Context:
        {context}

        Question: {input}

        Detailed Answer:
        """
    )
    # Hooking up the LLM with the prompt to make it answer questions
    return create_stuff_documents_chain(llm, qa_prompt)

def handle_user_query(query, retriever, document_chain):
    """Handles the user's query, explicitly getting context and then feeding to LLM."""
    with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
        console.print("[cyan]Performing hybrid search and re-ranking...[/cyan]")

        # Ask the retriever to find relevant documents for the query
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            console.print(Panel("[bold yellow]No relevant documents found for your query in the loaded context. Therefore, I cannot provide an answer based on the provided documents.[/bold yellow]",
                                 title="No Context Available", border_style="yellow"))
            return

        # Pass the retrieved_docs as context to the document_chain
        # Now, the LLM uses those retrieved docs to answer the question!
        answer = document_chain.invoke({"context": retrieved_docs, "input": query})

        console.print(Panel(Markdown(answer), title="💡 Answer", border_style="blue"))

        # Format and display sources (Citations)
        if retrieved_docs:
            console.print(Panel(format_sources(retrieved_docs), title="📚 Sources", border_style="yellow"))

def format_sources(docs):
    """Formats source documents for display, removing duplicates."""
    unique_sources = {}
    
    # Process each document to extract source information
    for doc in docs:
        # Extract source path and page number from document metadata
        source_path = doc.metadata.get('source', 'Unknown')
        source_name = os.path.basename(source_path)    # Get just the filename
        page_num = doc.metadata.get('page', -1)  # Default to -1 if no page number

        # Create a formatted string for this source
        source_info = f"📄 {source_name}"   
        
        # Add page number if available (page_num is not -1)
        if page_num != -1:
            source_info += f" (Page: {page_num + 1})"   # +1 because pages are 0-indexed

        # Use a tuple of (filename, page) as key to ensure uniqueness
        key = (source_name, page_num)
        unique_sources[key] = source_info

    # Handle case where no valid sources were found
    if not unique_sources:
        return "No specific sources found in the retrieved context."

    # Join all unique sources with newlines between them
    # Sorting ensures consistent ordering of sources in the output
    return "\n".join(source_text for key, source_text in sorted(unique_sources.items()))

#==============================================================================================================

# Front end with execution loops and error handling
def main():
    # Initialize the application with a welcome banner
    console.print(Panel(
        "[bold magenta]✨ Prob.lm - RAG Study Assistant ✨[/bold magenta]",
        subtitle="[default]Powered by Hybrid Search[/default]",
        expand=False, 
        width=80
    ))

    # Initialize core components
    uploader = DocumentUploader()   # Handles document storage and metadata
    llm = initialize_llm()  # Initialize the language model

    # Will be set up when documents are processed
    retriever = None
    document_chain = None

    # Main application loop
    while True:
        try:
            # Display main menu options
            console.print("\n[bold]Main Menu:[/bold]")
            console.print("[green]1. Upload Document(s)[/green]")
            console.print("[cyan]2. List Uploaded Documents[/cyan]")
            console.print("[blue]3. Process Documents & Start Q&A (RAG Pipeline)[/blue]")
            console.print("[red]4. Exit[/red]")

            # Get user choice with input validation
            choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4"])

            # Handle document upload
            if choice == "1":
                file_path = Prompt.ask(
                    "Enter the full path to the document file (e.g., /path/to/my_doc.pdf)"
                )
                # Validate file exists before proceeding
                if not Path(file_path).exists():
                    console.print(f"[bold red]Error: File not found at {file_path}[/bold red]")
                    continue
                # Allow custom title or use filename as default
                title = Prompt.ask(
                    "Enter a custom title for the document (optional)", 
                    default=None
                )
                uploader.upload_document(file_path, title)

            # List all uploaded documents
            elif choice == "2":
                uploader.list_documents()

            # Process documents and start Q&A
            elif choice == "3":
                # Get paths of documents that haven't been indexed yet
                unindexed_file_paths = uploader.get_unindexed_document_paths()

                new_split_docs = []
                newly_processed_file_paths = []

                if unindexed_file_paths:
                    with console.status("[bold cyan]Processing new unindexed documents...[/bold cyan]", spinner="dots"):
                        new_split_docs, newly_processed_file_paths = process_documents_for_rag(unindexed_file_paths)
                    if new_split_docs:
                        unique_sources_new = len({doc.metadata.get('source', '') for doc in new_split_docs if hasattr(doc, 'metadata')})
                        console.print(f"[green]✅ Processed {unique_sources_new} new document(s) and created {len(new_split_docs)} chunks for Chroma indexing.[/green]")
                    else:
                        console.print("[yellow]No supported new documents found to process for initial Chroma indexing.[/yellow]")
                else:
                    console.print("[cyan]No new documents to process. Loading existing indexed data.[/cyan]")
                    new_split_docs = []
                    newly_processed_file_paths = []

                # Create/update retriever. This function will handle loading existing Chroma
                # and adding `new_split_docs` to it, and rebuilding BM25 with *all* indexed docs.
                retriever = create_retriever(new_split_docs, newly_processed_file_paths, uploader)
                    
                if not retriever:
                    console.print("[bold red]Failed to create retriever. Exiting Q&A mode.[/bold red]")
                    continue
                
                document_chain = create_document_chain(llm)
                if not document_chain:
                    console.print("[bold red]Failed to create RAG document chain. Exiting Q&A mode.[/bold red]")
                    continue

                console.print("\n[bold green]RAG Pipeline is ready! You can now ask questions.[/bold green]")
                console.print("[italic]Type 'back' to return to the main menu.[/italic]")

                while True:
                    try:
                        query = Prompt.ask("[bold cyan]Ask a question (or 'back') [/bold cyan]")
                        if query.lower() == 'back':
                            break
                        if not query.strip():
                            continue
                            
                        # Process the user's question
                        handle_user_query(query, retriever, document_chain)
                        
                    except KeyboardInterrupt:
                        console.print("\n[yellow]Returning to main menu...[/yellow]")
                        break
                    except Exception as e:
                        # Log any errors during query processing
                        console.print(f"[bold red]An error occurred during query processing: {e}[/bold red]")
                        import traceback
                        traceback.print_exc()

            # Exit the application
            elif choice == "4":
                console.print("\n[bold magenta]Thank you! Hope you had a productive study session![/bold magenta]")
                break
                
        except KeyboardInterrupt:
            console.print("\n[bold magenta]Exiting... Thank You! [/bold magenta]")
            sys.exit(0)

if __name__ == "__main__":
    main()