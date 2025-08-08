# /problm_rag_app/app.py
"""
Main application file for the Prob.lm RAG Study Assistant.
Handles the Command-Line Interface (CLI), user interactions, and orchestrates
the document management and RAG pipeline modules.
"""
import sys
from pathlib import Path

# Rich UI Components
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

# Local module imports
from config import console, USE_SEMANTIC_CHUNKING, PDF_SUPPORT
from document_manager import DocumentUploader
from rag_pipeline import build_rag_pipeline

# LangChain components for context expansion
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- UI & Formatting Functions ---

def display_welcome_banner():
    """Displays the application's welcome banner."""
    chunking_type = "Semantic" if USE_SEMANTIC_CHUNKING else "Recursive"
    console.print(Panel(
        "[bold magenta]✨ Prob.lm - RAG Study Assistant ✨[/bold magenta]",
        subtitle="[cyan]Powered by Hybrid Search & Reranking[/cyan]",
        expand=False
    ))
    console.print(f"[green]Chunking Mode: {chunking_type}[/green]")

def format_sources(docs):
    """Formats source documents for display, grouping page numbers."""
    if not docs:
        return "No sources found."
    
    sources = {}
    for doc in docs:
        source_name = Path(doc.metadata.get('source', 'Unknown')).name
        page = doc.metadata.get('page', -1)
        if source_name not in sources:
            sources[source_name] = set()
        if page != -1:
            sources[source_name].add(page + 1)
    
    return "\n".join(
        f"📄 {name} (Pages: {', '.join(map(str, sorted(pages)))})"
        for name, pages in sorted(sources.items())
    )

# --- Query Processing ---

def expand_context(final_docs):
    """
    Expands the retrieved context by including neighboring chunks from the
    original documents to provide better continuity for the LLM.
    """
    if not final_docs: return []
    
    source_files = {doc.metadata.get('source') for doc in final_docs}
    all_pages_in_context = {
        (doc.metadata.get('source'), doc.metadata.get('page')) for doc in final_docs
    }

    # Load all chunks from the source documents of the retrieved docs
    full_doc_chunks_map = {}
    for source_file in source_files:
        if not source_file: continue
        if Path(source_file).suffix.lower() == ".pdf" and PDF_SUPPORT:
            loader = PyMuPDFLoader(str(source_file))
        else:
            loader = TextLoader(str(source_file), encoding='utf-8')
        
        full_doc = loader.load()
        if full_doc:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            full_doc_chunks_map[source_file] = text_splitter.split_documents(full_doc)

    expanded_chunks = {doc.page_content: doc for doc in final_docs}

    # Iterate through all chunks and add neighbors of retrieved chunks
    for source, chunks in full_doc_chunks_map.items():
        for i, chunk in enumerate(chunks):
            if (source, chunk.metadata.get('page')) in all_pages_in_context:
                for offset in [-1, 0, 1]: # Add previous, current, and next chunk
                    neighbor_idx = i + offset
                    if 0 <= neighbor_idx < len(chunks):
                        neighbor_chunk = chunks[neighbor_idx]
                        if neighbor_chunk.page_content not in expanded_chunks:
                            expanded_chunks[neighbor_chunk.page_content] = neighbor_chunk

    # Sort final chunks by source and page number
    sorted_docs = sorted(expanded_chunks.values(), key=lambda d: (d.metadata.get('source'), d.metadata.get('page', -1)))
    return sorted_docs


def handle_user_query(query, retriever, doc_chain):
    """Processes a user's query through the RAG pipeline and prints the result."""
    with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
        console.print("[cyan]Phase 1: Hybrid Search & Reranking...[/cyan]")
        retrieved_docs = retriever.invoke(query)
        if not retrieved_docs:
            console.print(Panel("[yellow]No relevant context found.[/yellow]", title="Warning"))
            return

        console.print("[cyan]Phase 2: Expanding context for continuity...[/cyan]")
        final_context = expand_context(retrieved_docs)
        
        console.print("[cyan]Phase 3: Generating answer with LLM...[/cyan]")
        if doc_chain:
            answer = doc_chain.invoke({"context": final_context, "input": query})
            console.print(Panel(Markdown(answer), title="💡 Answer", border_style="blue"))
        else:
            console.print(Panel("[yellow]LLM not available. Cannot generate an answer.[/yellow]", title="Warning"))

        console.print(Panel(format_sources(final_context), title="📚 Sources", border_style="yellow"))


# --- Main Application Flow ---

def handle_document_upload(uploader):
    """CLI flow for uploading a new document."""
    file_path_str = Prompt.ask("Enter the full path to your document")
    file_path = Path(file_path_str)
    if not file_path.exists():
        console.print(f"[bold red]Error: File not found at '{file_path_str}'[/bold red]")
        return
    title = Prompt.ask("Enter a custom title (optional)", default=file_path.name)
    uploader.upload_document(str(file_path), title)

def handle_qa_session(uploader):
    """Initiates the RAG pipeline and enters the Q&A loop."""
    retriever, doc_chain = build_rag_pipeline(uploader)
    if not retriever:
        console.print("[bold red]Failed to initialize RAG pipeline. Please ensure documents are uploaded.[/bold red]")
        return

    console.print("\n[bold green]Q&A Session Started.[/bold green] [italic]Type 'back' to return to menu.[/italic]")
    while True:
        query = Prompt.ask("[bold cyan]Ask a question (or type 'back' to go back to the menu)[/bold cyan]")
        if query.lower() == 'back':
            break
        if query.strip():
            handle_user_query(query, retriever, doc_chain)

def main():
    """Main application loop."""
    display_welcome_banner()
    uploader = DocumentUploader()

    while True:
        try:
            console.print("\n[bold]Main Menu:[/bold]")
            console.print("[green]1. Upload Document(s)[/green]")
            console.print("[cyan]2. List Uploaded Documents[/cyan]")
            console.print("[blue]3. Start Q&A Session[/blue]")
            console.print("[red]4. Exit[/red]")

            choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4"])

            if choice == "1":
                handle_document_upload(uploader)
            elif choice == "2":
                uploader.list_documents()
            elif choice == "3":
                handle_qa_session(uploader)
            elif choice == "4":
                break
        except KeyboardInterrupt:
            break

    console.print("\n[bold magenta]Goodbye! Hope you had a productive session.[/bold magenta]")
    sys.exit(0)

if __name__ == "__main__":
    main()