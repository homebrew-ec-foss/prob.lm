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
            console.print(f"[bold red]Error loading {file_path} with {loader._class.name_}: {e}[/bold red]")

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

#==============================================================================================================

# Front end with execution loops and error handling
def main():
    console.print(Panel("[bold magenta]✨ Prob.lm - RAG Study Assistant ✨[/bold magenta]",
                                 subtitle="[default]Powered by Hybrid Search[/default]",
                                 expand=False, width=80))

    uploader = DocumentUploader()
    llm = None # Placeholder
    retriever = None # Placeholder
    document_chain = None # Placeholder

    while True:
        try:
            console.print("\n[bold]Main Menu:[/bold]")
            console.print("[green]1. Upload Document(s)[/green]")
            console.print("[cyan]2. List Uploaded Documents[/cyan]")
            console.print("[blue]3. Process Documents & Start Q&A (RAG Pipeline)[/blue]")
            console.print("[red]4. Exit[/red]")

            choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4"])

            if choice == "1":
                file_path = Prompt.ask("Enter the full path to the document file (e.g., /path/to/my_doc.pdf)")
                if not Path(file_path).exists():
                    console.print(f"[bold red]Error: File not found at {file_path}[/bold red]")
                    continue
                title = Prompt.ask("Enter a custom title for the document (optional)", default=None)
                uploader.upload_document(file_path, title)

            elif choice == "2":
                uploader.list_documents()

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
                
                # These parts will be filled in by subsequent commits
                console.print("[yellow]RAG Pipeline processing not yet fully implemented.[/yellow]")
                continue

            elif choice == "4":
                console.print("\n[bold magenta]Thank you! Hope you had a productive study session![/bold magenta]")
                break
        except KeyboardInterrupt:
            console.print("\n[bold magenta]Exiting... Thank You! [/bold magenta]")
            sys.exit(0)
if __name__ == "_main_":
    main()
