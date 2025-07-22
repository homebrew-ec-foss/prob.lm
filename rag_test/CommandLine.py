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
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    console.print("[yellow]Warning: PyPDF2 not installed. PDF support disabled. Install with: pip install PyPDF2[/yellow]")

#===================================================================================================================
"""
RAG Document Upload CLI
Upload documents and extract text for RAG pipeline.
"""

# Front-End and Preprocessing
class DocumentUploader:
    def _init_(self, storage_dir: str = "rag_documents"):
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
                                 f"      Title: {self.documents[doc_id]['title']}",title="Upload Success", border_style="green"))
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
                # These parts will be filled in by subsequent commits
                console.print("[yellow]RAG Pipeline processing not yet fully implemented.[/yellow]")
                continue

            elif choice == "4":
                console.print("\n[bold magenta]Thank you! Hope you had a productive study session![/bold magenta]")
                break
        except KeyboardInterrupt:
            console.print("\n[bold magenta]Exiting... Thank You! [/bold magenta]")
            sys.exit(0)

if _name_ == "_main_":
    main()