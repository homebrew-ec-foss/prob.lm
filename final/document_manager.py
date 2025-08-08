# /problm_rag_app/document_manager.py
"""
Handles document uploading, storage, and tracking of processing status.
"""
import json
import shutil
from pathlib import Path
from datetime import datetime
from rich.table import Table, box
from rich.panel import Panel

# Local Imports
from config import console, STORAGE_DIR

class DocumentUploader:
    """Manages the lifecycle of documents for the RAG pipeline."""
    
    def __init__(self, storage_dir: Path = STORAGE_DIR):
        self.storage_dir = storage_dir
        self.docs_file = self.storage_dir / "uploaded_docs.json"
        self.documents = self._load_documents()

    def _load_documents(self) -> dict:
        """Loads document metadata from a JSON file."""
        if self.docs_file.exists():
            with open(self.docs_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure backward compatibility for the is_indexed flag
                for doc_info in data.values():
                    doc_info.setdefault('is_indexed', False)
                return data
        return {}

    def _save_documents(self):
        """Saves the current document metadata to the JSON file."""
        with open(self.docs_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, default=str)

    def upload_document(self, file_path: str, title: str = None) -> bool:
        """Copies a document to storage and records its metadata."""
        source_path = Path(file_path)
        if not source_path.exists():
            console.print(f"[bold red]Error: File not found at {file_path}[/bold red]")
            return False

        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{source_path.stem}"
        stored_path = self.storage_dir / f"{doc_id}{source_path.suffix}"

        shutil.copy2(source_path, stored_path)
        self.documents[doc_id] = {
            "title": title or source_path.name,
            "stored_file": str(stored_path),
            "file_type": source_path.suffix.lower(),
            "size": source_path.stat().st_size,
            "upload_date": datetime.now().isoformat(),
            "is_indexed": False
        }
        self._save_documents()
        console.print(Panel(f"[green]✓ Document uploaded: [bold]{doc_id}[/bold]\n"
                              f"       Title: {self.documents[doc_id]['title']}",
                              title="Upload Success", border_style="green"))
        return True

    def list_documents(self):
        """Displays a table of all uploaded documents."""
        if not self.documents:
            console.print("[yellow]No documents uploaded yet.[/yellow]")
            return

        table = Table(title="Uploaded Documents", border_style="blue", header_style="bold", box=box.SQUARE)
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="magenta")
        table.add_column("Size (MB)", style="yellow")
        table.add_column("Status", style="white")

        for doc_id, doc_info in self.documents.items():
            size_mb = f"{doc_info['size'] / (1024*1024):.2f}"
            status = "[green]Processed[/green]" if doc_info.get('is_indexed') else "[red]Not Processed[/red]"
            table.add_row(doc_id, doc_info['title'], size_mb, status)
        console.print(table)

    def get_unindexed_paths(self) -> list:
        """Returns file paths of documents not yet processed."""
        return [doc['stored_file'] for doc in self.documents.values() if not doc.get('is_indexed')]

    def get_all_paths(self) -> list:
        """Returns all stored document file paths."""
        return [doc['stored_file'] for doc in self.documents.values()]

    def mark_as_indexed(self, stored_file_path: str):
        """Marks a document as processed in the metadata."""
        for doc_info in self.documents.values():
            if doc_info['stored_file'] == stored_file_path:
                doc_info['is_indexed'] = True
        self._save_documents()