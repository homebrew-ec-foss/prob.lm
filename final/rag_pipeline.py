# /problm_rag_app/rag_pipeline.py
"""
Handles the creation of the RAG pipeline, including document processing,
retriever setup (Chroma, BM25, Reranker), and LLM chain initialization.
"""
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

# LangChain and related imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM

# Local Imports
from config import (
    console, DB_PATH, BM25_CACHE_FILE, USE_API_LLM, USE_SEMANTIC_CHUNKING,
    LOCAL_MODEL_NAME, API_MODEL_NAME, EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME,
    MODEL_KWARGS, PDF_SUPPORT, SEMANTIC_CHUNKING_AVAILABLE, GROQ_API_AVAILABLE, OLLAMA_AVAILABLE
)
if SEMANTIC_CHUNKING_AVAILABLE:
    from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()

# --- Document Processing ---

def _load_and_split_docs(file_paths, status):
    """Loads and splits document content into chunks for retrieval."""
    all_docs = []
    for file_path in file_paths:
        path = Path(file_path)
        loader = None
        if path.suffix.lower() == ".pdf" and PDF_SUPPORT:
            loader = PyMuPDFLoader(str(path))
        elif path.suffix.lower() in ['.txt', '.md', '.py', '.js', '.html', '.css']:
            loader = TextLoader(str(path), encoding='utf-8')
        else:
            console.print(f"[yellow]Skipping unsupported file: {path.name}[/yellow]")
            continue
        
        status.update(f"[bold cyan]Loading: {path.name}...[/bold cyan]")
        all_docs.extend(loader.load())

    if not all_docs:
        return []

    if USE_SEMANTIC_CHUNKING and SEMANTIC_CHUNKING_AVAILABLE:
        status.update("[bold cyan]Applying semantic chunking... (This can be slow)[/bold cyan]")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs=MODEL_KWARGS)
        text_splitter = SemanticChunker(embeddings=embeddings)
    else:
        status.update("[bold cyan]Applying fast recursive chunking...[/bold cyan]")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    split_docs = text_splitter.split_documents(all_docs)
    console.print(f"[green]✓ Created {len(split_docs)} chunks from {len(file_paths)} document(s).[/green]")
    return split_docs

# --- LLM & Chain Initialization ---

def _initialize_llm():
    """Initializes the LLM based on global configuration."""
    if USE_API_LLM:
        if not GROQ_API_AVAILABLE or not os.getenv("GROQ_API_KEY"):
            console.print("[bold red]Groq API key or library not found. LLM disabled.[/bold red]")
            return None
        console.print(f"[green]Using API Model: {API_MODEL_NAME}[/green]")
        return ChatGroq(model_name=API_MODEL_NAME, temperature=0.1, groq_api_key=os.getenv("GROQ_API_KEY"))
    else:
        if not OLLAMA_AVAILABLE:
            console.print("[bold red]'ollama' library not installed. Local LLM disabled.[/bold red]")
            return None
        console.print(f"[green]Using Local Model: {LOCAL_MODEL_NAME}[/green]")
        return OllamaLLM(model=LOCAL_MODEL_NAME, temperature=0.1)

def _create_document_chain(llm):
    """Creates the LangChain chain for answering questions based on context."""
    if not llm: return None
    qa_prompt = ChatPromptTemplate.from_template(
        """You are an expert assistant. Answer the question based *only* on the provided context.
        Synthesize information to form a cohesive, well-structured answer. Use markdown for clarity.
        If the answer is not in the context, state that clearly.

        Context:
        {context}

        Question: {input}

        Detailed Answer:
        """
    )
    return create_stuff_documents_chain(llm, qa_prompt)

# --- Retriever Creation ---

def _load_or_create_bm25(uploader, status):
    """Creates or loads a cached BM25 retriever."""
    all_doc_paths = uploader.get_all_paths()
    if not all_doc_paths: return None

    # Rebuild if cache is missing or if there are new, unindexed docs
    if not BM25_CACHE_FILE.exists() or uploader.get_unindexed_paths():
        status.update("[bold cyan]Building new BM25 index (keyword search)...[/bold cyan]")
        all_docs_for_bm25 = _load_and_split_docs(all_doc_paths, status)
        if not all_docs_for_bm25: return None
        bm25_retriever = BM25Retriever.from_documents(all_docs_for_bm25)
        with open(BM25_CACHE_FILE, "wb") as f: pickle.dump(bm25_retriever, f)
        console.print("[green]✓ BM25 retriever built and cached.[/green]")
    else:
        status.update("[bold cyan]Loading cached BM25 retriever...[/bold cyan]")
        with open(BM25_CACHE_FILE, "rb") as f: bm25_retriever = pickle.load(f)
    
    bm25_retriever.k = 15
    return bm25_retriever

def build_rag_pipeline(uploader):
    """
    The main function to build the entire RAG pipeline.
    Returns the retriever and the document chain.
    """
    with console.status("[bold cyan]Initializing RAG pipeline...[/bold cyan]", spinner="dots") as status:
        # Step 1: Initialize Embeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs=MODEL_KWARGS)

        # Step 2: Process new documents and update ChromaDB
        unindexed_paths = uploader.get_unindexed_paths()
        db_path_obj = Path(DB_PATH)
        
        if unindexed_paths:
            status.update("[bold cyan]Processing new documents for vector store...[/bold cyan]")
            new_docs_split = _load_and_split_docs(unindexed_paths, status)
            if db_path_obj.exists() and any(db_path_obj.iterdir()):
                vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
                vector_store.add_documents(new_docs_split)
            else:
                vector_store = Chroma.from_documents(new_docs_split, embeddings, persist_directory=DB_PATH)
            
            for path in unindexed_paths: uploader.mark_as_indexed(path)
            console.print("[green]✓ New documents indexed in Chroma DB.[/green]")
        
        # Step 3: Load ChromaDB and create vector retriever
        if not db_path_obj.exists() or not any(db_path_obj.iterdir()):
            console.print("[bold red]Error: No documents processed. Please upload a document first.[/bold red]")
            return None, None
        vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 15})

        # Step 4: Create BM25 retriever
        bm25_retriever = _load_or_create_bm25(uploader, status)
        if not bm25_retriever: return None, None

        # Step 5: Create Ensemble Retriever
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])
        
        # Step 6: Create Reranker
        cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME, model_kwargs=MODEL_KWARGS)
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
        
        # Step 7: Initialize LLM and Chain
        llm = _initialize_llm()
        doc_chain = _create_document_chain(llm)

        console.print("[green]✓ RAG pipeline is ready.[/green]")
        return compression_retriever, doc_chain