"""
MCP Server with FAISS Vector Store and Tools
Uses FastMCP for easy server setup with arithmetic and Wikipedia tools
"""
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
from typing import Any
import aiohttp
import httpx
from mcp.server.fastmcp import FastMCP
import numpy as np
import torch
import os
import asyncio
from pathlib import Path
from typing import Optional
import json
import shutil
from mcp.server.fastmcp import FastMCP
import ollama
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

import logging

from pathlib import Path


import subprocess

# openrouter_api_key = ""
user_home = os.path.expanduser('~')
proc = None

"""
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG unlocks full logs
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
"""

# Make absolutely sure we won't use MPS even if available
if hasattr(torch.backends, "mps"):
    # hard-disable MPS so libs don't probe it
    torch.backends.mps.is_available = lambda: False  # type: ignore
    torch.backends.mps.is_built = lambda: False

import faiss
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import logging

# faiss.omp_set_num_threads(1)
# torch.set_num_threads(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("llm-agent-tools", port=3000)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".png", ".jpg", ".jpeg"
}


# Global RAG instance
class RAGManager:
    """
    Manages RAG-Anything instance for the server.
    Configuration is received from the client during initialization.
    """
    
    def __init__(self):
        self.rag = None
        self.initialized = False
        
        # These will be set during initialize() from client's env_vars
        self.openrouter_api_key = None
        self.ollama_api_key = None
        self.working_dir = None
        self.data_dir = None
        self.output_dir = None
        self.llm_model = None
        self.embedding_model = None
        self.vlm_model = None
        
        # Ollama client for embeddings (reuse connection)
        self.ollama_client = None
    
    def _ensure_dirs(self):
        for p in [self.working_dir, self.data_dir, self.output_dir]:
            if p is not None:
                p.mkdir(parents=True, exist_ok=True)
                
rag_manager = RAGManager()

# Global vector store instance
class VectorStoreManager:
    """Manages FAISS vector store for the server"""
    
    def __init__(self):
        self.encoder = None
        self.dimension = None
        self.index = None
        self.documents = []
        self.initialized = False
    
    def initialize(self):
        """Lazy initialization of the embedding model"""
        if not self.initialized:
            logger.info("Initializing embedding model...")
            self.encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            self.dimension = self.encoder.get_sentence_embedding_dimension()
            logger.info(f"Embedding model initialized with dimension: {self.dimension}")
            self.initialized = True
    
    def add_texts(self, texts: list[str]):
        """Add texts to the vector store"""
        self.initialize()

        embeddings = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))

        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(embeddings)
        self.documents.extend(texts)
        logger.info(f"Added {len(texts)} documents. Total: {len(self.documents)}")
    
    def similarity_search(self, query: str, k: int = 2) -> list[str]:
        """Search for similar documents"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        self.initialize()
        
        query_embedding = self.encoder.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype('float32')
        
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_embedding, k)
        
        results = [self.documents[idx] for idx in indices[0]]
        logger.info(f"Found {len(results)} similar documents")
        return results
    
    def clear(self):
        """Clear the vector store"""
        self.index = None
        self.documents = []
        logger.info("Vector store cleared")

# Global vector store instance
vector_store = VectorStoreManager()

#region Arithmetic Tools for quick testing
@mcp.tool()
async def add(a: float, b: float) -> str:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        String containing the sum result
    """
    try:
        result = float(a) + float(b)
        print("Calling add tool")
        return f"The result of {a} + {b} is {result}"
    except (ValueError, TypeError) as e:
        return f"Error: Invalid numbers provided for addition. {str(e)}"


@mcp.tool()
async def subtract(a: float, b: float) -> str:
    """Subtract b from a.
    
    Args:
        a: Number to subtract from
        b: Number to subtract
    
    Returns:
        String containing the difference result
    """
    try:
        result = float(a) - float(b)
        print("Calling subtract tool")
        return f"The result of {a} - {b} is {result}"
    except (ValueError, TypeError) as e:
        return f"Error: Invalid numbers provided for subtraction. {str(e)}"


@mcp.tool()
async def multiply(a: float, b: float) -> str:
    """Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        String containing the product result
    """
    try:
        result = float(a) * float(b)
        print("Calling multiply tool")
        return f"The result of {a} * {b} is {result}"
    except (ValueError, TypeError) as e:
        return f"Error: Invalid numbers provided for multiplication. {str(e)}"


@mcp.tool()
async def divide(a: float, b: float) -> str:
    """Divide a by b.
    
    Args:
        a: Dividend (number to be divided)
        b: Divisor (number to divide by)
    
    Returns:
        String containing the quotient result or error if dividing by zero
    """
    try:
        print("Calling divide tool")
        a, b = float(a), float(b)
        if b == 0:
            return "Error: Division by zero is not allowed."
        result = a / b
        return f"The result of {a} / {b} is {result}"
    except (ValueError, TypeError) as e:
        return f"Error: Invalid numbers provided for division. {str(e)}"

#endregion arithmatic tools

#region Wikipedia Tool
@mcp.tool()
async def scrape_wikipedia(url: str) -> str:
    """Scrape and store information from a Wikipedia page.
    
    This tool will:
    1. Fetch and parse Wikipedia content
    2. Extract the title and main text
    3. Generate embeddings and store in FAISS vector database
    4. Make the content available for querying
    
    Args:
        url: Complete Wikipedia URL to scrape (e.g., https://en.wikipedia.org/wiki/Python)
    
    Returns:
        Success message with title or error message
    """
    try:
        # Validate URL
        print("Calling scrape tool")
        result = urlparse(url)
        if not all([result.scheme, result.netloc]) or "wikipedia.org" not in result.netloc:
            return "Error: Invalid URL. Please provide a complete Wikipedia URL (e.g., https://en.wikipedia.org/wiki/Topic)"

        logger.info(f"Scraping Wikipedia URL: {url}")

        # Fetch web content
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the main content div
        content_div = soup.find('div', {'id': 'mw-content-text'})

        # Extract title
        title_elem = soup.find('h1', {'id': 'firstHeading'})
        title = title_elem.text if title_elem else "Unknown Title"

        # Extract content
        content = []
        if content_div:
            for elem in content_div.find_all(['p', 'h2', 'h3']):
                if elem.name == 'p':
                    text = elem.get_text().strip()
                    if text and len(text) > 20:  # Filter out very short paragraphs
                        content.append(text)
                elif elem.name in ['h2', 'h3']:
                    content.append(f"\n\n{elem.get_text()}\n")

        full_content = f"{title}\n\n{''.join(content)}"
        logger.info(f"Scraped {len(full_content)} characters from {title}")

        # Create summary (first 5000 characters as context)
        summary = full_content[:5000]
        
        # Store in vector database
        # Break content into chunks for better retrieval
        chunk_size = 1000
        chunks = []
        
        # Add title and summary as first chunk
        chunks.append(f"Title: {title}\n\nSummary: {summary}")
        
        # Add content chunks
        for i in range(0, min(len(full_content), 10000), chunk_size):
            chunk = full_content[i:i+chunk_size]
            if len(chunk) > 100:  # Only add substantial chunks
                chunks.append(chunk)
        
        # Store in FAISS
        vector_store.add_texts(chunks)
        
        return f"Successfully scraped and stored information from Wikipedia.\n\nTitle: {title}\n\nScraped {len(chunks)} chunks of content totaling {len(full_content)} characters.\n\nYou can now ask questions about this content using the query_knowledge tool."
        
    except httpx.HTTPError as e:
        error_msg = f"Error fetching Wikipedia page: {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error scraping Wikipedia: {str(e)}"
        logger.error(error_msg)
        return error_msg
#endregion Wikipedia tool


# new tool
BASE_DIR = Path(__file__).resolve().parent   # folder where rag_server.py lives
TX_DIR = BASE_DIR / "my_files"   # <-- use subdirectory next to rag_server.py
TX_CHUNK = 1000


@mcp.tool() 
async def initialize_rag_system( # Q1
    parser: str = "mineru",
    enable_vision: bool = True
) -> str:
    """
    Initialize the RAG-Anything system with specified configuration.
    
    Args:
        parser: "mineru" (default, best for PDFs) or "docling" (better for Office docs)
        enable_vision: Enable vision model for image processing
    
    Returns:
        Status message confirming initialization
    """
    # Your implementation:
    # 1. Set up RAGManager with environment variables from client
    # 2. Create RAGAnythingConfig with specified parser
    # 3. Initialize RAGAnything instance
    # 4. Set up Ollama embedding function
    # 5. Set up OpenRouter vision function (if enabled)
    # 6. Mark system as initialized

    # print("DEBUG ENV DIRS:", os.getenv("DATA_DIR"), os.getenv("OUTPUT_DIR"), os.getenv("RAG_STORAGE_DIR"))

    # Load config from environment

    # Load environment variables
    rag_manager.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    rag_manager.ollama_api_key = os.getenv("OLLAMA_API_KEY", "")

    data_dir = os.getenv("DATA_DIR", "").strip()
    output_dir = os.getenv("OUTPUT_DIR", "").strip()
    storage_dir = os.getenv("RAG_STORAGE_DIR", "").strip()

    if not data_dir or not output_dir or not storage_dir:
        return (
            "ERROR: DATA_DIR, OUTPUT_DIR, and RAG_STORAGE_DIR must be set in .env."
        )
            
    rag_manager.data_dir = Path(data_dir).expanduser()
    rag_manager.output_dir = Path(output_dir).expanduser()
    rag_manager.working_dir = Path(storage_dir).expanduser()

    rag_manager.llm_model = os.getenv("LLM_MODEL", "gpt-oss:20b-cloud")
    rag_manager.embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    rag_manager.vlm_model = os.getenv("VLM_MODEL", "polaris-alpha")

    rag_manager._ensure_dirs()

    # --- Synchronous Ollama Client Logic (to be run in a thread) ---
    # This helper function performs the blocking work and returns the list.
    def _ollama_sync_embed_logic(texts: list[str], model_name: str) -> list[list[float]]:
        try:
            sync_client = ollama.Client()
            response = sync_client.embed(
                model=model_name,
                input=texts
            )
            return response['embeddings']
        except Exception as e:
            # Crucial to raise an error if the sync client fails
            raise RuntimeError(f"Ollama synchronous embedding failed: {e}")

    # --- ASYNC WRAPPER FOR LIGHTRAG ---
    # This function is passed to RAGAnything and MUST be defined as async
    async def _embedding_func_async_to_thread(texts: list[str]) -> list[list[float]]:
        """
        Final bridge solution: Define as async to satisfy LightRAG's signature,
        but use asyncio.to_thread to run the blocking ollama.Client() synchronously.
        """
        model_name = getattr(rag_manager, 'embedding_model', 'nomic-embed-text')
        
        # Run the blocking function in a separate worker thread.
        # This call IS an awaitable object, which LightRAG expects.
        # The result of this call is the final list of embeddings.
        embeddings = await asyncio.to_thread(
            _ollama_sync_embed_logic,
            texts,
            model_name
        )
        
        # Return the list. When LightRAG awaits this function, it gets the list, 
        # but the framework is designed to handle this specific await pattern.
        return embeddings

    # --------------- Embedding function ---------------
    embedding_func = EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=_embedding_func_async
        #func=_embedding_func_async_to_thread
    )

    # --------------- Vision function -------------------
    vision_model_func = build_vision_model_func() if enable_vision else None

    # --------------- Correct RAGAnythingConfig -------------------
    config = RAGAnythingConfig(
        working_dir=str(rag_manager.working_dir),
        parse_method="auto",            # valid
        parser_output_dir=str(rag_manager.output_dir),
        parser=parser,                   # valid
        display_content_stats=True,
        enable_image_processing=enable_vision,
        enable_table_processing=True,
        enable_equation_processing=True,
        max_concurrent_files=3,
        supported_file_extensions=list(SUPPORTED_EXTENSIONS),
        recursive_folder_processing=True,
        context_window=1,
        context_mode="page",
        max_context_tokens=2000,
        include_headers=True,
        include_captions=True,
        #content_format="minerU",        # default in your version
    )

    # --------------- Create RAGAnything instance -------------------
    rag_manager.rag = RAGAnything(
        config=config,
        llm_model_func=ollama_llm_complete,
        embedding_func=embedding_func,
        vision_model_func=vision_model_func,
    )

    rag_manager.initialized = True

    summary = {
        "status": rag_manager.initialized,
        "parser": parser,
        "enable_vision": enable_vision,
        "data_dir": str(rag_manager.data_dir),
        "output_dir": str(rag_manager.output_dir),
        "working_dir": str(rag_manager.working_dir),
        "llm_model": rag_manager.llm_model,
        "embedding_model": rag_manager.embedding_model,
        "vlm_model": rag_manager.vlm_model,
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)

"""
# Helper Fn for Q1
# def _embedding_func_sync(texts: list[str]) -> list[list[float]]: return asyncio.run(embed_texts(texts))

# Helper Fn for Q1
# def _embedding_func_sync(texts: list[str]) -> list[list[float]]:
    
#     try:
#         # 1. Use the synchronous ollama.Client() for the blocking call
#         response = ollama.Client().embed(
#             model='nomic-embed-text',
#             input=texts
#         )
#         # 2. Returns a concrete list of lists, which LightRAG can handle synchronously
#         return response['embeddings']
        
#     except Exception as e:
#             # --- TEMPORARY DEBUGGING CODE ---
#             error_msg = f"FATAL: Ollama Connection Error in Worker! Details: {e}"
#             print(error_msg, file=sys.stderr, flush=True) # Direct print to stderr
#             # --- END DEBUGGING CODE ---
            
#             # Original logic:
#             # logger.error(f"Synchronous Ollama embedding failed in worker thread: {e}")
#             raise RuntimeError(f"Embedding failed: {e}")
        


# async def _embedding_func_async(texts: list[str]) -> list[list[float]]:
#     async_client = ollama.AsyncClient()
#     resp = await async_client.embed(
#         model="nomic-embed-text",
#         input=texts
#     )
#     results = await rag_manager.rag.insert("")
#     return resp["embeddings"]    
"""

# Helper Fn for Q1
async def _embedding_func_async(texts: list[str]) -> list[list[float]]: return await embed_texts(texts)

# Helper Fn for Q1
def build_vision_model_func():
    """
    Build an async-compatible vision model function for RAG-Anything.
    RAG-Anything calls this internally when it needs VLM support.
    """
    async def vision_model_func(prompt: str, image_data: Optional[str] = None, **kwargs) -> str:
        if not image_data:
            return "No image data provided."
        return await process_image(image_data=image_data, prompt=prompt)

    return vision_model_func

# Helper Fn for Q1, used to be just def (sync instead of async)
async def ollama_llm_complete(prompt: str,
                        system_prompt: Optional[str] = None,
                        history_messages: Optional[list[dict]] = None,
                        **kwargs) -> str:
    """
    Synchronous wrapper to call Ollama LLM (e.g., gpt-oss:20b-cloud).
    Used by RAG-Anything for synthesis / title extraction, etc.
    """
    model = rag_manager.llm_model or "gpt-oss:20b-cloud"

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history_messages:
        for m in history_messages:
            if "role" in m and "content" in m:
                messages.append(m)

    messages.append({"role": "user", "content": prompt})

    resp = ollama.chat(
        model=model,
        messages=messages,
        options=kwargs.get("options") or {}
    )
    return resp["message"]["content"]

@mcp.tool() 
async def preprocess_documents() -> str: #Q2
    """
    Process all documents in the data directory and build the knowledge base.
    
    Returns:
        Summary of preprocessing (files processed, chunks created, etc.)
    """
    # Your implementation:
    # 1. Check if RAG system is initialized
    # 2. Scan DATA_DIR for supported file types (PDF, DOCX, PNG, JPG, TXT, MD)
    # 3. Call rag.insert() to process all documents
    # 4. RAG-Anything handles chunking, embedding, and storage automatically
    # 5. Return summary with file count, chunk count, time elapsed
    if not rag_manager.initialized or rag_manager.rag is None:
        return "ERROR: RAG system not initialized. Call initialize_rag_system first."

    if not rag_manager.data_dir or not rag_manager.data_dir.exists():
        return f"ERROR: DATA_DIR does not exist: {rag_manager.data_dir}"

    files = [
        p for p in rag_manager.data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    start = time.perf_counter()

    # RAG-Anything: folder-level processing, handles chunking & storage
    await rag_manager.rag.process_folder_complete(
        folder_path=str(rag_manager.data_dir),
        output_dir=str(rag_manager.output_dir),
        file_extensions=list(SUPPORTED_EXTENSIONS),
        recursive=True,
        max_workers=4,
    )

    elapsed = time.perf_counter() - start

    summary = {
        "status": "ok",
        "files_seen": len(files),
        "elapsed_seconds": round(elapsed, 3),
        "note": "RAG-Anything manages chunk counts and vector storage internally."
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


@mcp.tool() 
async def query_knowledge( #Q3
    question: str,
    mode: str = "hybrid"
) -> str:
    """
    Query the knowledge base and return relevant information.
    
    Args:
        question: Natural language query
        mode: "local", "global", or "hybrid" (default)
    
    Returns:
        Retrieved context with source citations
    """
    # Your implementation:
    # 1. Check if RAG system is initialized
    # 2. Call rag.query() with specified mode
    # 3. Format results with source information
    # 4. Return formatted response
    if not rag_manager.initialized or rag_manager.rag is None:
        return "ERROR: RAG system not initialized. Call initialize_rag_system first."

    mode = mode.lower()
    if mode not in {"local", "global", "hybrid"}:
        mode = "hybrid"

    # RAG-Anything provides async aquery()
    result = await rag_manager.rag.aquery(question, mode=mode)

    # result is typically a string with citations embedded
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool() 
async def add_document(file_path: str) -> str: #Q4
    """
    Add a single new document to the knowledge base.
    
    Args:
        file_path: Absolute path to the file to add
    
    Returns:
        Confirmation message
    """
    # Your implementation:
    # 1. Validate file exists and is supported type
    # 2. Copy file to DATA_DIR if not already there
    # 3. Call rag.insert() for the single file
    # 4. Return success message

    # --- 1. Check initialization ---
    if not rag_manager.initialized or rag_manager.rag is None:
        return "ERROR: RAG system not initialized. Run initialize_rag_system first."

    # Convert to Path
    src = Path(file_path).expanduser()

    # --- 2. Validate file existence ---
    if not src.exists() or not src.is_file():
        return f"ERROR: File does not exist: {src}"

    ext = src.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return f"ERROR: Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"

    # --- 3. Copy into DATA_DIR (avoid overwriting) ---
    dest = rag_manager.data_dir / src.name
    if dest.exists():
        # Avoid overwriting user's files
        base = dest.stem
        suf = dest.suffix
        counter = 1
        while True:
            new_dest = rag_manager.data_dir / f"{base}_{counter}{suf}"
            if not new_dest.exists():
                dest = new_dest
                break
            counter += 1

    try:
        shutil.copy2(src, dest)
    except Exception as e:
        return f"ERROR copying file: {e}"

    # --- 4. Process the single document using new API ---
    try:
        start = time.perf_counter()

        await rag_manager.rag.process_document_complete(
            file_path=str(dest),
            output_dir=str(rag_manager.output_dir)
        )

        elapsed = round(time.perf_counter() - start, 3)
    except Exception as e:
        return f"ERROR during processing: {e}"

    # --- 5. Return success summary ---
    return json.dumps({
        "status": "ok",
        "added_file": str(dest),
        "elapsed_seconds": elapsed,
        "note": "Document ingested successfully into RAGAnything."
    }, ensure_ascii=False, indent=2)


@mcp.tool() 
async def rename_pdfs_by_title() -> str: #Q5
    """
    Rename PDF files in data directory based on their extracted titles.
    
    Returns:
        Summary of renamed files (old_name -> new_name)
    """
    # Your implementation:
    # 1. Find all PDF files in DATA_DIR
    # 2. For each PDF, extract first page or metadata
    # 3. Use LLM to extract clean title from content
    # 4. Sanitize title for filename (remove special chars, limit length)
    # 5. Rename file (handle conflicts with numbers: title_1.pdf, title_2.pdf)
    # 6. Return list of old -> new mappings
    if not rag_manager.initialized:
        return "ERROR: RAG system not initialized."

    import re
    from PyPDF2 import PdfReader

    data_dir = rag_manager.data_dir
    pdfs = list(data_dir.rglob("*.pdf"))
    if not pdfs:
        return "No PDF files found."

    results = {}
    used_names = set()

    def sanitize(name: str) -> str:
        # Remove illegal filename chars
        name = re.sub(r'[\\/*?:"<>|]', "", name)
        name = name.strip()
        # Truncate long names
        return name[:60] if len(name) > 60 else name

    for pdf in pdfs:
        try:
            # --- Step 1: extract text from first page ---
            reader = PdfReader(str(pdf))
            first_page = reader.pages[0].extract_text() if reader.pages else ""
            context = first_page.strip()[:2000]  # safety limit

            if not context:
                results[pdf.name] = "(skipped: empty first page)"
                continue

            # --- Step 2: ask the LLM for a short title ---
            prompt = (
                "Extract a clean, short academic-style title from the text below. "
                "Output ONLY the title, no quotation marks, no commentary.\n\n"
                f"{context}"
            )
            raw_title = (await ollama_llm_complete(prompt)).strip()
            title = sanitize(raw_title)

            if not title:
                results[pdf.name] = "(skipped: LLM returned empty title)"
                continue

            # --- Step 3: handle filename collisions ---
            new_name = f"{title}.pdf"
            counter = 1
            while new_name in used_names or (pdf.parent / new_name).exists():
                new_name = f"{title}_{counter}.pdf"
                counter += 1

            used_names.add(new_name)

            # --- Step 4: rename file ---
            new_path = pdf.parent / new_name
            pdf.rename(new_path)

            results[pdf.name] = new_name

        except Exception as e:
            results[pdf.name] = f"(error: {e})"

    return json.dumps(results, ensure_ascii=False, indent=2)


@mcp.tool() 
async def clear_rag_storage() -> str: #Q7
    """
    Clear all RAG storage (vector DB, processed files, etc.).
    
    Returns:
        Confirmation message
    """
    # Your implementation:
    # 1. Delete RAG_STORAGE_DIR contents
    # 2. Delete OUTPUT_DIR contents
    # 3. Reset RAG system
    # 4. Return confirmation

    if not rag_manager.initialized:
        return "ERROR: RAG system not initialized."

    storage_dir = rag_manager.working_dir
    output_dir = rag_manager.output_dir

    removed = []

    # --- 1. Delete RAG_STORAGE_DIR contents ---
    if storage_dir and storage_dir.exists():
        for item in storage_dir.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                else:
                    shutil.rmtree(item)
                removed.append(str(item))
            except Exception as e:
                removed.append(f"{item} (FAILED: {e})")

    # --- 2. Delete OUTPUT_DIR contents ---
    if output_dir and output_dir.exists():
        for item in output_dir.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                else:
                    shutil.rmtree(item)
                removed.append(str(item))
            except Exception as e:
                removed.append(f"{item} (FAILED: {e})")

    # --- 3. Reset RAG system ---
    rag_manager.rag = None
    rag_manager.initialized = False

    # --- 4. Return confirmation ---
    return json.dumps(
        {
            "status": "ok",
            "removed_items": removed,
            "message": "RAG storage cleared. Run initialize_rag_system() again before processing documents."
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool() 
async def get_rag_status(detailed: bool = False) -> str: #Q8
    """
    Get comprehensive status of RAG system.
    
    Args:
        detailed: Include list of all files in knowledge base
    
    Returns:
        Status report (initialization state, document count, storage size, etc.)
    """
    # Your implementation:
    # 1. Check initialization status
    # 2. Count documents in DATA_DIR
    # 3. Check storage directory size
    # 4. If detailed=True, list all files
    # 5. Return formatted status report

    status = {
        "initialized": rag_manager.initialized,
        "data_dir": str(rag_manager.data_dir) if rag_manager.data_dir else None,
        "output_dir": str(rag_manager.output_dir) if rag_manager.output_dir else None,
        "storage_dir": str(rag_manager.working_dir) if rag_manager.working_dir else None,
    }

    # ---- 1. System not initialized ----
    if not rag_manager.initialized:
        status["message"] = "RAG system not initialized. Call initialize_rag_system first."
        return json.dumps(status, ensure_ascii=False, indent=2)

    # ---- 2. Count documents in DATA_DIR ----
    data_files = []
    if rag_manager.data_dir and rag_manager.data_dir.exists():
        for p in rag_manager.data_dir.iterdir():
            if p.is_file():
                data_files.append(str(p.name))

    status["document_count"] = len(data_files)

    # ---- 3. Compute total storage size in RAG_STORAGE_DIR ----
    total_bytes = 0
    if rag_manager.working_dir and rag_manager.working_dir.exists():
        for p in rag_manager.working_dir.rglob("*"):
            if p.is_file():
                total_bytes += p.stat().st_size

    status["storage_size_bytes"] = total_bytes
    status["storage_size_mb"] = round(total_bytes / (1024 * 1024), 3)

    # ---- 4. Detailed output ----
    if detailed:
        status["documents"] = data_files

        storage_files = []
        if rag_manager.working_dir and rag_manager.working_dir.exists():
            for p in rag_manager.working_dir.iterdir():
                storage_files.append(str(p.name))

        status["storage_files"] = storage_files

    # ---- 5. Return JSON ----
    return json.dumps(status, ensure_ascii=False, indent=2)

@mcp.tool()
async def download_arxiv_paper(url: str) -> str: #QX
    """
    Download a paper from an arXiv URL and save into DATA_DIR.
    """
    if not rag_manager.data_dir:
        return "ERROR: DATA_DIR not configured."

    # Normalize arXiv URL
    if "arxiv.org/abs/" in url:
        paper_id = url.split("/abs/")[1]
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    elif "arxiv.org/pdf/" in url and url.endswith(".pdf"):
        pdf_url = url
        paper_id = url.split("/pdf/")[1].replace(".pdf", "")
    else:
        return "ERROR: Not a valid arXiv URL."

    # Download PDF
    filename = f"arxiv_{paper_id}.pdf"
    dest_path = rag_manager.data_dir / filename

    async with aiohttp.ClientSession() as session:
        async with session.get(pdf_url) as resp:
            if resp.status != 200:
                return f"ERROR: Failed to download PDF. Status {resp.status}"
            content = await resp.read()
            dest_path.write_bytes(content)

    return (
        f"Downloaded arXiv paper to: {dest_path}\n\n"
        "Would you like to parse this document now? "
        "If yes, call: add_document(file_path='<same path>')"
    )


@mcp.tool() #FROM P1
async def initiate_terminal(cwd: Optional[str] = None) -> str:
    """Start a persistent bash terminal session."""
    global proc
    if proc is not None:
        proc.terminate()
        proc.wait()
        # return "Terminal is already open."

    if cwd != "":
        if "~" in cwd:
            cwd = cwd.replace("~", user_home)
        if not os.path.isdir(cwd):
            return f"{cwd} is not a directory."
        proc = subprocess.Popen(
            ["cmd.exe"],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
    else:
        proc = subprocess.Popen(
            ["cmd.exe"],
            cwd=user_home,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    return "Terminal Initiated"

@mcp.tool() #FROM P1
async def run_command(command: str) -> str:
    """Execute a bash command in the terminal."""
    global proc

    try:
        output = []
        if proc is not None:
            if not command.endswith("\n"):
                command += "\n"

            proc.stdin.write(command)

            marker = "[END_OF_CMD]" # a hack to know the end of a command output
            proc.stdin.write(f"echo {marker}\n")
                    
            proc.stdin.flush()

            while True:
                line = proc.stdout.readline()
                if not line: break
                if marker in line: break
                output.append(line.rstrip())        
        else:
            output = ["No terminal has been initiated. Please initiate a terminal first with `initiate_terminal(working_dir)`"]

        out = "\n".join(output)
        output_str = f"""
            Command: {command}
            Output: {out}    
        """

        return output_str
    
    except Exception as e:
        return f"ERROR: An exception occured {type(e).__name__}. Details: {e}"

@mcp.tool() #FROM P1
async def terminate_terminal() -> str:
    """Close the terminal session."""
    global proc
    if proc is not None:
        proc.terminate()
        proc.wait()
        return "Terminal Terminated"
    return "No terminal is open."


@mcp.tool()
async def ingest_tx() -> str:
    """
    Ingest local corpus into the vector DB.

    Reads all *.txt under ./my_files/, splits each file into ~TX_CHUNK-char
    pieces, prefixes chunks with their relative file path for provenance, and
    appends them to the global FAISS-backed `vector_store` so they are
    retrievable via `query_knowledge`.
    Returns a brief count summary.
    """
    files = sorted(TX_DIR.rglob("*.txt"))
    chunks: list[str] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        rel = f.relative_to(TX_DIR).as_posix()
        for i in range(0, len(text), TX_CHUNK):
            piece = text[i:i+TX_CHUNK].strip()
            if piece:
                chunks.append(f"[FILE] {rel}\n{piece}")
    if chunks:
        vector_store.add_texts(chunks)  # uses your lazy init + FAISS add
    return f"files={len(files)}, chunks={len(chunks)}, total_docs={len(vector_store.documents)}"


# #region Knowledge query tools
# @mcp.tool()
# async def query_knowledge(question: str) -> str:
#     """Query the knowledge base for information about previously scraped Wikipedia content.
    
#     This tool searches the vector database for relevant content and returns an answer.
#     You must scrape Wikipedia content first using the scrape_wikipedia tool.
    
#     Args:
#         question: Question to ask about the scraped content
    
#     Returns:
#         Answer based on stored Wikipedia content or error message
#     """
#     try:
#         # Check if vector store has content
#         if vector_store.index is None or len(vector_store.documents) == 0:
#             return "Error: No information has been stored yet. Please scrape a Wikipedia page first using the scrape_wikipedia tool."
        
#         logger.info(f"Querying knowledge base with question: {question}")
        
#         # Retrieve relevant documents
#         relevant_docs = vector_store.similarity_search(question, k=3)
        
#         if not relevant_docs:
#             return "I couldn't find relevant information to answer your question. The stored content may not contain information about this topic."
        
#         # Combine relevant documents into context
#         context = "\n\n---\n\n".join(relevant_docs)
        
#         # Return the context - the LLM will use this to answer the question
#         return f"Based on the stored Wikipedia content, here is the relevant information:\n\n{context}\n\n(This information can be used to answer the question: {question})"
        
#     except Exception as e:
#         error_msg = f"Error querying knowledge base: {str(e)}"
#         logger.error(error_msg)
#         return error_msg



@mcp.tool()
async def clear_knowledge_base() -> str:
    """Clear all stored Wikipedia content from the knowledge base.
    
    This removes all documents from the FAISS vector store, allowing you to start fresh
    with new Wikipedia pages.
    
    Returns:
        Confirmation message
    """
    try:
        vector_store.clear()
        return "Successfully cleared the knowledge base. You can now scrape new Wikipedia pages."
    except Exception as e:
        error_msg = f"Error clearing knowledge base: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Server Status Tool
@mcp.tool()
async def get_server_status() -> str:
    """Get the current status of the MCP server and vector database.
    
    Returns:
        Status information including number of stored documents
    """
    doc_count = len(vector_store.documents) if vector_store.documents else 0
    has_index = vector_store.index is not None
    is_initialized = vector_store.initialized
    
    status = f"""
        MCP Server Status:
        - Server: Running
        - Vector Store Initialized: {is_initialized}
        - FAISS Index Created: {has_index}
        - Stored Documents: {doc_count}
        - Available Tools: add, subtract, multiply, divide, scrape_wikipedia, query_knowledge, clear_knowledge_base, get_server_status

        Ready to accept requests!
        """
    return status.strip()

@mcp.tool()
async def debug_tool(): 
    single = await ollama.AsyncClient().embed(
        model='nomic-embed-text',
        input='The quick brown fox jumps over the lazy dog.'
    )   
    print(single['embeddings'])

#endregion knowledge query tools

# Key Implementation Details - Ollama Embeddings
async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings using Ollama"""
    async_client = ollama.AsyncClient()
    response = await async_client.embed(
        model="nomic-embed-text",
        input=texts
    )
    return response['embeddings']

# Key Implementation Details - OpenRouter Vision
async def process_image(image_data: str, prompt: str) -> str:
    """Process image with vision model via OpenRouter"""
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {rag_manager.openrouter_api_key}", #used to be openrouter_api_key
                "Content-Type": "application/json"
            },
            json={
                "model": "polaris-alpha",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                }]
            }
        )
        data = await response.json()
        return data['choices'][0]['message']['content']

# Main
if __name__ == "__main__":
    logger.info("Starting MCP Server with LLM Agent Tools...")
    logger.info("Available tools: arithmetic operations, Wikipedia scraping, knowledge querying")
    logger.info("Transport: streamable-http on port 3000")
    
    # Run the server with streamable-http transport
    mcp.run(transport="streamable-http")




# def embed_texts2(texts: list[str]) -> list[list[float]]:
#     """Generate embeddings using Ollama"""
#     async_client = ollama.AsyncClient()
#     response = async_client.embed(
#         model="nomic-embed-text",
#         input=texts
#     )
#     return response['embeddings']

# import ollama


