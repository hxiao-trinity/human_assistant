"""
MCP Server with FAISS Vector Store and Tools
Uses FastMCP for easy server setup with arithmetic and Wikipedia tools
"""
import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
import numpy as np
import torch
import subprocess
import re

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

#region Knowledge query tools
@mcp.tool()
async def query_knowledge(question: str) -> str:
    """Query the knowledge base for information about previously scraped Wikipedia content.
    
    This tool searches the vector database for relevant content and returns an answer.
    You must scrape Wikipedia content first using the scrape_wikipedia tool.
    
    Args:
        question: Question to ask about the scraped content
    
    Returns:
        Answer based on stored Wikipedia content or error message
    """
    try:
        # Check if vector store has content
        if vector_store.index is None or len(vector_store.documents) == 0:
            return "Error: No information has been stored yet. Please scrape a Wikipedia page first using the scrape_wikipedia tool."
        
        logger.info(f"Querying knowledge base with question: {question}")
        
        # Retrieve relevant documents
        relevant_docs = vector_store.similarity_search(question, k=3)
        
        if not relevant_docs:
            return "I couldn't find relevant information to answer your question. The stored content may not contain information about this topic."
        
        # Combine relevant documents into context
        context = "\n\n---\n\n".join(relevant_docs)
        
        # Return the context - the LLM will use this to answer the question
        return f"Based on the stored Wikipedia content, here is the relevant information:\n\n{context}\n\n(This information can be used to answer the question: {question})"
        
    except Exception as e:
        error_msg = f"Error querying knowledge base: {str(e)}"
        logger.error(error_msg)
        return error_msg


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

#endregion knowledge query tools

@mcp.tool()
async def ingest_my_files() -> str:
    SIZE = 1000
    if not os.path.exists("my_files"):
        return "1: my_files folder does not exist."
    files_processed = 0
    chunks_added = 0
    all_chunks = []

    # Walk and find all .txt files
    for root, _, files in os.walk("my_files"):
        for file in files:
            if file.lower().endswith(".txt"):
                
                file_path = os.path.join(root, file)

                # Read file content
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                if len(text) == 0:
                    continue

                # Chunk into ~1000 char segments
                for i in range(0, len(text), SIZE):
                    chunk = text[i:i + SIZE].strip()
                    if chunk:
                        prefixed = f"[my_files/{file}]\n{chunk}"
                        all_chunks.append(prefixed)
                        chunks_added += 1

                files_processed += 1

    # Store chunks into FAISS
    if chunks_added > 0:
        vector_store.add_texts(all_chunks)

    total_docs = len(vector_store.documents)

    return (
        f"Files processed: {files_processed}\n"
        f"Chunks created this round: {chunks_added}\n"
        f"Total documents stored: {total_docs}"
    )
    
@mcp.tool()
async def correct_command(query: str) -> str:
    """
    Translate natural voice commands into standardized structured commands.
    Args:
        query: The user's natural-language command.
    Returns:
        The translated voice command as a plain string.
    """
    try:
        # Run your finetuned model with conservative sampling (deterministic)
        res = subprocess.run(
            ["ollama", "run", "gemma-3-270m-vc-finetuned"], # "--temperature", "0.9", "--top_p", "1", "--top_k", "64"],
            input=query.strip() + " | selection:",
            text=True,
            capture_output=True,
            check=True,
        )
        out = res.stdout.strip()

        # cheap post-processing: keep first line, remove obvious junk/digits
        out = out.splitlines()[0].strip()
        out = re.sub(r"\s+", " ", out)
        out = re.sub(r"[^\w\s]", "", out)         # drop stray punctuation

        return out
    except Exception as e:
        return f"Error from VCC: {e}"


# Main
if __name__ == "__main__":
    logger.info("Starting MCP Server with LLM Agent Tools...")
    logger.info("Available tools: arithmetic operations, Wikipedia scraping, knowledge querying")
    logger.info("Transport: streamable-http on port 3000")
    
    # Run the server with streamable-http transport
    mcp.run(transport="streamable-http")