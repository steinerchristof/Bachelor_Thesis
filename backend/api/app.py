"""
Simplified FastAPI application - State of the art with minimal complexity
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import re

from .models import ChatRequest, ChatResponse
from .swiss_formatter import format_swiss_numbers

# Import QA system - adjust path as needed
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.backend.data_storage.qa import SimpleQA

# Configure professional logging
from app.backend.config.logging_config import setup_logging, API_LOGGER

# Setup logging for the entire application
root_logger = setup_logging(
    log_level="INFO",
    log_file="logs/backend.log",
    use_colors=False,  # Disable colors for now
    include_icons=False,  # Disable icons for clean thesis output
    enable_performance_filter=True
)

# Use component-specific logger
logger = API_LOGGER

# Global QA system instance
qa_system = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global qa_system
    
    # Startup
    logger.info("Initializing Question-Answering System")
    try:
        from app.backend.data_storage.qa import initialize_qa_system
        qa_system = initialize_qa_system()
        logger.info("Question-Answering System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Question-Answering System: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Question-Answering System")


# Create FastAPI app
app = FastAPI(
    title="Swiss RAG API",
    description="Simplified API for Swiss financial document Q&A",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Swiss RAG API",
        "version": "2.0.0"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for Q&A
    
    Args:
        request: Chat request with user's question
        
    Returns:
        Response with answer and sources
    """
    if not qa_system:
        raise HTTPException(status_code=503, detail="QA system not initialized")
    
    try:
        # Get answer from QA system
        result = qa_system.answer(request.text)
        
        # Format response with Swiss numbers
        formatted_answer = format_swiss_numbers(result.get("answer", ""))
        
        # Handle sources - convert strings to dicts if needed
        raw_sources = result.get("sources", [])
        sources = []
        for source in raw_sources:
            if isinstance(source, str):
                # Parse out the score information from the source string
                # The source format is: "Title (Score: X.XXX)" or similar variations
                title = source
                
                # Remove score information using various patterns
                # Remove patterns like "(Score: X.XXX)", "(nicht verwendet, Score: X.XXX)", 
                # "(X Dokumente verwendet, Ø Score: X.XXX)", etc.
                title = re.sub(r'\s*\([^)]*Score:[^)]*\)', '', source)
                title = title.strip()
                
                # Log the original source with score for backend visibility
                logger.info(f"Source: {source}")
                
                # Convert to dict format without score
                sources.append({"title": title, "metadata": {}})
            else:
                # Already a dict
                sources.append(source)
        
        return ChatResponse(
            answer=formatted_answer,
            sources=sources,
            confidence=result.get("confidence", 0.0),
            timing=result.get("timing", None)
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/document/{doc_id}")
async def get_document(doc_id: str):
    """
    Retrieve a document by ID
    
    Args:
        doc_id: Document identifier (title)
        
    Returns:
        Document file or 404 if not found
    """
    # Document directories to search
    doc_dirs = [
        "/Users/christof/RAG_Azure/downloads/files",
        "/Users/christof/RAG_Azure/downloads/bbConcept",
        "/Users/christof/RAG_Azure/downloads/goetzRufer"
    ]
    
    # Normalize the search query - handle multiple spaces and dashes better
    # First, replace multiple spaces with single space
    normalized_doc_id = re.sub(r'\s+', ' ', doc_id)
    # Replace " - " with space
    normalized_doc_id = normalized_doc_id.replace(" - ", " ")
    # Clean up any remaining multiple spaces and convert to lowercase
    normalized_doc_id = re.sub(r'\s+', ' ', normalized_doc_id).lower().strip()
    
    # Search for document
    for doc_dir in doc_dirs:
        doc_path = Path(doc_dir)
        if not doc_path.exists():
            continue
            
        # Search for PDF files matching the ID
        for pdf_file in doc_path.glob("**/*.pdf"):
            # Normalize the filename for comparison using the same logic
            filename = pdf_file.stem
            normalized_filename = re.sub(r'\s+', ' ', filename)
            normalized_filename = normalized_filename.replace(" - ", " ")
            normalized_filename = re.sub(r'\s+', ' ', normalized_filename).lower().strip()
            
            # Get search terms, filtering out single dashes
            search_terms = [term for term in normalized_doc_id.split() if term != '-']
            
            # Check if all search terms are in the normalized filename
            if all(term in normalized_filename for term in search_terms):
                return FileResponse(
                    path=str(pdf_file),
                    media_type="application/pdf",
                    filename=pdf_file.name
                )
    
    raise HTTPException(status_code=404, detail="Document not found")


@app.get("/document/{doc_id}/info")
async def get_document_info(doc_id: str):
    """
    Get document metadata without downloading
    
    Args:
        doc_id: Document identifier
        
    Returns:
        Document metadata if found
    """
    # Similar to get_document but returns metadata instead
    doc_dirs = [
        "/Users/christof/RAG_Azure/downloads/files",
        "/Users/christof/RAG_Azure/downloads/bbConcept",
        "/Users/christof/RAG_Azure/downloads/goetzRufer"
    ]
    
    # Normalize the search query - handle multiple spaces and dashes better
    # First, replace multiple spaces with single space
    normalized_doc_id = re.sub(r'\s+', ' ', doc_id)
    # Replace " - " with space
    normalized_doc_id = normalized_doc_id.replace(" - ", " ")
    # Clean up any remaining multiple spaces and convert to lowercase
    normalized_doc_id = re.sub(r'\s+', ' ', normalized_doc_id).lower().strip()
    
    for doc_dir in doc_dirs:
        doc_path = Path(doc_dir)
        if not doc_path.exists():
            continue
            
        for pdf_file in doc_path.glob("**/*.pdf"):
            # Normalize the filename for comparison using the same logic
            filename = pdf_file.stem
            normalized_filename = re.sub(r'\s+', ' ', filename)
            normalized_filename = normalized_filename.replace(" - ", " ")
            normalized_filename = re.sub(r'\s+', ' ', normalized_filename).lower().strip()
            
            # Get search terms, filtering out single dashes
            search_terms = [term for term in normalized_doc_id.split() if term != '-']
            
            # Check if all search terms are in the normalized filename
            if all(term in normalized_filename for term in search_terms):
                return {
                    "found": True,
                    "filename": pdf_file.name,
                    "title": pdf_file.stem,
                    "path": str(pdf_file),
                    "size_mb": pdf_file.stat().st_size / (1024 * 1024),
                    "client": pdf_file.parent.name
                }
    
    return {"found": False, "message": "Document not found"}