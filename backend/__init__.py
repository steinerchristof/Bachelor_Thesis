"""
RAG Azure Backend Package

This package contains all the core functionality for the RAG Azure system:
- config: Configuration management
- api: FastAPI backend
- data_storage: Data storage and retrieval components
- data_preparation: Data processing and indexing scripts

For convenience, commonly used items are available at the package level.
"""

# Version info
__version__ = "1.0.0"
__author__ = "RAG Azure Team"

# Expose commonly used functions for convenience
try:
    from .config import get_config, validate_config, load_openai_client
    from .data_storage import SimpleQA, initialize_qa_system
    
    # Make these available as: from backend import get_config, SimpleQA, etc.
    __all__ = [
        "get_config",
        "validate_config", 
        "load_openai_client",
        "SimpleQA",
        "initialize_qa_system",
        "__version__",
        "__author__"
    ]
    
except ImportError:
    # If dependencies aren't available, still function as a package
    __all__ = ["__version__", "__author__"]