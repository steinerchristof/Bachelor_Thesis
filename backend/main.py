#!/usr/bin/env python3
"""
Main entry point for the RAG API server

This script initializes and runs the FastAPI application.
It can be used in two ways:
1. Direct execution: python app/backend/main.py
2. With uvicorn: uvicorn app.backend.main:app
"""
import uvicorn
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the app - this makes it available as app.backend.main:app
from app.backend.api.app import app

if __name__ == "__main__":
    # Run the server with uvicorn when executed directly
    uvicorn.run(app, host="0.0.0.0", port=8000)
