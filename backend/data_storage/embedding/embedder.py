"""
Text embedding functionality using Azure OpenAI's embedding models
"""
import time
import numpy as np
from typing import List, Union
import openai
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
import re
import warnings

from sentence_transformers import SentenceTransformer
from app.backend.config.config import get_config, load_openai_client
from ..utils.text_utils import truncate_to_tokens, batch_iter

class Embedder:
    """
    Azure OpenAI text embedding model wrapper with batching and error handling
    """
    _instance = None
    _client = None
    _config = None
    
    def __new__(cls):
        """Implement as singleton to avoid multiple initializations"""
        if cls._instance is None:
            cls._instance = super(Embedder, cls).__new__(cls)
            cls._config = get_config()
            cls._client = load_openai_client()
            
            # Ensure we're using Azure OpenAI
            print(f"Initializing Azure OpenAI Embedder with model: {cls._config.get('embed_model', 'text-embedding-3-large')}")
        return cls._instance
    
    @classmethod
    def dim(cls) -> int:
        """Get the embedding dimension"""
        return cls._config["embed_dim"]
    
    @classmethod
    def encode(cls, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings with batching and error handling using Azure OpenAI
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            NumPy array of embeddings
        """
        vecs: List[List[float]] = []
        max_retries = cls._config["max_retries"]
        
        try:
            # Make sure texts is properly formatted
            if not texts or not all(isinstance(t, str) for t in texts):
                raise ValueError("Input must be a non-empty list of strings")
                
            # Process text in batches
            try:
                batches = list(batch_iter(
                    [truncate_to_tokens(t, cls._config["max_text_tok"]) for t in texts],
                    cls._config["max_req_tokens"]
                ))
                print(f"Batched {len(texts)} texts into {len(batches)} batches")
            except Exception as e:
                print(f"Error during text batching: {e}")
                # Fall back to simpler batching approach
                batch_size = 10
                batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
                print(f"Fallback batching: {len(batches)} batches of size {batch_size}")
            
            # Process each batch
            for batch in batches:
                for attempt in range(max_retries):
                    try:
                        res = cls._client.embeddings.create(
                            model = cls._config.get("embed_model", "text-embedding-3-large"),
                            input = batch,
                            encoding_format = "float",
                        )
                        vecs.extend([d.embedding for d in res.data])
                        break
                    except Exception as e:
                        if "rate limit" in str(e).lower():
                            # For rate limit errors, retry with exponential backoff
                            print(f"Rate limit hit, retrying in {2 ** attempt} seconds...")
                            time.sleep(2 ** attempt)
                        else:
                            print(f"Embedding error: {e}")
                            # Fallback: Add zero vector
                            vecs.extend([np.zeros(cls._config["embed_dim"]) for _ in batch])
                            break
            return np.asarray(vecs, dtype=np.float32)
        except Exception as e:
            print(f"General embedding error: {e}")
            if "tokeniser" in str(e) or "tokenizer" in str(e):
                print("Tokenization error detected. This is likely because the model 'text-embedding-3-large' needs the cl100k_base tokenizer.")
                print("Make sure text_utils.py's get_tokenizer function is updated to use tiktoken.get_encoding('cl100k_base') for this model.")
            # Fallback: Return zero vectors
            return np.asarray([np.zeros(cls._config["embed_dim"]) for _ in texts], dtype=np.float32) 