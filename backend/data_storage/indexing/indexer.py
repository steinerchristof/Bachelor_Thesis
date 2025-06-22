"""
Indexer for creating and managing document indexes
"""
import glob
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from sklearn.preprocessing import normalize
from dataclasses import dataclass

from app.backend.config.config import get_config
from ..embedding import Embedder
from ..search import BM25Searcher, VectorSearcher
from ..utils.metadata_utils import extract_payload
from ..utils.text_utils import truncate_to_tokens


class Indexer:
    """
    Document indexer that processes chunks and creates searchable indexes
    """
    _instance = None
    
    def __new__(cls):
        """Implement as singleton pattern"""
        if cls._instance is None:
            cls._instance = super(Indexer, cls).__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self) -> None:
        """Initialize the indexer"""
        self.config = get_config()
        self.chunks: List[Dict[str, Any]] = []
        self.embedder = Embedder()
        self.vector_searcher = VectorSearcher()
        self.bm25_searcher = BM25Searcher()
    
    def load_chunks(self, pattern: Optional[str] = None) -> None:
        """
        Load document chunks from files matching the pattern
        
        Args:
            pattern: Glob pattern for chunk files (default: from config)
        """
        if pattern is None:
            pattern = f"{self.config['chunks_dir']}/**/*.json"
            
        self.chunks = []
        for fp in glob.glob(pattern, recursive=True):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.chunks.extend(data if isinstance(data, list) else [data])
            except Exception as e:
                print(f"Error loading chunks from {fp}: {e}")
        
        print(f"✓ Loaded: {len(self.chunks)} chunks")
        
        # Update BM25 searcher with loaded chunks
        self._update_bm25_index()
    
    def _update_bm25_index(self) -> None:
        """Update BM25 index with currently loaded chunks"""
        # BM25SearcherImproved handles its own initialization
        pass
    
    def index(self) -> None:
        """
        Create full-text and vector indexes from loaded chunks
        """
        if not self.chunks:
            print("⚠️ No chunks loaded - aborting.")
            return
        
        # Clear existing LanceDB table if it exists
        self.vector_searcher.ensure_empty_table()
        
        # Set BM25 index
        dim = self.embedder.dim()
        print(f"→ Indexing {len(self.chunks):,} chunks (Vector dimension = {dim})...")
        
        # Process chunks and generate embeddings
        normalized_vectors, all_texts = self._generate_embeddings()
        
        # Save normalized vectors to disk
        vectors_path = self._save_normalized_vectors(normalized_vectors)
        
        # Create vector index
        self._create_vector_index(normalized_vectors, all_texts)
        
        print("✓ Indexing complete.")
        print(f"✓ Normalized vectors saved to {vectors_path}")
    
    def _generate_embeddings(self) -> tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for all chunks
        
        Returns:
            Tuple of (normalized vectors array, list of texts)
        """
        # Process all chunks and collect vectors
        all_vectors = []
        all_texts = []
        
        # Process in batches for embedding
        batch_size = self.config["batch_ui"]
        for start in tqdm(range(0, len(self.chunks), batch_size), desc="Embedding", unit="batch"):
            batch = self.chunks[start:start + batch_size]
            
            # Prepare texts for embedding
            texts = [truncate_to_tokens(
                (c.get("text", "") or " ")[:65_000], 
                self.config["max_text_tok"]
            ) for c in batch]
            all_texts.extend(texts)
            
            # Get embeddings
            vecs = self.embedder.encode(texts)
            all_vectors.append(vecs)
        
        # Combine all vectors into a single array
        vectors_array = np.vstack(all_vectors).astype(np.float32)
        print(f"→ Generated {vectors_array.shape[0]} vectors with dimension {vectors_array.shape[1]}")
        
        # Normalize vectors once before indexing (L2 normalization)
        print("→ Normalizing vectors...")
        normalized_vectors = normalize(vectors_array, norm='l2', axis=1).astype(np.float32)
        
        return normalized_vectors, all_texts
    
    def _save_normalized_vectors(self, normalized_vectors: np.ndarray) -> str:
        """
        Save normalized vectors to disk
        
        Args:
            normalized_vectors: Array of normalized vectors
            
        Returns:
            Path where vectors were saved
        """
        vectors_path = os.path.join(self.config.get('cache_dir', '.'), 'vectors_norm.npy')
        print(f"→ Saving normalized vectors to {vectors_path}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(vectors_path), exist_ok=True)
        
        # Save vectors
        np.save(vectors_path, normalized_vectors)
        return vectors_path
    
    def _create_vector_index(self, normalized_vectors: np.ndarray, all_texts: List[str]) -> None:
        """
        Create vector index in LanceDB
        
        Args:
            normalized_vectors: Array of normalized vectors
            all_texts: List of corresponding texts
        """
        print("→ Creating vector index...")
        batch_size = self.config["batch_ui"]
        
        for i in tqdm(range(0, len(self.chunks), batch_size), desc="Indexing", unit="batch"):
            batch_chunks = self.chunks[i:i + batch_size]
            batch_vectors = normalized_vectors[i:i + batch_size]
            batch_texts = all_texts[i:i + batch_size]
            
            # Prepare rows for LanceDB
            rows = self._prepare_lancedb_rows(i, batch_chunks, batch_vectors, batch_texts)
            
            # Add to LanceDB
            self.vector_searcher.create_or_append_to_table(rows)
        
        # Create vector index
        self.vector_searcher.create_index()
    
    def _prepare_lancedb_rows(self, start_idx: int, chunks: List[Dict[str, Any]], 
                             vectors: np.ndarray, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Prepare rows for insertion into LanceDB
        
        Args:
            start_idx: Starting index for this batch
            chunks: List of document chunks
            vectors: Array of normalized vectors
            texts: List of texts
            
        Returns:
            List of row dictionaries for LanceDB
        """
        return [{
            "id": start_idx + off,
            "vector": vec.tolist(),
            "text": text,
            **extract_payload(chunk.get("metadata", {})),
        } for off, (chunk, vec, text) in enumerate(zip(chunks, vectors, texts))]
    
    def get_chunks(self) -> List[Dict[str, Any]]:
        """Get the loaded chunks"""
        return self.chunks 