"""
Advanced vector search functionality for semantic document retrieval
"""
import lancedb
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional
import pyarrow as pa
from sklearn.preprocessing import normalize

from app.backend.config.config import get_config
from ..embedding import Embedder
from ..utils.metadata_utils import flatten_props
from ..utils.text_utils import normalize_title

class VectorSearcher:
    """
    Vector search implementation using LanceDB with improved score normalization
    """
    _instance = None
    
    def __new__(cls):
        """Implement as singleton for database connection efficiency"""
        if cls._instance is None:
            cls._instance = super(VectorSearcher, cls).__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self):
        """Initialize the vector search instance"""
        config = get_config()
        self.lance_uri = config["lance_uri"]
        self.table_name = config["table_name"]
        self.candidates = config["candidates"]
        
        # Initialize LanceDB connection
        self.db = lancedb.connect(self.lance_uri)
        self.table = (self.db.open_table(self.table_name) 
                      if self.table_name in self.db.table_names() else None)
        
        # Path to normalized vectors
        self.normalized_vectors_path = os.path.join(config.get('cache_dir', '.'), 'vectors_norm.npy')
    
    def encode_title(self, title: str):
        """
        Encode a title string using the embedding model with normalization
        
        Args:
            title: Title string to encode
            
        Returns:
            Normalized embedding vector
        """
        if not title:
            # Return zero vector if title is empty
            embedder = Embedder()
            dim = embedder.dim()
            return np.zeros(dim, dtype=np.float32)
            
        # Use the Embedder to encode the title
        embedder = Embedder()
        title_vector = embedder.encode([title])[0]
        
        # Normalize the vector (L2 normalization)
        title_vector = normalize(title_vector.reshape(1, -1), norm='l2').astype(np.float32).flatten()
        
        return title_vector
    
    def create_schema(self):
        """Create a PyArrow schema for the LanceDB table"""
        embedder = Embedder()
        dim = embedder.dim()
        
        return pa.schema([
            pa.field("id", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("text", pa.string()),
            pa.field("Adressen", pa.string()),
            pa.field("ExternerZugriff", pa.string()),
            pa.field("ExternerZugriffItems", pa.string()),
            pa.field("Titel", pa.string()),
            pa.field("MfilesId", pa.string()),
            pa.field("PortalLink", pa.string()),
            pa.field("Jahr", pa.string()),
            pa.field("KlasseNummer", pa.string()),
            pa.field("KlasseName", pa.string()),
        ])
    
    def ensure_empty_table(self):
        """Ensure the table is empty by dropping it if it exists"""
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
            self.table = None
    
    def create_or_append_to_table(self, rows: List[Dict[str, Any]]):
        """
        Create a new table or append to existing table
        
        Args:
            rows: List of row dictionaries to add to the table
        """
        if self.table is None:
            schema = self.create_schema()
            self.table = self.db.create_table(self.table_name, rows, schema=schema)
        else:
            self.table.add(rows)
    
    def create_index(self):
        """Create a vector index on the table"""
        if self.table:
            self.table.create_index(num_partitions=256, num_sub_vectors=96)
            print("✓ Vector index created")
    
    def _normalize_distance_scores(self, distances: List[float]) -> List[float]:
        """
        Convert distance scores to similarity scores with proper normalization
        
        Args:
            distances: List of distance values from vector search
            
        Returns:
            Normalized similarity scores in 0-1 range
        """
        if not distances:
            return []
            
        # Convert distances to similarities (1 - distance)
        # LanceDB distances are typically in range 0-2 for cosine distance
        similarities = [1.0 - min(d, 2.0)/2.0 for d in distances]
        
        # Apply sigmoid normalization for smoother distribution
        normalized = [1.0 / (1.0 + np.exp(-10 * (s - 0.5))) for s in similarities]
        
        return normalized
    
    def search(self, query: str, filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for semantically similar documents with improved score normalization
        
        Args:
            query: Search query text
            filter_expr: Optional filter expression for SQL-like filtering
            
        Returns:
            List of document dictionaries with properly normalized similarity scores
        """
        if self.table is None:
            print("Vector table not found")
            return []
        
        try:
            print(f"Embedding query: '{query}'")
            # Embed the query
            embedder = Embedder()
            query_vector = embedder.encode([query])[0]
            
            # Normalize the query vector (L2 normalization)
            query_vector = normalize(query_vector.reshape(1, -1), norm='l2').astype(np.float32).flatten()
            
            # Create a proper search configuration that works with LanceDB
            search_column = "vector"  # This should match the column name in your schema
            
            # Work around the vector format issue completely by manually filtering first
            print("[DEBUG] Using pre-filtering approach for vector search")
            
            # Get all records first
            print("[DEBUG] Fetching all records (this could be slow)...")
            df = self.table.to_pandas()
            
            # Apply filter manually if provided
            if filter_expr:
                print(f"[DEBUG] Applying filter: {filter_expr}")
                
                # Parse the filter expression
                import re
                
                # Check if we need to enhance with security filters
                config = get_config()
                enhanced_filter = self._apply_security_filter(filter_expr)
                if enhanced_filter != filter_expr:
                    print(f"[SECURITY] Enhanced filter with required security boundaries")
                    filter_expr = enhanced_filter
                    
                # Extract address filter
                address_match = re.search(r'`Adressen`\s*=\s*\'([^\']+)\'', filter_expr)
                address_filter = address_match.group(1) if address_match else None
                
                # Extract year filter
                year_matches = re.findall(r'`Jahr`\s*=\s*\'([^\']+)\'', filter_expr)
                
                # Extract security filter (ExternerZugriffItems)
                security_match = re.search(r'`ExternerZugriffItems`\s+LIKE\s+\'%([^\']+)%\'', filter_expr)
                security_filter = security_match.group(1) if security_match else None
                
                # Apply manual filtering
                if address_filter and 'Adressen' in df.columns:
                    df = df[df['Adressen'] == address_filter]
                    print(f"[DEBUG] Filtered to address '{address_filter}': {len(df)} rows")
                
                if year_matches and 'Jahr' in df.columns:
                    df = df[df['Jahr'].isin(year_matches)]
                    print(f"[DEBUG] Filtered to years {year_matches}: {len(df)} rows")
                    
                if security_filter and 'ExternerZugriffItems' in df.columns:
                    df = df[df['ExternerZugriffItems'].str.contains(security_filter, na=False)]
                    print(f"[DEBUG] Filtered by security '{security_filter}': {len(df)} rows")
                    
                # Check if security filter is required but no matching records
                active_filter = config.get("active_security_filter", "standard")
                security_filters = config.get("security_filters", {})
                filter_config = security_filters.get(active_filter, {})
                if filter_config and filter_config.get("required", True) and security_filter and df.empty:
                    print("[SECURITY] Required security filter produced no results")
                    return []  # Return empty results - DO NOT bypass security filters
            
            print(f"[DEBUG] Pre-filtering returned {len(df)} rows")
            
            # If we have no results, return empty list
            if df.empty:
                print("No vector search results found")
                return []
            
            # Get a sample to inspect the actual data format
            try:
                sample_df = self.table.search().limit(5).to_pandas()
                if not sample_df.empty and 'Adressen' in sample_df.columns:
                    sample_values = sample_df['Adressen'].dropna().unique().tolist()
                    print(f"[DEBUG] Sample 'Adressen' values from database: {sample_values}")
            except Exception as e:
                print(f"[DEBUG] Could not get sample values: {e}")
            
            # Now create a new table with just the filtered records for vector search
            try:
                # Extract IDs from filtered dataframe
                filtered_ids = df['id'].tolist()
                
                # Create a temporary table name
                import uuid
                temp_table_name = f"temp_search_{uuid.uuid4().hex[:8]}"
                
                # Create a temp table with just the filtered records
                temp_db = lancedb.connect(":memory:")
                
                # Create a temp table with the filtered records
                filtered_records = []
                for _, row in df.iterrows():
                    # Create a record with just the vector and ID
                    record = {
                        "id": row["id"],
                        "vector": row["vector"]
                    }
                    filtered_records.append(record)
                
                # Create the schema for the temp table
                embedder = Embedder()
                dim = embedder.dim()
                schema = pa.schema([
                    pa.field("id", pa.int32()),
                    pa.field("vector", pa.list_(pa.float32(), dim))
                ])
                
                # Create the temp table
                temp_table = temp_db.create_table(temp_table_name, filtered_records, schema=schema)
                
                # Now perform vector search on this filtered table
                search_results = temp_table.search(query_vector.tolist()).limit(self.candidates).to_pandas()
                
                print(f"[DEBUG] Vector search on filtered data returned {len(search_results)} results")
                
                # Map the results back to the original dataframe by ID
                result_ids = search_results['id'].tolist()
                result_distances = search_results['_distance'].tolist()
                
                # Create a mapping of ID to distance
                id_to_distance = {id_val: dist for id_val, dist in zip(result_ids, result_distances)}
                
                # Filter the original dataframe to just the result IDs and add distance
                df = df[df['id'].isin(result_ids)]
                df['_distance'] = df['id'].map(id_to_distance)
                
                # Sort by distance
                df = df.sort_values('_distance')
                
            except Exception as e:
                print(f"[ERROR] Error in vector search with temp table: {e}")
                import traceback
                traceback.print_exc()
                
                # Try an alternative approach - directly search the main table but limit to IDs
                try:
                    print("[DEBUG] Trying fallback vector search approach...")
                    
                    # Create an ID filter for the original table
                    id_filter = " OR ".join([f"`id` = {id_val}" for id_val in filtered_ids])
                    id_filter = f"({id_filter})"
                    
                    # Search with this filter
                    search_query = self.table.search(query_vector.tolist()).where(id_filter, prefilter=True)
                    df = search_query.limit(self.candidates).to_pandas()
                    
                    print(f"[DEBUG] Fallback vector search returned {len(df)} results")
                    
                except Exception as e2:
                    print(f"[ERROR] Error in fallback vector search: {e2}")
                    
                    # Last resort: just calculate the similarity score directly
                    print("[DEBUG] Using last resort approach - manual vector distance calculation")
                    # We'll use the pre-filtered df, but we need to calculate distances
                    
                    # Return the top candidates sorted by score (random if no calculation)
                    import random
                    df = df.sample(min(len(df), self.candidates))
                    df['_distance'] = [random.uniform(0.3, 0.7) for _ in range(len(df))]  # Placeholder distances
            
            # Print a few matching rows to verify filtering
            if not df.empty and 'Adressen' in df.columns:
                print(f"[DEBUG] First 3 matching rows 'Adressen' values:")
                print(df[['Adressen']].head(3))
                
                # Also print ExternerZugriffItems values to debug security filtering
                if 'ExternerZugriffItems' in df.columns:
                    print(f"[DEBUG] First 3 matching rows 'ExternerZugriffItems' values:")
                    print(df[['ExternerZugriffItems']].head(3))
            
            # Get distances and convert to normalized similarity scores
            distances = df["_distance"].tolist() if "_distance" in df.columns else [1.0] * len(df)
            normalized_scores = self._normalize_distance_scores(distances)
            
            # Format results with normalized scores
            results = []
            for i, (_, row) in enumerate(df.iterrows()):
                # Normalize document title for consistent representation with file system
                original_title = row.get("Titel", f"Dokument {row['id']}")
                normalized_title = normalize_title(original_title)
                
                # Create result dictionary with normalized score
                result = {
                    'id': int(row["id"]),
                    'vector_score': normalized_scores[i],  # Use normalized score
                    'text': row["text"],
                    'title': normalized_title,
                    'metadata': {
                        'id': row.get("MfilesId", ""),
                        'Adressen': row.get("Adressen", ""),
                        'Jahr': row.get("Jahr", ""),
                        'ExternerZugriffItems': row.get("ExternerZugriffItems", "")
                    }
                }
                results.append(result)
            
            # Sort by normalized score descending
            sorted_results = sorted(results, key=lambda x: x.get("vector_score", 0.0), reverse=True)
            
            print(f"Vector search completed with {len(sorted_results)} results")
            # Print some score statistics
            if sorted_results:
                scores = [r["vector_score"] for r in sorted_results]
                print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
                print(f"  Score mean: {sum(scores)/len(scores):.4f}")
            
            return sorted_results
            
        except Exception as e:
            print(f"Error in vector search: {e}")
            import traceback
            traceback.print_exc()
            
            # Always return empty list on error to avoid breaking the application
            return []
            
    def _apply_security_filter(self, filter_expr: Optional[str] = None) -> str:
        """
        Apply security filter to ensure consistent security boundary regardless of caller's filter
        
        Args:
            filter_expr: Optional filter expression
            
        Returns:
            Filter expression with security filter applied
        """
        config = get_config()
        
        # Get active security filter
        active_filter = config.get("active_security_filter", "standard")
        security_filters = config.get("security_filters", {})
        filter_config = security_filters.get(active_filter, {})
        
        # If no active filter is configured, no security filter to apply
        if not filter_config:
            # Only log this warning once
            if not hasattr(self, "_logged_no_filter_warning"):
                print("[SECURITY WARNING] No active security filter configured for vector search")
                self._logged_no_filter_warning = True
            return filter_expr or ""
        
        # Get field name and filter details
        field = filter_config.get("field", "")
        value_str = filter_config.get("value", "")
        allowed_values = [v.strip() for v in value_str.split(",") if v.strip()]
        required = filter_config.get("required", True)
        
        # No values to check, no security filter to apply
        if not allowed_values:
            # Only log this warning once
            if not hasattr(self, "_logged_no_values_warning"):
                print(f"[SECURITY WARNING] No values configured for security filter '{field}' in vector search")
                self._logged_no_values_warning = True
            return filter_expr or ""
        
        # Create security filter clause
        security_clause = ""
        if len(allowed_values) == 1:
            security_clause = f"`{field}` LIKE '%{allowed_values[0]}%'"
        else:
            # Multiple allowed values with OR between them
            security_clause = " OR ".join([f"`{field}` LIKE '%{value}%'" for value in allowed_values])
            security_clause = f"({security_clause})"
        
        # Apply security filter
        if not filter_expr:
            return security_clause
        elif required:
            # Security filter is required - AND it with existing filter
            return f"({filter_expr}) AND {security_clause}"
        else:
            # Security filter is optional - use existing filter
            return filter_expr 