"""
Advanced hybrid search combining BM25, vector search and cross-encoder reranking
with sophisticated score normalization and fusion techniques, including document type prioritization
"""
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import re
import time  # Add time module for detailed timing
from dataclasses import dataclass, field
from sentence_transformers import CrossEncoder
from sklearn.preprocessing import normalize
import logging

from app.backend.config.config import get_config
from .bm25 import get_bm25_searcher
from .vector import VectorSearcher
from ..embedding import Embedder

# Configure logging
logger = logging.getLogger("rag_system.search.hybrid")


class HybridSearcher:
    """
    Advanced hybrid search combining lexical and semantic search with state-of-the-art
    score normalization and fusion techniques
    """
    _instance = None
    
    def __new__(cls):
        """Implement as singleton for efficiency"""
        if cls._instance is None:
            cls._instance = super(HybridSearcher, cls).__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self) -> None:
        """Initialize the hybrid searcher"""
        # Load configuration
        config = get_config()
        self.top_k = config.get("top_k", 10)
        self.candidates = config.get("candidates", 70)  # Use configured value from config.py
        
        # Component weights for score calculation
        self.bm25_weight = config.get("bm25_weight", 0.15)
        self.vector_weight = config.get("vec_weight", 0.15)
        self.reranker_weight = config.get("rerank_weight", 0.7)
        self.title_dense_weight = config.get("title_dense_weight", 0.10)
        
        self.rerank_model_name = config.get("rerank_model", "BAAI/bge-reranker-base")
        self.device = config.get("device", "cpu")
        self.rrf_constant = config.get("rrf_constant", 60)
        self.max_rerank_text_length = config.get("max_rerank_text_length", 4096)
        self.rerank_batch_size = config.get("rerank_batch_size", 32)
        
        self.bm25_searcher = get_bm25_searcher()  # Use improved BM25
        self.vector_searcher = VectorSearcher()
        
        # No need to initialize from disk - improved BM25 handles this
        
        # Lazy-init for reranker
        self.reranker = None
        
        # Initialize timing dictionary
        self.timing = {}
    
    def get_reranker(self) -> CrossEncoder:
        """Get or initialize the reranker"""
        if self.reranker is None:
            print(f"Initializing reranker model: {self.rerank_model_name}")
            self.reranker = CrossEncoder(self.rerank_model_name, device=self.device, max_length=512)
        return self.reranker
    
    def set_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Set chunks for BM25 searcher - ensures BM25 has documents to work with
        """
        # Pass chunks to BM25 searcher so it can build its index
        if chunks:
            # Force initialization of BM25 with the provided chunks
            self.bm25_searcher._ensure_initialized(documents=chunks)
            logger.info(f"Initialized BM25 searcher with {len(chunks)} chunks")
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to 0-1 range using min-max normalization
        
        Args:
            scores: List of scores to normalize
            
        Returns:
            Normalized scores in 0-1 range
        """
        if not scores:
            return []
            
        min_score = min(scores)
        max_score = max(scores)
        
        # Handle case where all scores are the same
        if max_score == min_score:
            return [1.0] * len(scores)
            
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def _normalize_title(self, title: str) -> str:
        """
        Normalize document title to ensure consistent character representation
        between search results and actual files.
        
        Args:
            title: Document title to normalize
            
        Returns:
            Normalized title with consistent character representation
        """
        if not title:
            return ""
        
        # Replace special characters with a standard form
        normalized = title.replace(" / ", " _ ").replace("/", "_")
        
        # Replace any multiple spaces with a single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove any trailing/leading whitespace
        return normalized.strip()
    
    def _calculate_keyword_boost(self, query: str, title: str, metadata: Dict[str, Any] = None) -> float:
        """
        State-of-the-art document type boosting system for RAG search
        
        This advanced boosting system prioritizes exact document type matches
        and ensures that users get the most relevant documents first.
        
        Args:
            query: Search query
            title: Document title
            metadata: Document metadata containing Jahr field and other information
            
        Returns:
            Boost factor (1.0 = no boost, >1.0 = boost, can be very high for exact matches)
        """
        if not query or not title:
            return 1.0
        
        # Normalize both query and title for comparison
        query_lower = query.lower().strip()
        title_lower = title.lower().strip()
        
        # Define critical document types with their importance weights
        critical_document_types = {
            "jahresrechnung": 15.0,      # Highest priority - annual financial statements
            "jahresabschluss": 12.0,     # Annual closing documents
            "bilanz": 10.0,              # Balance sheet
            "erfolgsrechnung": 10.0,     # Income statement
            "gewinn": 8.0,               # Profit documents
            "verlust": 8.0,              # Loss documents
            "liquidität": 7.0,           # Liquidity analysis
            "eigenkapital": 6.0,         # Equity documents
            "fremdkapital": 6.0,         # Debt documents
            "umsatz": 5.0,               # Revenue documents
            "ertrag": 5.0,               # Income documents
            "aufwand": 4.0,              # Expense documents
            "abschreibung": 4.0,         # Depreciation documents
            "steuer": 4.0,               # Tax documents
            "lohnausweis": 8.0,          # Salary statements
            "lohnabrechnungen": 6.0,     # Payroll documents
            "deklaration": 5.0,          # Declaration documents
            "steuererklärung": 7.0,      # Tax returns
            "kontodetails": 3.0,         # Account details
            "protokoll": 4.0,            # Meeting minutes
            "revisionsbericht": 8.0,     # Audit reports
            "abschlussunterlagen": 9.0,  # Closing documents
            "mwst": 5.0,                 # VAT documents
            "ahv": 4.0,                  # Social security documents
            "uvg": 4.0,                  # Accident insurance
            "ktg": 4.0,                  # Daily allowance insurance
            "bvg": 4.0,                  # Occupational pension
        }
        
        boost_factor = 1.0
        
        # PHASE 1: Exact document type matching with massive boost
        for doc_type, importance in critical_document_types.items():
            if doc_type in query_lower:
                # Check if this document type appears in the title
                if doc_type in title_lower:
                    # Apply massive boost for exact document type match
                    boost_factor *= importance
                    # print(f"[BOOST] Exact match '{doc_type}' -> {importance}x boost")
                    
                    # Additional boost if it's the primary document type (appears early in title)
                    title_parts = title_lower.split(' - ')
                    if len(title_parts) >= 3 and doc_type in title_parts[1]:
                        boost_factor *= 2.0
                        # print(f"[BOOST] Primary document type -> additional 2.0x boost")
                    
                    break  # Only apply the most important match
        
        # PHASE 2: Year matching with contextual boost using metadata Jahr field
        query_years = set(re.findall(r'20[12]\d', query_lower))
        
        if query_years and metadata:
            # Get the year from metadata Jahr field
            document_year = metadata.get("Jahr", "")
            # Debug: show available fields in metadata
            available_fields = list(metadata.keys()) if isinstance(metadata, dict) else []
            # print(f"[BOOST] Query years: {query_years}, Document Jahr: '{document_year}', Available fields: {available_fields[:10]}")
            if document_year:
                # Convert to string for comparison
                doc_year_str = str(document_year).strip()
                
                # Check if any query year matches the document year
                if doc_year_str in query_years:
                    # Apply year boost for exact metadata year match
                    year_boost = 2.0  # Stronger boost for exact metadata match
                    boost_factor *= year_boost
                    # print(f"[BOOST] Metadata year match '{doc_year_str}' -> {year_boost}x boost")
                else:
                    # Check if document year is close to query years (±1 year tolerance)
                    try:
                        doc_year_int = int(doc_year_str)
                        for query_year_str in query_years:
                            query_year_int = int(query_year_str)
                            if abs(doc_year_int - query_year_int) == 1:
                                # Small boost for adjacent years
                                adjacent_boost = 1.2
                                boost_factor *= adjacent_boost
                                # print(f"[BOOST] Adjacent year match '{doc_year_str}' (query: {query_year_str}) -> {adjacent_boost}x boost")
                                break
                    except (ValueError, TypeError):
                        pass  # Skip if year conversion fails
            else:
                # Fallback to title-based year matching if metadata Jahr is not available
                title_years = set(re.findall(r'20[12]\d', title_lower))
                if title_years:
                    matching_years = query_years.intersection(title_years)
                    if matching_years:
                        # Weaker boost for title-based year matching
                        year_boost = 1.0 + (len(matching_years) * 0.3)
                        boost_factor *= year_boost
                        # print(f"[BOOST] Title year match {matching_years} (fallback) -> {year_boost}x boost")
        
        # PHASE 3: Document hierarchy boost
        # Prioritize main documents over sub-documents
        if any(main_type in title_lower for main_type in ["jahresrechnung", "jahresabschluss", "bilanz", "erfolgsrechnung"]):
            if "buchhaltungsunterlagen" not in title_lower:
                boost_factor *= 3.0
                # print(f"[BOOST] Main document type (not sub-document) -> 3.0x boost")
        
        # PHASE 4: Penalize irrelevant document types when searching for specific types
        if "jahresrechnung" in query_lower:
            # Heavily penalize non-financial documents when searching for Jahresrechnung
            penalty_keywords = ["buchhaltungsunterlagen", "kreditoren", "debitoren", "kreditkartenbelege"]
            for penalty in penalty_keywords:
                if penalty in title_lower:
                    boost_factor *= 0.1  # Heavy penalty
                    # print(f"[BOOST] Penalty for '{penalty}' when searching Jahresrechnung -> 0.1x")
                    break
        
        # PHASE 5: Client and context matching
        # Additional boost for documents that match the search context
        context_keywords = ["lucid", "gmbh", "ag"]
        query_context = [word for word in context_keywords if word in query_lower]
        title_context = [word for word in context_keywords if word in title_lower]
        
        if query_context and title_context:
            context_boost = 1.0 + (len(set(query_context).intersection(set(title_context))) * 0.2)
            boost_factor *= context_boost
        
        # PHASE 6: Apply intelligent caps based on document type
        if "jahresrechnung" in query_lower and "jahresrechnung" in title_lower:
            # For Jahresrechnung searches, allow very high boosts
            max_boost = 50.0
        elif any(critical_type in query_lower for critical_type in critical_document_types.keys()):
            # For other critical document types
            max_boost = 25.0
        else:
            # For general searches
            max_boost = 10.0
        
        final_boost = min(boost_factor, max_boost)
        
        if final_boost > 1.1:
            # print(f"[BOOST] Final boost factor: {final_boost:.1f}x (capped at {max_boost})")
            pass
        
        return final_boost
    
    def _perform_rank_fusion(self, 
                           bm25_docs: List[Dict[str, Any]], 
                           vector_docs: List[Dict[str, Any]],
                           query: str) -> List[Dict[str, Any]]:
        """
        Perform reciprocal rank fusion to combine results from different search methods
        
        Args:
            bm25_docs: BM25 search results
            vector_docs: Vector search results
            query: Search query
            
        Returns:
            Combined and re-ranked documents
        """
        # Get current config
        config = get_config()
        client_name = config.get("default_address_filter", "")
        
        # Safety check: validate that prefiltering worked correctly
        if client_name:
            # Only log if we find issues
            issues_found = False
            
            # Check vector docs
            filtered_vector_docs = []
            for doc in vector_docs:
                title = doc.get("title", "")
                if not title:
                    if not issues_found:
                        print(f"[DATA INTEGRITY CHECK] Validating results for client: {client_name}")
                        issues_found = True
                    print(f"[WARNING] Excluding vector result with empty title: ID={doc.get('id', 'unknown')}")
                    continue
                    
                # Check if metadata is available and contains correct client
                metadata = doc.get("metadata", {})
                doc_client = metadata.get("Adressen", "")
                if doc_client and doc_client != client_name:
                    if not issues_found:
                        print(f"[DATA INTEGRITY CHECK] Validating results for client: {client_name}")
                        issues_found = True
                    print(f"[WARNING] Excluding vector result with wrong client: '{doc_client}' (expected '{client_name}') - ID={doc.get('id', 'unknown')}")
                    continue
                    
                filtered_vector_docs.append(doc)
                
            if len(filtered_vector_docs) != len(vector_docs):
                print(f"[INFO] Filtered out {len(vector_docs) - len(filtered_vector_docs)} invalid vector results")
                vector_docs = filtered_vector_docs
                
            # Check BM25 docs
            filtered_bm25_docs = []
            for doc in bm25_docs:
                title = doc.get("title", "")
                if not title:
                    if not issues_found:
                        print(f"[DATA INTEGRITY CHECK] Validating results for client: {client_name}")
                        issues_found = True
                    print(f"[WARNING] Excluding BM25 result with empty title: ID={doc.get('id', 'unknown')}")
                    continue
                    
                # Check if metadata is available and contains correct client
                metadata = doc.get("metadata", {})
                doc_client = metadata.get("Adressen", "")
                if doc_client and doc_client != client_name:
                    if not issues_found:
                        print(f"[DATA INTEGRITY CHECK] Validating results for client: {client_name}")
                        issues_found = True
                    print(f"[WARNING] Excluding BM25 result with wrong client: '{doc_client}' (expected '{client_name}') - ID={doc.get('id', 'unknown')}")
                    continue
                    
                filtered_bm25_docs.append(doc)
                
            if len(filtered_bm25_docs) != len(bm25_docs):
                print(f"[INFO] Filtered out {len(bm25_docs) - len(filtered_bm25_docs)} invalid BM25 results")
                bm25_docs = filtered_bm25_docs
        
        # Create dictionaries for document lookup
        bm25_dict = {doc["id"]: doc for doc in bm25_docs}
        vector_dict = {doc["id"]: doc for doc in vector_docs}
        
        # Get all unique document IDs
        all_ids = set(bm25_dict.keys()) | set(vector_dict.keys())
        
        # Create ranks for BM25 and vector documents
        bm25_ranks = {doc["id"]: i+1 for i, doc in enumerate(
            sorted(bm25_docs, key=lambda x: x.get("bm25_score", 0), reverse=True))}
        vector_ranks = {doc["id"]: i+1 for i, doc in enumerate(
            sorted(vector_docs, key=lambda x: x.get("vector_score", 0), reverse=True))}
        
        # Calculate reciprocal rank fusion scores
        fusion_scores = {}
        for doc_id in all_ids:
            # Get ranks, using a high rank (1000) if document wasn't found in that search method
            bm25_rank = bm25_ranks.get(doc_id, 1000)
            vector_rank = vector_ranks.get(doc_id, 1000)
            
            # Calculate RRF score
            rrf_score = 1/(self.rrf_constant + bm25_rank) + 1/(self.rrf_constant + vector_rank)
            fusion_scores[doc_id] = rrf_score
        
        # Create combined documents with fusion scores
        combined_docs = []
        for doc_id in all_ids:
            # Prioritize document from vector search if available, otherwise use BM25
            base_doc_source = vector_dict.get(doc_id, bm25_dict.get(doc_id, {}))
            doc = base_doc_source.copy() # Make a copy to modify
            
            # Store the original full text - IMPORTANT: always preserve the full text
            doc["full_text"] = base_doc_source.get("text", "")

            # Add source information
            if doc_id in vector_dict and doc_id in bm25_dict:
                doc["source"] = "vector+bm25"  # Found by both search methods
            elif doc_id in vector_dict:
                doc["source"] = "vector"  # Found only by vector search
            else:
                doc["source"] = "bm25"  # Found only by BM25 search
            
            # Add scores from both methods
            doc["vector_score"] = vector_dict[doc_id].get("vector_score", 0) if doc_id in vector_dict else 0
            doc["bm25_score"] = bm25_dict[doc_id].get("bm25_score", 0) if doc_id in bm25_dict else 0
            doc["fusion_score"] = fusion_scores[doc_id]
            combined_docs.append(doc)
        
        # Sort by fusion score first to get only the top candidates to process titles
        sorted_docs = sorted(combined_docs, key=lambda x: x["fusion_score"], reverse=True)
        top_candidates = sorted_docs[:min(100, len(sorted_docs))]
        
        # Only encode titles for top candidates to save time
        if top_candidates:
            print(f"Encoding {len(top_candidates)} document titles in one batch")
            
            # Normalize query
            normalized_query = self._normalize_title(query)
            query_vec = self.vector_searcher.encode_title(normalized_query)
            
            # Extract titles in the same order as top_candidates
            titles = [self._normalize_title(doc.get("title", "")) for doc in top_candidates]
            
            # Encode all titles at once by batching with the vector searcher
            embedder = Embedder()
            title_vectors = embedder.encode(titles)
            
            # Calculate similarities between query and each title vector
            for i, doc in enumerate(top_candidates):
                title_vec = title_vectors[i]
                # Normalize the title vector (L2 normalization)
                title_vec = normalize(title_vec.reshape(1, -1), norm='l2').astype(np.float32).flatten()
                # Calculate cosine similarity
                sim = float(np.dot(query_vec, title_vec))
                doc["title_sim"] = sim  # raw cosine similarity
            
            # Normalize title similarity scores
            title_sims = [doc.get("title_sim", 0) for doc in top_candidates]
            normalized_title_sims = self._normalize_scores(title_sims)
            for doc, norm_sim in zip(top_candidates, normalized_title_sims):
                doc["title_sim"] = norm_sim
            
            # Set default title similarity for any documents not in top candidates
            for doc in sorted_docs:
                if "title_sim" not in doc:
                    doc["title_sim"] = 0.0
        
        # Return sorted results
        return sorted(sorted_docs, key=lambda x: x["fusion_score"], reverse=True)
    
    def _perform_search_with_filter(self, query: str, filter_expr: Optional[str]) -> List[Dict[str, Any]]:
        """
        Execute the search with the given filter expression
        
        Args:
            query: Search query
            filter_expr: Filter expression
            
        Returns:
            List of ranked documents
        """
        try:
            # Step 1: Get vector search results with prefiltering
            print("Performing search...")
            vector_start = time.time()
            vec_results = self.vector_searcher.search(query, filter_expr=filter_expr)
            vector_end = time.time()
            self.timing["vector_search"] = vector_end - vector_start
            
            # Check if the search returned no results and if it contained an ExternerZugriffItems filter
            if not vec_results and filter_expr and "ExternerZugriffItems" in filter_expr:
                print(f"No results found matching the security filter criteria")
                return []  # Return empty results when ExternerZugriffItems filter has no matches - SECURITY BOUNDARY
            
            # Ensure scores are properly normalized to 0-1 range
            if vec_results:
                # Extract vector scores
                vector_scores = [doc.get("vector_score", 0) for doc in vec_results]
                # Normalize vector scores
                normalized_vec_scores = self._normalize_scores(vector_scores)
                
                # Update vector scores with normalized values
                for doc, norm_score in zip(vec_results, normalized_vec_scores):
                    doc["vector_score"] = norm_score
                    
                print(f"Vector search: {len(vec_results)} results")
            else:
                print("No vector results found")
                return []  # Return empty results if no vector results
            
            # Step 2: Get BM25 results using the same prefiltering expression
            bm25_start = time.time()
            bm25_results = self._get_bm25_results(query, filter_expr)
            bm25_end = time.time()
            self.timing["bm25_search"] = bm25_end - bm25_start
            
            # Step 3: Combine results using reciprocal rank fusion
            fusion_start = time.time()
            combined_results = self._perform_rank_fusion(bm25_results, vec_results, query)
            fusion_end = time.time()
            self.timing["fusion"] = fusion_end - fusion_start
            
            # Print summary of filtering in compact format
            if filter_expr:
                print(f"Filter applied: {filter_expr}")
            
            # Limit candidates for reranking
            candidates = combined_results[:self.candidates]
            
            if not candidates:
                print("No documents found matching the criteria")
                return []
            
            # Apply document type priority system before reranking
            candidates = self._apply_document_type_priority(candidates, query)
            
            return self._apply_reranking_and_finalize(candidates, query)
                
        except Exception as e:
            print(f"Error in search: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _get_bm25_results(self, query: str, filter_expr: Optional[str]) -> List[Dict[str, Any]]:
        """
        Get BM25 search results with normalized scores
        
        Args:
            query: Search query
            filter_expr: Filter expression
            
        Returns:
            Normalized BM25 search results
        """
        try:
            # Apply the same filter expression to BM25 search
            # The improved BM25 will initialize itself on first use
            # Request 100 results to match vector search
            bm25_results = self.bm25_searcher.search(query, top_k=100, filter_expr=filter_expr)
            
            # Normalize BM25 scores to 0-1 range
            if bm25_results:
                # Extract BM25 scores
                bm25_scores = [doc.get("bm25_score", 0) for doc in bm25_results]
                # Normalize BM25 scores
                normalized_bm25_scores = self._normalize_scores(bm25_scores)
                
                # Update BM25 scores with normalized values
                for doc, norm_score in zip(bm25_results, normalized_bm25_scores):
                    doc["bm25_score"] = norm_score
                
                print(f"BM25 search: {len(bm25_results)} results")
            
            return bm25_results
                
        except Exception as e:
            print(f"BM25 search error: {e}")
            return []  # Use empty BM25 results if error occurs
    
    def _apply_reranking_and_finalize(self, candidates: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Apply reranking to candidates and finalize results
        
        Args:
            candidates: Candidate documents for reranking. Each doc should have 'text' (for reranker) and 'full_text'.
            query: Search query
            
        Returns:
            Final ranked documents
        """
        # Standard path: Apply cross-encoder reranking
        print("Neural reranking...")
        reranker_start = time.time()
        reranker = self.get_reranker()
        
        # Make sure all docs have a full_text field for later retrieval
        for doc in candidates:
            if "full_text" not in doc:
                doc["full_text"] = doc.get("text", "")
                
        # Use a truncated version of full_text for reranking to avoid exceeding model limits
        # But keep the full_text intact for later use in context to LLM
        pairs = [(query, doc.get("full_text", "")[:self.max_rerank_text_length]) for doc in candidates]
        
        try:
            # Get reranker scores with larger batch size for faster processing
            # Use configured batch size, or device-specific defaults
            if self.rerank_batch_size > 32 and self.device in ["cuda", "mps"]:
                batch_size = self.rerank_batch_size
            elif self.device in ["cuda", "mps"]:
                batch_size = 64  # Default for GPU
            else:
                batch_size = max(32, self.rerank_batch_size)  # At least 32 for CPU
            
            print(f"Reranking {len(pairs)} documents with batch size {batch_size}")
            reranker_scores = reranker.predict(pairs, batch_size=batch_size)
            
            # Add reranking scores and create ensemble score
            for doc, score in zip(candidates, reranker_scores):
                doc["reranking_score"] = float(score)
                
                # Calculate keyword boost for exact matches in title and metadata year
                # Pass the entire document as metadata since Jahr is a top-level field
                keyword_boost = self._calculate_keyword_boost(query, doc.get("title", ""), doc)
                doc["keyword_boost"] = keyword_boost
                
                # Calculate ensemble score using logarithmic combination
                # Emphasize reranking_score with increased weight (55%)
                bm25_contrib = np.log1p(doc.get("bm25_score", 0) * 5) * self.bm25_weight
                vector_contrib = np.log1p(doc.get("vector_score", 0) * 5) * self.vector_weight
                title_contrib = np.log1p(doc.get("title_sim", 0) * 5) * self.title_dense_weight
                rerank_contrib = score * self.reranker_weight
                
                # Apply keyword boost multiplicatively to BM25 contribution for exact keyword matches
                if keyword_boost > 1.0:
                    bm25_contrib *= keyword_boost
                
                # Calculate base ensemble score
                base_score = bm25_contrib + vector_contrib + title_contrib + rerank_contrib
                
                # Ensure scores are in 0-1 range
                doc["score"] = min(1.0, max(0.0, base_score))
            
            # Sort by final ensemble score
            final_results = sorted(candidates, key=lambda x: x["score"], reverse=True)[:self.top_k]
            
            # Add rank to results
            for i, doc in enumerate(final_results, 1):
                doc["rank"] = i
                # Ensure full_text is preserved for all final results
                if "full_text" not in doc:
                    doc["full_text"] = doc.get("text", "")
                
                # Create a text_preview field for display purposes
                if "text_preview" not in doc and "full_text" in doc:
                    preview_length = 150
                    text = doc["full_text"]
                    doc["text_preview"] = text[:preview_length] + "..." if len(text) > preview_length else text

            reranker_end = time.time()
            self.timing["reranking"] = reranker_end - reranker_start
            
            # Add detailed timing to results for debugging
            if final_results:
                final_results[0]["timing"] = self.timing
            
            self._log_search_results(final_results)
            
            return final_results
            
        except Exception as e:
            print(f"Reranking error: {e}")
            reranker_end = time.time()
            self.timing["reranking"] = reranker_end - reranker_start
            return self._fallback_ranking(candidates)
    
    def _fallback_ranking(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback ranking when reranking fails
        
        Args:
            candidates: Candidate documents
            
        Returns:
            Ranked documents using fusion scores
        """
        # Fallback to fusion scores if reranking fails
        for doc in candidates:
            # Use fusion score as the final score
            doc["score"] = doc.get("fusion_score", 0)
            # Ensure full_text is present here too
            if "full_text" not in doc:
                 doc["full_text"] = doc.get("text", "") # Fallback

        final_results = sorted(candidates, key=lambda x: x["score"], reverse=True)[:self.top_k]
        for i, doc in enumerate(final_results, 1):
            doc["rank"] = i
            # Ensure full_text is present in the final_results (again, for safety)
            if "full_text" not in doc:
                 original_candidate = {cand_doc["id"]: cand_doc for cand_doc in candidates}.get(doc["id"])
                 if original_candidate:
                     doc["full_text"] = original_candidate.get("full_text", original_candidate.get("text", ""))
            
            # Create a text_preview field for display purposes
            if "text_preview" not in doc and "full_text" in doc:
                preview_length = 150
                text = doc["full_text"]
                doc["text_preview"] = text[:preview_length] + "..." if len(text) > preview_length else text

        print(f"Used fallback ranking: {len(final_results)} results")
        return final_results
    
    def _log_search_results(self, final_results: List[Dict[str, Any]]) -> None:
        """
        Log search results with normalized scores - simplified version
        
        Args:
            final_results: Final search results
        """
        # Only log the total number of results found
        print(f"✓ Found {len(final_results)} results")
        
        # Validate final results for data integrity issues
        config = get_config()
        client_name = config.get("default_address_filter", "")
        
        # Log timing information in a compact format
        print(f"Search time: {self.timing.get('total', 0):.2f}s (Vector: {self.timing.get('vector_search', 0):.2f}s, BM25: {self.timing.get('bm25_search', 0):.2f}s, Rank fusion: {self.timing.get('fusion', 0):.2f}s, Rerank: {self.timing.get('reranking', 0):.2f}s)")
        
        # Check results for data integrity issues
        has_integrity_issues = False
        
        # Log the top 5 results in a compact format
        print("\n----- TOP RESULTS -----")
        for i, doc in enumerate(final_results[:5], 1):
            title = doc.get('title', f"ID={doc['id']}")
            doc_id = doc.get('id', 'unknown')
            score = doc.get('score', 0)
            
            # Flag any empty titles
            if not title or title.strip() == "":
                print(f"{i}. [EMPTY TITLE] ID={doc_id} [score: {score:.2f}] ⚠️")
                has_integrity_issues = True
                continue
                
            # Flag results with wrong client name if a filter is set
            if client_name:
                metadata = doc.get('metadata', {})
                doc_client = metadata.get('Adressen', '')
                
                if doc_client and doc_client != client_name:
                    print(f"{i}. {title} [score: {score:.2f}] ⚠️ WRONG CLIENT: {doc_client}")
                    has_integrity_issues = True
                    continue
            
            # Show keyword boost if significant
            keyword_boost = doc.get('keyword_boost', 1.0)
            boost_info = f" (boost: {keyword_boost:.1f}x)" if keyword_boost > 1.1 else ""
            
            # Simple compact output for valid results
            print(f"{i}. {title} [score: {score:.2f}]{boost_info}")
        
        # Only show "more results" message if there are more than 5 results
        if len(final_results) > 5:
            print(f"... and {len(final_results) - 5} more results")
            
        # Add data integrity warning if issues were found
        if has_integrity_issues:
            print("\n⚠️ WARNING: Some search results have data integrity issues (empty titles or wrong client).")
            print("   These may cause problems when retrieving documents.")
    
    def search(self, query: str, filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Execute advanced hybrid search with full score normalization and consistent prefiltering
        
        Args:
            query: Search query
            filter_expr: Optional filter expression for prefiltering documents
            
        Returns:
            List of ranked documents with normalized scores
        """
        # Reset timing dictionary
        self.timing = {
            "total": 0,
            "vector_search": 0,
            "bm25_search": 0,
            "fusion": 0,
            "reranking": 0
        }
        
        start_time = time.time()
        
        # Parse filters from filter_expr
        address_value = None
        externerzugriffitems_value = None
        config = get_config()
        
        # Use centralized security configuration
        active_filter = config.get("active_security_filter", "standard")
        security_filters = config.get("security_filters", {})
        filter_config = security_filters.get(active_filter, {})
        
        # Get default values from config
        if filter_expr is None:
            if config.get("default_address_filter"):
                address_value = config["default_address_filter"]
                print(f"Using address filter: {address_value}")
            
            # Get security filter from centralized config
            if filter_config:
                field = filter_config.get("field", "")
                value_str = filter_config.get("value", "")
                if field == "ExternerZugriffItems" and value_str:
                    externerzugriffitems_value = value_str
                    print(f"Using {field} security filter from config: {value_str}")
            else:
                # Fall back to legacy config for backward compatibility
                externerzugriffitems_config = config.get("default_externerzugriffitems_filter")
                if externerzugriffitems_config is not None:
                    # Even empty string values should be explicitly handled
                    if str(externerzugriffitems_config).strip() in ("", "None"):
                        externerzugriffitems_value = None
                        print("No ExternerZugriffItems security filter applied (empty in config)")
                    else:
                        externerzugriffitems_value = str(externerzugriffitems_config).strip()
                        print(f"Using ExternerZugriffItems security filter: {externerzugriffitems_value}")
                else:
                    # If the config value is completely missing
                    print("WARNING: ExternerZugriffItems filter not defined in config - no filter applied")
                    externerzugriffitems_value = None
                
        elif isinstance(filter_expr, str) and "'" in filter_expr:
            # Check if this is the old format with just address
            if "Adressen" in filter_expr:
                address_value = filter_expr.split("'")[1]
                
                # Apply security filter from centralized config
                if filter_config:
                    field = filter_config.get("field", "")
                    value_str = filter_config.get("value", "")
                    if field == "ExternerZugriffItems" and value_str and field not in filter_expr:
                        externerzugriffitems_value = value_str
                        print(f"Security notice: Adding {field} filter: {value_str}")
                else:
                    # Fall back to legacy config
                    externerzugriffitems_config = config.get("default_externerzugriffitems_filter")
                    if externerzugriffitems_config and str(externerzugriffitems_config).strip() not in ("", "None"):
                        externerzugriffitems_value = str(externerzugriffitems_config).strip()
                        print(f"Security notice: Adding ExternerZugriffItems filter: {externerzugriffitems_value}")
            else:
                # For complex filters, just use as is but verify it has required security filter if needed
                print(f"Using custom filter expression")
                
                # Check if the custom filter should have security filter but doesn't
                if filter_config:
                    field = filter_config.get("field", "")
                    value_str = filter_config.get("value", "")
                    required = filter_config.get("required", True)
                    
                    if required and field and value_str and field not in filter_expr:
                        print(f"WARNING: Custom filter expression missing required {field} security filter")
                        print("This could expose sensitive data - adding required security filter")
                        
                        # Enhance the filter with security filter
                        access_filter = f"`{field}` LIKE '%{value_str}%'"
                        enhanced_filter = f"{filter_expr} AND {access_filter}"
                        return self._perform_search_with_filter(query, enhanced_filter)
                else:
                    # Fall back to legacy config
                    externerzugriffitems_config = config.get("default_externerzugriffitems_filter")
                    if (externerzugriffitems_config and 
                        str(externerzugriffitems_config).strip() not in ("", "None") and
                        "ExternerZugriffItems" not in filter_expr):
                        print("WARNING: Custom filter expression missing ExternerZugriffItems security filter")
                        print("This could expose sensitive data - adding required security filter")
                        
                        # Enhance the filter with ExternerZugriffItems
                        access_filter = f"`ExternerZugriffItems` LIKE '%{str(externerzugriffitems_config).strip()}%'"
                        enhanced_filter = f"{filter_expr} AND {access_filter}"
                        return self._perform_search_with_filter(query, enhanced_filter)
                
                return self._perform_search_with_filter(query, filter_expr)
        else:
            address_value = filter_expr
            
            # Apply security filter from centralized config
            if filter_config:
                field = filter_config.get("field", "")
                value_str = filter_config.get("value", "")
                if field == "ExternerZugriffItems" and value_str:
                    externerzugriffitems_value = value_str
                    print(f"Security notice: Adding {field} filter: {value_str}")
            else:
                # Fall back to legacy config
                externerzugriffitems_config = config.get("default_externerzugriffitems_filter")
                if externerzugriffitems_config and str(externerzugriffitems_config).strip() not in ("", "None"):
                    externerzugriffitems_value = str(externerzugriffitems_config).strip()
                    print(f"Security notice: Adding ExternerZugriffItems filter: {externerzugriffitems_value}")
        
        # Log filter values in compact format
        print(f"Search: '{query}'" + 
              (f", Address: '{address_value}'" if address_value else "") + 
              (f", Security: '{externerzugriffitems_value}'" if externerzugriffitems_value else ""))
        
        # Build filter expression
        search_filter_expr = self._build_filter_expression(query, address_value, externerzugriffitems_value)
        
        results = self._perform_search_with_filter(query, search_filter_expr)
        
        # Record total search time
        self.timing["total"] = time.time() - start_time
        
        return results
    
    def _build_filter_expression(self, query: str, address_value: Optional[str], 
                               externerzugriffitems_value: Optional[str]) -> Optional[str]:
        """
        Build SQL filter expression for search
        
        Args:
            query: Original search query
            address_value: Address filter value
            externerzugriffitems_value: ExternerZugriffItems filter value
            
        Returns:
            SQL filter expression or None
        """
        filter_parts = []
        
        # Add address filter if provided
        if address_value:
            filter_parts.append(f"`Adressen` = '{address_value}'")
        
        # Add ExternerZugriffItems filter if provided and not empty
        # This is a SECURITY BOUNDARY - it must be properly enforced
        if externerzugriffitems_value and externerzugriffitems_value.strip() not in ("", "None"):
            try:
                # Check if it's a list format like "1; 2" or comma separated "1,2"
                if ";" in externerzugriffitems_value:
                    items = [item.strip() for item in externerzugriffitems_value.split(";") if item.strip()]
                    if items:
                        item_clauses = [f"`ExternerZugriffItems` LIKE '%{item}%'" for item in items]
                        filter_parts.append(f"({' OR '.join(item_clauses)})")
                        print(f"Added ExternerZugriffItems security filter with multiple values: {', '.join(items)}")
                    else:
                        # If we have an empty list after splitting, this is a configuration error
                        print("WARNING: ExternerZugriffItems filter was specified but contained no valid values")
                elif "," in externerzugriffitems_value:
                    # Handle comma-separated values from centralized config
                    items = [item.strip() for item in externerzugriffitems_value.split(",") if item.strip()]
                    if items:
                        item_clauses = [f"`ExternerZugriffItems` LIKE '%{item}%'" for item in items]
                        filter_parts.append(f"({' OR '.join(item_clauses)})")
                        print(f"Added ExternerZugriffItems security filter with multiple values: {', '.join(items)}")
                    else:
                        # If we have an empty list after splitting, this is a configuration error
                        print("WARNING: ExternerZugriffItems filter was specified but contained no valid values")
                else:
                    filter_parts.append(f"`ExternerZugriffItems` LIKE '%{externerzugriffitems_value.strip()}%'")
                    print(f"Added ExternerZugriffItems security filter: {externerzugriffitems_value.strip()}")
            except Exception as e:
                # Log the error but don't bypass the filter - security boundary
                print(f"ERROR: Invalid ExternerZugriffItems filter: {e}")
                print("Security notice: ExternerZugriffItems filter could not be properly applied")
                # Return a filter that will match nothing as a safety measure
                return "`id` = '-1'"  # Will match no documents
        else:
            print(f"No ExternerZugriffItems filter applied (value: '{externerzugriffitems_value}')")
        
        # Add year filtering if years are mentioned in the query
        years = re.findall(r'20[12]\d', query)  # Find years like 2020, 2021, 2022, 2023, etc.
        if years:
            print(f"Years mentioned in query: {', '.join(years)}")
            year_clauses = [f"`Jahr` = '{year}'" for year in years]
            filter_parts.append(f"({' OR '.join(year_clauses)})")
            print(f"Added year filtering for: {', '.join(years)}")
            print(f"Note: Years are used for filtering but NOT for title boosting")
        
        # Combine all filter parts with AND
        if filter_parts:
            search_filter_expr = " AND ".join(filter_parts)
            print(f"Combined filter expression: {search_filter_expr}")
            return search_filter_expr
        
        return None
    
    def _apply_document_type_priority(self, candidates: List[Dict], query: str) -> List[Dict]:
        """
        State-of-the-art document type priority system
        
        This system identifies critical financial documents and ensures they appear
        at the top of search results when users search for them specifically.
        
        Args:
            candidates: List of candidate documents
            query: Search query
            
        Returns:
            Reordered candidates with document type priority applied
        """
        if not query or not candidates:
            return candidates
        
        query_lower = query.lower().strip()
        
        # Define document type hierarchy (higher number = higher priority)
        document_type_hierarchy = {
            "jahresrechnung": 100,       # Highest priority
            "jahresabschluss": 95,       # Very high priority
            "bilanz": 90,                # High priority
            "erfolgsrechnung": 90,       # High priority
            "revisionsbericht": 85,      # High priority
            "abschlussunterlagen": 80,   # High priority (but lower than main documents)
            "steuererklärung": 75,       # Important
            "lohnausweis": 70,           # Important
            "protokoll": 65,             # Important
            "deklaration": 60,           # Moderate priority
            "lohnabrechnungen": 55,      # Moderate priority
            "kontodetails": 50,          # Moderate priority
            "buchhaltungsunterlagen": 20, # Low priority (supporting documents)
        }
        
        # Identify what document type the user is searching for
        target_doc_type = None
        target_priority = 0
        
        for doc_type, priority in document_type_hierarchy.items():
            if doc_type in query_lower:
                if priority > target_priority:
                    target_doc_type = doc_type
                    target_priority = priority
        
        if not target_doc_type:
            return candidates  # No specific document type detected
        
        # print(f"[PRIORITY] Detected target document type: '{target_doc_type}' (priority: {target_priority})")
        
        # Categorize candidates based on document type match
        exact_matches = []      # Documents that exactly match the target type
        related_matches = []    # Documents that are related but not exact
        other_documents = []    # All other documents
        
        for candidate in candidates:
            title = candidate.get('title', '').lower()
            
            # Check for exact match
            if target_doc_type in title:
                # Additional check: is this a main document or a sub-document?
                if target_doc_type == "jahresrechnung":
                    # For Jahresrechnung, prioritize main documents over Buchhaltungsunterlagen
                    if "buchhaltungsunterlagen" not in title:
                        candidate['_priority_score'] = 1000  # Highest priority
                        exact_matches.append(candidate)
                        # print(f"[PRIORITY] Exact match (main): {title[:80]}...")
                    else:
                        candidate['_priority_score'] = 100   # Lower priority
                        related_matches.append(candidate)
                        # print(f"[PRIORITY] Related match (sub): {title[:80]}...")
                else:
                    candidate['_priority_score'] = 500
                    exact_matches.append(candidate)
                    # print(f"[PRIORITY] Exact match: {title[:80]}...")
            else:
                # Check for related document types
                is_related = False
                for doc_type, priority in document_type_hierarchy.items():
                    if doc_type != target_doc_type and doc_type in title:
                        candidate['_priority_score'] = priority
                        related_matches.append(candidate)
                        is_related = True
                        break
                
                if not is_related:
                    candidate['_priority_score'] = 1
                    other_documents.append(candidate)
        
        # Sort each category by their original scores
        exact_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
        related_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
        other_documents.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Combine in priority order
        reordered_candidates = exact_matches + related_matches + other_documents
        
        # print(f"[PRIORITY] Reordered: {len(exact_matches)} exact, {len(related_matches)} related, {len(other_documents)} other")
        
        return reordered_candidates
    
    