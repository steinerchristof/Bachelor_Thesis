"""
Improved BM25 Searcher with efficient index management
No startup testing, no rebuilding unless necessary
"""
from typing import List, Dict, Any, Optional
import logging

from app.backend.data_preparation.bm25_index_manager import get_index_manager
from app.backend.data_storage.utils.text_utils import chunk_for_bm25


logger = logging.getLogger(__name__)


class BM25SearcherImproved:
    """
    State-of-the-art BM25 search with lazy loading and incremental updates
    """
    
    def __init__(self):
        self.index_manager = get_index_manager()
        self._initialized = False
    
    def _ensure_initialized(self, documents: Optional[List[Dict]] = None):
        """
        Ensure index is initialized, but only when first search is performed
        This avoids startup delays
        """
        if self._initialized:
            return
        
        # Try to load existing index
        if self.index_manager.load():
            logger.info("Loaded existing BM25 index")
            self._initialized = True
            return
        
        # Only build if we have documents and index doesn't exist
        if documents and self.index_manager.needs_rebuild(documents):
            logger.info("Building new BM25 index...")
            self.index_manager.build_index(documents)
            self._initialized = True
        else:
            logger.warning("BM25 index not available and no documents provided")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        documents: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using BM25 with optional filters
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_expr: SQL-like filter expression (parsed into dict)
            documents: Documents to index if not already indexed
            
        Returns:
            List of search results with scores
        """
        # Ensure index is ready (lazy initialization)
        self._ensure_initialized(documents)
        
        # Parse filter expression into dictionary
        filters = self._parse_filter_expr(filter_expr) if filter_expr else None
        
        # Perform search
        results = self.index_manager.search(query, top_k, filters)
        
        # Normalize scores to 0-1 range
        if results:
            max_score = max(r["score"] for r in results)
            if max_score > 0:
                for result in results:
                    result["score"] = result["score"] / max_score
        
        return results
    
    def update_document(self, doc_id: str, document: Dict):
        """Update a single document in the index"""
        self._ensure_initialized()
        self.index_manager.update_document(doc_id, document)
        logger.info(f"Updated document {doc_id} in BM25 index")
    
    def remove_document(self, doc_id: str):
        """Remove a document from the index"""
        self._ensure_initialized()
        self.index_manager.remove_document(doc_id)
        logger.info(f"Removed document {doc_id} from BM25 index")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return self.index_manager.get_stats()
    
    def _parse_filter_expr(self, filter_expr: str) -> Dict[str, Any]:
        """
        Parse SQL-like filter expression into dictionary
        Examples:
            "`Adressen` = 'Company A'" -> {"Adressen": "Company A"}
            "`Jahr` = '2023'" -> {"Jahr": "2023"}
            "`Adressen` = 'Company A' AND (`Jahr` = '2023')" -> {"Adressen": "Company A", "Jahr": "2023"}
            "`Adressen` = 'Company A' AND (`Jahr` = '2022' OR `Jahr` = '2023')" -> {"Adressen": "Company A", "Jahr": ["2022", "2023"]}
        """
        filters = {}
        
        # Handle complex expressions with parentheses
        import re
        
        # Find all conditions within parentheses
        paren_pattern = r'\(([^)]+)\)'
        paren_matches = re.findall(paren_pattern, filter_expr)
        
        for paren_content in paren_matches:
            if ' OR ' in paren_content:
                # Handle OR conditions
                or_parts = paren_content.split(' OR ')
                if or_parts:
                    # Get field name from first part
                    first_part = or_parts[0].strip()
                    if ' = ' in first_part:
                        field, _ = first_part.split(' = ', 1)
                        field = field.strip().strip('`')
                        
                        # Extract all values
                        values = []
                        for part in or_parts:
                            if ' = ' in part:
                                _, value = part.split(' = ', 1)
                                value = value.strip().strip("'")
                                values.append(value)
                        
                        if values:
                            filters[field] = values
            else:
                # Handle single condition in parentheses
                if ' = ' in paren_content:
                    field, value = paren_content.split(' = ', 1)
                    field = field.strip().strip('`')
                    value = value.strip().strip("'")
                    filters[field] = value
            
            # Remove the parenthesized expression from the main filter
            filter_expr = filter_expr.replace(f"({paren_content})", "")
        
        # Now handle remaining AND conditions
        # Split by AND, but be careful with any remaining parentheses
        parts = re.split(r'\s+AND\s+', filter_expr)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Handle equality
            if ' = ' in part and ' LIKE ' not in part:
                key, value = part.split(' = ', 1)
                key = key.strip().strip('`')
                value = value.strip().strip("'")
                # Only add if not already added
                if key not in filters:
                    filters[key] = value
            
            # Handle LIKE
            elif ' LIKE ' in part:
                key, value = part.split(' LIKE ', 1)
                key = key.strip().strip('`')
                value = value.strip().strip("'").strip("%")
                filters[key] = value
        
        return filters


# Global instance for backward compatibility
_searcher_instance = None


def get_bm25_searcher() -> BM25SearcherImproved:
    """Get or create the global BM25 searcher"""
    global _searcher_instance
    if _searcher_instance is None:
        _searcher_instance = BM25SearcherImproved()
    return _searcher_instance