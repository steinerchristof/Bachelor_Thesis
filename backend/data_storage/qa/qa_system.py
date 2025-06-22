"""
Question answering system using retrieval-augmented generation
"""
import os
import re
import json
import datetime
import time  # Add this import for timing measurements
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict

from app.backend.config.config import get_config, load_openai_client, filter_chunks_dynamically
from ..search import HybridSearcher
from ..indexing import Indexer

# Configure logger
logger = logging.getLogger(__name__)

class SimpleQA:
    """
    Simple QA system using hybrid search and OpenAI completion
    """
    def __init__(self, hybrid_searcher: HybridSearcher):
        """
        Initialize the QA system
        
        Args:
            hybrid_searcher: Hybrid search instance for retrieving documents
        """
        self.config = get_config()
        self.search = hybrid_searcher
        self.client = load_openai_client()
    
    def answer(self, question: str, search_method: str = "semantic") -> Dict[str, Any]:
        """
        Answer a question based on retrieved context
        
        Args:
            question: The question to answer
            search_method: Search method (semantic or hybrid, defaults to semantic)
            
        Returns:
            Dictionary with answer and sources
        """
        try:
            # Start timing
            start_time = time.time()
            timing = {"question_received": 0.0}
            
            # Create a timestamp for logging
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            query_id = re.sub(r'[^\w]', '_', question[:30]).strip('_')
            log_filename = f"logs/context/{timestamp}_{query_id}.json"
            
            # Log data structure
            log_data = {
                "timestamp": timestamp,
                "query": question,
                "search_method": search_method,
                "hits": [],
                "client_response": {},
                "timing": {}  # Add timing data to logs
            }
            
            # Search for relevant documents
            logger.info(f"Received chat request: {question}")
            search_start = time.time()
            hits = self.search.search(question)
            search_end = time.time()
            timing["search"] = search_end - search_start
            logger.info(f"Search completed in {timing['search']:.3f} seconds")
            
            # Get detailed search timing if available
            if hasattr(self.search, 'timing') and isinstance(self.search.timing, dict):
                for key, value in self.search.timing.items():
                    timing[f"search_{key}"] = value
                
                # Log detailed search timing
                logger.debug("Detailed search timing:")
                logger.debug(f"  Vector search: {self.search.timing.get('vector_search', 0):.3f}s")
                logger.debug(f"  BM25 search: {self.search.timing.get('bm25_search', 0):.3f}s")
                logger.debug(f"  Rank fusion: {self.search.timing.get('fusion', 0):.3f}s")
                logger.debug(f"  Title matching: {self.search.timing.get('title_matching', 0):.3f}s")
                logger.debug(f"  Reranking: {self.search.timing.get('reranking', 0):.3f}s")
            
            if not hits:
                response = {
                    "answer": f"Entschuldigung, dazu liegen mir keine Informationen für {self.config['default_address_filter'] or 'Ihre Firma'} vor.",
                    "sources": []
                }
                
                # Log no results
                log_data["hits"] = []
                log_data["client_response"] = response
                log_data["timing"] = timing
                self._save_log(log_filename, log_data)
                
                end_time = time.time()
                timing["total"] = end_time - start_time
                logger.info(f"No results response generated in {timing['total']:.3f} seconds")
                
                # Add timing to response
                response["timing"] = timing
                return response
            
            # Filter out invalid documents (empty titles, wrong client)
            client_name = self.config.get('default_address_filter', '')
            if client_name:
                filtered_hits = []
                for hit in hits:
                    title = hit.get('title', '')
                    if not title or title.strip() == "":
                        logger.warning(f"Excluding document with empty title: ID={hit.get('id', 'unknown')}")
                        continue
                        
                    # Check if metadata is available and contains correct client
                    metadata = hit.get('metadata', {})
                    doc_client = metadata.get('Adressen', '')
                    if doc_client and doc_client != client_name:
                        logger.warning(f"Excluding document with wrong client: '{doc_client}' (expected '{client_name}') - ID={hit.get('id', 'unknown')}")
                        continue
                        
                    filtered_hits.append(hit)
                
                if len(filtered_hits) != len(hits):
                    logger.info(f"Filtered out {len(hits) - len(filtered_hits)} invalid documents")
                    hits = filtered_hits
                    
                    # If we've filtered all documents, return no results
                    if not hits:
                        response = {
                            "answer": f"Entschuldigung, dazu liegen mir keine Informationen für {client_name} vor.",
                            "sources": []
                        }
                        
                        # Log no results
                        log_data["hits"] = []
                        log_data["client_response"] = response
                        log_data["timing"] = timing
                        self._save_log(log_filename, log_data)
                        
                        end_time = time.time()
                        timing["total"] = end_time - start_time
                        logger.info(f"No results response generated in {timing['total']:.3f} seconds")
                        
                        # Add timing to response
                        response["timing"] = timing
                        return response
            
            # Build context from hits - OPTIMIZED to limit context size
            context_start = time.time()
            
            # DYNAMIC TOKEN-AWARE FILTERING: Replace static TOP_K with intelligent space utilization
            filter_start = time.time()
            filtered_hits, removed_count, total_tokens_used, effective_k = filter_chunks_dynamically(
                hits, question, max_docs=50
            )
            filter_end = time.time()
            timing["dynamic_filtering"] = filter_end - filter_start
            
            logger.info(f"Dynamic context building: {effective_k} documents used from {len(hits)} total")
            logger.info(f"Token efficiency: {total_tokens_used:,} tokens used ({(total_tokens_used/120000)*100:.1f}% of 128k limit)")
            
            context_chunks = []
            sources = []
            
            for i, hit in enumerate(filtered_hits):
                title = hit.get('title', 'Kein Titel')
                # Use the metadata_id if available from the hit, otherwise the general id
                doc_id_for_header = hit.get('metadata', {}).get('id') or hit.get('id', '?')
                
                # Format score information
                main_score = hit.get('score', 0)
                reranking_score = hit.get('reranking_score', 0)
                bm25_score = hit.get('bm25_score', 0)
                vector_score = hit.get('vector_score', 0)
                
                # Create score info string
                score_info = f"Score: {main_score:.3f}"
                if 'reranking_score' in hit:
                    score_info += f" | Reranking: {reranking_score:.3f}"
                if 'bm25_score' in hit:
                    score_info += f" | BM25: {bm25_score:.3f}" 
                if 'vector_score' in hit:
                    score_info += f" | Vector: {vector_score:.3f}"
                
                # Build document header
                doc_header = f"[DOKUMENT {i+1}] Titel: {title} | ID: {doc_id_for_header}"
                
                # Always use the 'full_text' if available, otherwise fall back to 'text'
                # IMPORTANT: Do not truncate the text to ensure full chunks are used for context
                context_hit_text = hit.get('full_text', hit.get('text', ''))
                
                # Format and add to context
                context_chunks.append(f"{doc_header}\n\n{context_hit_text}")
            
            # Create source references - group duplicates and show usage
            source_groups = defaultdict(lambda: {'total': 0, 'used': 0, 'ids': [], 'scores': []})
            
            for i, hit in enumerate(hits):
                title = hit.get('title', 'Dokument')
                source_groups[title]['total'] += 1
                source_groups[title]['ids'].append(hit.get('metadata', {}).get('id') or hit.get('id', '?'))
                source_groups[title]['scores'].append(hit.get('score', 0))
                
                # Count if this document was actually used in context
                if i < effective_k:
                    source_groups[title]['used'] += 1
            
            # Create grouped source list
            sources = []
            for title, info in source_groups.items():
                if info['used'] > 0:
                    # Show used documents
                    if info['total'] == 1:
                        # Single document
                        avg_score = info['scores'][0]
                        sources.append(f"{title} (Score: {avg_score:.3f})")
                    else:
                        # Multiple documents - show usage statistics
                        avg_score = sum(info['scores'][:info['used']]) / info['used']  # Average of used docs only
                        if info['used'] == info['total']:
                            sources.append(f"{title} ({info['used']} Dokumente verwendet, Ø Score: {avg_score:.3f})")
                        else:
                            sources.append(f"{title} ({info['used']} von {info['total']} Dokumenten verwendet, Ø Score: {avg_score:.3f})")
                else:
                    # Show unused document groups (optional - you might want to skip these)
                    if info['total'] == 1:
                        avg_score = info['scores'][0]
                        sources.append(f"{title} (nicht verwendet, Score: {avg_score:.3f})")
                    else:
                        avg_score = sum(info['scores']) / len(info['scores'])
                        sources.append(f"{title} ({info['total']} Dokumente nicht verwendet, Ø Score: {avg_score:.3f})")
            
            # Join context chunks
            context = "\n\n" + "\n\n---\n\n".join(context_chunks) + "\n\n"
            
            context_end = time.time()
            timing["context_building"] = context_end - context_start
            logger.info(f"Context building completed in {timing['context_building']:.3f} seconds")
            
            # Add hits to log data - all hits regardless of context inclusion
            log_data["hits"] = [{
                "rank": hit.get("rank", i+1),
                "id": hit.get("id", "?"),
                "metadata_id": hit.get("metadata", {}).get("id") or hit.get("id", "?"),
                "title": hit.get("title", "Dokument"),
                "score": hit.get("score", 0),
                "reranking_score": hit.get("reranking_score", 0),
                "bm25_score": hit.get("bm25_score", 0),
                "vector_score": hit.get("vector_score", 0),
                # Use full_text for preview if available, otherwise text, then truncate for log preview
                "text_preview": (hit.get('full_text', hit.get('text', '')))[:200] + "..." if len(hit.get('full_text', hit.get('text', ''))) > 200 else hit.get('full_text', hit.get('text', ''))
            } for i, hit in enumerate(hits)]
            
            # Log which documents were selected
            logger.debug("Documents selected for answer:")
            for i, hit in enumerate(hits):
                # Only show documents that were actually included in the context
                if i < effective_k:
                    normalized_score = hit.get('score', 0)
                    
                    score_info = f"Norm. Score: {normalized_score:.4f}"
                    if 'reranking_score' in hit and hit.get('reranking_score', 0) > 0:
                        score_info += f" | Reranking: {hit.get('reranking_score', 0):.4f}"
                    if 'bm25_score' in hit and hit.get('bm25_score', 0) > 0:
                        score_info += f" | BM25: {hit.get('bm25_score', 0):.4f}" 
                    if 'vector_score' in hit and hit.get('vector_score', 0) > 0:
                        score_info += f" | Vector: {hit.get('vector_score', 0):.4f}"
                    
                    # First try metadata id, then fall back to the main document id
                    doc_id = hit.get('metadata', {}).get('id') or hit.get('id', '?')
                    logger.debug(f"  {i+1}. Title: {hit.get('title', 'No Title')} | ID: {doc_id} | {score_info}")
            
            # Generate answer using LLM with optimized settings
            llm_start = time.time()
            chat_completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": self.config["system_prompt"]},
                    {"role": "user", "content": f"Kontext:\n{context}\n\nFrage: {question}"},
                ],
            )
            llm_end = time.time()
            timing["llm_generation"] = llm_end - llm_start
            logger.info(f"LLM generation completed in {timing['llm_generation']:.3f} seconds")
            
            # Prepare response
            response = {
                "answer": chat_completion.choices[0].message.content.strip(),
                "sources": sources
            }
            
            # Add response to log data
            log_data["client_response"] = response
            log_data["context_sent_to_llm"] = f"Kontext:\n{context}\n\nFrage: {question}"
            
            # Add timing data
            end_time = time.time()
            timing["total"] = end_time - start_time
            log_data["timing"] = timing
            
            # Save log
            save_log_start = time.time()
            self._save_log(log_filename, log_data)
            save_log_end = time.time()
            timing["log_saving"] = save_log_end - save_log_start
            
            # Add timing data to response
            response["timing"] = timing
            
            # Log timing summary with detailed breakdown
            logger.info("="*60)
            logger.info("TIMING SUMMARY - Complete Request-to-Response Breakdown")
            logger.info("="*60)
            
            # Overall timing
            logger.info(f"Total Request Processing Time: {timing['total']:.3f} seconds")
            logger.info("")
            
            # Search phase breakdown
            logger.info(f"1. SEARCH PHASE: {timing['search']:.3f} seconds ({(timing['search']/timing['total']*100):.1f}% of total)")
            if "search_vector_search" in timing:
                logger.info(f"   a) Vector search: {timing['search_vector_search']:.3f} seconds")
                logger.info(f"   b) BM25 search: {timing['search_bm25_search']:.3f} seconds")
                logger.info(f"   c) Rank fusion: {timing['search_fusion']:.3f} seconds")
                logger.info(f"   d) Neural reranking: {timing['search_reranking']:.3f} seconds")
            logger.info("")
            
            # Context building phase
            logger.info(f"2. CONTEXT BUILDING: {timing['context_building']:.3f} seconds ({(timing['context_building']/timing['total']*100):.1f}% of total)")
            if "dynamic_filtering" in timing:
                logger.info(f"   - Dynamic token filtering: {timing['dynamic_filtering']:.3f} seconds")
            logger.info("")
            
            # LLM generation phase
            logger.info(f"3. LLM GENERATION: {timing['llm_generation']:.3f} seconds ({(timing['llm_generation']/timing['total']*100):.1f}% of total)")
            logger.info("")
            
            # Other operations
            logger.info(f"4. OTHER OPERATIONS:")
            logger.info(f"   - Log saving: {timing.get('log_saving', 0):.3f} seconds")
            
            # Calculate overhead
            accounted_time = (timing['search'] + timing['context_building'] + 
                            timing['llm_generation'] + timing.get('log_saving', 0))
            overhead = timing['total'] - accounted_time
            if overhead > 0.001:
                logger.info(f"   - System overhead: {overhead:.3f} seconds")
            
            logger.info("="*60)
            
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {e}", exc_info=True)
            end_time = time.time()
            return {
                "answer": f"Bei der Beantwortung Ihrer Frage ist ein Fehler aufgetreten: {str(e)}", 
                "sources": [],
                "timing": {"total": end_time - start_time, "error": True}
            }
    
    def _save_log(self, filename: str, data: dict):
        """
        Save log data to a file
        
        Args:
            filename: Path to log file
            data: Log data dictionary
        """
        try:
            # Ensure log directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Write log as pretty-printed JSON
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.debug(f"Log saved: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving log: {e}")

def initialize_qa_system() -> SimpleQA:
    """
    Initialize the QA system with necessary components
    
    Returns:
        Initialized SimpleQA instance
    """
    # Create indexer and load chunks
    indexer = Indexer()
    
    # Create hybrid searcher
    searcher = HybridSearcher()
    
    # Get config to check if table exists
    config = get_config()
    
    # Check if indexing is needed
    db = indexer.vector_searcher.db
    if config["table_name"] not in db.table_names():
        logger.info("Table not found. Indexing required.")
        indexer.load_chunks()
        indexer.index()
    else:
        logger.info("Loading existing index and chunks.")
        indexer.load_chunks()  # Still need chunks for BM25 search
    
    # Always set chunks for the searcher - the BM25SearcherImproved will handle
    # whether to build a new index or use an existing one
    logger.info("Setting chunks for hybrid searcher")
    searcher.set_chunks(indexer.get_chunks())
    
    # Create and return QA system
    qa = SimpleQA(searcher)
    logger.info("QA System initialized.")
    return qa 