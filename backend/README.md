# Swiss Financial Document RAG System - Backend Architecture

## Executive Summary

This backend implements a state-of-the-art Retrieval-Augmented Generation (RAG) system specifically designed for Swiss financial documents. It combines advanced search techniques, intelligent filtering, and Swiss-specific formatting to provide accurate, contextual answers from financial document repositories.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Technologies](#core-technologies)
3. [Key Mechanisms and Algorithms](#key-mechanisms-and-algorithms)
4. [Installation and Setup](#installation-and-setup)
5. [API Documentation](#api-documentation)
6. [System Components](#system-components)
7. [Performance Optimizations](#performance-optimizations)
8. [Security and Access Control](#security-and-access-control)
9. [Swiss-Specific Features](#swiss-specific-features)
10. [Deployment and Scaling](#deployment-and-scaling)

## Architecture Overview

The system follows a modular architecture with clear separation of concerns:

```
┌────────────────────────────────────────────────────────────┐
│                        Frontend                            │
└─────────────────────────────┬──────────────────────────────┘
                              │ HTTP/REST
┌─────────────────────────────▼──────────────────────────────┐
│                      FastAPI Server                        │
│                    (app/backend/api/)                      │
├────────────────────────────────────────────────────────────┤
│                     Core Components                        │
├─────────────┬────────────┬────────────┬────────────────────┤
│   Config    │   Search   │ Embedding  │  Data Storage      │
│  Management │   Engine   │   System   │   & Indexing       │
└─────────────┴────────────┴────────────┴────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
               ┌────▼─────┐      ┌─────▼─────┐
               │ LanceDB  │      │   Azure   │
               │ (Vector) │      │  OpenAI   │
               └──────────┘      └───────────┘
```

## Core Technologies

- **FastAPI**: Modern, high-performance web framework
- **LanceDB**: Vector database for semantic search
- **Azure OpenAI**: Embeddings (text-embedding-3-large) and completions (GPT-4o)
- **BM25**: Lexical search algorithm
- **Cross-Encoder**: Neural reranking (BAAI/bge-reranker-base)
- **Pydantic**: Data validation and settings management

## Key Mechanisms and Algorithms

### 1. Hybrid Search System

The hybrid search combines three search methods with sophisticated fusion:

```python
class HybridSearcher:
    """
    Advanced hybrid search combining lexical and semantic search with state-of-the-art
    score normalization and fusion techniques
    """
    
    def _perform_rank_fusion(self, 
                           bm25_docs: List[Dict[str, Any]], 
                           vector_docs: List[Dict[str, Any]],
                           query: str) -> List[Dict[str, Any]]:
        """
        Perform reciprocal rank fusion to combine results from different search methods
        """
        # Create ranks for BM25 and vector documents
        bm25_ranks = {doc["id"]: i+1 for i, doc in enumerate(
            sorted(bm25_docs, key=lambda x: x.get("bm25_score", 0), reverse=True))}
        vector_ranks = {doc["id"]: i+1 for i, doc in enumerate(
            sorted(vector_docs, key=lambda x: x.get("vector_score", 0), reverse=True))}
        
        # Calculate reciprocal rank fusion scores
        fusion_scores = {}
        for doc_id in all_ids:
            bm25_rank = bm25_ranks.get(doc_id, 1000)
            vector_rank = vector_ranks.get(doc_id, 1000)
            
            # RRF formula with configurable constant (default: 60)
            rrf_score = 1/(self.rrf_constant + bm25_rank) + 1/(self.rrf_constant + vector_rank)
            fusion_scores[doc_id] = rrf_score
```

**Key Features:**
- Reciprocal Rank Fusion (RRF) for combining BM25 and vector search
- Configurable weights: BM25 (15%), Vector (15%), Reranker (70%)
- Title similarity matching for additional relevance boost
- Document type prioritization for financial documents

### 2. Advanced Document Filtering in LanceDB

The system implements sophisticated prefiltering at the database level:

```python
def _build_filter_expression(self, query: str, address_value: Optional[str], 
                           externerzugriffitems_value: Optional[str]) -> Optional[str]:
    """
    Build SQL filter expression for search
    
    This creates a multi-layered filtering strategy:
    1. Client filtering (Adressen field)
    2. Security filtering (ExternerZugriffItems)
    3. Dynamic year filtering based on query
    """
    filter_parts = []
    
    # Client-specific filtering
    if address_value:
        filter_parts.append(f"`Adressen` = '{address_value}'")
    
    # Security boundary enforcement
    if externerzugriffitems_value and externerzugriffitems_value.strip() not in ("", "None"):
        # Handle comma-separated security values
        items = [item.strip() for item in externerzugriffitems_value.split(",") if item.strip()]
        if items:
            item_clauses = [f"`ExternerZugriffItems` LIKE '%{item}%'" for item in items]
            filter_parts.append(f"({' OR '.join(item_clauses)})")
    
    # Dynamic year filtering
    years = re.findall(r'20[12]\d', query)
    if years:
        year_clauses = [f"`Jahr` = '{year}'" for year in years]
        filter_parts.append(f"({' OR '.join(year_clauses)})")
    
    return " AND ".join(filter_parts) if filter_parts else None
```

**Security Features:**
- Mandatory security filters that cannot be bypassed
- Multi-value support for access control
- Fail-secure design (returns empty results on filter errors)

### 3. Document Type Boosting System

Intelligent boosting for financial document types:

```python
def _calculate_keyword_boost(self, query: str, title: str, metadata: Dict[str, Any] = None) -> float:
    """
    State-of-the-art document type boosting system for RAG search
    """
    critical_document_types = {
        "jahresrechnung": 15.0,      # Annual financial statements
        "jahresabschluss": 12.0,     # Annual closing documents
        "bilanz": 10.0,              # Balance sheet
        "erfolgsrechnung": 10.0,     # Income statement
        "revisionsbericht": 8.0,     # Audit reports
        # ... more document types
    }
    
    boost_factor = 1.0
    
    # Phase 1: Exact document type matching
    for doc_type, importance in critical_document_types.items():
        if doc_type in query_lower and doc_type in title_lower:
            boost_factor *= importance
            break
    
    # Phase 2: Year matching with metadata
    if query_years and metadata:
        document_year = metadata.get("Jahr", "")
        if document_year in query_years:
            boost_factor *= 2.0  # Exact year match boost
    
    return min(boost_factor, 50.0)  # Cap maximum boost
```

### 4. Dynamic Token-Aware Context Building

Intelligent chunk selection within token limits:

```python
def filter_chunks_dynamically(chunks: List[Dict], query: str, max_docs: int = 50) -> Tuple:
    """
    Dynamically filter chunks to maximize context quality within token limits
    """
    # Token budget allocation
    max_tokens_for_context = 120_000  # Leave space for system prompt
    
    selected_chunks = []
    total_tokens = 0
    
    for chunk in chunks[:max_docs]:
        chunk_tokens = count_tokens(chunk.get("full_text", ""))
        
        if total_tokens + chunk_tokens > max_tokens_for_context:
            break
            
        selected_chunks.append(chunk)
        total_tokens += chunk_tokens
    
    return selected_chunks, removed_count, total_tokens, len(selected_chunks)
```

### 5. Azure OpenAI Embedding System

Efficient embedding generation with batching:

```python
class Embedder:
    """
    Azure OpenAI text embedding model wrapper with batching and error handling
    """
    @classmethod
    def encode(cls, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings with batching and error handling
        """
        # Batch processing for efficiency
        batches = list(batch_iter(
            [truncate_to_tokens(t, cls._config["max_text_tok"]) for t in texts],
            cls._config["max_req_tokens"]
        ))
        
        for batch in batches:
            res = cls._client.embeddings.create(
                model="text-embedding-3-large",
                input=batch,
                encoding_format="float",
            )
            vecs.extend([d.embedding for d in res.data])
        
        return np.asarray(vecs, dtype=np.float32)
```

### 6. Swiss Number Formatting

Specialized formatting for Swiss financial standards:

```python
def format_swiss_numbers(text: str) -> str:
    """
    Format numbers in text to Swiss standard (1'234'567.89)
    """
    def _format_standard_number(match: re.Match) -> str:
        number = match.group(0)
        
        # Skip years
        if re.match(r'^(19|20)\d{2}$', number):
            return number
        
        # Format with apostrophes
        formatted = ''
        for i, digit in enumerate(reversed(integer_part)):
            if i > 0 and i % 3 == 0:
                formatted = "'" + formatted
            formatted = digit + formatted
        
        return formatted
```

## Installation and Setup

### Prerequisites

- Python 3.11+
- Azure OpenAI API access
- Sufficient disk space for LanceDB (varies by dataset size)

### Installation Steps

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

### Configuration

Create a `config.yaml` or use environment variables:

```yaml
# Azure OpenAI settings
azure_openai:
  api_key: ${AZURE_OPENAI_API_KEY}
  endpoint: ${AZURE_OPENAI_ENDPOINT}
  api_version: "2024-02-15-preview"
  deployment_name: "gpt-4o"
  embed_model: "text-embedding-3-large"
  embed_dim: 3072

# Search configuration
search:
  top_k: 10
  candidates: 70
  bm25_weight: 0.15
  vec_weight: 0.15
  rerank_weight: 0.70
  rrf_constant: 60

# Security filters
security_filters:
  standard:
    field: "ExternerZugriffItems"
    value: "1,2"
    required: true
```

## API Documentation

### Main Endpoints

#### Health Check
```http
GET /
```
Returns API status and version information.

#### Chat Endpoint
```http
POST /chat
Content-Type: application/json

{
  "question": "Was war der Gewinn im Jahr 2023?"
}
```

**Response:**
```json
{
  "answer": "Der Gewinn im Jahr 2023 betrug CHF 1'234'567.89",
  "sources": [
    "Jahresrechnung 2023 (3 Dokumente verwendet, Ø Score: 0.892)",
    "Bilanz 2023 (Score: 0.845)"
  ],
  "confidence": 0.95,
  "timing": {
    "total": 2.341,
    "search": 0.892,
    "llm_generation": 1.234
  }
}
```

## System Components

### 1. Data Preparation Pipeline

- **OCR and Chunking** (`ocr_chunking.py`): Converts PDFs to searchable chunks
- **M-Files Downloader** (`mfiles_downloader.py`): Retrieves documents from M-Files
- **Embedding and Indexing** (`embed_and_index.py`): Creates vector embeddings
- **BM25 Index Manager** (`bm25_index_manager.py`): Manages lexical search index

### 2. Search Components

- **Vector Search** (`vector.py`): Semantic similarity search using embeddings
- **BM25 Search** (`bm25.py`): Traditional keyword-based search
- **Hybrid Search** (`hybrid.py`): Combines and reranks results
- **Cross-Encoder Reranking**: Neural model for final relevance scoring

### 3. Question Answering

- **QA System** (`qa_system.py`): Orchestrates search and answer generation
- **Context Building**: Dynamic token-aware chunk selection
- **Answer Generation**: GPT-4 based response synthesis

## Performance Optimizations

### 1. Singleton Pattern
All major components use singleton pattern to avoid repeated initialization:
```python
class HybridSearcher:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HybridSearcher, cls).__new__(cls)
            cls._instance._init()
        return cls._instance
```

### 2. Batch Processing
- Embedding generation processes multiple texts in single API calls
- Reranking uses configurable batch sizes (32-64 documents)
- Title similarity computed in batches

### 3. Caching Strategies
- BM25 index persistence between searches
- Normalized vectors cached to disk
- LanceDB index precomputed for fast retrieval

### 4. Early Termination
- Dynamic filtering stops when token limit reached
- Search results limited to top candidates before reranking
- Prefiltering at database level reduces computation

## Security and Access Control

### Multi-Level Security

1. **Client Isolation**: Each client's documents are filtered by `Adressen` field
2. **Access Control**: `ExternerZugriffItems` field controls document visibility
3. **Fail-Secure Design**: Any filter errors result in empty results
4. **Audit Logging**: All queries and results logged with timestamps

### Security Configuration Example
```python
security_filters = {
    "standard": {
        "field": "ExternerZugriffItems",
        "value": "1,2",  # Comma-separated allowed values
        "required": true  # Cannot be bypassed
    }
}
```

## Swiss-Specific Features

### 1. Number Formatting
- Thousands separator: apostrophe (1'234'567)
- Decimal separator: period (0.89)
- Currency formatting: CHF 1'234.56

### 2. Language Support
- German-aware tokenization
- Swiss German stopword handling
- Multi-language metadata support

### 3. Document Types
Specialized handling for Swiss financial documents:
- Jahresrechnung (Annual Report)
- Bilanz (Balance Sheet)
- Erfolgsrechnung (Income Statement)
- Revisionsbericht (Audit Report)
- Steuererklärung (Tax Declaration)

## Deployment and Scaling

### Production Deployment

```bash
# Using Uvicorn with multiple workers
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --workers 4

# Or using Gunicorn with Uvicorn workers
gunicorn app.backend.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Scaling Considerations

1. **Horizontal Scaling**: Stateless design allows multiple instances
2. **Database Scaling**: LanceDB supports sharding for large datasets
3. **Caching Layer**: Redis can be added for session/result caching
4. **Load Balancing**: Use nginx or similar for distributing requests

### Monitoring and Observability

The system includes comprehensive timing instrumentation:
```python
timing = {
    "total": 2.341,
    "search": 0.892,
    "search_vector_search": 0.234,
    "search_bm25_search": 0.156,
    "search_fusion": 0.089,
    "search_reranking": 0.413,
    "context_building": 0.234,
    "llm_generation": 1.215
}
```

## Conclusion

This RAG system represents a production-ready solution for Swiss financial document analysis, combining state-of-the-art search techniques with robust security and Swiss-specific adaptations. The modular architecture allows for easy extension and customization while maintaining high performance and accuracy.

For detailed implementation examples and further documentation, refer to the individual component files in the repository.