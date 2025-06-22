# Bachelor_Thesis
Repository of my Bachelor Thesis at UZH in the Spring Semester of 2025

## Abstract
Swiss fiduciary firms process thousands of multilingual financial documents daily while complying with strict data protection regulations including revDSG. 
Current keyword-based systems cannot understand semantic relationships, forcing professionals to spend excessive time searching for information. Existing artificial intelligence (AI) solutions fail to maintain the data isolation required between client mandates in multi-tenant environments.
Using Design Science Research Methodology (DSRM), we developed a Retrieval-Augmented Generation (RAG) system for Dorean AG combining BM25 lexical search, vector embeddings, and reranking. The system uses GPT-4o on Swiss-only Azure infrastructure with metadata-based tenant isolation.
Testing with 60 fiduciary queries achieved 95\% accuracy and 20-second response times versus hours manually. Two external fiduciary firms validated the system's value for automated financial analysis, though noting limitations in document scope and need for user training. Both confirmed readiness for production deployment.

## Architecture

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


## Setup
A requirements.txt file is provided which contains all the needed libraries for the execution of the program.

## Usage

### Backend API (backend/api/app.py)
To run the backend API you will need an **API key** for the LLM service (AzureOpenAI).

### Document Processing (backend/data_preparation/ocr_chunking.py)
The document processing module handles the chunking of financial documents. To execute the processing, documents must be placed in the designated input directory and the chunking parameters can be adjusted accordingly.

### Embedding Service (backend/data_storage/embedding/embedder.py)
To run the embedding service you will need to have the **API key** for the embedding service (AzureOpenAI).

You can customize the behavior of the embedding by adjusting the parameters in the `generate_embeddings` function:

- **`batch_size`**: Controls how many documents are processed at once.  
  A higher value speeds up processing but requires more memory.

- **`model_name`**: Defines which sentence transformer model to use.  
  Different models provide different trade-offs between speed and quality.

- **`max_seq_length`**: Sets the maximum sequence length for the embeddings.  
  A higher value allows for longer text segments but increases processing time.

### Hybrid Search (backend/data_storage/search/hybrid.py)
To run the hybrid search you will need both the vector database and BM25 index to be initialized.

You can customize the search behavior by adjusting the parameters in the `hybrid_search` function:

- **`alpha`**: Controls the balance between vector search and BM25 keyword search.  
  A value closer to 1.0 gives more weight to semantic search.

- **`top_k`**: Defines how many results to retrieve.  
  A higher value returns more results but may include less relevant ones.

- **`rerank`**: Determines whether to apply re-ranking to the results.  
  Enabling this improves relevance but increases latency.

### Frontend (frontend/)
The frontend requires Node.js and npm to be installed.
