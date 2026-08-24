from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from configs.loader import load_config
from indexing.dense_encoder import DenseEncoder
from indexing.splade_encoder import SpladeEncoder
from indexing.faiss_indexer import FaissIndex
from indexing.es_client import get_es_client
from ingestion.loaders import load_corpus
from tracing import setup_logger, CircuitBreaker, JsonFormatter, TraceContext
from pathlib import Path
import logging
from dataclasses import dataclass
from retrieval.fusion import search_hybrid
from pydantic import BaseModel
from uuid import uuid4
import time
import asyncio

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

class DocumentResult(BaseModel):
    doc_id: str
    score: float
    title: str
    text: str

class SearchResponse(BaseModel):
    results: list[DocumentResult]
    trace_id: str
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Setup Production Logger
    formatter = JsonFormatter()
    logger = setup_logger(name="hybrid_api", formatter=formatter, level=logging.INFO)
    logger.info("Starting Hybrid RAG API server...")

    # 2. Initialize Circuit Breakers
    circuit_breaker_bm25 = CircuitBreaker(failure_threshold=3, cooldown_period=60.0)
    circuit_breaker_splade = CircuitBreaker(failure_threshold=3, cooldown_period=60.0)   
    circuit_breaker_dense = CircuitBreaker(failure_threshold=3, cooldown_period=60.0)

    # 3. Load Configuration
    cfg = load_config()
    data_cfg = cfg["data"]
    dense_cfg = cfg['dense']

    # 4. Load Heavy Resources (Once)
    logger.info("Loading corpus...")
    corpus = load_corpus(corpus_path=Path(data_cfg['corpus_path']))
    
    logger.info("Loading Splade encoder...")
    splade_encoder = SpladeEncoder()
    splade_encoder.encode_query("warmup")

    logger.info("Loading Dense encoder and FAISS index...")
    dense_encoder = DenseEncoder()
    dense_encoder.encode_query("warmup")
    
    faiss_index = FaissIndex.load(dense_cfg["index_path"], dense_cfg["ids_path"])

    logger.info("Connecting to Elasticsearch...")
    es_client = get_es_client()
    
    # 5. Attach to app.state
    app.state.logger = logger
    app.state.circuit_breaker_bm25 = circuit_breaker_bm25
    app.state.circuit_breaker_splade = circuit_breaker_splade
    app.state.circuit_breaker_dense = circuit_breaker_dense
    app.state.corpus = corpus
    app.state.dense_encoder = dense_encoder
    app.state.splade_encoder = splade_encoder
    app.state.faiss_index = faiss_index
    app.state.es_client = es_client
    app.state.cfg = cfg # Keep the whole config object for easy access to index names later
    
    yield  # Application runs here
    
    # Optional: Shutdown logic (e.g., logger.info("Shutting down API server..."))

app = FastAPI(lifespan=lifespan)

@app.post('/search', response_model=SearchResponse)
async def search_endpoint(request: Request, payload: SearchRequest):
    
    query_text = payload.query
    top_k = payload.top_k
    es_client = request.app.state.es_client
    faiss_index = request.app.state.faiss_index
    corpus = request.app.state.corpus
    circuit_breaker_bm25 = request.app.state.circuit_breaker_bm25
    circuit_breaker_splade = request.app.state.circuit_breaker_splade 
    circuit_breaker_dense = request.app.state.circuit_breaker_dense 
    dense_encoder = request.app.state.dense_encoder 
    splade_encoder = request.app.state.splade_encoder
    logger = request.app.state.logger 
    
    trace_id = str(uuid4())
    start_time = time.perf_counter()
    ctx = TraceContext(trace_id=trace_id, start_time=start_time)
    
    results = await asyncio.to_thread(search_hybrid,
                                    query_text=query_text,
                                    query_id=trace_id,
                                    bm25_client=es_client,
                                    splade_client=es_client, 
                                    faiss_index=faiss_index,
                                    corpus=corpus,
                                    bm25_breaker=circuit_breaker_bm25  ,
                                    splade_breaker=circuit_breaker_splade ,
                                    dense_breaker=circuit_breaker_dense ,
                                    dense_encoder=dense_encoder,
                                    splade_encoder=splade_encoder,
                                    bm25_index_name=request.app.state.cfg['bm25']['index_name'],
                                    splade_index_name=request.app.state.cfg['splade']['index_name'],
                                    ctx=ctx,
                                    logger=logger,
                                    top_k=top_k)
    
    pydantic_results = [
        DocumentResult(
            doc_id=res.doc_id,
            score=res.score,
            title=res.title,
            text=res.text
        )
        for res in results
    ]
    
    return SearchResponse(results=pydantic_results, trace_id=trace_id)


