from configs.loader import load_config
from ingestion.loaders import load_corpus, load_queries, load_qrels
from indexing.dense_encoder import DenseEncoder
from indexing.faiss_indexer import FaissIndex
from indexing.es_client import get_es_client
from retrieval.fusion import search_hybrid
from tracing import setup_logger, CircuitBreaker, MemoryLogHandler, JsonFormatter
from evaluation.runner import Evaluator
from pathlib import Path
import random
import logging


class FakeESClient:
    def search(self, **kwargs):
        raise ConnectionError("Simulated Elasticsearch downtime")
    
#create a logger
formatter = JsonFormatter()
memory_handler = MemoryLogHandler()
logger = setup_logger(name="regression_test", formatter=formatter, handler=memory_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

#create circuit breaker for each retrieval method
circuit_breaker_bm25 = CircuitBreaker(failure_threshold=3, cooldown_period=60.0)
circuit_breaker_splade = CircuitBreaker(failure_threshold=3, cooldown_period=60.0)   
circuit_breaker_dense = CircuitBreaker(failure_threshold=3, cooldown_period=60.0)

#load configuration
cfg = load_config()

data_cfg = cfg["data"]
bm25_cfg = cfg["bm25"]
splade_cfg = cfg["splade"]
dense_cfg = cfg['dense']

#load data
logger.info("Loading corpus...")

corpus = load_corpus(corpus_path=Path(data_cfg['corpus_path']))
queries = load_queries(queries_path=Path(data_cfg['queries_path']))
qrels = load_qrels(qrels_path=Path(data_cfg['qrels_path']))

#load dense encoder
logger.info("Loading Dense encoder...")
dense_encoder = DenseEncoder()
logger.info("Loading FAISS index ...")
faiss_index = FaissIndex.load(dense_cfg["index_path"], dense_cfg["ids_path"])

#get es client
logger.info("Connecting to Elasticsearch ...")
#es_client = get_es_client()
es_client = FakeESClient()

# 1. Sample 50 random queries
random.seed(42) 
sampled_q_ids = random.sample(list(qrels.keys()), 50)
sampled_qrels = {q_id: qrels[q_id] for q_id in sampled_q_ids}

# 2. Setup evaluator and query list
evaluator = Evaluator(qrels=sampled_qrels)
test_query_ids = list(sampled_qrels.keys())

logger.info(f"Starting regression test on {len(test_query_ids)} queries...")

# 3. Execution loop
for q_id in test_query_ids:
    query_text = queries[q_id].text
    
    fused_results = search_hybrid(
        query_text=query_text,
        query_id=q_id,
        bm25_client=es_client,
        splade_client= es_client, 
        faiss_index=faiss_index,
        corpus=corpus,
        dense_encoder=dense_encoder,
        bm25_breaker=circuit_breaker_bm25,
        splade_breaker=circuit_breaker_splade,
        dense_breaker=circuit_breaker_dense,
        bm25_index_name=bm25_cfg['index_name'],
        splade_index_name=splade_cfg['index_name'],
        logger=logger,
        top_k=100,      
        per_method_top_k=100    
    )
    
    ranked_doc_ids = [res.doc_id for res in fused_results]
    evaluator.add_run(query_id=q_id, ranked_doc_ids=ranked_doc_ids)

# 1. Core Metric
metrics = evaluator.aggregate()
ndcg_10 = metrics["ndcg"][10]
assert ndcg_10 >= 0.68, f"nDCG@10 dropped to {ndcg_10}"

# 2. Structured Logging
assert any("trace_id" in log and "timings" in log for log in memory_handler.log_records), \
    "Structured logging failed: missing trace_id or timings in logs"

# 3. Circuit Breakers
assert circuit_breaker_bm25.state == 'CLOSED', 'BM25 Circuit is not closed'
assert circuit_breaker_splade.state == 'CLOSED', 'SPLADE Circuit is not closed'
assert circuit_breaker_dense.state == 'CLOSED', 'Dense Circuit is not closed'

print("✅ All regression tests passed!")