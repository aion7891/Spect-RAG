"""LightRAG baseline runner (runs in its own venv). Usage:
  python lightrag_runner.py index <workdir> <docs.json>          docs.json = [{"id", "text"}, ...]
  python lightrag_runner.py query <workdir> <queries.json> <out.json> <mode> [max_total_tokens]
  python lightrag_runner.py retry <workdir> <docs.json>          re-insert documents that failed
Runs in its own virtual environment: pip install lightrag-hku ollama (Python 3.12). Uses Ollama
with gemma3:4b (extraction) and nomic-embed-text (embeddings, with the same prefixes as SpectraFuse).
"""
import asyncio, json, sys, time
from functools import partial
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc

HOST = "http://localhost:11434"
TOKENS = int(sys.argv[6]) if len(sys.argv) > 6 else 12000


async def make(workdir):
    rag = LightRAG(
        working_dir=workdir, llm_model_func=ollama_model_complete, llm_model_name="gemma3:4b",
        llm_model_kwargs={"host": HOST, "options": {"num_ctx": 16384, "temperature": 0}, "timeout": 1800},
        embedding_func=EmbeddingFunc(embedding_dim=768, max_token_size=8192, model_name="nomic-embed-text",
                                     supports_asymmetric=True,
                                     func=partial(ollama_embed.func, embed_model="nomic-embed-text", host=HOST,
                                                  query_prefix="search_query: ",
                                                  document_prefix="search_document: ")),
        chunk_token_size=1200, chunk_overlap_token_size=100, llm_model_max_async=2,
        default_llm_timeout=1800)  # a 4B local model can need minutes per chunk
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def index(workdir, docs_path):
    docs = json.load(open(docs_path))
    rag = await make(workdir)
    start = time.time()
    await rag.ainsert([d["text"] for d in docs], ids=[d["id"] for d in docs])
    print(json.dumps({"documents": len(docs), "seconds": round(time.time() - start, 1)}), flush=True)
    await rag.finalize_storages()


async def query(workdir, queries_path, out_path, mode):
    queries = json.load(open(queries_path))
    rag = await make(workdir)
    results = []
    for item in queries:
        start = time.time()
        context = await rag.aquery(item["question"], param=QueryParam(
            mode=mode, only_need_context=True, max_total_tokens=TOKENS, enable_rerank=False))
        results.append({"id": item["id"], "context": context if isinstance(context, str) else str(context),
                        "seconds": round(time.time() - start, 2)})
        json.dump(results, open(out_path, "w"))
        print(len(results), "/", len(queries), "%.1fs" % results[-1]["seconds"], flush=True)
    await rag.finalize_storages()




async def retry(workdir, docs_path):
    """Delete documents that failed (and duplicate records), then insert them again."""
    docs = {d["id"]: d for d in json.load(open(docs_path))}
    status = json.load(open(workdir + "/kv_store_doc_status.json"))
    failed = [doc_id for doc_id, info in status.items() if info.get("status") == "failed"]
    rag = await make(workdir)
    for doc_id in failed:
        await rag.adelete_by_doc_id(doc_id)
    again = [docs[d] for d in failed if d in docs]
    start = time.time()
    if again:
        await rag.ainsert([d["text"] for d in again], ids=[d["id"] for d in again])
    print(json.dumps({"retried": [d["id"] for d in again], "seconds": round(time.time() - start, 1)}), flush=True)
    await rag.finalize_storages()


if __name__ == "__main__":
    if sys.argv[1] == "index":
        asyncio.run(index(sys.argv[2], sys.argv[3]))
    elif sys.argv[1] == "retry":
        asyncio.run(retry(sys.argv[2], sys.argv[3]))
    else:
        asyncio.run(query(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]))
