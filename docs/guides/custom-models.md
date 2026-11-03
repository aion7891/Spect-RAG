---
description: >-
  Plug in any embedding model, any LLM, your own entity extractor, utility questions and a
  cross-encoder.
---

# Your own models

SpectRAG talks to models only through plain Python functions, so it works with any library or
API. You pass these functions in:

| Plug-in | Signature | Where to pass it |
|---|---|---|
| Embedder | `embed(texts) -> vectors` | `settings={"embedder": embed}` when ingesting, and `embed_fn=embed` when searching |
| LLM for answers | `llm_fn(prompt) -> text` | `search.answer(question, llm_fn)` |
| Entity extractor | `entity_fn(text) -> list of strings` | `SpectRAGIngestion(..., entity_fn=...)` |
| Utility questions | `question_fn(text) -> list of questions` | `SpectRAGIngestion(..., question_fn=make_question_fn(llm_fn))` |
| Cross-encoder | `rerank_fn(question, texts) -> scores` | `SpectRAGRetrieval(..., rerank_fn=...)` with `selector="anchor_expand_ce"` |

The code below uses third-party packages that SpectRAG does not install; install the ones you
use (`pip install sentence-transformers`, `openai`, `anthropic`, `spacy`, …).

## Embedders

Built in: `"hash"` (offline, the default) and `"ollama"` (`nomic-embed-text` through a local
Ollama server). Any other model is a function from a list of texts to a list of vectors (or a
2-D numpy array). SpectRAG scales the vectors to length 1.

{% tabs %}
{% tab title="sentence-transformers" %}
```python
from sentence_transformers import SentenceTransformer
from spectrag import SpectRAGIngestion, SpectRAGRetrieval

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    return model.encode(texts, batch_size=64)

SpectRAGIngestion("my_index", {"embedder": embed}).ingest("docs/")
search = SpectRAGRetrieval("my_index", embed_fn=embed)
```
{% endtab %}

{% tab title="OpenAI" %}
```python
from openai import OpenAI
from spectrag import SpectRAGIngestion, SpectRAGRetrieval

client = OpenAI()   # reads OPENAI_API_KEY

def embed(texts, batch_size=256):
    vectors = []
    for start in range(0, len(texts), batch_size):
        reply = client.embeddings.create(model="text-embedding-3-small",
                                         input=texts[start:start + batch_size])
        vectors += [item.embedding for item in reply.data]
    return vectors

SpectRAGIngestion("my_index", {"embedder": embed}).ingest("docs/")
search = SpectRAGRetrieval("my_index", embed_fn=embed)
```
{% endtab %}

{% tab title="Ollama, another model" %}
```python
SpectRAGIngestion("my_index", {"embedder": "ollama", "ollama_model": "mxbai-embed-large"}).ingest("docs/")
search = SpectRAGRetrieval("my_index")   # the index remembers the Ollama model
```
{% endtab %}
{% endtabs %}

{% hint style="warning" %}
**Pass your function again when you search.** An index stores only the name of a function, not
the function, so `SpectRAGRetrieval` asks for it: `SpectRAGRetrieval("my_index", embed_fn=embed)`.
The query must be embedded with the same model as the documents; a vector of the wrong size
raises `EmbedderError`.
{% endhint %}

Your function receives the same call for documents and for questions. The built-in Ollama
embedder adds the `search_document:` and `search_query:` prefixes that `nomic-embed-text`
expects; a custom function gets no such hint.

`semantic_embedder` can use a different (faster) embedder for the sentence distances used by
semantic chunking: for example `{"embedder": embed, "semantic_embedder": "hash"}`.

## LLMs for answers

`answer()` builds a grounded prompt from the selected context and passes it to `llm_fn`. Use
temperature 0 for repeatable answers, as the paper does.

{% tabs %}
{% tab title="Ollama" %}
```python
from spectrag.utils.llm import ollama_llm

llm_fn = ollama_llm("gemma3:4b")      # temperature 0, context window sized to the prompt
result = search.answer("my question", llm_fn)
```
`ollama_llm(model, url, num_ctx=None, timeout=600, max_tokens=None, retries=2)` sizes Ollama's
context window to each prompt (Ollama's default window silently drops the start of long
prompts), retries after timeouts, and removes the `<think>…</think>` part of reasoning models.
{% endtab %}

{% tab title="OpenAI" %}
```python
from openai import OpenAI

client = OpenAI()

def llm_fn(prompt):
    reply = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role": "user", "content": prompt}])
    return reply.choices[0].message.content

result = search.answer("my question", llm_fn)
```
{% endtab %}

{% tab title="Anthropic" %}
```python
import anthropic

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY

def llm_fn(prompt):
    reply = client.messages.create(
        model="claude-sonnet-5-5", max_tokens=1024,
        messages=[{"role": "user", "content": prompt}])
    return reply.content[0].text

result = search.answer("my question", llm_fn)
```
{% endtab %}
{% endtabs %}

To send the context to an LLM yourself, use `result = search.search(question)` and either
`result["prompt"]` (the grounded prompt) or `result["context"]` (the context only).

## Entities

Entity edges link chunks that mention the same names. The default extractor is rule-based:
capitalised phrases and their words, acronyms and codes, and years. For better entities, pass
any function that returns a list of strings:

```python
import spacy
from spectrag import SpectRAGIngestion

nlp = spacy.load("en_core_web_sm")

def entity_fn(text):
    return [entity.text.lower() for entity in nlp(text).ents]

SpectRAGIngestion("my_index", entity_fn=entity_fn).ingest("docs/")
```

Entities are compared as strings, so normalise them the same way for every chunk (here:
lowercase).

## Utility questions

Utility edges link chunks that answer similar questions, and the questions also take part in
seed retrieval. The paper generates them with an LLM. Without a `question_fn` there are no
utility questions, and the `utility` edge weight is shared by the other signals.

```python
from spectrag import SpectRAGIngestion
from spectrag.utils.llm import ollama_llm
from spectrag.utils.questions import make_question_fn

question_fn = make_question_fn(ollama_llm("gemma3:4b"), count=3)
SpectRAGIngestion("my_index", {"embedder": "ollama"}, question_fn=question_fn).ingest("docs/")
```

This calls the LLM once per chunk at ingestion time, so it is slow for large corpora.

## Cross-encoder reranking

The `anchor_expand_ce` selector keeps the best chunks as anchors, then adds the anchors' graph
neighbours in the order a cross-encoder ranks them:

```python
from sentence_transformers import CrossEncoder
from spectrag import SpectRAGRetrieval

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_fn(question, texts):
    return cross_encoder.predict([(question, text) for text in texts])

search = SpectRAGRetrieval("my_index", {"selector": "anchor_expand_ce"}, rerank_fn=rerank_fn)
```

Without a `rerank_fn`, `anchor_expand_ce` ranks the neighbours by their SpectRAG score.
