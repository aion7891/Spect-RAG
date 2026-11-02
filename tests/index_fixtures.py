"""Small enriched records with known geometry for Phase 4 behavioral tests."""

from hashlib import sha256

import numpy as np

from spect_rag.utility.chunking import count_tokens


def chunk(
    name: str, vector, *, doc: str | None = None, position: int = 0,
    section: list[str] | None = None, concepts: list[str] | None = None,
    questions=None, text: str | None = None,
) -> dict:
    doc = doc or name
    text = text or f"Evidence about {name}."
    vector = np.asarray(vector, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    question_vectors = np.asarray(questions or [], dtype=np.float32).reshape(-1, len(vector))
    if len(question_vectors):
        question_vectors /= np.linalg.norm(question_vectors, axis=1, keepdims=True)
    return {
        "chunk_id": name, "doc_id": doc, "position": position,
        "source": f"/corpus/{doc}.md", "content_hash": sha256(doc.encode()).hexdigest(),
        "text": text, "token_count": count_tokens(text),
        "embedding": vector, "embedding_model": "test-model", "embedding_dimension": len(vector),
        "embedding_prefix": "search_document: ", "extraction_model": "test-extractor",
        "metadata_extraction": "enabled", "concepts": concepts or [],
        "utility_questions": [f"Question {i} about {name}?" for i in range(len(question_vectors))],
        "utility_question_embeddings": question_vectors,
        "section_path": section or [], "citations": [],
        "pages": [], "page_start": None, "page_end": None, "block_indices": [position],
        "contains_table": False, "chunking_mode": "structural" if section else "sentence",
        "semantic_segment": None,
    }
