"""Shared checks for enriched records before graph building or index loading."""

import numpy as np

from .chunking import count_tokens
from .embeddings import normalize_embeddings


def validate_chunks(chunks: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate aligned records and return chunk/question matrices and owner rows."""
    if not chunks:
        raise ValueError("An index requires at least one enriched chunk.")
    required = {
        "chunk_id", "doc_id", "source", "content_hash", "position", "text", "token_count",
        "embedding", "embedding_model", "embedding_dimension", "embedding_prefix",
        "utility_questions", "utility_question_embeddings", "concepts", "section_path",
        "extraction_model", "metadata_extraction", "citations",
    }
    if any(not isinstance(chunk, dict) or not required <= chunk.keys() for chunk in chunks):
        raise ValueError("Incomplete enriched chunk record.")
    first = chunks[0]
    dimension = first["embedding_dimension"]
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("Embedding dimension must be a positive integer.")
    if not isinstance(first["embedding_model"], str) or not first["embedding_model"].strip():
        raise ValueError("An index requires an explicit embedding model identity.")
    identities = set()
    documents = {}
    vectors, questions, owners = [], [], []
    shared_fields = ("embedding_model", "embedding_dimension", "embedding_prefix",
                     "extraction_model", "metadata_extraction")
    for row, chunk in enumerate(chunks):
        if any(not isinstance(chunk[field], str) or not chunk[field] for field in (
            "chunk_id", "doc_id", "source", "content_hash",
        )):
            raise ValueError("Chunk/document/source identities must be non-empty strings.")
        if chunk["chunk_id"] in identities:
            raise ValueError("Duplicate chunk IDs are not allowed.")
        identities.add(chunk["chunk_id"])
        if any(chunk[field] != first[field] for field in shared_fields):
            raise ValueError("All chunks must use matching embedding/extraction settings.")
        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            raise ValueError("Chunks must have non-empty text.")
        if type(chunk["token_count"]) is not int or chunk["token_count"] != count_tokens(chunk["text"]):
            raise ValueError("Chunk token counts do not match their text.")
        if type(chunk["position"]) is not int or chunk["position"] < 0:
            raise ValueError("Chunk positions must be non-negative integers.")
        document = documents.setdefault(chunk["doc_id"], {
            "source": chunk["source"], "content_hash": chunk["content_hash"], "positions": [],
        })
        if any(chunk[field] != document[field] for field in ("source", "content_hash")):
            raise ValueError("A document ID must identify one source and content version.")
        document["positions"].append(chunk["position"])
        for field in ("concepts", "utility_questions", "section_path"):
            if not isinstance(chunk[field], list) or any(
                not isinstance(value, str) or not value.strip() for value in chunk[field]
            ):
                raise ValueError(f"{field} must be a list of non-empty strings.")
        if not isinstance(chunk["embedding_prefix"], str):
            raise ValueError("Embedding prefix must be a string.")
        if chunk["metadata_extraction"] not in {"enabled", "disabled"}:
            raise ValueError("Unknown metadata extraction state.")
        if chunk["metadata_extraction"] == "enabled":
            if not isinstance(chunk["extraction_model"], str) or not chunk["extraction_model"].strip():
                raise ValueError("Enabled extraction requires a model identity.")
        elif chunk["extraction_model"] is not None or chunk["concepts"] or chunk["utility_questions"]:
            raise ValueError("Disabled extraction must not contain fabricated model outputs.")
        vectors.append(chunk["embedding"])
        question_count = len(chunk["utility_questions"])
        question_vectors = np.asarray(chunk["utility_question_embeddings"])
        normalized = normalize_embeddings(question_vectors, rows=question_count, dimension=dimension)
        if not np.allclose(question_vectors, normalized, atol=1e-6, rtol=1e-6):
            raise ValueError("Question vectors must already be L2-normalized.")
        questions.extend(normalized)
        owners.extend([row] * question_count)
    for document in documents.values():
        if sorted(document["positions"]) != list(range(len(document["positions"]))):
            raise ValueError("Document chunk positions must be unique and contiguous from zero.")
    records = {chunk["chunk_id"]: chunk for chunk in chunks}
    for chunk in chunks:
        if not isinstance(chunk["citations"], list):
            raise ValueError("Citations must be a list of reference records.")
        for citation in chunk["citations"]:
            if not isinstance(citation, dict) or citation.get("status") not in {"resolved", "unresolved"}:
                raise ValueError("Invalid citation record.")
            if citation["status"] == "resolved":
                targets = citation.get("target_chunk_ids")
                if not isinstance(targets, list) or not targets or any(
                    target not in records or records[target]["doc_id"] != citation.get("target_doc_id")
                    for target in targets
                ):
                    raise ValueError("Resolved citation targets must match corpus chunk/document IDs.")
    normalized = normalize_embeddings(vectors, rows=len(chunks), dimension=dimension)
    if not np.allclose(vectors, normalized, atol=1e-6, rtol=1e-6):
        raise ValueError("Chunk vectors must already be L2-normalized.")
    return (
        normalized,
        np.asarray(questions, dtype=np.float32).reshape(-1, dimension),
        np.asarray(owners, dtype=np.int64),
    )
