"""Versioned local snapshots: JSON records, NumPy arrays, graph, FAISS, and BM25.

CURRENT is replaced only after a complete snapshot has been written. Old
snapshots remain intact. Loading verifies hashes and alignment before returning
any usable index; neither saving nor loading uses pickle.
"""

from datetime import datetime, timezone
from hashlib import file_digest
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import uuid4

import faiss
import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi

from .graph import DEFAULT_EDGE_WEIGHTS, graph_settings
from .spectral import normalized_laplacian
from .validation import validate_chunks


ARTIFACTS = (
    "chunks.json", "embeddings.npy", "question_embeddings.npy", "question_owners.npy",
    "spectral_coords.npy", "eigenvalues.npy", "graph.json", "lexical_tokens.json",
    "dense.faiss", "questions.faiss",
)


def tokenize(text: str) -> list[str]:
    """One shared Unicode word tokenizer for persisted BM25 data and later queries."""
    return re.findall(r"\w+", text.casefold())


def build_bm25(tokens: list[list[str]]):
    """No lexical signal exists when every document has an empty vocabulary."""
    return BM25Okapi(tokens) if any(tokens) else None


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _documents(chunks: list[dict]) -> list[dict]:
    result = {}
    for chunk in chunks:
        result.setdefault(chunk["doc_id"], {
            key: chunk[key] for key in ("doc_id", "source", "content_hash")
        })
    return list(result.values())


def _validate_graph(graph: nx.Graph, chunks: list[dict]) -> None:
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Index graph must be simple, undirected, and have no self-loops.")
    if list(graph) != [chunk["chunk_id"] for chunk in chunks]:
        raise ValueError("Graph node order must match the chunk/array order.")
    settings = graph_settings(**graph.graph["settings"])
    for _, _, data in graph.edges(data=True):
        if set(data) != {*DEFAULT_EDGE_WEIGHTS, "weight"}:
            raise ValueError("Every graph edge must retain all six signal scores.")
        if any(not np.isfinite(data[name]) or not 0 <= data[name] <= 1 for name in DEFAULT_EDGE_WEIGHTS):
            raise ValueError("Graph signal scores must lie in [0, 1].")
        expected = sum(settings["edge_weights"][name] * data[name] for name in DEFAULT_EDGE_WEIGHTS)
        if (
            not np.isfinite(data["weight"]) or data["weight"] <= 0
            or data["weight"] < settings["edge_threshold"]
            or not np.isclose(data["weight"], expected, atol=1e-10, rtol=1e-10)
        ):
            raise ValueError("Graph weights do not match fusion settings.")


def _validate_spectrum(graph: nx.Graph, coordinates: np.ndarray, values: np.ndarray, settings: dict) -> None:
    if coordinates.ndim != 2 or coordinates.shape != (len(graph), len(values)) or values.ndim != 1:
        raise ValueError("Spectral arrays do not align with graph nodes.")
    if not np.isfinite(coordinates).all() or not np.isfinite(values).all():
        raise ValueError("Spectral arrays must be finite.")
    if len(values) > settings["requested_dimension"]:
        raise ValueError("Saved spectrum exceeds its requested dimension.")
    if settings["zero_modes"] != nx.number_connected_components(graph):
        raise ValueError("Spectral zero-mode count does not match graph components.")
    if np.any(values <= settings["zero_tolerance"]) or np.any(values > 2 + 1e-7) or np.any(np.diff(values) < 0):
        raise ValueError("Eigenvalues must be positive, sorted normalized-Laplacian modes.")
    if not np.allclose(coordinates.T @ coordinates, np.eye(len(values)), atol=1e-6):
        raise ValueError("Spectral coordinates must have orthonormal columns.")
    laplacian = normalized_laplacian(graph)
    residuals = np.linalg.norm(laplacian @ coordinates - coordinates * values, axis=0)
    if np.any(residuals > 1e-6):
        raise ValueError("Spectral coordinates fail the graph eigenpair residual check.")


def save_index(
    index_dir: str | Path, chunks: list[dict], graph: nx.Graph, spectrum: dict, *,
    chunking_settings: dict,
) -> dict:
    """Write and publish a full immutable snapshot; return its manifest."""
    embeddings, questions, owners = validate_chunks(chunks)
    _validate_graph(graph, chunks)
    spectral_settings = {
        key: spectrum[key] for key in ("requested_dimension", "zero_tolerance", "zero_modes")
    }
    _validate_spectrum(graph, spectrum["coordinates"], spectrum["eigenvalues"], spectral_settings)
    dimension = embeddings.shape[1]
    dense_index, question_index = faiss.IndexFlatIP(dimension), faiss.IndexFlatIP(dimension)
    dense_index.add(embeddings)
    question_index.add(questions)
    tokens = [tokenize(chunk["text"]) for chunk in chunks]
    first = chunks[0]
    generation = uuid4().hex
    manifest = {
        "version": 1, "generation": generation,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
        "documents": _documents(chunks),
        "embedding": {
            "model": first["embedding_model"], "dimension": dimension,
            "document_prefix": first["embedding_prefix"],
        },
        "extraction": {
            "model": first["extraction_model"], "status": first["metadata_extraction"],
        },
        "chunking": {**chunking_settings, "tokenizer": "cl100k_base"},
        "graph": graph.graph["settings"], "spectral": spectral_settings,
        "counts": {
            "chunks": len(chunks), "questions": len(questions), "edges": graph.number_of_edges(),
            "spectral_dimensions": spectrum["coordinates"].shape[1],
        },
    }
    root = Path(index_dir)
    snapshots = root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".building-", dir=snapshots) as temporary:
        snapshot = Path(temporary)
        records = [
            {key: value for key, value in chunk.items()
             if key not in {"embedding", "utility_question_embeddings", "spectral_coords"}}
            for chunk in chunks
        ]
        _write_json(snapshot / "chunks.json", records)
        arrays = {
            "embeddings.npy": embeddings, "question_embeddings.npy": questions,
            "question_owners.npy": owners, "spectral_coords.npy": spectrum["coordinates"],
            "eigenvalues.npy": spectrum["eigenvalues"],
        }
        for name, array in arrays.items():
            np.save(snapshot / name, array, allow_pickle=False)
        _write_json(snapshot / "graph.json", nx.node_link_data(graph, edges="edges"))
        _write_json(snapshot / "lexical_tokens.json", tokens)
        faiss.write_index(dense_index, str(snapshot / "dense.faiss"))
        faiss.write_index(question_index, str(snapshot / "questions.faiss"))
        manifest["artifacts"] = {name: _digest(snapshot / name) for name in ARTIFACTS}
        _write_json(snapshot / "manifest.json", manifest)
        snapshot.rename(snapshots / generation)
    pointer = None
    try:
        with NamedTemporaryFile(mode="w", encoding="ascii", prefix=".CURRENT-", dir=root, delete=False) as stream:
            pointer = Path(stream.name)
            stream.write(generation + "\n")
        pointer.replace(root / "CURRENT")
    finally:
        if pointer is not None:
            pointer.unlink(missing_ok=True)
    return manifest


def _check_faiss(index, vectors: np.ndarray) -> None:
    if (
        not isinstance(index, faiss.IndexFlatIP) or index.d != vectors.shape[1]
        or index.ntotal != len(vectors) or index.metric_type != faiss.METRIC_INNER_PRODUCT
    ):
        raise ValueError("FAISS index type/dimension/count does not match its vectors.")
    if len(vectors) and not np.array_equal(index.reconstruct_n(0, len(vectors)), vectors):
        raise ValueError("FAISS vector order/content does not match its NumPy artifact.")


def load_index(
    index_dir: str | Path, *, embedding_model: str | None = None,
    embedding_dimension: int | None = None, document_prefix: str | None = None,
) -> dict:
    """Verify a snapshot and reconstruct graph, exact indexes, records, and BM25."""
    root = Path(index_dir)
    pointer = root / "CURRENT"
    if not pointer.is_file():
        raise ValueError(f"No published index at {root}; run ingestion first.")
    generation = pointer.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{32}", generation):
        raise ValueError("Invalid index snapshot pointer.")
    snapshot = root / "snapshots" / generation
    manifest = _read_json(snapshot / "manifest.json")
    if manifest.get("version") != 1 or manifest.get("generation") != generation:
        raise ValueError("Unsupported/mismatched index manifest version or generation.")
    for key, expected in (
        ("model", embedding_model), ("dimension", embedding_dimension), ("document_prefix", document_prefix),
    ):
        if expected is not None and manifest["embedding"][key] != expected:
            raise ValueError(f"Embedding {key} does not match the saved index.")
    if set(manifest["artifacts"]) != set(ARTIFACTS):
        raise ValueError("Index artifact list is incomplete or incompatible.")
    for name in ARTIFACTS:
        path = snapshot / name
        if not path.is_file() or _digest(path) != manifest["artifacts"][name]:
            raise ValueError(f"Missing or mismatched index artifact: {name}")
    embeddings = np.load(snapshot / "embeddings.npy", allow_pickle=False)
    questions = np.load(snapshot / "question_embeddings.npy", allow_pickle=False)
    owners = np.load(snapshot / "question_owners.npy", allow_pickle=False)
    coordinates = np.load(snapshot / "spectral_coords.npy", allow_pickle=False)
    values = np.load(snapshot / "eigenvalues.npy", allow_pickle=False)
    records = _read_json(snapshot / "chunks.json")
    if not isinstance(records, list) or not records or any(
        not isinstance(record, dict) or not isinstance(record.get("utility_questions"), list)
        for record in records
    ):
        raise ValueError("Invalid saved chunk records.")
    count, dimension = len(records), manifest["embedding"]["dimension"]
    if embeddings.shape != (count, dimension) or questions.ndim != 2 or questions.shape[1] != dimension:
        raise ValueError("Embedding artifacts have incompatible shapes.")
    if coordinates.ndim != 2 or coordinates.shape[0] != count:
        raise ValueError("Spectral artifact rows do not match the chunks.")
    chunks, offset = [], 0
    for row, record in enumerate(records):
        size = len(record["utility_questions"])
        chunks.append({
            **record, "embedding": embeddings[row],
            "utility_question_embeddings": questions[offset:offset + size],
            "spectral_coords": coordinates[row],
        })
        offset += size
    _, expected_questions, expected_owners = validate_chunks(chunks)
    if (
        offset != len(questions) or owners.dtype.kind not in "iu"
        or not np.array_equal(owners, expected_owners)
        or not np.allclose(questions, expected_questions, atol=1e-6)
    ):
        raise ValueError("Question vector rows/owners do not match the chunk records.")
    first = chunks[0]
    if manifest["embedding"] != {
        "model": first["embedding_model"], "dimension": first["embedding_dimension"],
        "document_prefix": first["embedding_prefix"],
    }:
        raise ValueError("Manifest embedding settings do not match the chunk records.")
    if manifest["extraction"] != {"model": first["extraction_model"], "status": first["metadata_extraction"]}:
        raise ValueError("Manifest extraction settings do not match the chunk records.")
    if manifest["chunk_ids"] != [chunk["chunk_id"] for chunk in chunks] or manifest["documents"] != _documents(chunks):
        raise ValueError("Manifest chunk order/document identities do not match the records.")
    if manifest["chunking"]["tokenizer"] != "cl100k_base" or any(
        chunk["token_count"] > manifest["chunking"]["chunk_size"] for chunk in chunks
    ):
        raise ValueError("Chunk text does not match the manifest token-budget settings.")
    graph = nx.node_link_graph(_read_json(snapshot / "graph.json"), edges="edges")
    _validate_graph(graph, chunks)
    if manifest["graph"] != graph.graph["settings"]:
        raise ValueError("Manifest graph settings do not match the graph.")
    _validate_spectrum(graph, coordinates, values, manifest["spectral"])
    counts = {
        "chunks": count, "questions": len(questions), "edges": graph.number_of_edges(),
        "spectral_dimensions": coordinates.shape[1],
    }
    if manifest["counts"] != counts:
        raise ValueError("Manifest counts do not match the index artifacts.")
    tokens = _read_json(snapshot / "lexical_tokens.json")
    if tokens != [tokenize(chunk["text"]) for chunk in chunks]:
        raise ValueError("BM25 tokens do not match the saved text.")
    dense_index = faiss.read_index(str(snapshot / "dense.faiss"))
    question_index = faiss.read_index(str(snapshot / "questions.faiss"))
    _check_faiss(dense_index, embeddings)
    _check_faiss(question_index, questions)
    return {
        "chunks": chunks, "graph": graph, "embeddings": embeddings,
        "question_embeddings": questions, "question_owners": owners,
        "spectral_coords": coordinates, "eigenvalues": values,
        "dense_index": dense_index, "question_index": question_index,
        "lexical_tokens": tokens, "bm25": build_bm25(tokens),
        "manifest": manifest, "snapshot_dir": snapshot,
    }
