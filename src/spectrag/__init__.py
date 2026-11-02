"""SpectRAG: spectral graph-augmented retrieval for RAG.

SpectRAGIngestion builds the index (offline, once per document set).
SpectRAGRetrieval finds the context for a question (online, per question).
"""

from spectrag.config import DEFAULTS, make_settings
from spectrag.ingestion import SpectRAGIngestion
from spectrag.retrieval import SpectRAGRetrieval

__version__ = "0.1.0"
