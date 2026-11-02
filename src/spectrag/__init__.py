"""SpectRAG: spectral graph-augmented retrieval for RAG.

SpectRAGIngestion builds the index (offline). SpectRAGRetrieval is added in a later phase.
"""

from spectrag.config import DEFAULTS, make_settings
from spectrag.ingestion import SpectRAGIngestion

__version__ = "0.1.0"
