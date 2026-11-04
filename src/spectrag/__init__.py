"""An unofficial implementation of SpectRAG (Sahinoglu, Mohammed, Kazangirler and Kazangirler,
arXiv:2610.36991): spectral graph-augmented retrieval for RAG. Not affiliated with the paper's
authors; their official implementation is https://github.com/SightAndCite/SpectRAG.

SpectRAGIngestion builds the index (offline, once per document set).
SpectRAGRetrieval finds the context for a question and can answer it (online, per question).

    from spectrag import SpectRAGIngestion, SpectRAGRetrieval
    SpectRAGIngestion("my_index").ingest("docs/")
    print(SpectRAGRetrieval("my_index").search("my question")["context"])
"""

from spectrag._version import INDEX_FORMAT, __version__
from spectrag.config import DEFAULTS, make_settings
from spectrag.errors import (DocumentError, DocumentNotFoundError, EmbedderError, IndexError_,
                             LLMError, MissingPackageError, NoDocumentsError, SettingsError,
                             SpectRAGError, UnknownSettingError)
from spectrag.ingestion import SpectRAGIngestion
from spectrag.log import enable_logging
from spectrag.retrieval import SpectRAGRetrieval

__all__ = ["SpectRAGIngestion", "SpectRAGRetrieval", "make_settings", "DEFAULTS",
           "enable_logging", "SpectRAGError", "SettingsError", "UnknownSettingError",
           "IndexError_", "EmbedderError", "DocumentError", "NoDocumentsError",
           "DocumentNotFoundError", "MissingPackageError", "LLMError",
           "__version__", "INDEX_FORMAT"]
