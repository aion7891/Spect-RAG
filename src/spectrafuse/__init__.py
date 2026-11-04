"""SpectraFuse: an independent implementation inspired by SpectRAG (Sahinoglu, Mohammed,
Kazangirler and Kazangirler, arXiv:2610.36991), spectral graph-augmented retrieval for RAG.
Not the official SpectRAG and not affiliated with its authors; the official implementation is
https://github.com/SightAndCite/SpectRAG. The two classes keep the paper's method name.

SpectRAGIngestion builds the index (offline, once per document set).
SpectRAGRetrieval finds the context for a question and can answer it (online, per question).

    from spectrafuse import SpectRAGIngestion, SpectRAGRetrieval
    SpectRAGIngestion("my_index").ingest("docs/")
    print(SpectRAGRetrieval("my_index").search("my question")["context"])
"""

from spectrafuse._version import INDEX_FORMAT, __version__
from spectrafuse.config import DEFAULTS, make_settings
from spectrafuse.errors import (DocumentError, DocumentNotFoundError, EmbedderError, IndexError_,
                             LLMError, MissingPackageError, NoDocumentsError, SettingsError,
                             SpectRAGError, UnknownSettingError)
from spectrafuse.ingestion import SpectRAGIngestion
from spectrafuse.log import enable_logging
from spectrafuse.retrieval import SpectRAGRetrieval

__all__ = ["SpectRAGIngestion", "SpectRAGRetrieval", "make_settings", "DEFAULTS",
           "enable_logging", "SpectRAGError", "SettingsError", "UnknownSettingError",
           "IndexError_", "EmbedderError", "DocumentError", "NoDocumentsError",
           "DocumentNotFoundError", "MissingPackageError", "LLMError",
           "__version__", "INDEX_FORMAT"]
