"""SpectRAG's exceptions. Each one also inherits the matching built-in exception, so code that
catches ValueError, KeyError, FileNotFoundError or ConnectionError keeps working.

    try:
        search = SpectRAGRetrieval("my_index")
    except spectrag.SpectRAGError as error:
        print(error)
"""


class SpectRAGError(Exception):
    """Base class for every error raised on purpose by SpectRAG."""


class SettingsError(SpectRAGError, ValueError):
    """A setting has the wrong type, is out of range, or is not one of the allowed choices."""


class UnknownSettingError(SettingsError, KeyError):
    """A setting name that SpectRAG does not know (often a typo)."""

    def __str__(self):  # KeyError would otherwise print the message with quotes around it
        return str(self.args[0]) if self.args else ""


class IndexError_(SpectRAGError, FileNotFoundError):
    """The index folder is missing, incomplete, or was made by an incompatible version.
    (The trailing underscore avoids hiding Python's built-in IndexError.)"""


class EmbedderError(SpectRAGError, ConnectionError):
    """The embedder cannot be reached, or its vectors do not match the index."""


class DocumentError(SpectRAGError, ValueError):
    """Documents given to ingest_documents() are not in the expected form."""


class NoDocumentsError(DocumentError):
    """There was no document with text to ingest."""


class DocumentNotFoundError(DocumentError, FileNotFoundError):
    """The file or folder given to ingest() does not exist."""


class MissingPackageError(SpectRAGError, ImportError):
    """An optional package is needed (e.g. faiss-cpu or pypdf); the message says what to install."""


class LLMError(SpectRAGError, ConnectionError):
    """The LLM (e.g. a local Ollama model) cannot be reached or did not reply in time."""
