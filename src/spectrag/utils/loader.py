"""Load documents from a file or a folder."""

from pathlib import Path

SUPPORTED = (".txt", ".md", ".markdown", ".pdf")


def read_pdf(path):
    """Read the text of a PDF file (needs the optional 'pypdf' package)."""
    try:
        import pypdf
    except ImportError:
        raise ImportError("Reading PDFs needs pypdf: pip install pypdf")
    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def read_file(path):
    """Read one supported file and return its text."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    return path.read_text(encoding="utf-8", errors="replace")


def list_files(path):
    """Return a sorted list of supported files under 'path'."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("No such file or folder: " + str(path))
    if path.is_file():
        return [path]
    files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED]
    return sorted(files)


def make_doc_id(file_path, root):
    """Build a readable document id, e.g. 'reports/q1.md'."""
    file_path = Path(file_path)
    root = Path(root)
    if root.is_file():
        return file_path.name
    return file_path.relative_to(root).as_posix()


def load_docs(path):
    """Load every supported file into a list of {'doc_id', 'path', 'text'} dicts."""
    docs = []
    for file_path in list_files(path):
        text = read_file(file_path).strip()
        if not text:
            continue  # skip empty files
        docs.append({
            "doc_id": make_doc_id(file_path, path),
            "path": str(file_path),
            "text": text,
        })
    return docs
