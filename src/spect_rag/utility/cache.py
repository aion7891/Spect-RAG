"""Small, versioned JSON cache for model outputs; no pickle or model calls."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def cache_key(kind: str, settings: dict, text: str) -> str:
    payload = {"version": 1, "kind": kind, "settings": settings, "text": text}
    return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_cache(directory: str | Path | None, kind: str, key: str):
    """Return a cached value or None for a miss; reject corrupt/incompatible data."""
    if directory is None:
        return None
    path = Path(directory) / kind / f"{key}.json"
    if not path.exists():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(entry, dict) or entry.get("version") != 1
            or entry.get("key") != key or "data" not in entry
        ):
            raise ValueError("cache schema/key mismatch")
        return entry["data"]
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid cache file {path}; remove it to recompute.") from exc


def write_cache(directory: str | Path | None, kind: str, key: str, data) -> None:
    """Atomically publish a validated model output when disk caching is enabled."""
    if directory is None:
        return
    path = Path(directory) / kind / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"version": 1, "key": key, "data": data}, ensure_ascii=False, allow_nan=False)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
