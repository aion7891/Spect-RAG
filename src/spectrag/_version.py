"""The package version (read by pyproject.toml) and the index format version."""

__version__ = "0.1.2"

# Bump INDEX_FORMAT when the files of an index change in a way older code cannot read.
# 1 = JSON edge lists (before the Phase 1 audit), 2 = binary .npy/.npz edges and signals.
INDEX_FORMAT = 2
