"""Utilities for isolated SQLite files in tests on Windows and POSIX."""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4


_TEST_DB_ROOT = Path(__file__).resolve().parents[1] / "tmp"


def new_test_db_path(prefix: str) -> Path:
    """Return a unique database filename without creating a temporary subfolder."""
    _TEST_DB_ROOT.mkdir(parents=True, exist_ok=True)
    return _TEST_DB_ROOT / f"{prefix}-{uuid4().hex}.sqlite3"


def remove_test_db(path: Path) -> None:
    """Remove the database and SQLite sidecar files after connections are closed."""
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)


@contextmanager
def isolated_test_db(prefix: str) -> Iterator[Path]:
    path = new_test_db_path(prefix)
    try:
        yield path
    finally:
        remove_test_db(path)
