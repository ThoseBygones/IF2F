"""Small shared utilities for public IF2F entry points."""

from pathlib import Path


def require_files(**named_paths):
    missing = []
    resolved = {}
    for name, value in named_paths.items():
        path = Path(value).expanduser().resolve()
        resolved[name] = path
        if not path.is_file():
            missing.append(f"{name}: {path}")
    if missing:
        details = "\n  - ".join(missing)
        raise FileNotFoundError(f"Required model files were not found:\n  - {details}")
    return resolved


def is_within(path, directory):
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
        return True
    except ValueError:
        return False
