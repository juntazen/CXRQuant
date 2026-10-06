"""Atomic run files and content identities, independent of CUDA."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def tree_hash(path):
    files = {
        str(p.relative_to(path)): sha256_file(p)
        for p in sorted(Path(path).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    return {"sha256": object_hash(files), "files": files}


def code_hash(root):
    root = Path(root)
    files = {}
    for folder in ("src", "scripts", "configs"):
        for name, value in tree_hash(root / folder)["files"].items():
            files[folder + "/" + name] = value
    files["pyproject.toml"] = sha256_file(root / "pyproject.toml")
    return object_hash(files)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def require_identity(actual, expected):
    if actual != expected:
        raise ValueError("Run identity changed; create a new run directory")


def checked_result(path, identity, checkpoint_hash=None):
    value = json.loads(Path(path).read_text())
    require_identity(value.get("identity"), identity)
    if value.get("status") != "complete":
        raise ValueError("Result is not complete")
    if checkpoint_hash is not None and value.get("checkpoint_hash") != checkpoint_hash:
        raise ValueError("Checkpoint hash mismatch")
    return value
