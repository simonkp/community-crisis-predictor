import hashlib
import json
from pathlib import Path
from typing import Any


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {"files": {}, "subreddits": {}}
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    if "files" not in data:
        data["files"] = {}
    if "subreddits" not in data:
        data["subreddits"] = {}
    return data


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def compute_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def record_file_entry(manifest: dict[str, Any], file_path: str | Path) -> dict[str, Any]:
    p = Path(file_path)
    entry = {
        "path": str(p),
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else 0,
        "sha256": compute_sha256(p) if p.exists() else "",
    }
    manifest.setdefault("files", {})[str(p)] = entry
    return entry


def is_file_entry_valid(manifest: dict[str, Any], file_path: str | Path) -> bool:
    p = Path(file_path)
    entry = manifest.get("files", {}).get(str(p))
    if not entry or not p.exists():
        return False
    if int(entry.get("size_bytes", -1)) != p.stat().st_size:
        return False
    recorded_sha = entry.get("sha256", "")
    if not recorded_sha:
        return False
    return recorded_sha == compute_sha256(p)


def record_subreddit_ingestion(
    manifest: dict[str, Any],
    subreddit: str,
    rows: int,
    min_created_utc: int | None,
    max_created_utc: int | None,
    status: str = "ingested",
) -> None:
    manifest.setdefault("subreddits", {})[subreddit] = {
        "rows": int(rows),
        "min_created_utc": int(min_created_utc) if min_created_utc is not None else None,
        "max_created_utc": int(max_created_utc) if max_created_utc is not None else None,
        "status": status,
    }

