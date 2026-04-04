from pathlib import Path

from src.collector.manifest import (
    is_file_entry_valid,
    load_manifest,
    record_file_entry,
    record_subreddit_ingestion,
    save_manifest,
)


def test_manifest_load_save_roundtrip(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = load_manifest(manifest_path)
    assert manifest["files"] == {}
    assert manifest["subreddits"] == {}

    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")
    entry = record_file_entry(manifest, file_path)
    assert entry["exists"] is True
    assert entry["size_bytes"] > 0
    assert entry["sha256"]

    record_subreddit_ingestion(manifest, "depression", 123, 1700000000, 1700600000)
    save_manifest(manifest_path, manifest)

    loaded = load_manifest(manifest_path)
    assert loaded["subreddits"]["depression"]["rows"] == 123
    assert is_file_entry_valid(loaded, file_path) is True


def test_manifest_invalid_after_file_change(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")
    manifest = load_manifest(manifest_path)
    record_file_entry(manifest, file_path)
    save_manifest(manifest_path, manifest)

    file_path.write_text("changed", encoding="utf-8")
    loaded = load_manifest(manifest_path)
    assert is_file_entry_valid(loaded, file_path) is False

