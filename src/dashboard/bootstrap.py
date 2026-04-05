"""Shared Streamlit dashboard bootstrap: repo root on sys.path + secrets/env config."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_repo_root_on_path(start_file: str | Path | None = None) -> Path:
    """Walk upward from start_file (or this file) until pyproject.toml is found; insert repo root on sys.path."""
    if start_file is None:
        p = Path(__file__).resolve().parent
    else:
        p = Path(start_file).resolve()
        if p.is_file():
            p = p.parent
    root: Path | None = None
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists():
            root = cand
            break
    if root is None:
        root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def cfg_value_from_secrets_or_env(key: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then process environment."""
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)
