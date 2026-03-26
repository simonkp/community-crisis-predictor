import json
import re
from pathlib import Path

import pandas as pd


_FILENAME_RE = re.compile(
    r"^arctic_shift_(?P<subreddit>[A-Za-z0-9_]+)_(?P<start>\d{6})_(?P<end>\d{6})\.jsonl$"
)
_SKIP_SELFTEXT = {"[deleted]", "[removed]", ""}


def parse_arctic_shift_filename(path: str | Path) -> str | None:
    m = _FILENAME_RE.match(Path(path).name)
    if not m:
        return None
    return m.group("subreddit").lower()


class ArcticShiftLoader:
    """Stream line-by-line Arctic Shift JSONL into canonical raw schema."""

    def __init__(self, min_selftext_chars: int = 10):
        self.min_selftext_chars = int(min_selftext_chars)

    def load_jsonl(self, path: str | Path, subreddit: str | None = None) -> tuple[pd.DataFrame, dict]:
        target_path = Path(path)
        rows: list[dict] = []
        stats = {
            "file": target_path.name,
            "parsed": 0,
            "kept": 0,
            "skipped_json": 0,
            "skipped_non_self": 0,
            "skipped_body": 0,
            "skipped_subreddit": 0,
        }
        with open(target_path, encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    rec = json.loads(payload)
                except json.JSONDecodeError:
                    stats["skipped_json"] += 1
                    print(f"[arctic_shift] warn: JSON parse failed in {target_path.name}:{i}")
                    continue
                stats["parsed"] += 1
                if not bool(rec.get("is_self", False)):
                    stats["skipped_non_self"] += 1
                    continue

                selftext_raw = rec.get("selftext")
                selftext = "" if selftext_raw is None else str(selftext_raw).strip()
                if selftext.lower() in _SKIP_SELFTEXT or len(selftext) < self.min_selftext_chars:
                    stats["skipped_body"] += 1
                    continue

                # Reddit JSON uses inconsistent casing; pipeline + Zenodo use lowercase slugs.
                sub = str(rec.get("subreddit", "")).strip().lower()
                if subreddit and sub != subreddit.lower():
                    stats["skipped_subreddit"] += 1
                    continue

                author = rec.get("author")
                author_s = "" if author is None else str(author).strip()
                if author_s in {"[deleted]", "[removed]", "None"}:
                    author_s = ""

                created = pd.to_numeric(rec.get("created_utc"), errors="coerce")
                if pd.isna(created):
                    stats["skipped_json"] += 1
                    continue

                rows.append(
                    {
                        "post_id": str(rec.get("id", "")).strip(),
                        "created_utc": int(created),
                        "selftext": selftext,
                        "subreddit": sub,
                        "author": author_s,
                        "data_source": "arctic_shift",
                    }
                )
                stats["kept"] += 1

        out = pd.DataFrame(rows, columns=["post_id", "created_utc", "selftext", "subreddit", "author", "data_source"])
        return out, stats
