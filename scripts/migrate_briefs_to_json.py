"""One-time migration: consolidate weekly_briefs/*.txt → weekly_briefs.json per subreddit.

Usage:
    python scripts/migrate_briefs_to_json.py [--reports-dir data/reports] [--dry-run]

After migration the original .txt files can be safely deleted (the script does NOT
delete them automatically; review the JSON output first).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def migrate(reports_dir: Path, dry_run: bool) -> None:
    sub_dirs = [d for d in reports_dir.iterdir() if d.is_dir()]
    if not sub_dirs:
        print(f"No subdirectories found in {reports_dir}")
        return

    for sub_dir in sorted(sub_dirs):
        brief_dir = sub_dir / "weekly_briefs"
        txt_files = sorted(brief_dir.glob("*.txt")) if brief_dir.exists() else []

        if not txt_files:
            continue

        json_path = sub_dir / "weekly_briefs.json"

        # Load existing JSON if present (don't overwrite newer entries)
        existing: dict = {}
        if json_path.exists():
            try:
                with open(json_path, encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}

        added = 0
        for txt in txt_files:
            week_key = txt.stem
            if week_key in existing:
                continue  # already migrated, don't overwrite
            try:
                text = txt.read_text(encoding="utf-8").strip()
            except OSError as e:
                print(f"  WARN: could not read {txt}: {e}")
                continue
            existing[week_key] = {
                "text": text,
                "source": "legacy_txt",
                "generated_at": "",
            }
            added += 1

        merged = dict(sorted(existing.items()))

        if dry_run:
            print(
                f"[DRY RUN] {sub_dir.name}: would write {len(merged)} entries "
                f"({added} new) to {json_path}"
            )
        else:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            print(f"  {sub_dir.name}: wrote {len(merged)} entries ({added} new) -> {json_path}")

    if not dry_run:
        print(
            "\nMigration complete. Review the JSON files, then optionally remove "
            "the old weekly_briefs/ directories with:\n"
            "  find data/reports -type d -name weekly_briefs -exec rm -rf {} +"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        default="data/reports",
        help="Path to the reports directory (default: data/reports)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        print(f"ERROR: reports dir not found: {reports_dir}", file=sys.stderr)
        sys.exit(1)

    migrate(reports_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
