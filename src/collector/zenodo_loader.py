import io
import re
import time
from pathlib import Path

import pandas as pd
import requests


class ZenodoLoader:
    def __init__(
        self,
        dataset_url: str,
        archive_dir: str,
        staging_dir: str,
        record_id: int = 3941387,
        timeframes: list[str] | None = None,
    ):
        self.dataset_url = dataset_url
        self.archive_dir = Path(archive_dir)
        self.staging_dir = Path(staging_dir)
        self.record_id = int(record_id)
        self.timeframes = [t.lower() for t in (timeframes or ["2018", "2019", "pre", "post"])]
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_record_metadata(self, timeout_seconds: int = 60) -> dict:
        api_url = f"https://zenodo.org/api/records/{self.record_id}"
        print(f"  [Zenodo] Fetching record metadata: {api_url}")
        try:
            resp = requests.get(api_url, timeout=timeout_seconds)
            resp.raise_for_status()
        except requests.ConnectionError:
            raise RuntimeError(
                f"Cannot reach Zenodo ({api_url}). Check your internet connection.\n"
                "Tip: run with --synthetic for an offline demo."
            ) from None
        except requests.Timeout:
            raise RuntimeError(
                f"Zenodo request timed out after {timeout_seconds}s. "
                "Try again later or run with --synthetic."
            ) from None
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Zenodo returned HTTP {exc.response.status_code} for record {self.record_id}. "
                "The dataset may be temporarily unavailable. "
                "Tip: run with --synthetic for an offline demo."
            ) from exc
        payload = resp.json()
        if not payload:
            raise RuntimeError(
                f"Zenodo returned an empty response for record {self.record_id}."
            )
        print(f"  [Zenodo] Record metadata fetched ({len(payload.get('files', []))} files listed)")
        return payload

    def ensure_subreddit_files(self, subreddit: str, timeout_seconds: int = 120) -> list[Path]:
        meta = self._fetch_record_metadata(timeout_seconds=timeout_seconds)
        files = meta.get("files", [])
        sub = subreddit.lower()
        downloaded: list[Path] = []
        matched = 0

        for f in files:
            key = f.get("key", "")
            key_lower = key.lower()
            if not key_lower.endswith(".csv"):
                continue
            if not key_lower.startswith(f"{sub}_"):
                continue
            if not any(f"_{tf}_" in key_lower for tf in self.timeframes):
                continue
            if "_features_" not in key_lower:
                continue
            matched += 1

            links = f.get("links", {}) if isinstance(f, dict) else {}
            url = links.get("content") or links.get("self")
            if not url:
                continue

            target = self.staging_dir / key
            if not target.exists():
                print(f"  [Zenodo] Downloading {key} ...")
                t0 = time.perf_counter()
                try:
                    r = requests.get(url, timeout=timeout_seconds)
                    r.raise_for_status()
                except (requests.ConnectionError, requests.Timeout) as exc:
                    print(f"  [Zenodo] WARNING: failed to download {key}: {exc}")
                    print("  Skipping this file; other files will continue.")
                    continue
                except requests.HTTPError as exc:
                    print(f"  [Zenodo] WARNING: HTTP {exc.response.status_code} for {key}")
                    continue
                if len(r.content) == 0:
                    print(f"  [Zenodo] WARNING: empty response for {key}, skipping.")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(r.content)
                elapsed = time.perf_counter() - t0
                print(
                    f"  [Zenodo] Downloaded {key} ({target.stat().st_size / (1024 * 1024):.1f} MB) "
                    f"in {elapsed:.2f}s"
                )
            else:
                print(f"  [Zenodo] Using cached file {key}")
            downloaded.append(target)
        print(f"  [Zenodo] Matched {matched} files for r/{subreddit}; ready files: {len(downloaded)}")
        return downloaded

    def discover_data_files(self) -> list[Path]:
        out: list[Path] = []
        for ext in ("*.csv", "*.parquet", "*.json", "*.jsonl"):
            out.extend(self.staging_dir.rglob(ext))
        return sorted(set(out))

    def load_subreddit_posts(
        self,
        subreddit: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        frames = []
        total_files = 0
        for p in self.discover_data_files():
            if not p.name.lower().startswith(f"{subreddit.lower()}_"):
                continue
            total_files += 1
            t0 = time.perf_counter()
            print(f"  [Zenodo] Loading {p.name} ...")
            df = self._try_load_file(p)
            if df is None or df.empty:
                print(f"  [Zenodo] Skipped {p.name}: unreadable or empty")
                continue
            sdf = self._normalize_schema(df, subreddit, source_name=p.name)
            if sdf.empty:
                print(f"  [Zenodo] Skipped {p.name}: no rows after subreddit/schema filtering")
                continue
            elapsed = time.perf_counter() - t0
            print(f"  [Zenodo] Parsed {p.name}: {len(sdf)} rows in {elapsed:.2f}s")
            frames.append(sdf)

        if not frames:
            print(f"  [Zenodo] No usable rows found for r/{subreddit} across {total_files} staged files")
            return pd.DataFrame(
                columns=["post_id", "created_utc", "selftext", "subreddit", "author", "data_source"]
            )

        out = pd.concat(frames, ignore_index=True)
        out = out.dropna(subset=["created_utc", "selftext"])
        out["created_utc"] = pd.to_numeric(out["created_utc"], errors="coerce")
        out = out.dropna(subset=["created_utc"])
        out["created_utc"] = out["created_utc"].astype(int)

        # Optional date filtering.
        if start_date or end_date:
            created_dt = pd.to_datetime(out["created_utc"], unit="s", errors="coerce")
            mask = pd.Series([True] * len(out))
            if start_date:
                mask &= created_dt >= pd.to_datetime(start_date)
            if end_date:
                mask &= created_dt <= pd.to_datetime(end_date)
            out = out[mask]

        out = out.drop_duplicates(subset=["post_id"])
        out["created_utc_dt"] = pd.to_datetime(out["created_utc"], unit="s", errors="coerce")
        print(f"  [Zenodo] Final normalized rows for r/{subreddit}: {len(out)}")
        return out.reset_index(drop=True)

    def _try_load_file(self, path: Path) -> pd.DataFrame | None:
        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                return pd.read_csv(path, low_memory=False)
            if suffix == ".parquet":
                return pd.read_parquet(path)
            if suffix in {".json", ".jsonl"}:
                if suffix == ".jsonl":
                    return pd.read_json(path, lines=True)
                # Accept either list-like JSON or object per line fallback.
                txt = path.read_text(encoding="utf-8")
                if txt.strip().startswith("["):
                    return pd.read_json(io.StringIO(txt))
                return pd.read_json(path, lines=True)
        except Exception:
            return None
        return None

    def _normalize_schema(self, df: pd.DataFrame, subreddit: str, source_name: str = "") -> pd.DataFrame:
        cols = {c.lower(): c for c in df.columns}
        subreddit_col = self._pick_col(cols, ["subreddit"])
        post_col = self._pick_col(cols, ["post", "selftext", "body", "text"])
        date_col = self._pick_col(cols, ["date", "created_utc", "created", "timestamp"])
        author_col = self._pick_col(cols, ["author_hash", "author", "user", "username"])
        id_col = self._pick_col(cols, ["post_id", "id", "submission_id"])

        if not post_col or not date_col:
            return pd.DataFrame()

        work = df.copy()
        if subreddit_col:
            work = work[work[subreddit_col].astype(str).str.lower() == subreddit.lower()]
        elif "subreddit" in work.columns:
            work = work[work["subreddit"].astype(str).str.lower() == subreddit.lower()]

        if work.empty:
            return pd.DataFrame()

        source_slug = re.sub(r"[^a-zA-Z0-9]+", "_", source_name).strip("_").lower() or "unknown_file"
        out = pd.DataFrame(
            {
                "selftext": work[post_col].astype(str),
                "created_utc": self._to_unix(work[date_col]),
                "subreddit": subreddit,
                "author": work[author_col].astype(str) if author_col else "[deleted]",
                "post_id": work[id_col].astype(str)
                if id_col
                else work.index.to_series().astype(str).map(
                    lambda x: f"zenodo_{subreddit}_{source_slug}_{x}"
                ),
                "data_source": "zenodo_covid",
            }
        )
        # Keep only canonical raw columns; ignore precomputed features.
        return out[["post_id", "created_utc", "selftext", "subreddit", "author", "data_source"]]

    @staticmethod
    def _pick_col(lower_to_original: dict[str, str], candidates: list[str]) -> str | None:
        for c in candidates:
            if c in lower_to_original:
                return lower_to_original[c]
        return None

    @staticmethod
    def _to_unix(series: pd.Series) -> pd.Series:
        # If already numeric unix-like keep as-is; else parse datetime.
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() > 0.8:
            return numeric
        dt = pd.to_datetime(series, errors="coerce", utc=True)
        # Use total_seconds() so this works across pandas versions (ns, us, ms resolution)
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        return (dt - epoch).dt.total_seconds()

