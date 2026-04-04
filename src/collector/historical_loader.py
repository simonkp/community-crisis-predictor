import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


@dataclass
class RetryPolicy:
    max_retries: int = 3
    backoff_base: float = 2.0
    timeout_seconds: int = 30
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)
    jitter_seconds: float = 0.15


@dataclass
class CollectionSummary:
    requested_after: int
    requested_before: int
    fetched_posts: int = 0
    request_count: int = 0
    retry_count: int = 0
    truncated: bool = False
    terminal_error: str = ""


class PushshiftLoader:
    def __init__(
        self,
        base_url: str = "https://api.pullpush.io",
        rate_limit_rps: float = 1.0,
        retry_policy: RetryPolicy | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._rate_limit_rps = max(rate_limit_rps, 0.1)
        self.retry_policy = retry_policy or RetryPolicy()

    def load_range(
        self, subreddit: str, after: int, before: int, batch_size: int = 500
    ) -> tuple[pd.DataFrame, CollectionSummary]:
        all_posts = []
        current_after = after
        summary = CollectionSummary(requested_after=after, requested_before=before)

        while current_after < before:
            params = {
                "subreddit": subreddit,
                "after": current_after,
                "before": before,
                "size": batch_size,
                "sort": "asc",
                "sort_type": "created_utc",
            }
            data: list[dict[str, Any]] = []
            done = False
            terminal_error = ""

            for attempt in range(self.retry_policy.max_retries + 1):
                summary.request_count += 1
                try:
                    resp = requests.get(
                        f"{self.base_url}/reddit/search/submission",
                        params=params,
                        timeout=self.retry_policy.timeout_seconds,
                    )
                    if resp.status_code in self.retry_policy.retryable_status_codes:
                        raise requests.HTTPError(
                            f"Retryable status: {resp.status_code}",
                            response=resp,
                        )
                    resp.raise_for_status()
                    data = resp.json().get("data", [])
                    done = True
                    break
                except (requests.Timeout, requests.ConnectionError) as e:
                    if attempt >= self.retry_policy.max_retries:
                        terminal_error = f"Terminal network failure: {e}"
                        break
                    summary.retry_count += 1
                    self._sleep_with_backoff(attempt)
                except requests.HTTPError as e:
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    retryable = status in self.retry_policy.retryable_status_codes
                    if retryable and attempt < self.retry_policy.max_retries:
                        summary.retry_count += 1
                        self._sleep_with_backoff(attempt)
                        continue
                    terminal_error = f"Terminal HTTP failure: {e}"
                    break
                except ValueError as e:
                    terminal_error = f"Response parse failure: {e}"
                    break

            if not done:
                summary.truncated = True
                summary.terminal_error = terminal_error or "Unknown failure"
                print(f"Pushshift request failed for r/{subreddit}: {summary.terminal_error}")
                break

            if not data:
                break

            for post in data:
                all_posts.append({
                    "post_id": post.get("id", ""),
                    "created_utc": post.get("created_utc", 0),
                    "title": post.get("title", ""),
                    "selftext": post.get("selftext", ""),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "subreddit": subreddit,
                    "author": post.get("author", "[deleted]"),
                    "is_self": post.get("is_self", True),
                })

            current_after = data[-1]["created_utc"] + 1
            time.sleep(1.0 / self._rate_limit_rps)

        df = pd.DataFrame(all_posts)
        summary.fetched_posts = len(df)
        if not df.empty:
            df["created_utc_dt"] = pd.to_datetime(df["created_utc"], unit="s")
        return df, summary

    def _sleep_with_backoff(self, attempt: int) -> None:
        # Small deterministic jitter avoids synchronized retries across workers.
        delay = (self.retry_policy.backoff_base**attempt) + (
            self.retry_policy.jitter_seconds * (attempt + 1)
        )
        time.sleep(delay)
