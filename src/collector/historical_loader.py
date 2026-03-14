import time

import pandas as pd
import requests


class PushshiftLoader:
    def __init__(self, base_url: str = "https://api.pullpush.io"):
        self.base_url = base_url.rstrip("/")
        self._rate_limit_rps = 1

    def load_range(
        self, subreddit: str, after: int, before: int, batch_size: int = 500
    ) -> pd.DataFrame:
        all_posts = []
        current_after = after

        while current_after < before:
            params = {
                "subreddit": subreddit,
                "after": current_after,
                "before": before,
                "size": batch_size,
                "sort": "asc",
                "sort_type": "created_utc",
            }

            try:
                resp = requests.get(
                    f"{self.base_url}/reddit/search/submission",
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
            except (requests.RequestException, ValueError) as e:
                print(f"Pushshift request failed: {e}")
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
        if not df.empty:
            df["created_utc_dt"] = pd.to_datetime(df["created_utc"], unit="s")
        return df
