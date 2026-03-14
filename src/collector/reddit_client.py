import time
from datetime import datetime

import pandas as pd


class RedditCollector:
    def __init__(self, config: dict):
        import praw
        reddit_cfg = config["reddit"]
        self.reddit = praw.Reddit(
            client_id=reddit_cfg["client_id"],
            client_secret=reddit_cfg["client_secret"],
            user_agent=reddit_cfg.get("user_agent", "CrisisPredictor/1.0"),
        )
        self._max_retries = 3

    def collect_subreddit(
        self, subreddit: str, after: datetime, before: datetime, limit: int = None
    ) -> pd.DataFrame:
        sub = self.reddit.subreddit(subreddit)
        posts = []
        after_ts = after.timestamp()
        before_ts = before.timestamp()

        for attempt in range(self._max_retries):
            try:
                for submission in sub.new(limit=limit or 10000):
                    created = submission.created_utc
                    if created < after_ts:
                        break
                    if created > before_ts:
                        continue

                    posts.append({
                        "post_id": submission.id,
                        "created_utc": int(created),
                        "title": submission.title or "",
                        "selftext": submission.selftext or "",
                        "score": submission.score,
                        "num_comments": submission.num_comments,
                        "subreddit": subreddit,
                        "author": str(submission.author) if submission.author else "[deleted]",
                        "is_self": submission.is_self,
                    })
                break
            except Exception as e:
                if attempt < self._max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"Rate limited, waiting {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise

        df = pd.DataFrame(posts)
        if not df.empty:
            df["created_utc_dt"] = pd.to_datetime(df["created_utc"], unit="s")
        return df
