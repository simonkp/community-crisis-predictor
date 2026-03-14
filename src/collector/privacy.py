import hashlib
import re

import pandas as pd


_URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+", re.IGNORECASE
)
_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE
)


def hash_author(author: str, salt: str) -> str:
    if not author or author in ("[deleted]", "[removed]", "None"):
        return "anonymous"
    return hashlib.sha256(f"{author}{salt}".encode()).hexdigest()[:16]


def remove_urls(text: str) -> str:
    if not text:
        return ""
    return _URL_PATTERN.sub("", text)


def remove_emails(text: str) -> str:
    if not text:
        return ""
    return _EMAIL_PATTERN.sub("", text)


def strip_pii(df: pd.DataFrame, salt: str) -> pd.DataFrame:
    df = df.copy()

    if "author" in df.columns:
        df["author_hash"] = df["author"].astype(str).apply(
            lambda a: hash_author(a, salt)
        )
        df = df.drop(columns=["author"])

    for col in ["title", "selftext"]:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(remove_urls).apply(remove_emails)

    return df
