import re

import pandas as pd

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"\s+")
_DELETED_VALUES = {"[deleted]", "[removed]", ""}


def clean_text(text: str) -> str:
    if not text or text in _DELETED_VALUES:
        return ""
    text = _URL_PATTERN.sub("", text)
    text = text.lower()
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def clean_post(title: str, selftext: str) -> str:
    title = clean_text(title or "")
    body = clean_text(selftext or "")
    if title and body:
        return f"{title}. {body}"
    return title or body


def filter_deleted(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~df["selftext"].astype(str).isin(_DELETED_VALUES)
    return df[mask].copy()


def process_posts(df: pd.DataFrame, min_length: int = 20) -> pd.DataFrame:
    df = df.copy()
    df = filter_deleted(df)
    df["clean_text"] = df.apply(
        lambda r: clean_post(r.get("title", ""), r.get("selftext", "")), axis=1
    )
    df = df[df["clean_text"].str.len() >= min_length].copy()
    return df
