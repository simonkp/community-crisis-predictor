import pandas as pd
from src.processing.text_cleaner import clean_text, clean_post, filter_deleted, process_posts


def test_clean_text_removes_urls():
    text = "Visit https://example.com for more info"
    result = clean_text(text)
    assert "https://example.com" not in result
    assert "visit" in result


def test_clean_text_lowercases():
    assert clean_text("Hello WORLD") == "hello world"


def test_clean_text_normalizes_whitespace():
    assert clean_text("too   many    spaces") == "too many spaces"


def test_clean_text_handles_deleted():
    assert clean_text("[deleted]") == ""
    assert clean_text("[removed]") == ""


def test_clean_post_combines_title_body():
    result = clean_post("My Title", "My body text")
    assert "my title" in result
    assert "my body text" in result


def test_filter_deleted():
    df = pd.DataFrame({
        "selftext": ["real post", "[deleted]", "[removed]", "another real post"],
    })
    result = filter_deleted(df)
    assert len(result) == 2


def test_process_posts_min_length():
    df = pd.DataFrame({
        "title": ["Short", "A longer title that meets minimum"],
        "selftext": ["Hi", "This is a much longer post body that definitely meets the minimum length requirement"],
        "score": [1, 2],
        "num_comments": [0, 1],
    })
    result = process_posts(df, min_length=20)
    assert len(result) == 1
    assert "clean_text" in result.columns
