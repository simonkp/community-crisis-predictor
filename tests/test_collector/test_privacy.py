import pandas as pd
from src.collector.privacy import hash_author, remove_urls, remove_emails, strip_pii


def test_hash_author_deterministic():
    h1 = hash_author("testuser", "salt")
    h2 = hash_author("testuser", "salt")
    assert h1 == h2


def test_hash_author_different_salt():
    h1 = hash_author("testuser", "salt1")
    h2 = hash_author("testuser", "salt2")
    assert h1 != h2


def test_hash_author_deleted():
    assert hash_author("[deleted]", "salt") == "anonymous"
    assert hash_author("None", "salt") == "anonymous"
    assert hash_author("", "salt") == "anonymous"


def test_remove_urls():
    text = "Check this out https://example.com and www.test.com for more"
    result = remove_urls(text)
    assert "https://example.com" not in result
    assert "www.test.com" not in result


def test_remove_emails():
    text = "Contact me at user@example.com for help"
    result = remove_emails(text)
    assert "user@example.com" not in result


def test_strip_pii():
    df = pd.DataFrame({
        "author": ["user1", "user2"],
        "title": ["Hello https://link.com", "Test"],
        "selftext": ["Email me@test.com", "Body text"],
    })
    result = strip_pii(df, "salt")

    assert "author" not in result.columns
    assert "author_hash" in result.columns
    assert "https://link.com" not in result["title"].iloc[0]
    assert "me@test.com" not in result["selftext"].iloc[0]
