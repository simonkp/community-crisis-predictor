import pandas as pd

from src.collector.zenodo_loader import ZenodoLoader


def test_zenodo_loader_maps_raw_schema_and_filters_subreddit(tmp_path):
    archive_dir = tmp_path / "archive"
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    csv_path = staging_dir / "depression_2019_features_tfidf_256.csv"

    df = pd.DataFrame(
        {
            "subreddit": ["depression", "anxiety"],
            "post": ["text a", "text b"],
            "date": ["2020-03-01T00:00:00Z", "2020-03-02T00:00:00Z"],
            "author_hash": ["a1", "a2"],
            "tfidf_fake": [0.1, 0.2],
            "liwc_fake": [1, 2],
        }
    )
    df.to_csv(csv_path, index=False)

    loader = ZenodoLoader(
        dataset_url="https://example.com/dataset.zip",
        archive_dir=str(archive_dir),
        staging_dir=str(staging_dir),
    )
    out = loader.load_subreddit_posts("depression")
    assert len(out) == 1
    assert set(out.columns) == {
        "post_id",
        "created_utc",
        "selftext",
        "subreddit",
        "author",
        "data_source",
        "created_utc_dt",
    }
    assert out.iloc[0]["selftext"] == "text a"
    assert out.iloc[0]["data_source"] == "zenodo_covid"


def test_zenodo_loader_date_filter(tmp_path):
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    csv_path = staging_dir / "depression_post_features_tfidf_256.csv"
    pd.DataFrame(
        {
            "subreddit": ["depression", "depression"],
            "post": ["old", "new"],
            "date": ["2019-01-01T00:00:00Z", "2020-06-01T00:00:00Z"],
            "author_hash": ["u1", "u2"],
        }
    ).to_csv(csv_path, index=False)

    loader = ZenodoLoader("https://example.com", str(tmp_path / "archive"), str(staging_dir))
    out = loader.load_subreddit_posts("depression", start_date="2020-01-01", end_date="2020-12-31")
    assert len(out) == 1
    assert out.iloc[0]["selftext"] == "new"

