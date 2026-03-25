from src.collector.arctic_shift_loader import ArcticShiftLoader, parse_arctic_shift_filename


def test_parse_arctic_shift_filename():
    assert (
        parse_arctic_shift_filename("arctic_shift_depression_201805_201810.jsonl")
        == "depression"
    )
    assert parse_arctic_shift_filename("wrong_name.jsonl") is None


def test_arctic_shift_loader_filters_and_maps(tmp_path):
    p = tmp_path / "arctic_shift_depression_201805_201810.jsonl"
    lines = [
        # valid
        '{"id":"a1","created_utc":1525132800,"selftext":"this is a valid post body",'
        '"subreddit":"depression","author":"user1","is_self":true}\n',
        # deleted body
        '{"id":"a2","created_utc":1525132801,"selftext":"[deleted]",'
        '"subreddit":"depression","author":"u2","is_self":true}\n',
        # link post
        '{"id":"a3","created_utc":1525132802,"selftext":"valid enough body",'
        '"subreddit":"depression","author":"u3","is_self":false}\n',
        # invalid json
        '{"id":"a4",\n',
        # deleted author should become empty author string but still kept
        '{"id":"a5","created_utc":1525132803,"selftext":"another valid text body",'
        '"subreddit":"depression","author":"[deleted]","is_self":true}\n',
    ]
    p.write_text("".join(lines), encoding="utf-8")

    loader = ArcticShiftLoader(min_selftext_chars=10)
    df, stats = loader.load_jsonl(p, subreddit="depression")
    assert len(df) == 2
    assert set(df.columns) == {"post_id", "created_utc", "selftext", "subreddit", "author", "data_source"}
    assert set(df["post_id"].tolist()) == {"a1", "a5"}
    assert (df["data_source"] == "arctic_shift").all()
    assert df.loc[df["post_id"] == "a5", "author"].iloc[0] == ""
    assert stats["parsed"] == 4
    assert stats["kept"] == 2
    assert stats["skipped_body"] == 1
    assert stats["skipped_non_self"] == 1
    assert stats["skipped_json"] == 1
