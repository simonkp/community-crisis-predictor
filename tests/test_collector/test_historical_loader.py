from unittest.mock import Mock, patch

import requests

from src.collector.historical_loader import PushshiftLoader, RetryPolicy


def _mock_response(status_code=200, data=None):
    resp = Mock()
    resp.status_code = status_code
    if status_code >= 400:
        err = requests.HTTPError(f"status {status_code}")
        err.response = resp
        resp.raise_for_status.side_effect = err
    else:
        resp.raise_for_status.return_value = None
    resp.json.return_value = {"data": data or []}
    return resp


@patch("src.collector.historical_loader.time.sleep", return_value=None)
@patch("src.collector.historical_loader.requests.get")
def test_loader_retries_on_429(mock_get, _sleep):
    mock_get.side_effect = [
        _mock_response(status_code=429),
        _mock_response(status_code=200, data=[]),
    ]
    loader = PushshiftLoader(retry_policy=RetryPolicy(max_retries=2))
    _, summary = loader.load_range("depression", 1, 2, batch_size=10)
    assert summary.retry_count >= 1


@patch("src.collector.historical_loader.time.sleep", return_value=None)
@patch("src.collector.historical_loader.requests.get")
def test_loader_retries_on_timeout(mock_get, _sleep):
    mock_get.side_effect = [
        requests.Timeout("timeout"),
        _mock_response(status_code=200, data=[]),
    ]
    loader = PushshiftLoader(retry_policy=RetryPolicy(max_retries=2))
    _, summary = loader.load_range("depression", 1, 2, batch_size=10)
    assert summary.retry_count >= 1


@patch("src.collector.historical_loader.time.sleep", return_value=None)
@patch("src.collector.historical_loader.requests.get")
def test_loader_marks_truncated_after_terminal_failure(mock_get, _sleep):
    mock_get.side_effect = requests.Timeout("timeout")
    loader = PushshiftLoader(retry_policy=RetryPolicy(max_retries=1))
    _, summary = loader.load_range("depression", 1, 2, batch_size=10)
    assert summary.truncated is True
    assert "Terminal" in summary.terminal_error
