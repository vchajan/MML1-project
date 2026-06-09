from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import download_nasa_power_daily as nasa  # noqa: E402


def payload_for_dates(dates: list[str], value: float = 1.0) -> dict:
    return {
        "properties": {
            "parameter": {
                param: {date.replace("-", ""): value for date in dates}
                for param in nasa.PARAMETERS
            }
        }
    }


def test_cache_complete_is_reused() -> None:
    with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temp_dir:
        path = Path(temp_dir) / "WPT.csv"
        frame = nasa.parse_nasa_response(payload_for_dates(["2001-01-01", "2001-01-02"]), "WPT", 10.0, 20.0, "2001-01-01", "2001-01-02")
        frame.to_csv(path, index=False)
        complete, message = nasa.cache_is_complete(path, "WPT", "2001-01-01", "2001-01-02")
    assert complete
    assert message == "complete"


def test_incomplete_cache_is_identified() -> None:
    with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temp_dir:
        path = Path(temp_dir) / "WPT.csv"
        pd.DataFrame({"weather_point_id": ["WPT"], "date": ["2001-01-01"]}).to_csv(path, index=False)
        complete, message = nasa.cache_is_complete(path, "WPT", "2001-01-01", "2001-01-02")
    assert not complete
    assert "missing_columns" in message


def test_sentinel_minus_999_becomes_missing() -> None:
    payload = payload_for_dates(["2001-01-01"], value=-999)
    frame = nasa.parse_nasa_response(payload, "WPT", 10.0, 20.0, "2001-01-01", "2001-01-01")
    assert pd.isna(frame.loc[0, "PRECTOTCORR"])


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = "error"

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = responses
        self.calls = 0

    def get(self, *_args, **_kwargs) -> FakeResponse:
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_retry_succeeds_after_mock_http_failure() -> None:
    payload = payload_for_dates(["2001-01-01"])
    session = FakeSession([FakeResponse(503), FakeResponse(200, payload)])
    result, status = nasa.fetch_with_retry(
        session, 10.0, 20.0, "2001-01-01", "2001-01-01", sleep_func=lambda _seconds: None, max_retries=2
    )
    assert status == 200
    assert session.calls == 2
    assert result == payload
