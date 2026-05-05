from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from bot.config import Settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in (
        "PAPER_MODE",
        "KALSHI_DEMO_API_BASE",
        "KALSHI_DEMO_KEY_ID",
        "KALSHI_DEMO_PRIVATE_KEY_PATH",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


def test_defaults_load_with_paper_mode_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KALSHI_DEMO_API_BASE", "https://demo-api.kalshi.co/trade-api/v2")
    monkeypatch.setenv("PAPER_MODE", "true")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    s = Settings()

    assert s.paper_mode is True
    assert s.kalshi_demo_api_base == "https://demo-api.kalshi.co/trade-api/v2"
    assert s.kalshi_demo_key_id is None
    assert s.log_level == "INFO"


def test_live_mode_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER_MODE", "false")

    with pytest.raises(ValidationError):
        Settings()


def test_live_mode_with_key_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("KALSHI_DEMO_KEY_ID", "abc-123")

    s = Settings()

    assert s.paper_mode is False
    assert s.kalshi_demo_key_id == "abc-123"
