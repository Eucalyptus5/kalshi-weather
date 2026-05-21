from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from bot.config import Settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in (
        "MODE",
        "KALSHI_DEMO_API_BASE",
        "KALSHI_DEMO_KEY_ID",
        "KALSHI_DEMO_PRIVATE_KEY_PATH",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


def test_default_mode_is_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KALSHI_DEMO_API_BASE", "https://demo-api.kalshi.co/trade-api/v2")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    s = Settings()

    assert s.mode == "paper"
    assert s.kalshi_demo_api_base == "https://demo-api.kalshi.co/trade-api/v2"
    assert s.kalshi_demo_key_id is None
    assert s.log_level == "INFO"


def test_mode_paper_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODE", "paper")

    s = Settings()

    assert s.mode == "paper"


def test_demo_mode_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODE", "demo")

    with pytest.raises(ValidationError):
        Settings()


def test_demo_mode_with_key_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODE", "demo")
    monkeypatch.setenv("KALSHI_DEMO_KEY_ID", "abc-123")

    s = Settings()

    assert s.mode == "demo"
    assert s.kalshi_demo_key_id == "abc-123"


def test_live_mode_rejected_by_literal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODE", "live")

    with pytest.raises(ValidationError):
        Settings()


def test_env_var_mode_demo_with_key_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODE", "demo")
    monkeypatch.setenv("KALSHI_DEMO_KEY_ID", "demo-id")

    s = Settings()

    assert s.mode == "demo"
