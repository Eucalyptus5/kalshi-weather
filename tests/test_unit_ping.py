from __future__ import annotations

import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from scripts import unit_ping, ws_watchdog


REPO_ROOT = Path(__file__).resolve().parent.parent
TOPIC = "wd-topic-c9f3e"
UNIT = "kalshi-replay-forward.service"

SUCCESS_ENV = {"SERVICE_RESULT": "success", "EXIT_CODE": "exited", "EXIT_STATUS": "0"}
FAILURE_ENV = {"SERVICE_RESULT": "exit-code", "EXIT_CODE": "exited", "EXIT_STATUS": "1"}
ABORT_ENV = {"SERVICE_RESULT": "core-dump", "EXIT_CODE": "dumped", "EXIT_STATUS": "ABRT"}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SERVICE_RESULT", "EXIT_CODE", "EXIT_STATUS", "KW_WATCHDOG_NTFY_TOPIC"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def posted(monkeypatch: pytest.MonkeyPatch) -> list[urllib.request.Request]:
    requests: list[urllib.request.Request] = []

    class Resp:
        def __enter__(self) -> Resp:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> Resp:
        requests.append(request)
        return Resp()

    monkeypatch.setattr(ws_watchdog.urllib.request, "urlopen", fake_urlopen)
    return requests


def set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        (
            SUCCESS_ENV,
            b"unit_stopped verdict=success unit=kalshi-replay-forward.service "
            b"result=success exit_code=exited exit_status=0",
        ),
        (
            FAILURE_ENV,
            b"unit_stopped verdict=failed unit=kalshi-replay-forward.service "
            b"result=exit-code exit_code=exited exit_status=1",
        ),
        (
            ABORT_ENV,
            b"unit_stopped verdict=failed unit=kalshi-replay-forward.service "
            b"result=core-dump exit_code=dumped exit_status=ABRT",
        ),
    ],
    ids=["success", "failure", "abort"],
)
def test_posts_golden_line(
    monkeypatch: pytest.MonkeyPatch,
    posted: list[urllib.request.Request],
    env: dict[str, str],
    expected: bytes,
) -> None:
    set_env(monkeypatch, env)
    monkeypatch.setenv("KW_WATCHDOG_NTFY_TOPIC", TOPIC)

    rc = unit_ping.main(["--unit", UNIT])

    assert rc == 0
    assert len(posted) == 1
    assert posted[0].full_url == f"https://ntfy.sh/{TOPIC}"
    assert posted[0].data == expected


def test_verdict_separates_success_from_failures(
    monkeypatch: pytest.MonkeyPatch, posted: list[urllib.request.Request]
) -> None:
    monkeypatch.setenv("KW_WATCHDOG_NTFY_TOPIC", TOPIC)
    for env in (SUCCESS_ENV, FAILURE_ENV, ABORT_ENV):
        set_env(monkeypatch, env)
        unit_ping.main(["--unit", UNIT])

    bodies = [request.data.decode() for request in posted]
    assert "verdict=success" in bodies[0]
    assert "verdict=success" not in bodies[1]
    assert "verdict=success" not in bodies[2]
    assert len(set(bodies)) == 3


def test_unit_name_reaches_body(
    monkeypatch: pytest.MonkeyPatch, posted: list[urllib.request.Request]
) -> None:
    set_env(monkeypatch, SUCCESS_ENV)
    monkeypatch.setenv("KW_WATCHDOG_NTFY_TOPIC", TOPIC)

    assert unit_ping.main(["--unit", "kalshi-replay-drain.service"]) == 0
    assert b"unit=kalshi-replay-drain.service" in posted[0].data


def test_missing_environment_renders_none(
    monkeypatch: pytest.MonkeyPatch, posted: list[urllib.request.Request]
) -> None:
    monkeypatch.setenv("KW_WATCHDOG_NTFY_TOPIC", TOPIC)

    rc = unit_ping.main(["--unit", UNIT])

    assert rc == 0
    assert posted[0].data == (
        b"unit_stopped verdict=failed unit=kalshi-replay-forward.service "
        b"result=none exit_code=none exit_status=none"
    )


@pytest.mark.parametrize("topic", [None, ""], ids=["absent", "empty"])
def test_no_topic_skips_post(
    monkeypatch: pytest.MonkeyPatch,
    posted: list[urllib.request.Request],
    topic: str | None,
) -> None:
    set_env(monkeypatch, SUCCESS_ENV)
    if topic is not None:
        monkeypatch.setenv("KW_WATCHDOG_NTFY_TOPIC", topic)

    rc = unit_ping.main(["--unit", UNIT])

    assert rc == 0
    assert posted == []


def test_post_failure_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    def raising_urlopen(request: urllib.request.Request, timeout: float | None = None) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(ws_watchdog.urllib.request, "urlopen", raising_urlopen)
    set_env(monkeypatch, FAILURE_ENV)
    monkeypatch.setenv("KW_WATCHDOG_NTFY_TOPIC", TOPIC)

    assert unit_ping.main(["--unit", UNIT]) == 1


def test_stdout_reports_sent_without_topic_value(
    monkeypatch: pytest.MonkeyPatch,
    posted: list[urllib.request.Request],
    capsys: pytest.CaptureFixture[str],
) -> None:
    set_env(monkeypatch, FAILURE_ENV)
    monkeypatch.setenv("KW_WATCHDOG_NTFY_TOPIC", TOPIC)

    assert unit_ping.main(["--unit", UNIT]) == 0

    assert len(posted) == 1
    out = capsys.readouterr().out
    assert TOPIC not in out
    assert "verdict=failed" in out
    assert f"unit={UNIT}" in out
    assert "ntfy=sent" in out


def test_stdout_reports_skipped_without_topic_value(
    monkeypatch: pytest.MonkeyPatch,
    posted: list[urllib.request.Request],
    capsys: pytest.CaptureFixture[str],
) -> None:
    set_env(monkeypatch, SUCCESS_ENV)
    monkeypatch.setenv("KW_WATCHDOG_NTFY_TOPIC", "")

    assert unit_ping.main(["--unit", UNIT]) == 0

    assert posted == []
    out = capsys.readouterr().out
    assert TOPIC not in out
    assert "verdict=success" in out
    assert "ntfy=skipped" in out


def test_help_smoke() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "unit_ping.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--unit" in result.stdout
