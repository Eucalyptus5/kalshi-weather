import argparse
import json
import os
import sqlite3
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

STALE_HEARTBEAT_S = 300
MAX_GAP_DELTA = 2
MIN_DISK_FREE_FRACTION = 0.20
# restart drifts about +20s/day from 08:04, so the window carries months of slack
RESTART_WINDOW_START_MIN = 7 * 60 + 55
RESTART_WINDOW_END_MIN = 8 * 60 + 40


def evaluate(
    service_state: str,
    heartbeat_age_s: float | None,
    gap_delta: int | None,
    disk_free_fraction: float,
    in_window: bool,
    db_ok: bool,
) -> list[str]:
    alarms: list[str] = []
    if service_state not in ("active", "activating"):
        alarms.append(f"service_not_active state={service_state}")
    if not db_ok:
        alarms.append("db_unreadable")
    else:
        if not in_window and (heartbeat_age_s is None or heartbeat_age_s > STALE_HEARTBEAT_S):
            age = "none" if heartbeat_age_s is None else f"{heartbeat_age_s:.0f}"
            alarms.append(f"heartbeat_stale age_seconds={age}")
        if gap_delta is not None and gap_delta > MAX_GAP_DELTA:
            alarms.append(f"ws_gaps_delta delta={gap_delta}")
    if disk_free_fraction < MIN_DISK_FREE_FRACTION:
        alarms.append(f"disk_low free_fraction={disk_free_fraction:.2f}")
    return alarms


def in_restart_window(now: datetime) -> bool:
    minute = now.hour * 60 + now.minute
    return RESTART_WINDOW_START_MIN <= minute < RESTART_WINDOW_END_MIN


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def read_service_state(unit: str) -> str:
    result = subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True)
    return result.stdout.strip()


def connect_ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def read_db(db: Path) -> tuple[datetime | None, int]:
    conn = connect_ro(db)
    try:
        beat = conn.execute("SELECT max(beat_at) FROM ws_heartbeats").fetchone()[0]
        gap_count = conn.execute("SELECT count(*) FROM ws_gaps").fetchone()[0]
    finally:
        conn.close()
    latest = None if beat is None else datetime.fromisoformat(beat).replace(tzinfo=timezone.utc)
    return latest, gap_count


def read_disk_free_fraction(directory: Path) -> float:
    stats = os.statvfs(directory)
    return stats.f_bavail / stats.f_blocks


def load_previous_gap_count(state_path: Path) -> int | None:
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text())["gap_count"]


def save_gap_count(state_path: Path, gap_count: int | None) -> None:
    state_path.write_text(json.dumps({"gap_count": gap_count}))


def post_alarms(topic: str, alarms: list[str]) -> None:
    request = urllib.request.Request(
        f"https://ntfy.sh/{topic}",
        data="\n".join(alarms).encode(),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30):
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ws recorder health check")
    parser.add_argument("--db", type=Path, default=Path("data/state.db"))
    parser.add_argument("--unit", default="kalshi-ws.service")
    parser.add_argument(
        "--state", type=Path, default=Path(__file__).with_name("ws_watchdog_state.json")
    )
    parser.add_argument("--log", type=Path, default=Path(__file__).with_name("ws_watchdog.log"))
    return parser


def run(args: argparse.Namespace) -> int:
    now = utc_now()
    service_state = read_service_state(args.unit)

    db_ok = True
    latest_beat: datetime | None = None
    gap_count: int | None = None
    try:
        latest_beat, gap_count = read_db(args.db)
    except sqlite3.Error:
        db_ok = False

    heartbeat_age_s = None if latest_beat is None else (now - latest_beat).total_seconds()
    previous = load_previous_gap_count(args.state)
    gap_delta = None if gap_count is None or previous is None else gap_count - previous
    free_fraction = read_disk_free_fraction(args.db.parent)

    alarms = evaluate(
        service_state, heartbeat_age_s, gap_delta, free_fraction, in_restart_window(now), db_ok
    )
    save_gap_count(args.state, gap_count)

    ntfy = "none"
    if alarms:
        topic = os.environ.get("KW_WATCHDOG_NTFY_TOPIC", "")
        if topic:
            try:
                post_alarms(topic, alarms)
                ntfy = "sent"
            except OSError:
                ntfy = "failed"
        else:
            ntfy = "skipped"

    verdict = "alarm" if alarms else "ok"
    age_text = "none" if heartbeat_age_s is None else f"{heartbeat_age_s:.0f}"
    gap_count_text = "none" if gap_count is None else str(gap_count)
    gap_delta_text = "none" if gap_delta is None else str(gap_delta)
    line = (
        f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')} verdict={verdict} service={service_state} "
        f"heartbeat_age_s={age_text} gap_count={gap_count_text} gap_delta={gap_delta_text} "
        f"disk_free_fraction={free_fraction:.3f} ntfy={ntfy}"
    )
    if alarms:
        line += " alarms=" + "; ".join(alarms)
    with args.log.open("a") as handle:
        handle.write(line + "\n")
    return 1 if ntfy == "failed" else 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
