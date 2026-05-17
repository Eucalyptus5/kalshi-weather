from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import update
from sqlalchemy.orm import Session, sessionmaker

from bot.storage.sqlite import GateFailure


# Anchored at least 10 minutes after the planned SIGTERM so a re-run never re-stamps
# post-fix rows even if the operator runs the script while the bot is back up.
FIX_TIMESTAMP: datetime = datetime(2026, 5, 28, 0, 0, tzinfo=timezone.utc)
NOTE_LABEL: str = "pre_fix_status_string_bug"


def annotate(session_factory: sessionmaker[Session]) -> int:
    with session_factory() as session:
        result = session.execute(
            update(GateFailure)
            .where(GateFailure.gate_name == "market_open")
            .where(GateFailure.evaluated_at < FIX_TIMESTAMP)
            .where(GateFailure.notes.is_(None))
            .values(notes=NOTE_LABEL)
        )
        session.commit()
        return int(result.rowcount or 0)
