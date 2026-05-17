from __future__ import annotations

from datetime import date, datetime
from datetime import timezone as _timezone
from decimal import Decimal
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    TypeDecorator,
    UniqueConstraint,
    create_engine,
    event,
    inspect,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class UtcDateTime(TypeDecorator):
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("UtcDateTime requires tz-aware datetime, got naive")
        return value.astimezone(_timezone.utc)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=_timezone.utc)
        return value.astimezone(_timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(tz=_timezone.utc)


class Base(DeclarativeBase):
    pass


class Forecast(Base):
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station: Mapped[str] = mapped_column(String(16))
    run_time: Mapped[datetime] = mapped_column(UtcDateTime())
    valid_date: Mapped[date] = mapped_column(Date)
    members_json: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    __table_args__ = (
        UniqueConstraint(
            "station", "run_time", "valid_date", name="uq_forecasts_station_run_valid"
        ),
        Index("ix_forecasts_station_valid_date", "station", "valid_date"),
    )


class Market(Base):
    __tablename__ = "markets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(64), unique=True)
    series: Mapped[str] = mapped_column(String(32))
    event_date: Mapped[date] = mapped_column(Date)
    is_monthly: Mapped[bool] = mapped_column(Boolean)
    is_tail: Mapped[bool] = mapped_column(Boolean)
    strike_low: Mapped[Decimal] = mapped_column(Numeric(10, 4))
    strike_high: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    close_time: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    status: Mapped[str] = mapped_column(String(16))
    last_seen_at: Mapped[datetime] = mapped_column(UtcDateTime())
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    __table_args__ = (Index("ix_markets_series_event_date", "series", "event_date"),)


class OrderbookSnapshot(Base):
    __tablename__ = "orderbook_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(64))
    snapshot_at: Mapped[datetime] = mapped_column(UtcDateTime())
    yes_ask: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    yes_bid: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    no_ask: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    no_bid: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    yes_ask_depth: Mapped[int | None] = mapped_column(Integer, nullable=True)
    yes_bid_depth: Mapped[int | None] = mapped_column(Integer, nullable=True)
    no_ask_depth: Mapped[int | None] = mapped_column(Integer, nullable=True)
    no_bid_depth: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    __table_args__ = (Index("ix_orderbook_snapshots_ticker_snapshot_at", "ticker", "snapshot_at"),)


class PaperTradeRow(Base):
    __tablename__ = "paper_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    intended_at: Mapped[datetime] = mapped_column(UtcDateTime())
    market_ticker: Mapped[str] = mapped_column(String(64))
    side: Mapped[str] = mapped_column(String(16))
    contracts: Mapped[int] = mapped_column(Integer)
    simulated_price: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    fee_dollars: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    fair_at_entry: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    strategy: Mapped[str] = mapped_column(String(32))
    attempted_contracts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ensemble_spread_sigma_t: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    lead_time_hours: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    nbm_divergence: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    __table_args__ = (Index("ix_paper_trades_strategy_intended_at", "strategy", "intended_at"),)


class SimulatedPnl(Base):
    __tablename__ = "simulated_pnl"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_trade_id: Mapped[int] = mapped_column(
        ForeignKey("paper_trades.id", ondelete="CASCADE"), index=True
    )
    settled_at: Mapped[datetime] = mapped_column(UtcDateTime())
    outcome: Mapped[str] = mapped_column(String(8))
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)


class GateFailure(Base):
    __tablename__ = "gate_failures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluated_at: Mapped[datetime] = mapped_column(UtcDateTime())
    gate_name: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(String(256))
    mode: Mapped[str] = mapped_column(String(8))
    market_ticker: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    __table_args__ = (
        Index("ix_gate_failures_evaluated_at_gate_name", "evaluated_at", "gate_name"),
    )


def make_engine(db_path: Path | str) -> Engine:
    if isinstance(db_path, Path):
        url = f"sqlite:///{db_path}"
    elif db_path == ":memory:":
        url = "sqlite:///:memory:"
    else:
        url = f"sqlite:///{db_path}"

    engine = create_engine(url, future=True, connect_args={"timeout": 30})

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def ensure_baseline_stamped(engine: Engine, baseline: str) -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    with engine.begin() as connection:
        if "alembic_version" in tables:
            existing = connection.execute(text("SELECT version_num FROM alembic_version")).first()
            if existing is not None:
                return
        if "forecasts" not in tables:
            return
        if "alembic_version" not in tables:
            connection.execute(
                text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
            )
        connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:rev)"),
            {"rev": baseline},
        )
