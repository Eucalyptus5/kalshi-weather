from __future__ import annotations

from datetime import date, datetime
from datetime import timezone as _timezone
from decimal import Decimal
from pathlib import Path

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
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


_REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent


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


class DecimalText(TypeDecorator):
    # TEXT, not Numeric: sqlite Numeric paths through float and drops trailing
    # zeros, so Decimal("0.500000") would come back as Decimal("0.5").
    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return Decimal(value)


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
    q_raw: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    strategy: Mapped[str] = mapped_column(String(32))
    attempted_contracts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ensemble_spread_sigma_t: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    lead_time_hours: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    nbm_divergence: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    demo_order_client_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, unique=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    __table_args__ = (Index("ix_paper_trades_strategy_intended_at", "strategy", "intended_at"),)


class DemoOrder(Base):
    __tablename__ = "demo_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_order_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    market_ticker: Mapped[str] = mapped_column(String(64), index=True)
    strategy: Mapped[str | None] = mapped_column(String(32), nullable=True)
    side: Mapped[str] = mapped_column(String(16))
    requested_contracts: Mapped[int] = mapped_column(Integer)
    filled_contracts: Mapped[int] = mapped_column(Integer, default=0)
    requested_yes_price_dollars: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 6), nullable=True
    )
    fair_at_entry: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    q_raw: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    intended_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    avg_fill_price: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    fee_dollars: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    realized_pnl_dollars: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    placed_at: Mapped[datetime] = mapped_column(UtcDateTime())
    last_status_at: Mapped[datetime] = mapped_column(UtcDateTime())
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)


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


class ReconcilerState(Base):
    __tablename__ = "reconciler_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now, onupdate=_utc_now)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False, index=True)
    cash_dollars: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    # collateral basis: cash + sum(market_exposure)
    total_collateral_dollars: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    # mark-to-market: cash + wire portfolio_value; matches Kalshi UI
    portfolio_value_mtm_dollars: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 6), nullable=True
    )
    total_exposure_dollars: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    realized_pnl_dollars: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    fees_paid_dollars: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    open_positions_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False, default=_utc_now)


class GateFailure(Base):
    __tablename__ = "gate_failures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluated_at: Mapped[datetime] = mapped_column(UtcDateTime())
    gate_name: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(String(512))
    mode: Mapped[str] = mapped_column(String(8))
    market_ticker: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(String(64), nullable=True)
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    last_seen_at: Mapped[datetime] = mapped_column(UtcDateTime(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    __table_args__ = (
        Index("ix_gate_failures_evaluated_at_gate_name", "evaluated_at", "gate_name"),
        Index(
            "ux_gate_failures_dedup",
            "gate_name",
            "market_ticker",
            "reason",
            unique=True,
        ),
    )


class WsBookEvent(Base):
    __tablename__ = "ws_book_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(64))
    received_at: Mapped[datetime] = mapped_column(UtcDateTime())
    seq: Mapped[int] = mapped_column(Integer)
    side: Mapped[str] = mapped_column(String(8))
    price: Mapped[Decimal] = mapped_column(DecimalText())
    size: Mapped[Decimal] = mapped_column(DecimalText())
    is_snapshot: Mapped[bool] = mapped_column(Boolean)
    ts_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    __table_args__ = (Index("ix_ws_book_events_ticker_received_at", "ticker", "received_at"),)


class WsTrade(Base):
    __tablename__ = "ws_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(64))
    trade_id: Mapped[str] = mapped_column(String(), nullable=False)
    received_at: Mapped[datetime] = mapped_column(UtcDateTime())
    yes_price: Mapped[Decimal] = mapped_column(DecimalText())
    count: Mapped[Decimal] = mapped_column(DecimalText())
    taker_side: Mapped[str] = mapped_column(String(8))
    ts_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    __table_args__ = (Index("ix_ws_trades_ticker_received_at", "ticker", "received_at"),)


class WsGap(Base):
    __tablename__ = "ws_gaps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(64))
    detected_at: Mapped[datetime] = mapped_column(UtcDateTime())
    last_seq: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    __table_args__ = (Index("ix_ws_gaps_ticker_detected_at", "ticker", "detected_at"),)


class WsHeartbeat(Base):
    __tablename__ = "ws_heartbeats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    beat_at: Mapped[datetime] = mapped_column(UtcDateTime())
    book_events: Mapped[int] = mapped_column(Integer)
    trades: Mapped[int] = mapped_column(Integer)
    gaps: Mapped[int] = mapped_column(Integer)
    subscribed: Mapped[int] = mapped_column(Integer)
    raw_bytes: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    __table_args__ = (Index("ix_ws_heartbeats_beat_at", "beat_at"),)


class WsObsArrival(Base):
    __tablename__ = "ws_obs_arrivals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(16))
    obs_time: Mapped[datetime] = mapped_column(UtcDateTime())
    tmpf: Mapped[Decimal] = mapped_column(DecimalText())
    received_at: Mapped[datetime] = mapped_column(UtcDateTime())
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    __table_args__ = (Index("ix_ws_obs_arrivals_station_obs_time", "station", "obs_time"),)


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


_DEPTH_COLUMNS: frozenset[str] = frozenset(
    {"yes_ask_depth", "yes_bid_depth", "no_ask_depth", "no_bid_depth"}
)


def upgrade_schema(
    db_path: Path | str,
    script_location: Path | None = None,
) -> None:
    abs_db_path = Path(db_path).resolve()
    abs_script_location = (
        script_location.resolve() if script_location is not None else _REPO_ROOT / "alembic"
    )
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(abs_script_location))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{abs_db_path}")

    head = ScriptDirectory.from_config(cfg).get_current_head()

    engine = make_engine(abs_db_path)
    with engine.begin() as conn:
        tables = set(inspect(conn).get_table_names())
        ob_cols: set[str] = set()
        if "orderbook_snapshots" in tables:
            ob_cols = {c["name"] for c in inspect(conn).get_columns("orderbook_snapshots")}

        if "alembic_version" not in tables and "forecasts" not in tables:
            conn.execute(
                text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)")
            )
        elif "alembic_version" not in tables or (
            "alembic_version" in tables
            and conn.execute(text("SELECT version_num FROM alembic_version")).first() is None
        ):
            if "alembic_version" not in tables:
                conn.execute(
                    text(
                        "CREATE TABLE alembic_version "
                        "(version_num VARCHAR(32) NOT NULL PRIMARY KEY)"
                    )
                )
            if "orderbook_snapshots" in tables and _DEPTH_COLUMNS <= ob_cols:
                # physical schema already matches head: a create_all-materialized DB
                # needs head stamp, not baseline.
                conn.execute(
                    text("INSERT INTO alembic_version (version_num) VALUES (:v)"),
                    {"v": head},
                )
            else:
                # baseline, not head: a 0001-era physical schema with empty
                # alembic_version needs to walk 0002+ migrations.
                conn.execute(
                    text("INSERT INTO alembic_version (version_num) VALUES (:v)"),
                    {"v": "0001"},
                )
        else:
            current = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
            if (
                current == "0001"
                and "orderbook_snapshots" in tables
                and _DEPTH_COLUMNS <= ob_cols
                and current != head
            ):
                # physical schema already matches head: a create_all-materialized DB
                # needs head stamp, not baseline.
                conn.execute(
                    text("UPDATE alembic_version SET version_num = :v"),
                    {"v": head},
                )

    engine.dispose()
    alembic_command.upgrade(cfg, "head")
