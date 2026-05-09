"""initial

Revision ID: 0001
Revises:
Create Date: 2026-05-06 01:41:58.594674

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "forecasts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("station", sa.String(length=16), nullable=False),
        sa.Column("run_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_date", sa.Date(), nullable=False),
        sa.Column("members_json", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "station", "run_time", "valid_date", name="uq_forecasts_station_run_valid"
        ),
    )
    with op.batch_alter_table("forecasts", schema=None) as batch_op:
        batch_op.create_index(
            "ix_forecasts_station_valid_date", ["station", "valid_date"], unique=False
        )

    op.create_table(
        "gate_failures",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gate_name", sa.String(length=64), nullable=False),
        sa.Column("reason", sa.String(length=256), nullable=False),
        sa.Column("mode", sa.String(length=8), nullable=False),
        sa.Column("market_ticker", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("gate_failures", schema=None) as batch_op:
        batch_op.create_index(
            "ix_gate_failures_evaluated_at_gate_name",
            ["evaluated_at", "gate_name"],
            unique=False,
        )

    op.create_table(
        "markets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(length=64), nullable=False),
        sa.Column("series", sa.String(length=32), nullable=False),
        sa.Column("event_date", sa.Date(), nullable=False),
        sa.Column("is_monthly", sa.Boolean(), nullable=False),
        sa.Column("is_tail", sa.Boolean(), nullable=False),
        sa.Column("strike_low", sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column("strike_high", sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column("close_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ticker"),
    )
    with op.batch_alter_table("markets", schema=None) as batch_op:
        batch_op.create_index(
            "ix_markets_series_event_date", ["series", "event_date"], unique=False
        )

    op.create_table(
        "orderbook_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(length=64), nullable=False),
        sa.Column("snapshot_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("yes_ask", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("yes_bid", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("no_ask", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("no_bid", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("orderbook_snapshots", schema=None) as batch_op:
        batch_op.create_index(
            "ix_orderbook_snapshots_ticker_snapshot_at",
            ["ticker", "snapshot_at"],
            unique=False,
        )

    op.create_table(
        "paper_trades",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("intended_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("market_ticker", sa.String(length=64), nullable=False),
        sa.Column("side", sa.String(length=16), nullable=False),
        sa.Column("contracts", sa.Integer(), nullable=False),
        sa.Column("simulated_price", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("fee_dollars", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("fair_at_entry", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("strategy", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("paper_trades", schema=None) as batch_op:
        batch_op.create_index(
            "ix_paper_trades_strategy_intended_at",
            ["strategy", "intended_at"],
            unique=False,
        )

    op.create_table(
        "simulated_pnl",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("paper_trade_id", sa.Integer(), nullable=False),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("outcome", sa.String(length=8), nullable=False),
        sa.Column("realized_pnl", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["paper_trade_id"], ["paper_trades.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("simulated_pnl", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_simulated_pnl_paper_trade_id"),
            ["paper_trade_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("simulated_pnl", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_simulated_pnl_paper_trade_id"))
    op.drop_table("simulated_pnl")

    with op.batch_alter_table("gate_failures", schema=None) as batch_op:
        batch_op.drop_index("ix_gate_failures_evaluated_at_gate_name")
    op.drop_table("gate_failures")

    with op.batch_alter_table("orderbook_snapshots", schema=None) as batch_op:
        batch_op.drop_index("ix_orderbook_snapshots_ticker_snapshot_at")
    op.drop_table("orderbook_snapshots")

    with op.batch_alter_table("paper_trades", schema=None) as batch_op:
        batch_op.drop_index("ix_paper_trades_strategy_intended_at")
    op.drop_table("paper_trades")

    with op.batch_alter_table("markets", schema=None) as batch_op:
        batch_op.drop_index("ix_markets_series_event_date")
    op.drop_table("markets")

    with op.batch_alter_table("forecasts", schema=None) as batch_op:
        batch_op.drop_index("ix_forecasts_station_valid_date")
    op.drop_table("forecasts")
