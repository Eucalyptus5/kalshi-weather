"""ws recorder tables

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-07 12:00:00.000000

Weather books are low-traffic; if a smoke day exceeds ~500k rows/day across all series,
add per-day pruning of non-DEN/CHI series before the recorder deploy signs off.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0009"
down_revision: Union[str, Sequence[str], None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ws_book_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("ticker", sa.String(64), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("price", sa.String(), nullable=False),
        sa.Column("size", sa.String(), nullable=False),
        sa.Column("is_snapshot", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_ws_book_events_ticker_received_at", "ws_book_events", ["ticker", "received_at"]
    )
    op.create_table(
        "ws_trades",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("ticker", sa.String(64), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("yes_price", sa.String(), nullable=False),
        sa.Column("count", sa.String(), nullable=False),
        sa.Column("taker_side", sa.String(8), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ws_trades_ticker_received_at", "ws_trades", ["ticker", "received_at"])
    op.create_table(
        "ws_gaps",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("ticker", sa.String(64), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seq", sa.Integer(), nullable=False),
        sa.Column("reason", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ws_gaps_ticker_detected_at", "ws_gaps", ["ticker", "detected_at"])


def downgrade() -> None:
    op.drop_index("ix_ws_gaps_ticker_detected_at", table_name="ws_gaps")
    op.drop_table("ws_gaps")
    op.drop_index("ix_ws_trades_ticker_received_at", table_name="ws_trades")
    op.drop_table("ws_trades")
    op.drop_index("ix_ws_book_events_ticker_received_at", table_name="ws_book_events")
    op.drop_table("ws_book_events")
