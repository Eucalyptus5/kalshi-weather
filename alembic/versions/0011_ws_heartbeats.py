"""ws heartbeats table

Revision ID: 0011
Revises: 0010
Create Date: 2026-07-07 14:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0011"
down_revision: Union[str, Sequence[str], None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ws_heartbeats",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("beat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("book_events", sa.Integer(), nullable=False),
        sa.Column("trades", sa.Integer(), nullable=False),
        sa.Column("gaps", sa.Integer(), nullable=False),
        sa.Column("subscribed", sa.Integer(), nullable=False),
        sa.Column("raw_bytes", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ws_heartbeats_beat_at", "ws_heartbeats", ["beat_at"])


def downgrade() -> None:
    op.drop_index("ix_ws_heartbeats_beat_at", table_name="ws_heartbeats")
    op.drop_table("ws_heartbeats")
