"""portfolio snapshots

Revision ID: 0007
Revises: 0006
Create Date: 2026-06-02 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0007"
down_revision: Union[str, Sequence[str], None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "portfolio_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("snapshot_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cash_dollars", sa.Numeric(12, 6), nullable=False),
        sa.Column("portfolio_value_dollars", sa.Numeric(12, 6), nullable=False),
        sa.Column("total_exposure_dollars", sa.Numeric(12, 6), nullable=False),
        sa.Column("realized_pnl_dollars", sa.Numeric(12, 6), nullable=False),
        sa.Column("fees_paid_dollars", sa.Numeric(12, 6), nullable=False),
        sa.Column("open_positions_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_portfolio_snapshots_snapshot_at", "portfolio_snapshots", ["snapshot_at"])


def downgrade() -> None:
    op.drop_index("ix_portfolio_snapshots_snapshot_at", table_name="portfolio_snapshots")
    op.drop_table("portfolio_snapshots")
