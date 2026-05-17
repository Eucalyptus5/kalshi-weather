"""orderbook depth and trade features

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-27 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("orderbook_snapshots", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("yes_ask_depth", sa.Integer(), nullable=True, server_default="0")
        )
        batch_op.add_column(
            sa.Column("yes_bid_depth", sa.Integer(), nullable=True, server_default="0")
        )
        batch_op.add_column(
            sa.Column("no_ask_depth", sa.Integer(), nullable=True, server_default="0")
        )
        batch_op.add_column(
            sa.Column("no_bid_depth", sa.Integer(), nullable=True, server_default="0")
        )

    with op.batch_alter_table("paper_trades", schema=None) as batch_op:
        batch_op.add_column(sa.Column("attempted_contracts", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("ensemble_spread_sigma_t", sa.Numeric(10, 6), nullable=True))
        batch_op.add_column(sa.Column("lead_time_hours", sa.Numeric(10, 4), nullable=True))
        batch_op.add_column(sa.Column("nbm_divergence", sa.Numeric(10, 6), nullable=True))

    op.execute(
        "UPDATE paper_trades SET attempted_contracts = contracts WHERE attempted_contracts IS NULL"
    )

    with op.batch_alter_table("gate_failures", schema=None) as batch_op:
        batch_op.add_column(sa.Column("notes", sa.String(length=64), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("gate_failures", schema=None) as batch_op:
        batch_op.drop_column("notes")

    with op.batch_alter_table("paper_trades", schema=None) as batch_op:
        batch_op.drop_column("nbm_divergence")
        batch_op.drop_column("lead_time_hours")
        batch_op.drop_column("ensemble_spread_sigma_t")
        batch_op.drop_column("attempted_contracts")

    with op.batch_alter_table("orderbook_snapshots", schema=None) as batch_op:
        batch_op.drop_column("no_bid_depth")
        batch_op.drop_column("no_ask_depth")
        batch_op.drop_column("yes_bid_depth")
        batch_op.drop_column("yes_ask_depth")
