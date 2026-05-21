"""demo orders

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-28 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "demo_orders",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("client_order_id", sa.String(length=64), nullable=False),
        sa.Column("exchange_order_id", sa.String(length=64), nullable=True),
        sa.Column("market_ticker", sa.String(length=64), nullable=False),
        sa.Column("strategy", sa.String(length=32), nullable=True),
        sa.Column("side", sa.String(length=16), nullable=False),
        sa.Column("requested_contracts", sa.Integer(), nullable=False),
        sa.Column("filled_contracts", sa.Integer(), nullable=False),
        sa.Column("requested_yes_price_dollars", sa.Numeric(10, 6), nullable=True),
        sa.Column("fair_at_entry", sa.Numeric(10, 6), nullable=True),
        sa.Column("intended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("avg_fill_price", sa.Numeric(10, 6), nullable=True),
        sa.Column("fee_dollars", sa.Numeric(10, 6), nullable=True),
        sa.Column("realized_pnl_dollars", sa.Numeric(10, 6), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("placed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_status_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("client_order_id"),
        sa.UniqueConstraint("exchange_order_id"),
    )
    op.create_index("ix_demo_orders_client_order_id", "demo_orders", ["client_order_id"])
    op.create_index("ix_demo_orders_market_ticker", "demo_orders", ["market_ticker"])
    op.create_index("ix_demo_orders_status", "demo_orders", ["status"])

    with op.batch_alter_table("paper_trades", schema=None) as batch_op:
        batch_op.add_column(sa.Column("demo_order_client_id", sa.String(length=64), nullable=True))
        batch_op.create_index(
            "ix_paper_trades_demo_order_client_id",
            ["demo_order_client_id"],
            unique=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("paper_trades", schema=None) as batch_op:
        batch_op.drop_index("ix_paper_trades_demo_order_client_id")
        batch_op.drop_column("demo_order_client_id")

    op.drop_index("ix_demo_orders_status", table_name="demo_orders")
    op.drop_index("ix_demo_orders_market_ticker", table_name="demo_orders")
    op.drop_index("ix_demo_orders_client_order_id", table_name="demo_orders")
    op.drop_table("demo_orders")
