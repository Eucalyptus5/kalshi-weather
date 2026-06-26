"""ws ts_ms and trade_id columns

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-07 13:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0010"
down_revision: Union[str, Sequence[str], None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("ws_book_events", schema=None) as batch_op:
        batch_op.add_column(sa.Column("ts_ms", sa.Integer(), nullable=True))

    with op.batch_alter_table("ws_trades", schema=None) as batch_op:
        batch_op.add_column(sa.Column("trade_id", sa.String(), nullable=False))
        batch_op.add_column(sa.Column("ts_ms", sa.Integer(), nullable=False))


def downgrade() -> None:
    with op.batch_alter_table("ws_trades", schema=None) as batch_op:
        batch_op.drop_column("ts_ms")
        batch_op.drop_column("trade_id")

    with op.batch_alter_table("ws_book_events", schema=None) as batch_op:
        batch_op.drop_column("ts_ms")
