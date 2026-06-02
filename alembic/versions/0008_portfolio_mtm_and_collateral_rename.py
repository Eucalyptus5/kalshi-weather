"""portfolio mtm and collateral rename

Revision ID: 0008
Revises: 0007
Create Date: 2026-06-02 18:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0008"
down_revision: Union[str, Sequence[str], None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("portfolio_snapshots", schema=None) as batch_op:
        batch_op.alter_column(
            "portfolio_value_dollars",
            new_column_name="total_collateral_dollars",
            existing_type=sa.Numeric(12, 6),
            existing_nullable=False,
        )
        batch_op.add_column(
            sa.Column("portfolio_value_mtm_dollars", sa.Numeric(12, 6), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("portfolio_snapshots", schema=None) as batch_op:
        batch_op.drop_column("portfolio_value_mtm_dollars")
        batch_op.alter_column(
            "total_collateral_dollars",
            new_column_name="portfolio_value_dollars",
            existing_type=sa.Numeric(12, 6),
            existing_nullable=False,
        )
