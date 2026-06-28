"""ws obs arrivals table

Revision ID: 0012
Revises: 0011
Create Date: 2026-07-07 16:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0012"
down_revision: Union[str, Sequence[str], None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ws_obs_arrivals",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("station", sa.String(16), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("obs_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("tmpf", sa.String(), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_ws_obs_arrivals_station_obs_time", "ws_obs_arrivals", ["station", "obs_time"]
    )


def downgrade() -> None:
    op.drop_index("ix_ws_obs_arrivals_station_obs_time", table_name="ws_obs_arrivals")
    op.drop_table("ws_obs_arrivals")
