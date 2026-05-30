"""gate failures dedupe

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-31 13:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0006"
down_revision: Union[str, Sequence[str], None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing_cols = {c["name"] for c in sa.inspect(bind).get_columns("gate_failures")}
    add_count = "count" not in existing_cols
    add_last_seen = "last_seen_at" not in existing_cols
    if add_count or add_last_seen:
        with op.batch_alter_table("gate_failures", schema=None) as batch_op:
            if add_count:
                batch_op.add_column(
                    sa.Column("count", sa.Integer(), nullable=False, server_default="1")
                )
            if add_last_seen:
                batch_op.add_column(
                    sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True)
                )

    op.execute(
        """
        CREATE TEMP TABLE _gf_agg AS
        SELECT gate_name, market_ticker, reason,
               COUNT(*) AS c, MAX(created_at) AS lsa
          FROM gate_failures
         GROUP BY gate_name, market_ticker, reason
        """
    )
    op.execute("CREATE INDEX _gf_agg_idx ON _gf_agg (gate_name, market_ticker, reason)")
    op.execute(
        """
        UPDATE gate_failures
           SET count = (
                   SELECT c FROM _gf_agg
                    WHERE _gf_agg.gate_name = gate_failures.gate_name
                      AND (
                          (_gf_agg.market_ticker IS NULL AND gate_failures.market_ticker IS NULL)
                          OR _gf_agg.market_ticker = gate_failures.market_ticker
                      )
                      AND _gf_agg.reason = gate_failures.reason
               ),
               last_seen_at = (
                   SELECT lsa FROM _gf_agg
                    WHERE _gf_agg.gate_name = gate_failures.gate_name
                      AND (
                          (_gf_agg.market_ticker IS NULL AND gate_failures.market_ticker IS NULL)
                          OR _gf_agg.market_ticker = gate_failures.market_ticker
                      )
                      AND _gf_agg.reason = gate_failures.reason
               )
        """
    )
    op.execute("DROP TABLE _gf_agg")

    op.execute("UPDATE gate_failures SET last_seen_at = created_at WHERE last_seen_at IS NULL")

    op.execute(
        """
        DELETE FROM gate_failures
         WHERE id NOT IN (
             SELECT MIN(id)
               FROM gate_failures
              GROUP BY gate_name, market_ticker, reason
         )
        """
    )

    with op.batch_alter_table("gate_failures", schema=None) as batch_op:
        batch_op.alter_column(
            "last_seen_at",
            existing_type=sa.DateTime(timezone=True),
            nullable=False,
        )

    existing_indexes = {ix["name"] for ix in sa.inspect(bind).get_indexes("gate_failures")}
    if "ux_gate_failures_dedup" not in existing_indexes:
        with op.batch_alter_table("gate_failures", schema=None) as batch_op:
            batch_op.create_index(
                "ux_gate_failures_dedup",
                ["gate_name", "market_ticker", "reason"],
                unique=True,
            )


def downgrade() -> None:
    with op.batch_alter_table("gate_failures", schema=None) as batch_op:
        batch_op.drop_index("ux_gate_failures_dedup")
        batch_op.drop_column("last_seen_at")
        batch_op.drop_column("count")
