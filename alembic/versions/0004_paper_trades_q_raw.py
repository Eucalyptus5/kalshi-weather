"""paper_trades q_raw

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-29 12:00:00.000000

paper_trades.q_raw backfills from fair_at_entry under the pre-correction-layer
invariant fair_at_entry == q_raw. A future maintainer who introduces a different
correction layer onto fair_at_entry must not re-run this backfill against shifted
data. demo_orders.q_raw stays nullable because upsert_exchange_record writes
orphan exchange-side rows with no raw value; the translator boundary in
_paper_trade_row_from_demo_row enforces the no-null contract on resurrection.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0004"
down_revision: Union[str, Sequence[str], None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()

    paper_bad = bind.execute(
        sa.text("SELECT COUNT(*) FROM paper_trades WHERE fair_at_entry NOT BETWEEN 0 AND 1")
    ).scalar()
    if paper_bad and paper_bad > 0:
        raise RuntimeError(
            f"migration_0004_aborted: {paper_bad} paper_trades rows have "
            "fair_at_entry out of [0, 1]; backfill would corrupt q_raw"
        )

    demo_bad = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM demo_orders "
            "WHERE fair_at_entry IS NOT NULL AND fair_at_entry NOT BETWEEN 0 AND 1"
        )
    ).scalar()
    if demo_bad and demo_bad > 0:
        raise RuntimeError(
            f"migration_0004_aborted: {demo_bad} demo_orders rows have "
            "fair_at_entry out of [0, 1]; backfill would corrupt q_raw"
        )

    with op.batch_alter_table("paper_trades", schema=None) as batch_op:
        batch_op.add_column(sa.Column("q_raw", sa.Numeric(10, 6), nullable=True))

    op.execute("UPDATE paper_trades SET q_raw = fair_at_entry WHERE q_raw IS NULL")

    with op.batch_alter_table("paper_trades", schema=None) as batch_op:
        batch_op.alter_column("q_raw", existing_type=sa.Numeric(10, 6), nullable=False)

    with op.batch_alter_table("demo_orders", schema=None) as batch_op:
        batch_op.add_column(sa.Column("q_raw", sa.Numeric(10, 6), nullable=True))

    op.execute("UPDATE demo_orders SET q_raw = fair_at_entry WHERE fair_at_entry IS NOT NULL")

    with op.batch_alter_table("gate_failures", schema=None) as batch_op:
        batch_op.alter_column(
            "reason",
            existing_type=sa.String(length=256),
            type_=sa.String(length=512),
        )


def downgrade() -> None:
    with op.batch_alter_table("gate_failures", schema=None) as batch_op:
        batch_op.alter_column(
            "reason",
            existing_type=sa.String(length=512),
            type_=sa.String(length=256),
        )

    with op.batch_alter_table("demo_orders", schema=None) as batch_op:
        batch_op.drop_column("q_raw")

    with op.batch_alter_table("paper_trades", schema=None) as batch_op:
        batch_op.drop_column("q_raw")
