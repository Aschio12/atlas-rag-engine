"""job queue lease columns

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-03 16:17:37.119997

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    with op.batch_alter_table("ingest_jobs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("run_after", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""

    with op.batch_alter_table("ingest_jobs", schema=None) as batch_op:
        batch_op.drop_column("locked_until")
        batch_op.drop_column("run_after")
