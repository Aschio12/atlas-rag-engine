"""job options and enrichment config

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-04 12:21:16.734375

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    with op.batch_alter_table("collections", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "enrichment_config", sa.JSON(), server_default=sa.text("'{}'"), nullable=False
            )
        )

    with op.batch_alter_table("ingest_jobs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("options", sa.JSON(), server_default=sa.text("'{}'"), nullable=False)
        )


def downgrade() -> None:
    """Downgrade schema."""

    with op.batch_alter_table("ingest_jobs", schema=None) as batch_op:
        batch_op.drop_column("options")

    with op.batch_alter_table("collections", schema=None) as batch_op:
        batch_op.drop_column("enrichment_config")
