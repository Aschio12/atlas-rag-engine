"""collection chunking config

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-04 11:34:50.842649

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    with op.batch_alter_table("collections", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("chunking_config", sa.JSON(), server_default=sa.text("'{}'"), nullable=False)
        )


def downgrade() -> None:
    """Downgrade schema."""

    with op.batch_alter_table("collections", schema=None) as batch_op:
        batch_op.drop_column("chunking_config")
