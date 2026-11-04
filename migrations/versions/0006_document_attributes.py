"""document attributes

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-04 14:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    op.add_column(
        "documents",
        sa.Column("attributes", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_column("documents", "attributes")
