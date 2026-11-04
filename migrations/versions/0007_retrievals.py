"""saved retrievals

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-04 15:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table(
        "retrievals",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("hit_count", sa.Integer(), nullable=False),
        sa.Column("took_ms", sa.Float(), nullable=False),
        sa.Column("record", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_retrievals_user_id_users"), ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_retrievals_workspace_id_workspaces"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_retrievals")),
    )
    op.create_index(op.f("ix_retrievals_created_at"), "retrievals", ["created_at"])
    op.create_index(op.f("ix_retrievals_user_id"), "retrievals", ["user_id"])
    op.create_index(op.f("ix_retrievals_workspace_id"), "retrievals", ["workspace_id"])


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index(op.f("ix_retrievals_workspace_id"), table_name="retrievals")
    op.drop_index(op.f("ix_retrievals_user_id"), table_name="retrievals")
    op.drop_index(op.f("ix_retrievals_created_at"), table_name="retrievals")
    op.drop_table("retrievals")
