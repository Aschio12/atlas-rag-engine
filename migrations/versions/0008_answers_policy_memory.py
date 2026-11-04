"""workspace policy, conversation summaries, message metadata, memory facts

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-04 18:20:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | Sequence[str] | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    op.add_column(
        "workspaces",
        sa.Column("policy", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )
    op.add_column(
        "conversations",
        sa.Column("summary", sa.Text(), nullable=False, server_default=sa.text("''")),
    )
    op.add_column(
        "conversations",
        sa.Column("summarized_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column("messages", sa.Column("metadata", sa.JSON(), nullable=True))
    op.create_table(
        "memory_facts",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("text", sa.String(length=500), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_by", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["users.id"],
            name=op.f("fk_memory_facts_created_by_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_memory_facts_workspace_id_workspaces"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_memory_facts")),
    )
    op.create_index(op.f("ix_memory_facts_workspace_id"), "memory_facts", ["workspace_id"])


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index(op.f("ix_memory_facts_workspace_id"), table_name="memory_facts")
    op.drop_table("memory_facts")
    op.drop_column("messages", "metadata")
    op.drop_column("conversations", "summarized_count")
    op.drop_column("conversations", "summary")
    op.drop_column("workspaces", "policy")
