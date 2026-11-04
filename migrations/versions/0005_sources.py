"""sources

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-04 12:41:03.963467

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table(
        "sources",
        sa.Column("collection_id", sa.Uuid(), nullable=False),
        sa.Column(
            "type",
            sa.Enum("web", "github", name="sourcetype", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "idle",
                "running",
                "done",
                "failed",
                name="sourcestatus",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("stats", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["collection_id"],
            ["collections.id"],
            name=op.f("fk_sources_collection_id_collections"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sources")),
        sa.UniqueConstraint("collection_id", "name", name=op.f("uq_sources_collection_id")),
    )
    with op.batch_alter_table("sources", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_sources_collection_id"), ["collection_id"], unique=False
        )

    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.add_column(sa.Column("source_id", sa.Uuid(), nullable=True))
        batch_op.create_index(batch_op.f("ix_documents_source_id"), ["source_id"], unique=False)
        batch_op.create_foreign_key(
            batch_op.f("fk_documents_source_id_sources"),
            "sources",
            ["source_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    """Downgrade schema."""

    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f("fk_documents_source_id_sources"), type_="foreignkey")
        batch_op.drop_index(batch_op.f("ix_documents_source_id"))
        batch_op.drop_column("source_id")

    with op.batch_alter_table("sources", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_sources_collection_id"))

    op.drop_table("sources")
