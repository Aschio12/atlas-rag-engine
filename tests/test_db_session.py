import pytest
from sqlalchemy import text

from app.core.config import Settings
from app.db.models import User, Workspace
from app.db.session import make_engine, make_sessionmaker


async def test_sqlite_engine_enforces_foreign_keys(app):
    async with app.state.engine.connect() as conn:
        assert (await conn.execute(text("PRAGMA foreign_keys"))).scalar() == 1


async def test_in_memory_sqlite_shares_one_connection_across_sessions(app):
    maker = app.state.sessionmaker
    async with maker() as s:
        s.add(User(email="a@b.co", password_hash="x"))
        await s.commit()
    async with maker() as s:
        assert (await s.execute(text("SELECT count(*) FROM users"))).scalar() == 1


async def test_uncommitted_work_is_discarded_when_session_closes(app):
    maker = app.state.sessionmaker
    async with maker() as s:  # what session_dep does: no commit -> discarded on close
        s.add(User(email="gone@b.co", password_hash="x"))
        await s.flush()
    async with maker() as s:
        assert (await s.execute(text("SELECT count(*) FROM users"))).scalar() == 0


async def test_database_restricts_deleting_a_workspace_owner(app):
    maker = app.state.sessionmaker
    async with maker() as s:
        u = User(email="o@b.co", password_hash="x")
        s.add(u)
        await s.flush()
        s.add(Workspace(name="W", slug="w", owner_id=u.id))
        await s.commit()
    async with maker() as s:
        # RESTRICT: deleting a user who owns a workspace must fail at the database level
        with pytest.raises(Exception, match="FOREIGN KEY"):
            await s.execute(text("DELETE FROM users"))


def test_engine_for_default_url_builds_lazily_without_connecting():
    engine = make_engine(Settings(_env_file=None))
    assert make_sessionmaker(engine) is not None
