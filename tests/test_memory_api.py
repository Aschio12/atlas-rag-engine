import uuid

import pytest
from answer_helpers import SpyLLM, add_doc, add_member, ask, use_llm

from app.db.models import Role


def url(space, fact=""):
    return f"/v1/workspaces/{space['ws']}/memory" + (f"/{fact}" if fact else "")


async def add(client, space, text="Answer in British English."):
    r = await client.post(url(space), json={"text": text}, headers=space["headers"])
    assert r.status_code == 201, r.text
    return r.json()


async def test_facts_are_added_listed_edited_and_deleted(client, space):
    fact = await add(client, space, "  Our fiscal year starts in April. ")
    assert fact["text"] == "Our fiscal year starts in April." and fact["enabled"] is True
    assert fact["created_by"] is not None
    other = await add(client, space, "Prefer short answers.")
    listed = (await client.get(url(space), headers=space["headers"])).json()
    assert [f["text"] for f in listed] == [fact["text"], other["text"]]  # oldest first
    patched = await client.patch(
        url(space, fact["id"]),
        json={"text": "Fiscal year starts in March."},
        headers=space["headers"],
    )
    assert patched.json()["text"] == "Fiscal year starts in March."
    off = await client.patch(
        url(space, fact["id"]), json={"enabled": False}, headers=space["headers"]
    )
    assert off.json()["enabled"] is False and off.json()["text"] == "Fiscal year starts in March."
    assert (
        await client.delete(url(space, fact["id"]), headers=space["headers"])
    ).status_code == 204
    assert [f["id"] for f in (await client.get(url(space), headers=space["headers"])).json()] == [
        other["id"]
    ]


@pytest.mark.parametrize("body", [{"text": ""}, {"text": "   "}, {"text": "x" * 501}, {}])
async def test_invalid_facts_are_422(client, space, body):
    r = await client.post(url(space), json=body, headers=space["headers"])
    assert r.status_code == 422


async def test_the_number_of_facts_is_limited(client, app, space):
    app.state.settings.memory_max_facts = 2
    await add(client, space, "one")
    await add(client, space, "two")
    r = await client.post(url(space), json={"text": "three"}, headers=space["headers"])
    assert r.status_code == 409 and "at most 2" in r.json()["detail"]


async def test_members_write_viewers_only_read(client, app, space):
    viewer = await add_member(client, app, space, "vera@example.com", Role.VIEWER)
    member = await add_member(client, app, space, "mia@example.com", Role.MEMBER)
    fact = await add(client, member, "Written by a member.")
    assert (await client.get(url(viewer), headers=viewer["headers"])).status_code == 200
    assert (
        await client.post(url(viewer), json={"text": "x"}, headers=viewer["headers"])
    ).status_code == 403
    assert (
        await client.patch(
            url(viewer, fact["id"]), json={"enabled": False}, headers=viewer["headers"]
        )
    ).status_code == 403
    assert (
        await client.delete(url(viewer, fact["id"]), headers=viewer["headers"])
    ).status_code == 403


async def test_facts_are_private_to_their_workspace(client, space, signup):
    fact = await add(client, space)
    stranger = await signup("eve@example.com")
    assert (await client.get(url(space), headers=stranger)).status_code == 404
    assert (await client.delete(url(space, fact["id"]), headers=stranger)).status_code == 404
    assert (
        await client.patch(
            url(space, str(uuid.uuid4())), json={"enabled": False}, headers=space["headers"]
        )
    ).status_code == 404


async def test_enabled_facts_reach_the_prompt_and_disabled_ones_do_not(client, app, space):
    await add_doc(client, app, space)
    a = await add(client, space, "Answer in British English.")
    await add(client, space, "Our fiscal year starts in April.")
    await client.patch(url(space, a["id"]), json={"enabled": False}, headers=space["headers"])
    llm = use_llm(app, SpyLLM())
    out = await ask(client, space, "How long do refunds take?")
    (call,) = llm.answer_calls()
    system = call[0].content
    assert "## Known facts" in system and "- Our fiscal year starts in April." in system
    assert "British English" not in system
    assert out["memory"] == {"facts": 1, "summary": False, "turns": 0}


async def test_without_facts_the_prompt_has_no_facts_section(client, app, space):
    await add_doc(client, app, space)
    llm = use_llm(app, SpyLLM())
    await ask(client, space, "How long do refunds take?")
    assert "Known facts" not in llm.answer_calls()[0][0].content


async def test_deleting_the_workspace_deletes_its_facts(client, app, space):
    await add(client, space)
    r = await client.delete(f"/v1/workspaces/{space['ws']}", headers=space["headers"])
    assert r.status_code == 204
    from sqlalchemy import func, select

    from app.db.models import MemoryFact

    async with app.state.sessionmaker() as s:
        assert (await s.execute(select(func.count()).select_from(MemoryFact))).scalar_one() == 0
