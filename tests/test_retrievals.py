"""The retrieval trace and the saved-search history behind the Inspector."""

import json
import uuid

import pytest
from search_helpers import REFUND, find, search_url
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import Retrieval, Role, User, WorkspaceMember
from app.retrieve.history import strip_text


def stage(out, name):
    return next(s for s in out["trace"]["stages"] if s["name"] == name)


def url(space, rid=""):
    return f"/v1/workspaces/{space['ws']}/retrievals" + (f"/{rid}" if rid else "")


async def detail(client, space, rid):
    r = await client.get(url(space, rid), headers=space["headers"])
    assert r.status_code == 200, r.text
    return r.json()


async def count_rows(app):
    async with app.state.sessionmaker() as s:
        return (await s.execute(select(func.count()).select_from(Retrieval))).scalar_one()


async def add_member(client, app, space, email, role):
    from search_helpers import POLICY  # noqa: F401  (keeps the helper import in one place)

    headers = await _signup(client, email)
    async with app.state.sessionmaker() as s:
        uid = (await s.execute(select(User.id).where(User.email == email))).scalar_one()
        s.add(WorkspaceMember(workspace_id=uuid.UUID(space["ws"]), user_id=uid, role=role))
        await s.commit()
    return {**space, "headers": headers}


async def _signup(client, email):
    pw = "correct-horse-9"
    await client.post("/v1/auth/register", json={"email": email, "password": pw})
    r = await client.post("/v1/auth/login", json={"email": email, "password": pw})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ---- the trace object -------------------------------------------------------------------------


async def test_the_trace_is_a_versioned_self_contained_object(client, indexed):
    out = await find(client, indexed, REFUND, mode="hybrid", top_k=3)
    trace = out["trace"]
    assert trace["schema_version"] == 1 and trace["query"] == REFUND and trace["total_ms"] > 0
    assert trace["options"]["mode"] == "hybrid" and trace["options"]["top_k"] == 3
    assert trace["options"]["fusion"]["method"] == "rrf"
    json.dumps(trace)  # plain JSON all the way down
    # every passage the trace mentions can be shown from the trace alone
    mentioned = set()
    for s in trace["stages"]:
        d = s["details"]
        for ch in d.get("channels", {}).values():
            mentioned |= {e["chunk_id"] for e in ch.get("top", [])}
        mentioned |= {e["chunk_id"] for e in d.get("top", [])}
        mentioned |= {e["chunk_id"] for e in d.get("hits", [])}
    assert mentioned and mentioned <= set(trace["chunks"])
    chunk = trace["chunks"][out["hits"][0]["chunk_id"]]
    assert chunk["title"] == "policy" and 0 < len(chunk["preview"]) <= 160
    assert set(chunk) == {"title", "page", "heading_path", "preview"}


async def test_each_ranking_stage_lists_its_best_candidates_with_scores(client, indexed):
    out = await find(client, indexed, REFUND, mode="hybrid", top_k=3)
    channels = stage(out, "resolve")["details"]["channels"]
    for label in ("dense", "sparse"):
        top = channels[label]["top"]
        assert top[0]["rank"] == 1 and top[0]["via"] == "chunk"
        assert [e["score"] for e in top] == sorted((e["score"] for e in top), reverse=True)
    fused = stage(out, "fusion")["details"]["top"]
    assert fused[0]["chunk_id"] == out["hits"][0]["chunk_id"]
    assert fused[0]["ranks"] == {"dense": 1, "sparse": 1}
    assert abs(sum(fused[0]["contributions"].values()) - fused[0]["score"]) < 1e-5
    final = stage(out, "result")["details"]["hits"]
    assert [h["chunk_id"] for h in final] == [h["chunk_id"] for h in out["hits"]]
    assert final[0]["scores"] == out["hits"][0]["scores"]


async def test_snapshots_are_bounded(client, indexed):
    out = await find(client, indexed, "customer care", mode="hybrid", top_k=50)
    assert len(stage(out, "fusion")["details"]["top"]) <= 20
    assert all(len(c["top"]) <= 20 for c in stage(out, "resolve")["details"]["channels"].values())


# ---- saving and reading back -----------------------------------------------------------------


async def test_a_search_is_saved_and_can_be_read_back_without_passage_text(client, app, indexed):
    out = await find(
        client, indexed, REFUND, mode="hybrid", top_k=3,
        context={"enabled": True, "pool": 5},
    )  # fmt: skip
    rid = out["retrieval_id"]
    assert rid and await count_rows(app) == 1
    saved = await detail(client, indexed, rid)
    assert saved["query"] == REFUND and saved["mode"] == "hybrid" and saved["hit_count"] == 3
    assert saved["took_ms"] > 0 and saved["user_id"]
    record = saved["record"]
    assert record["trace"] == out["trace"]
    assert [h["chunk_id"] for h in record["hits"]] == [h["chunk_id"] for h in out["hits"]]
    assert all("text" not in h for h in record["hits"])
    assert record["hits"][0]["scores"] == out["hits"][0]["scores"]
    assert record["hits"][0]["reasons"] == out["hits"][0]["reasons"]
    assert record["context"]["blocks"] and all("text" not in b for b in record["context"]["blocks"])
    assert record["context"]["blocks"][0]["tokens"] > 0 and "retrieval_id" in record


async def test_record_false_saves_nothing(client, app, indexed):
    out = await find(client, indexed, "refunds", record=False)
    assert out["retrieval_id"] is None and await count_rows(app) == 0


async def test_a_zero_history_limit_turns_saving_off(client, app, indexed):
    app.state.settings.retrieval_history_limit = 0
    out = await find(client, indexed, "refunds")
    assert out["retrieval_id"] is None and await count_rows(app) == 0


async def test_only_the_newest_searches_are_kept(client, app, indexed):
    app.state.settings.retrieval_history_limit = 3
    ids = [(await find(client, indexed, f"refund {i}"))["retrieval_id"] for i in range(5)]
    assert await count_rows(app) == 3
    listed = (await client.get(url(indexed), headers=indexed["headers"])).json()
    assert [i["id"] for i in listed["items"]] == ids[::-1][:3] and listed["total"] == 3
    gone = await client.get(url(indexed, ids[0]), headers=indexed["headers"])
    assert gone.status_code == 404


async def test_a_history_write_failure_does_not_fail_the_search(client, indexed, monkeypatch):
    async def broken(*a, **kw):
        raise SQLAlchemyError("disk full")

    monkeypatch.setattr("app.api.search.save_retrieval", broken)
    out = await find(client, indexed, "refunds")
    assert out["hits"] and out["retrieval_id"] is None
    assert "could not be saved" in out["warnings"][0]


def test_strip_text_removes_text_but_not_the_original():
    response = {
        "hits": [{"chunk_id": "a", "text": "secret"}],
        "context": {"blocks": [{"label": "S1", "text": "secret"}], "tokens": 1},
        "query": "q",
    }
    out = strip_text(response)
    assert out["hits"] == [{"chunk_id": "a"}] and out["context"]["blocks"] == [{"label": "S1"}]
    assert response["hits"][0]["text"] == "secret"  # the input is untouched
    assert strip_text({"hits": [], "context": None})["context"] is None


# ---- listing, access, deleting ----------------------------------------------------------------


async def test_the_list_is_newest_first_and_paged(client, indexed):
    ids = [(await find(client, indexed, f"question {i}"))["retrieval_id"] for i in range(4)]
    page = await client.get(url(indexed) + "?limit=2", headers=indexed["headers"])
    body = page.json()
    assert [i["id"] for i in body["items"]] == ids[::-1][:2] and body["total"] == 4
    assert {"id", "query", "mode", "hit_count", "took_ms", "created_at", "user_id"} <= set(
        body["items"][0]
    )
    second = await client.get(url(indexed) + "?limit=2&offset=2", headers=indexed["headers"])
    assert [i["id"] for i in second.json()["items"]] == ids[::-1][2:]


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1"])
async def test_bad_paging_is_422(client, indexed, query):
    r = await client.get(url(indexed) + "?" + query, headers=indexed["headers"])
    assert r.status_code == 422


async def test_members_see_their_own_searches_and_admins_see_all(client, app, indexed):
    mine = (await find(client, indexed, "owner question"))["retrieval_id"]
    member = await add_member(client, app, indexed, "mia@example.com", Role.MEMBER)
    hers = (await find(client, member, "member question"))["retrieval_id"]
    seen = (await client.get(url(member), headers=member["headers"])).json()
    assert [i["id"] for i in seen["items"]] == [hers]
    assert (await client.get(url(member, mine), headers=member["headers"])).status_code == 404
    owner_view = (await client.get(url(indexed), headers=indexed["headers"])).json()
    assert {i["id"] for i in owner_view["items"]} == {mine, hers} and owner_view["total"] == 2
    assert (await client.get(url(indexed, hers), headers=indexed["headers"])).status_code == 200


async def test_other_workspaces_and_anonymous_callers_get_nothing(client, indexed, signup):
    rid = (await find(client, indexed, "refunds"))["retrieval_id"]
    stranger = await signup("eve@example.com")
    assert (await client.get(url(indexed), headers=stranger)).status_code == 404
    assert (await client.get(url(indexed, rid), headers=stranger)).status_code == 404
    assert (await client.get(url(indexed))).status_code == 401
    assert (
        await client.get(url(indexed, str(uuid.uuid4())), headers=indexed["headers"])
    ).status_code == 404
    assert (
        await client.get(url(indexed, "not-a-uuid"), headers=indexed["headers"])
    ).status_code == 422


async def test_a_saved_search_can_be_deleted_by_its_owner_only(client, app, indexed):
    rid = (await find(client, indexed, "refunds"))["retrieval_id"]
    member = await add_member(client, app, indexed, "mia@example.com", Role.MEMBER)
    assert (await client.delete(url(member, rid), headers=member["headers"])).status_code == 404
    assert await count_rows(app) == 1
    assert (await client.delete(url(indexed, rid), headers=indexed["headers"])).status_code == 204
    assert await count_rows(app) == 0
    assert (await client.delete(url(indexed, rid), headers=indexed["headers"])).status_code == 404


async def test_deleting_the_workspace_deletes_its_saved_searches(client, app, indexed):
    await find(client, indexed, "refunds")
    assert await count_rows(app) == 1
    r = await client.delete(f"/v1/workspaces/{indexed['ws']}", headers=indexed["headers"])
    assert r.status_code == 204
    assert await count_rows(app) == 0


async def test_search_still_validates_the_record_flag(client, indexed):
    r = await client.post(
        search_url(indexed), json={"query": "x", "record": "maybe"}, headers=indexed["headers"]
    )
    assert r.status_code == 422
