"""Query transformation inside the search pipeline."""

import json

import pytest
from search_helpers import REFUND, SHIPPING, ScriptedLLM, find, search_url, use_llm

from app.providers.errors import ProviderError

HISTORY = [
    {"role": "user", "content": "Tell me about shipping."},
    {"role": "assistant", "content": "Orders ship from Rotterdam within two business days."},
]


def reply(**fields):
    return json.dumps(fields)


def stage(out, name):
    return next(s for s in out["trace"]["stages"] if s["name"] == name)


async def test_rewrite_turns_a_follow_up_into_a_question_that_can_be_searched(client, app, indexed):
    use_llm(app, ScriptedLLM(rewrite=reply(query=SHIPPING)))
    plain = await find(client, indexed, "how long does it take?", mode="hybrid", top_k=3)
    out = await find(
        client, indexed, "how long does it take?", mode="hybrid", top_k=3,
        history=HISTORY, transforms={"rewrite": True},
    )  # fmt: skip
    assert "Rotterdam" in out["hits"][0]["text"]
    assert "Rotterdam" not in plain["hits"][0]["text"]
    t = stage(out, "transform")["details"]
    assert t["original"] == "how long does it take?" and t["primary"] == SHIPPING
    assert t["enabled"] == ["rewrite"] and t["llm_calls"] == 1
    assert out["query"] == "how long does it take?"  # the response echoes what the user typed


async def test_rewrite_without_history_changes_nothing(client, app, indexed):
    llm = use_llm(app, ScriptedLLM(rewrite=reply(query="never used")))
    out = await find(client, indexed, "refund policy", mode="hybrid", transforms={"rewrite": True})
    assert llm.calls == [] and "history" in stage(out, "transform")["details"]["skipped"]["rewrite"]


async def test_multi_query_finds_passages_the_original_wording_misses(client, app, indexed):
    use_llm(
        app, ScriptedLLM(multi_query=reply(queries=[SHIPPING, "dispatch from the Rotterdam site"]))
    )
    plain = await find(client, indexed, "zzzyx qqqqw", mode="hybrid", top_k=3)
    out = await find(
        client, indexed, "zzzyx qqqqw", mode="hybrid", top_k=3,
        transforms={"multi_query": True, "multi_query_count": 2},
    )  # fmt: skip
    top = out["hits"][0]
    assert "Rotterdam" in top["text"] and "Rotterdam" not in plain["hits"][0]["text"]
    assert {"dense[multi_query:1]", "sparse[multi_query:1]"} <= set(top["scores"])
    assert any("multi_query:1" in r and "Orders ship" in r for r in top["reasons"])
    searches = stage(out, "variant_search")["details"]["channels"]
    assert set(searches) >= {"dense[multi_query:1]", "sparse[multi_query:2]"}
    fusion = stage(out, "fusion")["details"]
    assert len(fusion["list_sizes"]) == 2 + 4 and fusion["weights"]["dense[multi_query:1]"] == 1.0


async def test_the_variant_weight_scales_expansions_and_zero_ignores_them(client, app, indexed):
    use_llm(app, ScriptedLLM(multi_query=reply(queries=[SHIPPING])))
    base = await find(client, indexed, REFUND, mode="hybrid", top_k=5)
    off = await find(
        client, indexed, REFUND, mode="hybrid", top_k=5,
        transforms={"multi_query": True}, fusion={"variant_weight": 0},
    )  # fmt: skip
    assert [h["chunk_id"] for h in off["hits"]][:3] == [h["chunk_id"] for h in base["hits"]][:3]
    half = await find(
        client, indexed, REFUND, mode="hybrid", top_k=5,
        transforms={"multi_query": True}, fusion={"variant_weight": 0.5},
    )  # fmt: skip
    assert stage(half, "fusion")["details"]["weights"]["sparse[multi_query:1]"] == 0.5


async def test_hyde_searches_with_the_hypothetical_answer_by_meaning_only(client, app, indexed):
    use_llm(app, ScriptedLLM(hyde=reply(passage=f"{SHIPPING}. Parcels leave quickly.")))
    out = await find(
        client, indexed, "zzzyx qqqqw", mode="hybrid", top_k=3,
        transforms={"hyde": True}, fusion={"variant_weight": 2},
    )  # fmt: skip
    # the question itself carries no signal, so the hypothetical answer decides (weighted up)
    top = out["hits"][0]
    assert "Rotterdam" in top["text"] and "dense[hyde]" in top["scores"]
    assert "sparse[hyde]" not in stage(out, "variant_search")["details"]["channels"]
    assert "sparse[hyde]" not in top["scores"]


async def test_decompose_gives_every_part_of_a_question_its_own_search(client, app, indexed):
    use_llm(
        app,
        ScriptedLLM(decompose=reply(questions=[REFUND, SHIPPING])),
    )
    question = "zzzyx qqqqw and also vvvvk jjjjh"  # carries no signal of its own
    out = await find(
        client, indexed, question, mode="hybrid", top_k=4, transforms={"decompose": True}
    )
    texts = " ".join(h["text"] for h in out["hits"][:2])
    assert "Refunds are issued" in texts and "Rotterdam" in texts
    assert {"dense[sub:1]", "dense[sub:2]"} <= set(
        stage(out, "variant_search")["details"]["channels"]
    )


async def test_step_back_adds_a_broader_search(client, app, indexed):
    use_llm(
        app, ScriptedLLM(step_back=reply(query="What does the customer handbook say about orders?"))
    )
    out = await find(client, indexed, "shipping", mode="hybrid", transforms={"step_back": True})
    assert "dense[step_back]" in stage(out, "variant_search")["details"]["channels"]


async def test_transforms_work_in_dense_and_sparse_modes(client, app, indexed):
    use_llm(app, ScriptedLLM(multi_query=reply(queries=[SHIPPING])))
    dense = await find(client, indexed, "zzzyx", mode="dense", transforms={"multi_query": True})
    assert set(stage(dense, "variant_search")["details"]["channels"]) == {"dense[multi_query:1]"}
    sparse = await find(client, indexed, "zzzyx", mode="sparse", transforms={"multi_query": True})
    assert set(stage(sparse, "variant_search")["details"]["channels"]) == {"sparse[multi_query:1]"}
    assert "Rotterdam" in sparse["hits"][0]["text"]


async def test_a_broken_llm_degrades_to_a_plain_search_with_warnings(client, app, indexed):
    use_llm(app, ScriptedLLM(multi_query=ProviderError("llm", "HTTP 503"), hyde="not json"))
    plain = await find(client, indexed, REFUND, mode="hybrid", top_k=3)
    out = await find(
        client,
        indexed,
        REFUND,
        mode="hybrid",
        top_k=3,
        transforms={"multi_query": True, "hyde": True},
    )
    assert [h["chunk_id"] for h in out["hits"]] == [h["chunk_id"] for h in plain["hits"]]
    assert len(out["warnings"]) == 2 and "503" in out["warnings"][0] + out["warnings"][1]
    assert "variant_search" not in [s["name"] for s in out["trace"]["stages"]]


async def test_transforms_respect_filters_and_scope(client, app, indexed):
    use_llm(app, ScriptedLLM(multi_query=reply(queries=[SHIPPING])))
    out = await find(
        client, indexed, "zzzyx", mode="hybrid", top_k=10, transforms={"multi_query": True},
        filters=[{"field": "language", "op": "eq", "value": "de"}],
    )  # fmt: skip
    assert out["hits"] == []


@pytest.mark.parametrize(
    "body",
    [
        {"transforms": {"multi_query_count": 0}},
        {"transforms": {"multi_query_count": 6}},
        {"transforms": {"max_sub_questions": 7}},
        {"transforms": {"rewrite": "sometimes"}},
        {"history": [{"role": "system", "content": "x"}]},
        {"history": [{"role": "user", "content": ""}]},
        {"history": [{"role": "user", "content": "x"}] * 21},
    ],
)
async def test_invalid_transform_options_are_422(client, indexed, body):
    r = await client.post(
        search_url(indexed), json={"query": "x", **body}, headers=indexed["headers"]
    )
    assert r.status_code == 422
