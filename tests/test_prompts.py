import pytest

from app.generate import prompts
from app.generate.prompts import Prompt, PromptError, load_directory, parse

GOOD = """---
name: greet
version: 2
description: Say hello
variables: who, extra
---
Hello {{who}}.{{ extra }}
"""


def test_the_shipped_prompts_load_and_are_consistent():
    available = prompts.available()
    assert {"answer", "summarize"} <= set(available)
    for name, versions in available.items():
        for v in versions:
            p = prompts.get(name, v)
            assert p.name == name and p.version == v and p.description and p.text.strip()
            assert p.variables  # every shipped prompt is a template


def test_a_prompt_is_parsed_with_its_header():
    p = parse(GOOD, "greet.v2.md")
    assert (p.name, p.version, p.description) == ("greet", 2, "Say hello")
    assert p.variables == ("who", "extra") and p.text == "Hello {{who}}.{{ extra }}\n"


def test_rendering_substitutes_literally_and_collapses_empty_sections():
    p = parse(GOOD, "greet.v2.md")
    assert p.render(who="Ada", extra="") == "Hello Ada.\n"
    assert (
        p.render(who="{{extra}}", extra="!") == "Hello {{extra}}.!\n"
    )  # values are not re-expanded
    sections = parse(
        "---\nname: s\nversion: 1\ndescription: d\nvariables: a, b\n---\n"
        "Top\n\n{{a}}\n\n{{b}}\n\nEnd\n",
        "s.v1.md",
    )
    assert sections.render(a="", b="") == "Top\n\nEnd\n"


def test_rendering_rejects_missing_and_unexpected_variables():
    p = parse(GOOD, "greet.v2.md")
    with pytest.raises(PromptError, match="missing extra"):
        p.render(who="Ada")
    with pytest.raises(PromptError, match="unexpected other"):
        p.render(who="Ada", extra="", other="x")


def test_the_reference_records_name_version_and_content_hash():
    p = parse(GOOD, "greet.v2.md")
    assert p.ref == f"greet@2#{p.sha}" and len(p.sha) == 8
    edited = parse(GOOD.replace("Hello", "Hi"), "greet.v2.md")
    assert edited.sha != p.sha and edited.ref != p.ref


@pytest.mark.parametrize(
    ("source", "filename", "message"),
    [
        (GOOD, "greet.md", "file names look like"),
        (GOOD, "Greet.v2.md", "file names look like"),
        (GOOD, "greet.v0.md", "file names look like"),
        ("no header\n", "greet.v2.md", "header"),
        ("text before\n---\nname: greet\n---\nbody\n", "greet.v2.md", "header"),
        (GOOD.replace("name: greet", "name: other"), "greet.v2.md", "must match"),
        (GOOD.replace("version: 2", "version: 3"), "greet.v2.md", "must match"),
        (GOOD.replace("description: Say hello\n", ""), "greet.v2.md", "description"),
        (GOOD.replace("variables: who, extra", "variables: who"), "greet.v2.md", "undeclared"),
        (
            GOOD.replace("variables: who, extra", "variables: who, extra, more"),
            "greet.v2.md",
            "unused",
        ),
        (GOOD.replace("name: greet", "name greet"), "greet.v2.md", "key: value"),
    ],
)
def test_malformed_prompt_files_are_rejected(source, filename, message):
    with pytest.raises(PromptError, match=message):
        parse(source, filename)


def test_a_directory_is_loaded_by_name_and_version(tmp_path):
    (tmp_path / "greet.v1.md").write_text(GOOD.replace("version: 2", "version: 1"))
    (tmp_path / "greet.v2.md").write_text(GOOD)
    loaded = load_directory(tmp_path)
    assert sorted(loaded["greet"]) == [1, 2] and isinstance(loaded["greet"][2], Prompt)


def test_get_returns_the_latest_unless_pinned(monkeypatch, tmp_path):
    (tmp_path / "greet.v1.md").write_text(GOOD.replace("version: 2", "version: 1"))
    (tmp_path / "greet.v2.md").write_text(GOOD)
    monkeypatch.setattr(prompts, "_registry", lambda: load_directory(tmp_path))
    assert prompts.get("greet").version == 2 and prompts.get("greet", 1).version == 1
    with pytest.raises(PromptError, match="no version 3"):
        prompts.get("greet", 3)
    with pytest.raises(PromptError, match="no prompt named"):
        prompts.get("nothing")


def test_the_answer_prompt_states_the_grounding_rules():
    text = prompts.get("answer").text
    for must in ("only", "NOT_FOUND", "[S1]", "untrusted", "never reveal", "<sources>"):
        assert must.lower() in text.lower(), must
    rendered = prompts.get("answer").render(
        today="2026-11-04", canary="CANARY-1", instructions="", memory="", summary=""
    )
    assert "2026-11-04" in rendered and "CANARY-1" in rendered and "{{" not in rendered


async def test_the_prompt_catalogue_is_served_to_signed_in_users(client, signup):
    headers = await signup()
    r = await client.get("/v1/prompts", headers=headers)
    assert r.status_code == 200
    by_name = {p["name"]: p for p in r.json()}
    answer = by_name["answer"]
    assert answer["version"] == 1 and answer["ref"].startswith("answer@1#")
    assert answer["ref"].endswith(answer["sha"]) and "{{canary}}" in answer["text"]
    assert set(answer["variables"]) == {"today", "canary", "instructions", "memory", "summary"}
    assert (await client.get("/v1/prompts")).status_code == 401


async def test_an_answer_names_the_prompt_that_produced_it(client, signup, space):
    headers = await signup("other@example.com")
    catalogue = {p["ref"] for p in (await client.get("/v1/prompts", headers=headers)).json()}
    r = await client.post(
        f"/v1/workspaces/{space['ws']}/answer",
        json={"question": "anything"},
        headers=space["headers"],
    )
    assert r.json()["prompt"] in catalogue
