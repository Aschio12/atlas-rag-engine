"""Helpers for the answering tests: documents, scripted models and an SSE reader."""

import io
import json

from app.providers.fake import FakeLLM

REFUNDS = "Refunds are issued within 14 days of purchase to the original payment method."
SHIPPING = "Orders ship from the warehouse in Rotterdam within two business days."
HANDBOOK = f"# Policies\n\n## Refunds\n\n{REFUNDS}\n\n## Shipping\n\n{SHIPPING}\n"


async def add_doc(client, app, space, text=HANDBOOK, name="handbook.md", **form):
    files = {"file": (name, io.BytesIO(text.encode()), "text/markdown")}
    r = await client.post(space["url"], files=files, data=form, headers=space["headers"])
    assert r.status_code == 201, r.text
    while await app.state.worker.run_once():
        pass
    return r.json()["id"]


def answer_url(space):
    return f"/v1/workspaces/{space['ws']}/answer"


async def ask(client, space, question, expect=200, **body):
    r = await client.post(
        answer_url(space), json={"question": question, **body}, headers=space["headers"]
    )
    assert r.status_code == expect, r.text
    return r.json()


def sse_events(text: str) -> list[tuple[str, dict]]:
    """Parse a server-sent event body into (event, data) pairs, skipping comments."""
    events = []
    for block in text.strip().split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:])
        if name:
            events.append((name, data))
    return events


async def stream_answer(client, space, question, **body):
    async with client.stream(
        "POST",
        answer_url(space),
        json={"question": question, "stream": True, **body},
        headers=space["headers"],
    ) as r:
        assert r.status_code == 200, await r.aread()
        text = (await r.aread()).decode()
    return sse_events(text), text


class SpyLLM(FakeLLM):
    """The fake LLM, remembering every request; `reply` may be a function of the messages."""

    def __init__(self, reply=None):
        super().__init__()
        self._fn = reply if callable(reply) else None
        if not callable(reply):
            self._reply = reply

    def _text(self, messages):
        return self._fn(messages) if self._fn else super()._text(messages)

    def answer_calls(self):
        return [c for c in self.calls if "<sources>" in c[-1].content]


class LeakyLLM(SpyLLM):
    """Repeats its system prompt, hidden reference code included."""

    def _text(self, messages):
        return messages[0].content


def use_llm(app, llm):
    app.state.providers._llm = llm
    return llm


async def add_member(client, app, space, email, role):
    """Sign `email` up, make them a member of the workspace with `role`, return a space dict for
    them (same workspace, their own headers)."""
    import uuid

    from sqlalchemy import select

    from app.db.models import User, WorkspaceMember

    password = "correct-horse-9"
    await client.post("/v1/auth/register", json={"email": email, "password": password})
    login = await client.post("/v1/auth/login", json={"email": email, "password": password})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    async with app.state.sessionmaker() as s:
        uid = (await s.execute(select(User.id).where(User.email == email))).scalar_one()
        s.add(WorkspaceMember(workspace_id=uuid.UUID(space["ws"]), user_id=uid, role=role))
        await s.commit()
    return {**space, "headers": headers}
