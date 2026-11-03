import io
import json
import uuid

import pytest
from sqlalchemy import update

from app.db.models import IngestJob, JobState
from app.ingest.pipeline import PermanentError


@pytest.fixture
def settings(settings, tmp_path):
    """A file-backed SQLite database. The default in-memory one shares a single connection, so
    a streaming reader and a concurrent writer would trample each other's transactions."""
    return settings.model_copy(update={"database_url": f"sqlite+aiosqlite:///{tmp_path}/jobs.db"})


def _file(data=b"Refunds are issued within 14 days."):
    return {"file": ("notes.txt", io.BytesIO(data), "text/plain")}


def jobs_url(space, *rest):
    return "/".join([f"/v1/workspaces/{space['ws']}/jobs", *rest])


@pytest.fixture
async def uploaded(client, space):
    return (await client.post(space["url"], files=_file(), headers=space["headers"])).json()


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        if block.startswith(":"):
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


async def test_get_job(client, space, uploaded):
    r = await client.get(jobs_url(space, uploaded["job"]["id"]), headers=space["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["document_id"] == uploaded["id"]
    assert body["document_title"] == "notes"
    assert (body["state"], body["progress"], body["attempts"]) == ("queued", 0, 0)


async def test_list_jobs_filters(client, space, uploaded):
    h = space["headers"]
    r = await client.get(jobs_url(space), headers=h)
    assert r.json()["total"] == 1
    assert (await client.get(jobs_url(space), params={"state": "done"}, headers=h)).json()[
        "total"
    ] == 0
    r = await client.get(jobs_url(space), params={"document_id": uploaded["id"]}, headers=h)
    assert r.json()["total"] == 1
    r = await client.get(jobs_url(space), params={"document_id": str(uuid.uuid4())}, headers=h)
    assert r.json()["items"] == []
    assert (await client.get(jobs_url(space), params={"state": "x"}, headers=h)).status_code == 422


async def test_document_detail_includes_its_job(client, space, uploaded):
    r = await client.get(f"{space['url']}/{uploaded['id']}", headers=space["headers"])
    assert r.json()["job"]["id"] == uploaded["job"]["id"]
    r = await client.get(space["url"], headers=space["headers"])
    assert r.json()["items"][0]["job"]["state"] == "queued"


async def test_jobs_are_hidden_from_other_workspaces_and_users(client, signup, space, uploaded):
    job_id = uploaded["job"]["id"]
    eve = await signup("eve@example.com")
    assert (await client.get(jobs_url(space, job_id), headers=eve)).status_code == 404
    assert (await client.get(jobs_url(space, job_id, "events"), headers=eve)).status_code == 404
    assert (await client.post(jobs_url(space, job_id, "retry"), headers=eve)).status_code == 404

    other = (
        await client.post("/v1/workspaces", json={"name": "Other"}, headers=space["headers"])
    ).json()
    crossed = f"/v1/workspaces/{other['id']}/jobs/{job_id}"
    assert (await client.get(crossed, headers=space["headers"])).status_code == 404
    assert (
        await client.get(f"/v1/workspaces/{other['id']}/jobs", headers=space["headers"])
    ).json()["total"] == 0


async def test_requires_auth_and_valid_ids(client, space, uploaded):
    assert (await client.get(jobs_url(space))).status_code == 401
    assert (
        await client.get(jobs_url(space, "not-a-uuid"), headers=space["headers"])
    ).status_code == 422
    missing = jobs_url(space, str(uuid.uuid4()))
    assert (await client.get(missing, headers=space["headers"])).status_code == 404


async def test_retry_only_applies_to_failed_jobs(client, space, uploaded):
    r = await client.post(jobs_url(space, uploaded["job"]["id"], "retry"), headers=space["headers"])
    assert r.status_code == 409
    assert "queued" in r.json()["detail"]


async def test_retry_requeues_a_failed_job_and_the_worker_finishes_it(client, app, space, uploaded):
    worker = app.state.worker

    async def boom(ctx):
        raise PermanentError("could not read file")

    worker.stages = [(JobState.PARSING, boom)]
    await worker.run_once()
    job_url = jobs_url(space, uploaded["job"]["id"])
    failed = (await client.get(job_url, headers=space["headers"])).json()
    assert (failed["state"], failed["error"]) == ("failed", "could not read file")
    doc = (await client.get(f"{space['url']}/{uploaded['id']}", headers=space["headers"])).json()
    assert doc["status"] == "failed"

    worker.stages = []  # the "fix"
    r = await client.post(job_url + "/retry", headers=space["headers"])
    assert r.status_code == 200
    assert (r.json()["state"], r.json()["attempts"], r.json()["error"]) == ("queued", 0, None)
    doc = (await client.get(f"{space['url']}/{uploaded['id']}", headers=space["headers"])).json()
    assert doc["status"] == "pending"

    await worker.run_once()
    done = (await client.get(job_url, headers=space["headers"])).json()
    assert done["state"] == "done"
    doc = (await client.get(f"{space['url']}/{uploaded['id']}", headers=space["headers"])).json()
    assert doc["status"] == "ready"


async def test_events_stream_a_finished_job_and_close(client, app, space, uploaded):
    await app.state.worker.run_once()
    r = await client.get(jobs_url(space, uploaded["job"]["id"], "events"), headers=space["headers"])
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    events = parse_sse(r.text)
    assert len(events) == 1
    name, data = events[0]
    assert name == "job"
    assert (data["state"], data["progress"], data["document_status"]) == ("done", 100, "ready")


async def test_events_follow_progress_until_done(client, app, settings, space, uploaded):
    """Stream while the job advances through states; every change is delivered, then it ends."""
    import asyncio

    settings.sse_poll_s = 0.01
    sm = app.state.sessionmaker
    job_id = uuid.UUID(uploaded["job"]["id"])

    async def drive():
        for state, progress in [
            (JobState.PARSING, 0),
            (JobState.CHUNKING, 40),
            (JobState.EMBEDDING, 70),
            (JobState.DONE, 100),
        ]:
            await asyncio.sleep(0.15)
            async with sm() as s:
                await s.execute(
                    update(IngestJob)
                    .where(IngestJob.id == job_id)
                    .values(state=state, progress=progress)
                )
                await s.commit()

    driver = asyncio.create_task(drive())
    r = await asyncio.wait_for(
        client.get(jobs_url(space, str(job_id), "events"), headers=space["headers"]), timeout=10
    )
    await driver
    states = [d["state"] for _, d in parse_sse(r.text)]
    assert states[0] == "queued"
    assert states[-1] == "done"
    assert states == sorted(set(states), key=states.index)  # no duplicates
    assert {"chunking", "embedding"} <= set(states)


async def test_events_end_with_failed_and_show_the_error(client, app, space, uploaded):
    async def boom(ctx):
        raise PermanentError("encrypted PDF")

    app.state.worker.stages = [(JobState.PARSING, boom)]
    await app.state.worker.run_once()
    r = await client.get(jobs_url(space, uploaded["job"]["id"], "events"), headers=space["headers"])
    ((name, data),) = parse_sse(r.text)
    assert (data["state"], data["error"], data["document_status"]) == (
        "failed",
        "encrypted PDF",
        "failed",
    )


async def test_events_report_gone_when_the_document_is_deleted(
    client, app, settings, space, uploaded
):
    import asyncio

    settings.sse_poll_s = 0.01

    async def delete_soon():
        await asyncio.sleep(0.1)
        await client.delete(f"{space['url']}/{uploaded['id']}", headers=space["headers"])

    task = asyncio.create_task(delete_soon())
    r = await asyncio.wait_for(
        client.get(jobs_url(space, uploaded["job"]["id"], "events"), headers=space["headers"]),
        timeout=10,
    )
    await task
    names = [n for n, _ in parse_sse(r.text)]
    assert names == ["job", "gone"]
