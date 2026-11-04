"""Helpers shared by the search tests: ingest real documents through the worker, then search."""

import io

SMALL = {"target_tokens": 64, "overlap_tokens": 8, "max_tokens": 128}
PASTA = "Boil salted water, cook the spaghetti for nine minutes and toss with garlic and olive oil"
REFUND = "Refunds are issued within 14 days of purchase to the original payment method"
SHIPPING = "Orders ship from the warehouse in Rotterdam within two business days"
FILLER = " ".join(f"Detail {i} is handled by the customer care team." for i in range(8))
POLICY = (
    "# Policies\n\nThis handbook lists the customer policies.\n\n"
    f"## Refunds\n\n{REFUND}. {FILLER}\n\n"
    f"## Shipping\n\n{SHIPPING}. {FILLER}\n\n"
    f"## Warranty\n\nHardware is covered by a two year limited warranty. {FILLER}"
)
RECIPES = (
    "# Recipes\n\nSimple dishes.\n\n"
    f"## Pasta\n\n{PASTA}. {FILLER}\n\n"
    f"## Soup\n\nSimmer onions, carrots and celery with vegetable stock for forty minutes. {FILLER}"
)


def search_url(space):
    return f"/v1/workspaces/{space['ws']}/search"


async def make_collection(client, space, name="docs", **body):
    r = await client.post(
        f"/v1/workspaces/{space['ws']}/collections",
        json={"name": name, "chunking_config": SMALL, **body},
        headers=space["headers"],
    )
    assert r.status_code == 201, r.text
    cid = r.json()["id"]
    return cid, f"/v1/workspaces/{space['ws']}/collections/{cid}/documents"


async def ingest(client, app, space, url, text=POLICY, name="policy.md"):
    files = {"file": (name, io.BytesIO(text.encode()), "text/markdown")}
    r = await client.post(url, files=files, headers=space["headers"])
    assert r.status_code == 201, r.text
    while await app.state.worker.run_once():
        pass
    return r.json()["id"]


async def find(client, space, query, **body):
    """Search and return the JSON; dense mode unless the test says otherwise."""
    body.setdefault("mode", "dense")
    r = await client.post(
        search_url(space), json={"query": query, **body}, headers=space["headers"]
    )
    assert r.status_code == 200, r.text
    return r.json()


async def build_indexed(client, app, space):
    cid, url = await make_collection(client, space)
    doc_id = await ingest(client, app, space, url)
    return {**space, "cid": cid, "url": url, "doc": doc_id}
