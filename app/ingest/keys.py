"""Where a document's blobs live: `<workspace>/<collection>/<document>/<name>`.

Nesting by tenant makes cleanup a single prefix delete and keeps one tenant's files from ever
sharing a folder with another's.
"""

from __future__ import annotations

import uuid


def collection_prefix(workspace_id: uuid.UUID, collection_id: uuid.UUID) -> str:
    return f"{workspace_id}/{collection_id}"


def workspace_prefix(workspace_id: uuid.UUID) -> str:
    return str(workspace_id)


def document_prefix(workspace_id: uuid.UUID, collection_id: uuid.UUID, document_id: uuid.UUID):
    return f"{collection_prefix(workspace_id, collection_id)}/{document_id}"


def original_key(workspace_id: uuid.UUID, collection_id: uuid.UUID, document_id: uuid.UUID) -> str:
    return f"{document_prefix(workspace_id, collection_id, document_id)}/original"


def parsed_key(workspace_id: uuid.UUID, collection_id: uuid.UUID, document_id: uuid.UUID) -> str:
    return f"{document_prefix(workspace_id, collection_id, document_id)}/parsed.json"
