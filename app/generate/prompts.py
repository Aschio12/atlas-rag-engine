"""Versioned prompt templates kept in files (Phase 4.1).

Prompts live in `app/generate/templates/<name>.v<N>.md`: a small header (name, version,
description, the variables the text uses) between `---` lines, then the template with
`{{variable}}` placeholders. Keeping them in files means a prompt change is a reviewable diff,
every answer records which prompt produced it (`answer@1#3f2a9c1d`: name, version, content hash),
and an old version stays available to reproduce an old answer or to compare two versions.

The loader is strict on purpose: the header must match the file name, every placeholder must be
declared and every declared variable used, and rendering fails on a missing or unexpected
variable, so a typo is a test failure, not a silently odd prompt. Substituted values are inserted
literally and never re-expanded, so text from documents or users cannot inject placeholders.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "templates"
_FILE = re.compile(r"^(?P<name>[a-z][a-z0-9_]*)\.v(?P<version>[1-9][0-9]*)\.md$")
_PLACEHOLDER = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")


class PromptError(ValueError):
    """A prompt file is malformed, or a prompt was rendered with the wrong variables."""


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    description: str
    variables: tuple[str, ...]
    text: str

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:8]

    @property
    def ref(self) -> str:
        """What is recorded with an answer: `answer@1#3f2a9c1d`."""
        return f"{self.name}@{self.version}#{self.sha}"

    def render(self, **values: str) -> str:
        missing = [v for v in self.variables if v not in values]
        extra = [v for v in values if v not in self.variables]
        if missing or extra:
            problems = []
            if missing:
                problems.append(f"missing {', '.join(missing)}")
            if extra:
                problems.append(f"unexpected {', '.join(extra)}")
            raise PromptError(f"prompt {self.name}@{self.version}: {'; '.join(problems)}")
        out = _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), self.text)
        return re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"  # empty optional sections collapse


def parse(source: str, filename: str) -> Prompt:
    match = _FILE.match(filename)
    if not match:
        raise PromptError(f"{filename}: file names look like name.v1.md")
    parts = source.split("---\n", 2)
    if len(parts) != 3 or parts[0].strip():
        raise PromptError(f"{filename}: a header between --- lines must come first")
    header: dict[str, str] = {}
    for line in parts[1].splitlines():
        if line.strip():
            key, sep, value = line.partition(":")
            if not sep:
                raise PromptError(f"{filename}: header line {line!r} is not 'key: value'")
            header[key.strip()] = value.strip()
    if header.get("name") != match["name"] or header.get("version") != match["version"]:
        raise PromptError(f"{filename}: the header's name and version must match the file name")
    if not header.get("description"):
        raise PromptError(f"{filename}: a description is required")
    declared = tuple(v.strip() for v in header.get("variables", "").split(",") if v.strip())
    text = parts[2].strip("\n") + "\n"
    used = set(_PLACEHOLDER.findall(text))
    if undeclared := sorted(used - set(declared)):
        raise PromptError(f"{filename}: undeclared variables {', '.join(undeclared)}")
    if unused := sorted(set(declared) - used):
        raise PromptError(f"{filename}: declared but unused variables {', '.join(unused)}")
    return Prompt(match["name"], int(match["version"]), header["description"], declared, text)


def load_directory(directory: Path) -> dict[str, dict[int, Prompt]]:
    prompts: dict[str, dict[int, Prompt]] = {}
    for path in sorted(directory.glob("*.md")):
        prompt = parse(path.read_text(encoding="utf-8"), path.name)
        prompts.setdefault(prompt.name, {})[prompt.version] = prompt
    return prompts


@lru_cache(maxsize=1)
def _registry() -> dict[str, dict[int, Prompt]]:
    return load_directory(PROMPT_DIR)


def get(name: str, version: int | None = None) -> Prompt:
    """A prompt by name; the latest version unless one is pinned."""
    versions = _registry().get(name)
    if not versions:
        raise PromptError(f"no prompt named {name!r}")
    if version is None:
        return versions[max(versions)]
    if version not in versions:
        raise PromptError(f"prompt {name!r} has no version {version} (has {sorted(versions)})")
    return versions[version]


def available() -> dict[str, list[int]]:
    return {name: sorted(versions) for name, versions in sorted(_registry().items())}
