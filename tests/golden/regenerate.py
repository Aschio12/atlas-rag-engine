"""Rewrite the expected snapshots from the current parsers and chunkers.

    python tests/golden/regenerate.py

Review `git diff tests/golden/expected` before committing: every line that changed is a change
in what users will get back.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import golden  # noqa: E402


def main() -> None:
    golden.EXPECTED.mkdir(exist_ok=True)
    for path in golden.input_files():
        golden.expected_path(path).write_text(golden.dump(golden.snapshot(path)), encoding="utf-8")
        print("updated", path.name)


if __name__ == "__main__":
    main()
