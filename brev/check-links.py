#!/usr/bin/env python3
"""Check that every relative link in the Markdown files and the syllabus notebooks resolves.

Usage: check-links.py [PATH ...]    (default: the repository)

External links are not fetched. The exit code is the number of broken links.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\]\(([^)\s#?]+)(?:[#?][^)]*)?\)")
CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)


def markdown_of(path):
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text())["cells"]
        return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    return path.read_text()


def broken_links(path):
    broken = []
    for target in LINK.findall(CODE.sub("", markdown_of(path))):
        if "://" in target or target.startswith("mailto:"):
            continue
        if not (path.parent / target).exists():
            broken.append(target)
    return broken


def main():
    roots = [Path(p).resolve() for p in sys.argv[1:]] or [REPO]
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files += sorted(p for p in root.rglob("*.md") if ".git" not in p.parts)
            files += sorted(p for p in root.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)
    total = 0
    for path in files:
        for target in broken_links(path):
            print(f"{path.relative_to(REPO) if path.is_relative_to(REPO) else path}: broken link {target}")
            total += 1
    print(f"{len(files)} files checked, {total} broken links")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
