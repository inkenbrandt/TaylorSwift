"""
Tests that keep the documentation honest.

The docs promise that every ``from TaylorSwift.x import y`` in a code block
resolves, and that the nav in ``mkdocs.yml`` matches the files on disk.  Both
have drifted before: the quickstart imported ``despike_dataframe`` and
``InstrumentConfig`` from :mod:`TaylorSwift.corrections`, where neither has
ever lived.

Signature checking is deliberately out of scope — these tests catch stale
import paths and missing pages, not wrong argument names.
"""

import importlib
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
MKDOCS_YML = Path(__file__).resolve().parent.parent / "mkdocs.yml"

# ``from TaylorSwift.module import a, b`` inside a fenced code block.
_FROM_IMPORT = re.compile(
    r"^from\s+(TaylorSwift(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s+import\s+(.+)$",
    re.MULTILINE,
)
# ``::: TaylorSwift.module`` mkdocstrings directives.
_AUTODOC = re.compile(
    r"^:::\s+(TaylorSwift(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*$", re.MULTILINE
)


def _markdown_files():
    return sorted(DOCS.rglob("*.md"))


def _iter_from_imports():
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for match in _FROM_IMPORT.finditer(text):
            module = match.group(1)
            names = match.group(2)
            if names.strip().startswith("("):  # no parenthesised forms in the docs
                continue
            for name in (n.strip() for n in names.split(",")):
                if name and name != "*":
                    yield path, module, name


def test_docs_directory_is_present():
    assert DOCS.is_dir(), f"docs directory missing at {DOCS}"
    assert _markdown_files(), "no markdown files found under docs/"


@pytest.mark.parametrize(
    "path, module, name",
    [
        pytest.param(p, m, n, id=f"{p.name}::{m}.{n}")
        for p, m, n in _iter_from_imports()
    ],
)
def test_documented_imports_resolve(path, module, name):
    """Every ``from TaylorSwift.x import y`` in the docs must actually work."""
    mod = importlib.import_module(module)
    assert hasattr(mod, name), (
        f"{path.relative_to(DOCS.parent)} documents "
        f"`from {module} import {name}`, but {module} has no attribute {name!r}"
    )


def test_top_level_attribute_references_resolve():
    """``tswift.<name>`` used in the docs must be a real top-level export."""
    import TaylorSwift

    pattern = re.compile(r"\btswift\.([A-Za-z_][A-Za-z0-9_]*)")
    missing = []
    for path in _markdown_files():
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            name = match.group(1)
            if name.startswith("__"):
                continue
            if not hasattr(TaylorSwift, name):
                missing.append(f"{path.name}: tswift.{name}")

    assert not missing, "documented top-level names that do not exist:\n" + "\n".join(
        sorted(set(missing))
    )


def test_autodoc_targets_are_importable():
    """Every ``::: TaylorSwift.x`` directive must point at a real module."""
    for path in _markdown_files():
        for match in _AUTODOC.finditer(path.read_text(encoding="utf-8")):
            module = match.group(1)
            importlib.import_module(module)  # raises if the module is gone


def test_every_public_module_has_an_api_page():
    """A new module in src/ must not silently miss the API reference."""
    src = Path(__file__).resolve().parent.parent / "src" / "TaylorSwift"
    documented = set()
    for path in (DOCS / "api").glob("*.md"):
        for match in _AUTODOC.finditer(path.read_text(encoding="utf-8")):
            documented.add(match.group(1).split(".")[-1])

    on_disk = {p.stem for p in src.glob("*.py") if not p.stem.startswith("_")}

    undocumented = on_disk - documented
    assert not undocumented, (
        "modules without an API page (add docs/api/<name>.md and a nav entry): "
        f"{sorted(undocumented)}"
    )


def test_nav_entries_exist_on_disk():
    """Every markdown path named in mkdocs.yml nav must exist."""
    text = MKDOCS_YML.read_text(encoding="utf-8")
    nav = text.split("\nnav:", 1)[1]

    missing = [
        ref
        for ref in re.findall(r":\s*([A-Za-z0-9_\-/]+\.md)\s*$", nav, re.MULTILINE)
        if not (DOCS / ref).is_file()
    ]
    assert not missing, f"nav references missing files: {missing}"


def test_all_docs_pages_are_in_nav():
    """A page not in the nav is unreachable; catch orphans early."""
    text = MKDOCS_YML.read_text(encoding="utf-8")
    nav = text.split("\nnav:", 1)[1]
    referenced = set(re.findall(r":\s*([A-Za-z0-9_\-/]+\.md)\s*$", nav, re.MULTILINE))

    on_disk = {p.relative_to(DOCS).as_posix() for p in _markdown_files()}

    orphans = on_disk - referenced
    assert not orphans, f"docs pages missing from mkdocs.yml nav: {sorted(orphans)}"
