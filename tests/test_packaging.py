"""Keep the lazy runtime API and the installed static API in agreement."""

import ast
from pathlib import Path

import TaylorSwift


def test_stub_exports_match_lazy_exports():
    stub = Path(TaylorSwift.__file__).with_suffix(".pyi")
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    exports = {
        alias.asname: ("." * node.level + node.module, alias.name)
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert exports == TaylorSwift._EXPORTS
