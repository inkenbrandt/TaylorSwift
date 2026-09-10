"""Reject version tags that disagree with the distribution metadata."""

import os
from pathlib import Path

import tomllib


def main() -> None:
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    expected = f"v{project['version']}"
    actual = os.environ["GITHUB_REF_NAME"]
    if actual != expected:
        raise SystemExit(f"Release tag {actual!r} does not match {expected!r}")
    print(f"Release tag agrees with {project['name']} {project['version']}")


if __name__ == "__main__":
    main()
