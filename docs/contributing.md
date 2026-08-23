# Contributing

## Development setup

```bash
git clone https://github.com/inkenbrandt/TaylorSwift
cd TaylorSwift
pip install -e ".[dev,docs]"
```

## Tests

```bash
pytest                                          # the suite
pytest --cov=TaylorSwift --cov-report=term-missing
pytest benchmarks --benchmark-only              # performance suite
```

CI runs the suite on Python 3.10 through 3.14. Benchmarks run on every push but
report rather than gate — shared-runner timings are too noisy to fail on.
Compare locally with `--benchmark-compare`.

## Linting and typing

```bash
ruff check .
ruff format .
mypy
```

`mypy` is configured to analyse under Python 3.12 even though the package
supports 3.10+, because NumPy 2.x ships stubs using `type` statements that
older mypy targets reject.

## Documentation

```bash
mkdocs serve      # live reload at http://127.0.0.1:8000
mkdocs build --strict
```

`mkdocs build --strict` is what Read the Docs runs (`fail_on_warning: true`),
so if it passes locally it will pass there. Broken internal links and
unresolvable mkdocstrings references both fail the build.

`watch: [src/TaylorSwift]` in `mkdocs.yml` means editing a docstring rebuilds
the API pages without restarting the server.

### Docstring style

The codebase uses **NumPy-style** docstrings, and `mkdocs.yml` sets
`docstring_style: numpy`. Mixing in Google style will silently render as
unstyled prose:

```python
def example(freq, tau):
    """
    One-line summary.

    Longer explanation, including the physics and any references.

    Parameters
    ----------
    freq : np.ndarray
        Natural frequency [Hz].
    tau : float
        Sensor time constant [s].

    Returns
    -------
    np.ndarray
        Transfer function values in [0, 1].

    References
    ----------
    Moore, C.J. (1986). Frequency response corrections for eddy correlation
        systems. Boundary-Layer Meteorol., 37, 17–35.
    """
```

Conventions worth matching:

- State **units** in square brackets for every physical quantity.
- Cite the source for any implemented method, in the module docstring at
  minimum.
- Public API goes in the module's `__all__`, and new top-level names go in
  `_EXPORTS` in `__init__.py` (which is what keeps imports lazy).

### Adding a module to the docs

1. Create `docs/api/<module>.md` with a short intro and a
   `::: TaylorSwift.<module>` block.
2. Add it to the `nav:` list in `mkdocs.yml`.
3. If it introduces a new top-level export, add it to the tables in
   `docs/api/index.md`.

### Documentation examples must run

Every code block in the docs is meant to be runnable as written. Before
committing a doc change, check the imports and signatures against the source —
`tests/test_docs.py` verifies that the documented import paths resolve, but it
cannot check argument names for you.

## Pull requests

- Branch off `main`.
- Keep the change focused; unrelated cleanups belong in their own PR.
- Add or update tests for behaviour changes.
- Update the [changelog](changelog.md) under **Unreleased**.
- Make sure `pytest`, `ruff check`, `mypy`, and `mkdocs build --strict` all
  pass.

## Reporting bugs

Open an issue at
[github.com/inkenbrandt/TaylorSwift/issues](https://github.com/inkenbrandt/TaylorSwift/issues)
with the version (`TaylorSwift.__version__`), Python version, a minimal
reproducer, and what you expected instead.

For a numerical disagreement, say which reference implementation or paper you
are comparing against — that is usually the fastest route to a diagnosis.
