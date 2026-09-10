# Contributing

## Development setup

```bash
git clone https://github.com/inkenbrandt/TaylorSwift
cd TaylorSwift
pip install -e ".[dev,docs]"
```

For a reproducible environment, use `uv sync --locked --extra dev --extra docs`.
After intentionally editing dependencies or project metadata, run `uv lock`
and commit `uv.lock`. CI uses `uv lock --check` and `uv sync --locked`; it
fails on stale metadata without rewriting the lockfile.

## Tests

```bash
pytest                                          # the suite
pytest --cov=TaylorSwift --cov-report=term-missing
pytest benchmarks --benchmark-only              # performance suite
```

CI runs the suite on Linux with Python 3.10 through 3.14 and Windows with
Python 3.12. Windows also runs uncaptured I/O tests with strict cp1252 output.
JUnit XML and coverage XML are generated explicitly and retained as artifacts.
Benchmarks run on every push but
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

CI also builds and installs the wheel into a fresh environment outside the
checkout, checks lazy imports, distribution metadata and plotting, and runs
a strict mypy consumer example. The example checks inferred types and proves
that invalid plot inputs, tuple `savefig` calls and unknown exports are rejected.
To run it locally after `uv build`, use `python tools/check_wheel.py`.
Top-level static exports live in `src/TaylorSwift/__init__.pyi`; keep them in
sync with `_EXPORTS`. The `py.typed` marker enables inline typing, but does not
promise that every legacy function has a complete annotation.

## Dependency audit and compatibility

The `dev` extra retains the tools used by tests, benchmarks, lint and typing.
The `docs` extra retains only MkDocs, Material, mkdocstrings and
pymdown-extensions, all used in `mkdocs.yml`. Neither the MkDocs configuration
nor Read the Docs uses Sphinx, mknotebooks or mkdocs-jupyter, so those packages
have been removed. `tob2toa` has no imports in the package, tests or examples
and has also been removed from `dev`.

The example notebooks are interactive authoring assets, not part of the docs
build. Install `.[notebooks]` for Notebook and ipykernel; users who previously
obtained these through `dev` or `docs` should add this extra explicitly. The
Jupyter umbrella package is unnecessary for these examples.

Plotting and legacy pipeline dependencies remain required runtime dependencies.
A future split into `plotting` or `legacy` extras needs a dependency/import map
(including pandas/Arrow conversion and statsmodels-backed despiking), guarded
imports with actionable installation errors, and tests for base-only, each
extra, and full installs. Preserve existing import names and default-install
behavior during a documented deprecation period; change the default dependency
set only in an announced breaking release with migration instructions. Lazy
top-level imports alone do not make these dependencies optional.

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
