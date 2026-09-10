# Installation

## From PyPI

```bash
pip install taylorswift-spectra
```

!!! warning "The distribution name and the import name differ"
    The project is published as **`taylorswift-spectra`** because the short
    name was already claimed on PyPI by an unrelated project. The import name
    is unchanged:

    ```python
    import TaylorSwift as tswift
    ```

    Installing a package called `taylorswift` will *not* give you this
    library.

## Requirements

`TaylorSwift` requires **Python 3.10 or newer** and pulls in:

| Package | Minimum | Used for |
| --- | --- | --- |
| `numpy` | 1.24 | array maths, FFT input |
| `scipy` | 1.10 | FFT, signal processing, optimisation |
| `polars` | 1.0 | TOA5 reading, tidy result tables |
| `pandas` | 1.5 | legacy `CalcFlux` pipelines |
| `pyarrow` | — | Parquet export |
| `KDEpy` | 1.1 | UKDE despiking |
| `matplotlib` | 3.7 | plotting |
| `statsmodels` | 0.14 | robust regression in despiking |

## Optional extras

=== "Development"

    ```bash
    pip install "taylorswift-spectra[dev]"
    ```

    Adds `pytest`, `pytest-cov`, `pytest-benchmark`, `ruff`, and `mypy`.

=== "Documentation"

    ```bash
    pip install "taylorswift-spectra[docs]"
    ```

    Adds `mkdocs`, `mkdocs-material`, and `mkdocstrings[python]`.

=== "Interactive notebooks"

    ```bash
    pip install "taylorswift-spectra[notebooks]"
    ```

    Adds Notebook and ipykernel for the examples. These tools are no longer
    included in the development or documentation extras.

## Development checkout

```bash
git clone https://github.com/inkenbrandt/TaylorSwift
cd TaylorSwift
pip install -e ".[dev,docs]"
```

Run the test suite:

```bash
pytest
```

Build the documentation locally with live reload:

```bash
mkdocs serve
```

Then open <http://127.0.0.1:8000>. The `watch:` entry in `mkdocs.yml` points
at `src/TaylorSwift`, so editing a docstring rebuilds the API pages
immediately.

## Verifying the install

```python
import TaylorSwift as tswift

print(tswift.__version__)
```

!!! note
    `__version__` is read from the installed distribution metadata. Running
    from a source tree that was never installed reports `0.0.0.dev0`.
