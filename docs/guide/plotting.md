# Plotting

Three helpers produce the standard micrometeorological diagnostic figures.

!!! important "They return `(fig, axes)`"
    Every plotting function returns a **tuple**, not a bare figure:

    ```python
    fig, axes = plot_cospectra(results)
    fig.savefig("cospectra.pdf", dpi=150)
    ```

## Cospectra

```python
from TaylorSwift.plotting import plot_cospectra

fig, axes = plot_cospectra(
    results,
    stability_range=(-2.0, 2.0),
    show_model=True,
    show_slope=True,
    figsize=(14, 10),
)
```

A 2×2 grid — $\overline{w'T'}$, $\overline{w'u'}$, $\overline{w'CO_2'}$,
$\overline{w'H_2O'}$ — with curves coloured by stability class $z/L$ on a
diverging red–blue scale centred at neutral.

| Argument | Default | Effect |
| --- | --- | --- |
| `stability_range` | `(-2.0, 2.0)` | Only plot intervals whose $z/L$ falls inside |
| `show_model` | `True` | Overlay the Kaimal (1972) model curves |
| `show_slope` | `True` | Draw the $-4/3$ inertial-subrange reference line |
| `figsize` | `(14, 10)` | Figure size |
| `save_path` | `None` | If given, save to this path |

The model overlay is the point of the figure: measured curves should track
Kaimal through the inertial subrange and fall below it at high frequency by
exactly the amount your [corrections](corrections.md) are putting back.

## Power spectra

```python
from TaylorSwift.plotting import plot_spectra

fig, axes = plot_spectra(results, stability_range=(-2.0, 2.0), show_model=True)
```

Normalised spectra $n S(n)/\sigma^2$ for $u$, $v$, $w$, and $T$. The expected
inertial-subrange slope here is $-2/3$, not $-4/3$.

A `w` spectrum that flattens at high frequency instead of rolling off is the
classic signature of white noise in the sonic — worth catching before it
propagates into a season of fluxes.

## Ogives

```python
from TaylorSwift.plotting import plot_ogive

fig, axes = plot_ogive(results, stability_range=(-2.0, 2.0), figsize=(14, 5))
```

Cumulative cospectra, integrated from high to low frequency. **The diagnostic
question is whether the curve flattens.** If it is still climbing at the lowest
resolved frequency, your averaging period is too short to capture the whole
flux — the standard justification for moving from 30 to 60 minutes.

## Saving

Either capture the figure or pass `save_path`:

```python
fig, axes = plot_cospectra(results)
fig.savefig("cospectra.pdf", dpi=300, bbox_inches="tight")

plot_cospectra(results, save_path="cospectra.png")
```

## Customising

`axes` is a normal Matplotlib array, so post-hoc tweaks work as usual:

```python
fig, axes = plot_cospectra(results)

for ax in axes.flat:
    ax.set_xlim(1e-3, 1e2)
    ax.grid(True, which="both", alpha=0.3)

axes[0, 0].set_title("Sensible heat cospectrum")
fig.suptitle("Site A — June 2023", fontsize=14)
```

## Headless environments

On a server or in CI, select a non-interactive backend **before** importing
anything that pulls in pyplot:

```python
import matplotlib
matplotlib.use("Agg")

from TaylorSwift.plotting import plot_cospectra
```

## Filter before you plot

The plotting helpers filter on `stability_range` only — they do not consult
`qc_flags`. Apply your [quality control](quality-control.md) first, or bad
intervals will be drawn alongside good ones:

```python
clean = [r for r in results if not r.qc_flags.get("vm97_hard_flag", False)]
fig, axes = plot_cospectra(clean)
```
