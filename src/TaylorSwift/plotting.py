"""
plotting.py — Publication-quality spectral / cospectral plots.

Follows the conventions used in the micromet literature:

  - Log–log axes with dimensionless frequency f = nz/U on the x-axis.
  - Area-preserving form: n·Co(n)/cov(w'x') on the y-axis for cospectra,
    or n·S(n)/var(x) for power spectra.
  - Reference slope lines for the inertial subrange (-2/3 for spectra,
    -4/3 for cospectra — i.e., -5/3 and -7/3 when plotting n·S vs n).
  - Optional Kaimal (1972) model curves for comparison.

References
----------
Kaimal, J.C. et al. (1972). Spectral characteristics of surface-layer
    turbulence. Quart. J. Roy. Meteor. Soc., 98, 563–589.
Moraes, O.L.L. et al. (2008). Physica A, 387, 4927–4939.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import LogLocator, NullFormatter


# ---------------------------------------------------------------------------
# Kaimal (1972) model cospectra  (empirical fits for neutral conditions)
# ---------------------------------------------------------------------------
def _kaimal_cospec_wT(f):
    """Kaimal et al. (1972) model for n·Co_wT / <w'T'> vs f = nz/U."""
    # Eq from Kaimal et al. (1972) for near-neutral:
    # n Co_wT / <w'T'> = 12.92 f / (1 + 26.7 f)^(7/4)
    return 12.92 * f / (1.0 + 26.7 * f) ** (7.0 / 4.0)


def _kaimal_cospec_wu(f):
    """Kaimal et al. (1972) model for n·Co_wu / <w'u'> vs f = nz/U."""
    # n Co_wu / <w'u'> = 9.6 f / (1 + 14.0 f)^(7/4)
    # (approximation from Horst, 1997 / Massman, 2000)
    return 9.6 * f / (1.0 + 14.0 * f) ** (7.0 / 4.0)


def _kaimal_spec_w(f):
    """Kaimal (1972) model for n·S_w / u*² vs f = nz/U (neutral)."""
    return 2.1 * f / (1.0 + 5.3 * f ** (5.0 / 3.0))


def _kaimal_spec_u(f):
    """Kaimal (1972) model for n·S_u / u*² vs f = nz/U (neutral)."""
    return 105.0 * f / (1.0 + 33.0 * f) ** (5.0 / 3.0)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _setup_spectral_axes(ax, xlabel='$f = nz/U$', ylabel=None, title=None):
    """Configure log-log axes with micromet conventions."""
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_locator(LogLocator(base=10))


def _add_slope_line(ax, slope, f_range=(0.5, 5.0), label=None,
                    anchor_y=None, **kwargs):
    """Draw a reference power-law slope on a log-log axis."""
    f = np.logspace(np.log10(f_range[0]), np.log10(f_range[1]), 50)
    y = f ** slope
    if anchor_y is not None:
        y = y * (anchor_y / y[0])
    defaults = dict(ls='--', color='grey', lw=1.2, alpha=0.7)
    defaults.update(kwargs)
    ax.plot(f, y, label=label, **defaults)


# ---------------------------------------------------------------------------
# Main cospectral plot
# ---------------------------------------------------------------------------
def plot_cospectra(
    results,
    stability_range=(-2.0, 2.0),
    show_model: bool = True,
    show_slope: bool = True,
    figsize=(14, 10),
    save_path=None,
):
    """
    Plot normalised cospectra (w'T', w'u', w'CO₂', w'H₂O') in a 2×2 grid.

    Data are coloured by stability class (z/L).

    Parameters
    ----------
    results : list[SpectralResult]
        Output of process_file or a list of process_interval calls.
    stability_range : tuple
        (min z/L, max z/L) — only plot intervals within this range.
    show_model : bool
        Overlay Kaimal (1972) model curves (default True).
    show_slope : bool
        Show -4/3 inertial subrange reference slope (default True).
    figsize : tuple
        Figure size.
    save_path : str or None
        If given, save figure to this path.

    Returns
    -------
    fig, axes
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_wT, ax_wu = axes[0]
    ax_wCO2, ax_wH2O = axes[1]

    # Stability colormap
    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)

    panels = [
        (ax_wT,   'ncosp_wT',   "$n \\cdot Co_{wT} \\,/\\, \\overline{w'T'}$",
         "Sensible heat flux cospectrum", _kaimal_cospec_wT),
        (ax_wu,   'ncosp_wu',   "$n \\cdot Co_{wu} \\,/\\, \\overline{w'u'}$",
         "Momentum flux cospectrum", _kaimal_cospec_wu),
        (ax_wCO2, 'ncosp_wCO2', "$n \\cdot Co_{wCO_2} \\,/\\, \\overline{w'CO_2'}$",
         "CO$_2$ flux cospectrum", None),
        (ax_wH2O, 'ncosp_wH2O', "$n \\cdot Co_{wH_2O} \\,/\\, \\overline{w'H_2O'}$",
         "H$_2$O flux cospectrum", None),
    ]

    for ax, attr, ylabel, title, model_fn in panels:
        _setup_spectral_axes(ax, ylabel=ylabel, title=title)

        for res in results:
            zL = res.zL
            if not np.isfinite(zL):
                continue
            if zL < stability_range[0] or zL > stability_range[1]:
                continue

            f_nd = res.freq_nd
            cosp = np.abs(getattr(res, attr))

            if len(f_nd) == 0 or not np.any(np.isfinite(f_nd)):
                continue

            color = cmap(norm(np.clip(zL, -2, 2)))
            ax.plot(f_nd, cosp, color=color, alpha=0.35, lw=0.7)

        if show_model and model_fn is not None:
            f_model = np.logspace(-3, 2, 500)
            ax.plot(f_model, model_fn(f_model), 'k-', lw=2.0, alpha=0.8,
                    label='Kaimal (1972)')
            ax.legend(fontsize=9, loc='upper right')

        if show_slope:
            _add_slope_line(ax, -4.0/3.0, f_range=(1, 10),
                            label='$f^{-4/3}$', anchor_y=0.1)
            ax.legend(fontsize=9, loc='upper right')

        ax.set_xlim(1e-3, 20)
        ax.set_ylim(1e-3, 10)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                        fraction=0.02, pad=0.04)
    cbar.set_label('$z/L$', fontsize=11)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig, axes


# ---------------------------------------------------------------------------
# Power spectra plot
# ---------------------------------------------------------------------------
def plot_spectra(
    results,
    stability_range=(-2.0, 2.0),
    show_model: bool = True,
    figsize=(14, 10),
    save_path=None,
):
    """
    Plot normalised power spectra (u, v, w, T) in a 2×2 grid.

    Normalization: n·S(n) / σ² where σ² is the variance of the component.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_u, ax_v = axes[0]
    ax_w, ax_T = axes[1]

    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)

    panels = [
        (ax_u, 'spec_u', "$n \\cdot S_u \\,/\\, \\sigma_u^2$", "Longitudinal ($u$)"),
        (ax_v, 'spec_v', "$n \\cdot S_v \\,/\\, \\sigma_v^2$", "Lateral ($v$)"),
        (ax_w, 'spec_w', "$n \\cdot S_w \\,/\\, \\sigma_w^2$", "Vertical ($w$)"),
        (ax_T, 'spec_T', "$n \\cdot S_T \\,/\\, \\sigma_T^2$", "Temperature ($T_s$)"),
    ]

    for ax, attr, ylabel, title in panels:
        _setup_spectral_axes(ax, ylabel=ylabel, title=title)

        for res in results:
            zL = res.zL
            if not np.isfinite(zL):
                continue
            if zL < stability_range[0] or zL > stability_range[1]:
                continue

            f_nd = res.freq_nd
            spec = getattr(res, attr)

            if len(f_nd) == 0 or len(spec) == 0:
                continue

            # Normalise by variance (integral of n·S(n) d(ln n) ≈ σ²)
            var_est = np.trapz(spec, np.log(res.freq + 1e-30))
            if abs(var_est) > 1e-12:
                spec_norm = spec / var_est
            else:
                continue

            color = cmap(norm(np.clip(zL, -2, 2)))
            ax.plot(f_nd, spec_norm, color=color, alpha=0.35, lw=0.7)

        # Reference -2/3 slope (inertial subrange for n·S vs f)
        _add_slope_line(ax, -2.0/3.0, f_range=(1, 10),
                        label='$f^{-2/3}$', anchor_y=0.05)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_xlim(1e-3, 20)
        ax.set_ylim(1e-4, 5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                        fraction=0.02, pad=0.04)
    cbar.set_label('$z/L$', fontsize=11)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig, axes


# ---------------------------------------------------------------------------
# Ogive plot
# ---------------------------------------------------------------------------
def plot_ogive(
    results,
    stability_range=(-2.0, 2.0),
    figsize=(14, 5),
    save_path=None,
):
    """
    Plot ogives (cumulative cospectra from high to low frequency).

    Ogives that flatten at low frequencies indicate that the chosen averaging
    period captures all of the turbulent flux.  If the ogive is still rising
    at the lowest resolved frequency, flux is being lost.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=False)

    labels = [
        ("ogive_wT", "cov_wT", "$Og_{wT}$ / $\\overline{w'T'}$",
         "Heat flux ogive"),
        ("ogive_wu", "cov_wu", "$Og_{wu}$ / $\\overline{w'u'}$",
         "Momentum ogive"),
        ("ogive_wCO2", "cov_wCO2", "$Og_{wCO_2}$ / $\\overline{w'CO_2'}$",
         "CO$_2$ ogive"),
        ("ogive_wH2O", "cov_wH2O", "$Og_{wH_2O}$ / $\\overline{w'H_2O'}$",
         "H$_2$O ogive"),
    ]

    for ax, (og_attr, cov_attr, ylabel, title) in zip(axes, labels):
        ax.set_xscale('log')
        ax.set_xlabel('$f = nz/U$', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, which='both', ls=':', alpha=0.4)
        ax.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5)

        for res in results:
            zL = res.zL
            if not np.isfinite(zL):
                continue
            if zL < stability_range[0] or zL > stability_range[1]:
                continue

            f_nd = res.freq_nd
            ogive = getattr(res, og_attr)
            cov = getattr(res, cov_attr)

            if len(f_nd) == 0 or len(ogive) == 0 or abs(cov) < 1e-12:
                continue

            ax.plot(f_nd, ogive / cov, alpha=0.4, lw=0.8)

        ax.set_xlim(1e-3, 20)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig, axes


# ---------------------------------------------------------------------------
# Summary time-series plot
# ---------------------------------------------------------------------------
def plot_summary_timeseries(results, figsize=(14, 10), save_path=None):
    """
    Plot key turbulence parameters vs time for QC overview.

    Shows: wind speed, u*, z/L, sensible heat flux, and covariances.
    """
    import matplotlib.dates as mdates

    times = [r.timestamp_start for r in results if r.timestamp_start is not None]
    if not times:
        return None, None

    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)

    def _vals(attr):
        return [getattr(r, attr) for r in results if r.timestamp_start is not None]

    axes[0].plot(times, _vals('u_mean'), 'k.-', ms=3)
    axes[0].set_ylabel('$\\bar{U}$ [m/s]')
    axes[0].set_title('Mean wind speed')

    axes[1].plot(times, _vals('ustar'), 'b.-', ms=3)
    axes[1].set_ylabel('$u_*$ [m/s]')
    axes[1].set_title('Friction velocity')

    zL_vals = _vals('zL')
    axes[2].plot(times, zL_vals, 'r.-', ms=3)
    axes[2].set_ylabel('$z/L$')
    axes[2].set_ylim(-3, 3)
    axes[2].axhline(0, color='k', ls='-', lw=0.5)
    axes[2].set_title('Stability parameter')

    axes[3].plot(times, _vals('H'), 'g.-', ms=3)
    axes[3].set_ylabel('$H$ [W/m²]')
    axes[3].axhline(0, color='k', ls='-', lw=0.5)
    axes[3].set_title('Sensible heat flux')

    axes[4].plot(times, _vals('cov_wCO2'), 'm.-', ms=3, label="$\\overline{w'CO_2'}$")
    axes[4].set_ylabel("$\\overline{w'CO_2'}$\n[mg m⁻² s⁻¹]")
    axes[4].axhline(0, color='k', ls='-', lw=0.5)
    axes[4].set_title('CO₂ flux (covariance)')

    for ax in axes:
        ax.grid(True, ls=':', alpha=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    axes[-1].set_xlabel('Time (UTC)')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig, axes
