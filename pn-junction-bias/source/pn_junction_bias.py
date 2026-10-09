# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.5.3",
#     "matplotlib==3.11.2",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(np):
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.linewidth": 1.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "mathtext.fontset": "dejavusans",
        }
    )

    # Silicon at 300 K. n_i and kT/e match the extrinsic companion and Neamen's zero-bias example.
    KT_E = 0.0259
    NI_SI = 1.5e10
    EG_EV = 1.12
    E_CHARGE = 1.6e-19
    EPS_S = 11.7 * 8.85e-14
    VA_MIN = -4.0
    VA_MAX = 1.0

    BLUE = "#0072B2"
    ORANGE = "#D55E00"
    GREEN = "#009E73"
    NAVY = "#0f4c81"
    GRAY = "#6B6B6B"
    PINK = "#CC79A7"

    def sci_tex(x, digits=2):
        x = float(x)
        if not np.isfinite(x) or x == 0.0:
            return "0"
        sign = "-" if x < 0 else ""
        ax = abs(x)
        exp = int(np.floor(np.log10(ax)))
        man = ax / (10.0**exp)
        return rf"{sign}{man:.{digits}f}\times 10^{{{exp}}}"

    def style_ax(ax):
        ax.tick_params(direction="out", length=4, width=1.0)
        ax.grid(True, alpha=0.28, color=GRAY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def built_in_voltage(Na, Nd):
        """V_bi = (kT/e) ln(N_a N_d / n_i^2), volts."""
        return float(KT_E * np.log(float(Na) * float(Nd) / NI_SI**2))

    def depletion_edges(Na, Nd, Vbarrier):
        """x_n and x_p (cm) for a positive barrier V_bi - V_a."""
        factor = 2.0 * EPS_S / E_CHARGE * float(Vbarrier)
        xn = np.sqrt(factor * float(Na) / (float(Nd) * (float(Na) + float(Nd))))
        xp = np.sqrt(factor * float(Nd) / (float(Na) * (float(Na) + float(Nd))))
        return float(xn), float(xp)

    def peak_field(Na, xp):
        """|E_max| = e N_a x_p / ε_s, V/cm. The field points from n toward p."""
        return float(E_CHARGE * float(Na) * float(xp) / EPS_S)

    def abrupt_profiles(x_cm, xp, xn, Na, Nd, V_n):
        """ρ, E, and V for the abrupt depletion approximation.

        V = 0 on the neutral p-side and V = V_n on the neutral n-side.
        """
        x_cm = np.asarray(x_cm, dtype=float)
        rho = np.zeros_like(x_cm)
        field = np.zeros_like(x_cm)
        potential = np.zeros_like(x_cm)
        on_p = (x_cm >= -xp) & (x_cm <= 0.0)
        on_n = (x_cm > 0.0) & (x_cm <= xn)
        rho[on_p] = -E_CHARGE * Na
        rho[on_n] = E_CHARGE * Nd
        field[on_p] = -E_CHARGE * Na / EPS_S * (x_cm[on_p] + xp)
        field[on_n] = E_CHARGE * Nd / EPS_S * (x_cm[on_n] - xn)
        potential[on_p] = E_CHARGE * Na / (2.0 * EPS_S) * (x_cm[on_p] + xp) ** 2
        potential[on_n] = V_n - E_CHARGE * Nd / (2.0 * EPS_S) * (x_cm[on_n] - xn) ** 2
        potential[x_cm > xn] = V_n
        return rho, field, potential

    return (
        BLUE,
        EG_EV,
        E_CHARGE,
        GRAY,
        GREEN,
        KT_E,
        NAVY,
        NI_SI,
        ORANGE,
        PINK,
        VA_MAX,
        VA_MIN,
        abrupt_profiles,
        built_in_voltage,
        depletion_edges,
        peak_field,
        plt,
        sci_tex,
        style_ax,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # pn Junction under Applied Bias
    **ECE335 - companion to lectures**

    An abrupt silicon junction. This computes \(\rho(x)\), \(\mathcal{E}(x)\), \(V(x)\), and the band edges when a voltage is applied. The derivation stays on the slides.

    \(V_a\) is the potential of the p-side relative to the n-side. Reverse bias is \(V_a < 0\). On the slides that reverse bias is a positive \(V_R\), so \(V_a = -V_R\) and the barrier in the depletion formulas is \(V_{bi} - V_a\).

    \(V(x)\) is Neamen's electrostatic potential: \(0\) on the neutral p-side and \(V_{bi} - V_a\) on the neutral n-side. Silicon at \(300\,\mathrm{K}\), with \(n_i = 1.5\times 10^{10}\,\mathrm{cm}^{-3}\) and \(kT/e = 0.0259\,\mathrm{V}\). The applied voltage is taken to drop entirely across the space-charge region.
    """)
    return


@app.cell
def _(VA_MAX, VA_MIN, mo):
    Na_log_slider = mo.ui.slider(
        start=14.0,
        stop=18.0,
        step=0.1,
        value=16.0,
        show_value=True,
        label=r"log₁₀(N_a / cm⁻³)",
    )
    Nd_log_slider = mo.ui.slider(
        start=14.0,
        stop=18.0,
        step=0.1,
        value=15.0,
        show_value=True,
        label=r"log₁₀(N_d / cm⁻³)",
    )
    Va_slider = mo.ui.slider(
        start=VA_MIN,
        stop=VA_MAX,
        step=0.1,
        value=0.0,
        show_value=True,
        label=r"V_a (V)",
    )
    _controls = mo.vstack(
        [
            mo.md(r"### Junction parameters"),
            mo.hstack(
                [Na_log_slider, Nd_log_slider, Va_slider],
                justify="start",
                gap=1.0,
                wrap=True,
            ),
        ]
    )
    _controls
    return Na_log_slider, Nd_log_slider, Va_slider


@app.cell
def _(
    BLUE,
    EG_EV,
    E_CHARGE,
    GRAY,
    GREEN,
    KT_E,
    NAVY,
    NI_SI,
    Na_log_slider,
    Nd_log_slider,
    ORANGE,
    PINK,
    Va_slider,
    abrupt_profiles,
    built_in_voltage,
    depletion_edges,
    mo,
    np,
    peak_field,
    plt,
    sci_tex,
    style_ax,
):
    _Na = 10.0 ** float(Na_log_slider.value)
    _Nd = 10.0 ** float(Nd_log_slider.value)
    _Va = float(Va_slider.value)
    _Vbi = built_in_voltage(_Na, _Nd)
    _model_ok = _Va < _Vbi
    _Vbarrier = (_Vbi - _Va) if _model_ok else None

    _xn_eq, _xp_eq = depletion_edges(_Na, _Nd, _Vbi)
    if _model_ok:
        _xn, _xp = depletion_edges(_Na, _Nd, _Vbarrier)
        _Emax = peak_field(_Na, _xp)
        _W = _xn + _xp
    else:
        _xn, _xp, _Emax, _W = None, None, None, None

    _span = max(_xn_eq, _xp_eq)
    if _model_ok:
        _span = max(_span, _xn, _xp)
    _x_extent = max(2.5 * _span, 0.2e-4)
    _x = np.linspace(-_x_extent, _x_extent, 1000)
    _x_um = _x * 1.0e4

    _rho_eq, _E_eq, _V_eq = abrupt_profiles(_x, _xp_eq, _xn_eq, _Na, _Nd, _Vbi)
    if _model_ok:
        _rho, _E, _V = abrupt_profiles(_x, _xp, _xn, _Na, _Nd, _Vbarrier)
        _xp_um = _xp * 1.0e4
        _xn_um = _xn * 1.0e4
        _rho_plot, _E_plot, _V_plot = _rho, _E, _V
        _curve_label = rf"$V_a = {_Va:.1f}\ \mathrm{{V}}$"
    else:
        _rho_plot, _E_plot, _V_plot = _rho_eq, _E_eq, _V_eq
        _xp_um = _xp_eq * 1.0e4
        _xn_um = _xn_eq * 1.0e4
        _curve_label = "thermal equilibrium"

    _fig, _axes = plt.subplots(
        2, 2, figsize=(12.4, 9.8), sharex=True, layout="constrained"
    )
    (_ax_rho, _ax_E), (_ax_V, _ax_band) = _axes

    def _mark_junction(ax):
        ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.axvline(-_xp_um, color=GRAY, linestyle="--", linewidth=1.1)
        ax.axvline(_xn_um, color=GRAY, linestyle="--", linewidth=1.1)

    if _model_ok:
        _ax_rho.plot(
            _x_um,
            _rho_eq / E_CHARGE,
            color=GRAY,
            linestyle="--",
            linewidth=1.6,
            label="zero bias",
        )
    _ax_rho.plot(
        _x_um,
        _rho_plot / E_CHARGE,
        color=BLUE,
        linewidth=2.2,
        label=_curve_label,
    )
    _ax_rho.fill_between(_x_um, _rho_plot / E_CHARGE, 0.0, color=BLUE, alpha=0.15)
    _ax_rho.axhline(0.0, color=GRAY, linewidth=0.6)
    _mark_junction(_ax_rho)
    _ax_rho.set_title(r"(a)  Charge density")
    _ax_rho.set_ylabel(r"$\rho/e$ (cm$^{-3}$)")
    _ax_rho.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    _ax_rho.legend(fontsize=16, loc="upper left")
    style_ax(_ax_rho)

    if _model_ok:
        _ax_E.plot(
            _x_um, _E_eq, color=GRAY, linestyle="--", linewidth=1.6, label="zero bias"
        )
    _ax_E.plot(_x_um, _E_plot, color=ORANGE, linewidth=2.2, label=_curve_label)
    _ax_E.fill_between(_x_um, _E_plot, 0.0, color=ORANGE, alpha=0.15)
    if _model_ok:
        _ax_E.text(
            0.97,
            0.08,
            rf"$|\mathcal{{E}}_{{\max}}| = {_Emax:.2e}$ V/cm",
            transform=_ax_E.transAxes,
            ha="right",
            va="bottom",
            fontsize=16,
            color=ORANGE,
        )
    _ax_E.axhline(0.0, color=GRAY, linewidth=0.6)
    _mark_junction(_ax_E)
    _ax_E.set_title(r"(b)  Electric field")
    _ax_E.set_ylabel(r"$\mathcal{E}$ (V/cm)")
    _ax_E.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    style_ax(_ax_E)

    if _model_ok:
        _ax_V.plot(
            _x_um, _V_eq, color=GRAY, linestyle="--", linewidth=1.6, label="zero bias"
        )
    _ax_V.plot(_x_um, _V_plot, color=GREEN, linewidth=2.2, label=_curve_label)
    if _model_ok:
        _ax_V.axhline(
            _Vbarrier,
            color=GREEN,
            linestyle="--",
            linewidth=1.1,
            label=rf"$V_{{bi}}-V_a = {_Vbarrier:.3f}$ V",
        )
    _ax_V.fill_between(_x_um, _V_plot, 0.0, color=GREEN, alpha=0.10)
    _ax_V.axhline(0.0, color=GRAY, linewidth=0.6)
    _mark_junction(_ax_V)
    _ax_V.set_title(r"(c)  Electrostatic potential")
    _ax_V.set_ylabel(r"$V$ (V)")
    _ax_V.set_xlabel(r"$x$ (μm)")
    _ax_V.legend(fontsize=16, loc="lower right")
    style_ax(_ax_V)

    _Ec_offset = EG_EV / 2.0 + KT_E * np.log(_Na / NI_SI)

    def _band_edges(potential):
        _Ec = _Ec_offset - potential
        _Ev = _Ec - EG_EV
        _Ei = 0.5 * (_Ec + _Ev)
        return _Ec, _Ev, _Ei

    if _model_ok:
        _Ec, _Ev, _Ei = _band_edges(_V)
    else:
        _Ec, _Ev, _Ei = _band_edges(_V_eq)
    _ax_band.plot(_x_um, _Ec, color=BLUE, linewidth=2.2, label=r"$E_c$")
    _ax_band.plot(_x_um, _Ev, color=ORANGE, linewidth=2.2, label=r"$E_v$")
    _ax_band.plot(
        _x_um, _Ei, color=GREEN, linestyle=":", linewidth=1.6, label=r"$E_{Fi}$"
    )
    if _model_ok and abs(_Va) >= 0.001:
        _EFp_line = np.where(_x <= _xn, 0.0, np.nan)
        _EFn_line = np.where(_x >= -_xp, _Va, np.nan)
        _ax_band.plot(_x_um, _EFp_line, color=PINK, linestyle="--", linewidth=2.0, label=r"$E_{Fp}$")
        _ax_band.plot(_x_um, _EFn_line, color=NAVY, linestyle="--", linewidth=2.0, label=r"$E_{Fn}$")
    else:
        _ax_band.axhline(0.0, color=NAVY, linestyle="--", linewidth=1.8, label=r"$E_F$")
    _ax_band.axvspan(-_xp_um, _xn_um, color=GRAY, alpha=0.08)
    if _model_ok:
        _ax_band.text(
            0.98,
            0.96,
            rf"$e(V_{{bi}}-V_a) = {_Vbarrier:.3f}$ eV",
            transform=_ax_band.transAxes,
            ha="right",
            va="top",
            fontsize=16,
            color=GREEN,
        )
    _mark_junction(_ax_band)
    _ax_band.set_title(r"(d)  Energy bands")
    _ax_band.set_ylabel("Energy (eV)")
    _ax_band.set_xlabel(r"$x$ (μm)")
    _ax_band.legend(fontsize=16, loc="upper right")
    style_ax(_ax_band)

    if _Va > _Vbi:
        _tail = (
            r". $V_a > V_{bi}$, so the equations are not valid. " 
            r"Below are the plots for zero bias."
        )
    else:
        _tail = (
            r". Dashed gray curves are zero bias. "
            r"Dashed vertical lines mark \(-x_p\) and \(x_n\)."
        )
    _caption = mo.md(
        rf"""
    $N_a = {sci_tex(_Na)}\ \mathrm{{cm}}^{{-3}}$,
    $N_d = {sci_tex(_Nd)}\ \mathrm{{cm}}^{{-3}}$,
    $V_a = {_Va:.1f}\ \mathrm{{V}}${_tail}
    """
    )
    _eq_vbi = mo.md(
        rf"""
    $$
    V_{{bi}} = \frac{{kT}}{{e}}\ln\left(\frac{{N_a N_d}}{{n_i^2}}\right) = {_Vbi:.3f}\ \mathrm{{V}}
    $$
    """
    )
    if _model_ok:
        _eq_w = mo.md(
            rf"""
    $$
    W = \left[\frac{{2\varepsilon_s}}{{e}}\frac{{N_a+N_d}}{{N_a N_d}}(V_{{bi}}-V_a)\right]^{{1/2}} = {_W * 1.0e4:.3f}\ \mu\mathrm{{m}}
    $$
    """
        )
        _barrier_txt = rf"{_Vbarrier:.3f} V"
        _xn_txt = rf"{_xn * 1.0e4:.4f} μm"
        _xp_txt = rf"{_xp * 1.0e4:.4f} μm"
        _W_txt = rf"{_W * 1.0e4:.3f} μm"
        _E_txt = rf"${sci_tex(_Emax)}$ V/cm"
        _note = None
    else:
        _eq_w = None
        _barrier_txt = "—"
        _xn_txt = "—"
        _xp_txt = "—"
        _W_txt = "—"
        _E_txt = "—"
        _note = None
    if _model_ok and _Va > 0.0 and _Vbarrier < 0.1:
        _note = mo.md(
            r"""
    **Near the edge of the model.** \(V_a\) is close to \(V_{bi}\). The plot still uses \(V_{bi}-V_a\), and the assumption that the whole applied voltage drops across the space-charge region is then a poor one.
    """
        )

    _table = mo.md(
        rf"""
    | Quantity | Symbol | Value |
    |:---|:---:|---:|
    | Built-in potential | \(V_{{bi}}\) | {_Vbi:.3f} V |
    | Applied voltage | \(V_a\) | {_Va:.1f} V |
    | Potential barrier | \(V_{{bi}}-V_a\) | {_barrier_txt} |
    | N-side space-charge edge | \(x_n\) | {_xn_txt} |
    | P-side space-charge edge | \(x_p\) | {_xp_txt} |
    | Total space-charge width | \(W\) | {_W_txt} |
    | Peak electric field | \(\lvert\mathcal{{E}}_{{\max}}\rvert\) | {_E_txt} |
    """
    )
    _blocks = [_caption, _fig, _eq_vbi]
    if _eq_w is not None:
        _blocks.append(_eq_w)
    if _note is not None:
        _blocks.append(_note)
    _blocks.append(_table)
    mo.vstack(_blocks)
    return


@app.cell
def _(mo):
    _explain = mo.accordion(
        {
            "What the panels are doing": mo.md(
                r"""
    \(\rho/e\) equals \(-N_a\) between \(-x_p\) and the junction, and \(+N_d\) between the junction and \(x_n\). It is zero in the neutral regions. \(x_n/x_p = N_a/N_d\), so the space-charge region reaches farther into the lighter-doped side.

    \(\mathcal{E}\) points from the n-side toward the p-side, so the trace is negative. Its peak sits at \(x = 0\), and \(|\mathcal{E}_{\max}| = e N_a x_p/\varepsilon_s\).

    \(V(x)\) is parabolic inside the space-charge region. \(E_c\) follows \(-V\), and \(E_v = E_c - E_g\). With \(E_{Fp}\) set to \(0\), the quasi-Fermi split on the plot is \(E_{Fn}-E_{Fp} = eV_a\) in eV.

    A more negative \(V_a\) raises the barrier, widens \(W\), and increases \(|\mathcal{E}_{\max}|\). A positive \(V_a\) does the opposite, while \(V_a\) stays below \(V_{bi}\).
    """
            )
        }
    )
    _explain
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding

    Use the sliders and the table, then open the answers.

    1. At the default dopings and \(V_a = 0\), which side holds most of \(W\)?
    2. Set \(V_a = -4\,\mathrm{V}\) at those dopings. What happens to \(W\) and \(|\mathcal{E}_{\max}|\)?
    3. Slide \(V_a\) up through \(V_{bi}\). Why do the biased curves drop out?
    """)
    return


@app.cell
def _(built_in_voltage, depletion_edges, mo, peak_field):
    _Na = 1.0e16
    _Nd = 1.0e15
    _Vbi = built_in_voltage(_Na, _Nd)
    _xn0, _xp0 = depletion_edges(_Na, _Nd, _Vbi)
    _W0 = (_xn0 + _xp0) * 1.0e4
    _E0 = peak_field(_Na, _xp0)
    _Vb = _Vbi - (-4.0)
    _xn4, _xp4 = depletion_edges(_Na, _Nd, _Vb)
    _W4 = (_xn4 + _xp4) * 1.0e4
    _E4 = peak_field(_Na, _xp4)
    mo.accordion(
        {
            "Answers": mo.md(
                rf"""
    1. The n-side. \(N_d\) is ten times smaller than \(N_a\), so \(x_n/x_p = 10\). At \(V_a = 0\), \(x_n = {_xn0 * 1.0e4:.3f}\,\mu\mathrm{{m}}\), \(x_p = {_xp0 * 1.0e4:.4f}\,\mu\mathrm{{m}}\), and \(W = {_W0:.3f}\,\mu\mathrm{{m}}\).
    2. The barrier becomes \(V_{{bi}}+4 = {_Vb:.3f}\,\mathrm{{V}}\). Both \(W\) and \(|\mathcal{{E}}_{{\max}}|\) grow as the square root of the barrier: \(W = {_W4:.3f}\,\mu\mathrm{{m}}\) and \(|\mathcal{{E}}_{{\max}}| = {_E4:.2e}\,\mathrm{{V/cm}}\), from \({_W0:.3f}\,\mu\mathrm{{m}}\) and \({_E0:.2e}\,\mathrm{{V/cm}}\) at zero bias.
    3. \(x_n\), \(x_p\), and \(W\) are square roots of \(V_{{bi}}-V_a\). Once \(V_a\) reaches \(V_{{bi}}\), that barrier is no longer positive and the depletion approximation used here stops.
    """
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
