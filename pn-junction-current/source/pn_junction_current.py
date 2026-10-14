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

    # Silicon at 300 K, as on the slides and in the other pn-junction companions.
    VT_300 = 0.0259
    NI_300 = 1.5e10
    EG_EV = 1.12
    K_B_EV = VT_300 / 300.0  # k/e in V/K, consistent with V_t = 0.0259 V at 300 K
    E_CHARGE = 1.6e-19
    EPS_S = 11.7 * 8.85e-14
    # Neamen Example 8.2 transport parameters, held fixed in temperature.
    D_N = 25.0
    D_P = 10.0

    BLUE = "#0072B2"
    ORANGE = "#D55E00"
    GREEN = "#009E73"
    NAVY = "#0f4c81"
    GRAY = "#6B6B6B"

    def sci_tex(x, digits=2):
        x = float(x)
        if not np.isfinite(x) or x == 0.0:
            return "0"
        sign = "-" if x < 0 else ""
        ax = abs(x)
        exp = int(np.floor(np.log10(ax)))
        man = ax / (10.0**exp)
        if round(man, digits) >= 10.0:
            man /= 10.0
            exp += 1
        return rf"{sign}{man:.{digits}f}\times 10^{{{exp}}}"

    def style_ax(ax):
        ax.tick_params(direction="out", length=4, width=1.0)
        ax.grid(True, alpha=0.28, color=GRAY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def thermal_voltage(T):
        """V_t = kT/e, scaled from 0.0259 V at 300 K."""
        return VT_300 * np.asarray(T, dtype=float) / 300.0

    def ni_at(T):
        """n_i(T) from n_i^2 ∝ T^3 exp(-E_g/kT), pinned to 1.5e10 cm^-3 at 300 K."""
        T = np.asarray(T, dtype=float)
        return NI_300 * (T / 300.0) ** 1.5 * np.exp(
            -EG_EV / (2.0 * K_B_EV) * (1.0 / T - 1.0 / 300.0)
        )

    def built_in_voltage(Na, Nd):
        """V_bi = V_t ln(N_a N_d / n_i^2) at 300 K, volts."""
        return float(VT_300 * np.log(float(Na) * float(Nd) / NI_300**2))

    def depletion_edges(Na, Nd, Vbarrier):
        """x_n and x_p (cm) for a positive barrier V_bi - V_a."""
        factor = 2.0 * EPS_S / E_CHARGE * float(Vbarrier)
        xn = np.sqrt(factor * float(Na) / (float(Nd) * (float(Na) + float(Nd))))
        xp = np.sqrt(factor * float(Nd) / (float(Na) * (float(Na) + float(Nd))))
        return float(xn), float(xp)

    def model_limit(Na, Nd, T):
        """Largest V_a in the ideal-diode model at T: the lower of V_bi(T) and the
        low-injection limit, where the excess minority carriers on the lighter-doped
        side reach 0.1 of the majority concentration."""
        T = np.asarray(T, dtype=float)
        Vt = thermal_voltage(T)
        ni2 = ni_at(T) ** 2
        Nmin = min(float(Na), float(Nd))
        V_bi = Vt * np.log(float(Na) * float(Nd) / ni2)
        V_li = Vt * np.log1p(0.1 * Nmin**2 / ni2)
        return np.minimum(V_bi, V_li)

    def saturation_current(Na, Nd, Ln, Lp, T=300.0):
        """J_s = e n_i^2 [D_n/(L_n N_a) + D_p/(L_p N_d)], A/cm^2."""
        ni2 = ni_at(T) ** 2
        return E_CHARGE * ni2 * (D_N / (Ln * Na) + D_P / (Lp * Nd))

    return (
        BLUE,
        D_N,
        D_P,
        EG_EV,
        E_CHARGE,
        GRAY,
        GREEN,
        NAVY,
        NI_300,
        ORANGE,
        VT_300,
        built_in_voltage,
        depletion_edges,
        model_limit,
        ni_at,
        plt,
        saturation_current,
        sci_tex,
        style_ax,
        thermal_voltage,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # pn Junction Current
    **ECE335 - companion to lectures**

    The ideal-diode analysis of an abrupt silicon pn junction, computed step by step. Minority carriers at the space charge edges, their distributions in the neutral regions, the hole and electron current components, and the \(J\)–\(V\) curve versus temperature. The derivation stays on the slides.

    Silicon, long-diode limit, with \(n_i = 1.5\times 10^{10}\,\mathrm{cm}^{-3}\) and \(V_t = kT/e = 0.0259\,\mathrm{V}\) at \(300\,\mathrm{K}\). \(D_n = 25\) and \(D_p = 10\,\mathrm{cm^2/s}\) as in Neamen Example 8.2. \(V_a\) is the voltage of the p side relative to the n side.
    """)
    return


@app.cell
def _(mo):
    Na_edge_slider = mo.ui.slider(
        start=14.0,
        stop=19.0,
        step=0.1,
        value=19.0,
        show_value=True,
        label=r"log₁₀(N_a / cm⁻³)",
    )
    Nd_edge_slider = mo.ui.slider(
        start=14.0,
        stop=19.0,
        step=0.1,
        value=16.0,
        show_value=True,
        label=r"log₁₀(N_d / cm⁻³)",
    )
    Va_edge_slider = mo.ui.slider(
        start=-0.5,
        stop=0.7,
        step=0.01,
        value=0.6,
        show_value=True,
        label=r"V_a (V)",
    )
    _controls = mo.vstack(
        [
            mo.md(r"## 1. Minority and Excess Carrier Concentrations"),
            mo.md(
                r"Concentrations at the space charge edges, \(x = -x_p\) and \(x = x_n\). The defaults are the slide example: \(N_a = 10^{19}\), \(N_d = 10^{16}\,\mathrm{cm}^{-3}\), \(V_a = 0.60\,\mathrm{V}\)."
            ),
            mo.hstack(
                [Na_edge_slider, Nd_edge_slider, Va_edge_slider],
                justify="start",
                gap=1.0,
                wrap=True,
            ),
        ]
    )
    _controls
    return Na_edge_slider, Nd_edge_slider, Va_edge_slider


@app.cell
def _(
    NI_300,
    Na_edge_slider,
    Nd_edge_slider,
    VT_300,
    Va_edge_slider,
    mo,
    np,
    sci_tex,
):
    _Na = 10.0 ** float(Na_edge_slider.value)
    _Nd = 10.0 ** float(Nd_edge_slider.value)
    _Va = float(Va_edge_slider.value)
    _np0 = NI_300**2 / _Na
    _pn0 = NI_300**2 / _Nd
    _boltz = float(np.exp(_Va / VT_300))
    _np_edge = _np0 * _boltz
    _pn_edge = _pn0 * _boltz
    _dnp = _np0 * np.expm1(_Va / VT_300)
    _dpn = _pn0 * np.expm1(_Va / VT_300)

    _eq_equil = mo.md(rf"""
    $$
    \begin{{aligned}}
    n_{{p0}} &= \frac{{n_i^2}}{{N_a}} = \frac{{({sci_tex(NI_300, 1)})^2}}{{{sci_tex(_Na)}}} = {sci_tex(_np0)}\ \mathrm{{cm^{{-3}}}} \\
    p_{{n0}} &= \frac{{n_i^2}}{{N_d}} = \frac{{({sci_tex(NI_300, 1)})^2}}{{{sci_tex(_Nd)}}} = {sci_tex(_pn0)}\ \mathrm{{cm^{{-3}}}}
    \end{{aligned}}
    $$
    """)
    _eq_boltz = mo.md(rf"""
    $$
    \exp\!\left(\frac{{eV_a}}{{kT}}\right) = \exp\!\left(\frac{{{_Va:.2f}}}{{{VT_300}}}\right) = {sci_tex(_boltz)}
    $$
    """)
    _eq_edges = mo.md(rf"""
    $$
    \begin{{aligned}}
    n_p(-x_p) &= n_{{p0}}\exp\!\left(\frac{{eV_a}}{{kT}}\right) = {sci_tex(_np_edge)}\ \mathrm{{cm^{{-3}}}} \\
    p_n(x_n) &= p_{{n0}}\exp\!\left(\frac{{eV_a}}{{kT}}\right) = {sci_tex(_pn_edge)}\ \mathrm{{cm^{{-3}}}}
    \end{{aligned}}
    $$
    """)
    _eq_excess = mo.md(rf"""
    $$
    \begin{{aligned}}
    \delta n_p(-x_p) &= n_{{p0}}\left[\exp\!\left(\frac{{eV_a}}{{kT}}\right)-1\right] = {sci_tex(_dnp)}\ \mathrm{{cm^{{-3}}}} \\
    \delta p_n(x_n) &= p_{{n0}}\left[\exp\!\left(\frac{{eV_a}}{{kT}}\right)-1\right] = {sci_tex(_dpn)}\ \mathrm{{cm^{{-3}}}}
    \end{{aligned}}
    $$
    """)

    _worst = max(_dnp / _Na, _dpn / _Nd)
    if _Va >= 4.0 * VT_300:
        _sense = r"Forward bias: minority carriers are injected. \(n_p(-x_p) \approx \delta n_p(-x_p)\) and \(p_n(x_n) \approx \delta p_n(x_n)\): the minority carriers at the space charge edges are almost all excess carriers."
    elif _Va > 0.0:
        _sense = r"Forward bias: minority carriers are injected. Below about \(4V_t \approx 0.1\,\mathrm{V}\), \(\delta n_p(-x_p)\) and \(\delta p_n(x_n)\) are still comparable to \(n_{p0}\) and \(p_{n0}\)."
    elif _Va < 0.0:
        _sense = r"Reverse bias: minority carriers are extracted. The excess is negative and approaches \(-n_{p0}\), \(-p_{n0}\)."
    else:
        _sense = r"Thermal equilibrium: no excess carriers."
    if _worst < 0.1:
        _inj = rf"Low injection holds: \(\delta n_p/p_{{p0}} = {sci_tex(_dnp / _Na)}\) and \(\delta p_n/n_{{n0}} = {sci_tex(_dpn / _Nd)}\), both \(\ll 1\)."
    else:
        _inj = rf"**Outside the model.** \(\delta n_p/p_{{p0}} = {sci_tex(_dnp / _Na)}\) and \(\delta p_n/n_{{n0}} = {sci_tex(_dpn / _Nd)}\). Low injection needs both \(\ll 1\)."

    mo.vstack(
        [
            mo.md(r"**Equilibrium minority carriers** (neutral regions, far from the junction)"),
            _eq_equil,
            mo.md(r"**(i) Minority carriers at the space charge edges**"),
            _eq_boltz,
            _eq_edges,
            mo.md(r"**(ii) Excess minority carriers**, \(\delta n_p = n_p - n_{p0}\) and \(\delta p_n = p_n - p_{n0}\)"),
            _eq_excess,
            mo.md(_sense + " " + _inj),
        ]
    )
    return


@app.cell
def _(mo):
    Na_slider = mo.ui.slider(
        start=14.0,
        stop=18.0,
        step=0.1,
        value=16.0,
        show_value=True,
        label=r"log₁₀(N_a / cm⁻³)",
    )
    Nd_slider = mo.ui.slider(
        start=14.0,
        stop=18.0,
        step=0.1,
        value=16.0,
        show_value=True,
        label=r"log₁₀(N_d / cm⁻³)",
    )
    Va_slider = mo.ui.slider(
        start=-0.5,
        stop=0.65,
        step=0.01,
        value=0.4,
        show_value=True,
        label=r"V_a (V)",
    )
    taun_log_slider = mo.ui.slider(
        start=-8.0,
        stop=-5.0,
        step=0.1,
        value=-6.3,
        show_value=True,
        label=r"log₁₀(τ_n0 / s)",
    )
    taup_log_slider = mo.ui.slider(
        start=-8.0,
        stop=-5.0,
        step=0.1,
        value=-6.3,
        show_value=True,
        label=r"log₁₀(τ_p0 / s)",
    )
    junction_controls = mo.vstack(
        [
            mo.hstack(
                [Na_slider, Nd_slider, Va_slider],
                justify="start",
                gap=1.0,
                wrap=True,
            ),
            mo.hstack(
                [taun_log_slider, taup_log_slider],
                justify="start",
                gap=1.0,
                wrap=True,
            ),
        ]
    )
    mo.vstack(
        [
            mo.md(r"## 2. Minority Carrier Distributions"),
            mo.md(
                r"These sliders also drive Section 3, and the dopings and lifetimes drive Section 4. The defaults are Neamen Example 8.2: \(N_a = N_d = 10^{16}\,\mathrm{cm}^{-3}\), \(\tau_{n0} = \tau_{p0} = 5\times 10^{-7}\,\mathrm{s}\), here \(10^{-6.3} = 5.01\times 10^{-7}\,\mathrm{s}\)."
            ),
            junction_controls,
        ]
    )
    return (
        Na_slider,
        Nd_slider,
        Va_slider,
        junction_controls,
        taun_log_slider,
        taup_log_slider,
    )


@app.cell
def _(
    D_N,
    D_P,
    E_CHARGE,
    NI_300,
    Na_slider,
    Nd_slider,
    VT_300,
    Va_slider,
    built_in_voltage,
    depletion_edges,
    np,
    taun_log_slider,
    taup_log_slider,
):
    # Shared state for Sections 2 and 3: the long-diode solution of Neamen Eqs. (8.14)-(8.15).
    jn_Na = 10.0 ** float(Na_slider.value)
    jn_Nd = 10.0 ** float(Nd_slider.value)
    jn_Va = float(Va_slider.value)
    jn_taun = 10.0 ** float(taun_log_slider.value)
    jn_taup = 10.0 ** float(taup_log_slider.value)
    jn_Ln = float(np.sqrt(D_N * jn_taun))
    jn_Lp = float(np.sqrt(D_P * jn_taup))
    jn_Vbi = built_in_voltage(jn_Na, jn_Nd)
    jn_np0 = NI_300**2 / jn_Na
    jn_pn0 = NI_300**2 / jn_Nd
    jn_dnp_edge = jn_np0 * float(np.expm1(jn_Va / VT_300))
    jn_dpn_edge = jn_pn0 * float(np.expm1(jn_Va / VT_300))
    jn_valid = jn_Va < jn_Vbi
    if jn_valid:
        jn_xn, jn_xp = depletion_edges(jn_Na, jn_Nd, jn_Vbi - jn_Va)
    else:
        # Depletion approximation has no solution; draw with W = 0 so the plots stay visible.
        jn_xn, jn_xp = 0.0, 0.0
    jn_low_inj = max(jn_dnp_edge / jn_Na, jn_dpn_edge / jn_Nd) < 0.1
    jn_Jp_edge = E_CHARGE * D_P * jn_dpn_edge / jn_Lp
    jn_Jn_edge = E_CHARGE * D_N * jn_dnp_edge / jn_Ln
    jn_J = jn_Jp_edge + jn_Jn_edge
    # Share of J carried by holes across the junction. Independent of V_a, so safe at V_a = 0.
    _gp = D_P * jn_pn0 / jn_Lp
    _gn = D_N * jn_np0 / jn_Ln
    jn_hole_frac = _gp / (_gp + _gn)
    return (
        jn_J,
        jn_Jn_edge,
        jn_Jp_edge,
        jn_Ln,
        jn_Lp,
        jn_Na,
        jn_Nd,
        jn_Va,
        jn_Vbi,
        jn_dnp_edge,
        jn_dpn_edge,
        jn_hole_frac,
        jn_low_inj,
        jn_np0,
        jn_pn0,
        jn_valid,
        jn_xn,
        jn_xp,
    )


@app.cell
def _(mo):
    def model_warning(valid, low_inj, Va, Vbi, Na, Nd, dnp, dpn, sci_tex):
        if not valid:
            return mo.md(
                rf"**Outside the model.** \(V_a = {Va:.2f}\,\mathrm{{V}}\) is not below \(V_{{bi}} = {Vbi:.3f}\,\mathrm{{V}}\), so there is no space charge region left in the depletion approximation. The curves below are drawn with \(W = 0\). Lower \(V_a\) or raise the doping."
            )
        if not low_inj:
            return mo.md(
                rf"**Outside the model.** Low injection fails: \(\delta n_p(-x_p)/p_{{p0}} = {sci_tex(dnp / Na)}\) and \(\delta p_n(x_n)/n_{{n0}} = {sci_tex(dpn / Nd)}\). The curves still follow the ideal-diode equations, but a real diode departs from them here."
            )
        return None

    return (model_warning,)


@app.cell
def _(
    BLUE,
    GRAY,
    ORANGE,
    jn_Ln,
    jn_Lp,
    jn_Na,
    jn_Nd,
    jn_Va,
    jn_Vbi,
    jn_dnp_edge,
    jn_dpn_edge,
    jn_low_inj,
    jn_np0,
    jn_pn0,
    jn_valid,
    jn_xn,
    jn_xp,
    mo,
    model_warning,
    np,
    plt,
    sci_tex,
    style_ax,
):
    _warn = model_warning(
        jn_valid, jn_low_inj, jn_Va, jn_Vbi, jn_Na, jn_Nd, jn_dnp_edge, jn_dpn_edge, sci_tex
    )
    _xp_r = np.linspace(-jn_xp - 3.0 * jn_Ln, -jn_xp, 400)
    _xn_r = np.linspace(jn_xn, jn_xn + 3.0 * jn_Lp, 400)
    # The log panel runs until |excess| falls to ~1% of equilibrium (3 to 30 diffusion lengths).
    _span_n = float(np.clip(np.log(max(abs(jn_dnp_edge) / jn_np0, 1.0) / 0.01), 3.0, 30.0))
    _span_p = float(np.clip(np.log(max(abs(jn_dpn_edge) / jn_pn0, 1.0) / 0.01), 3.0, 30.0))
    _xp_log = np.linspace(-jn_xp - _span_n * jn_Ln, -jn_xp, 600)
    _xn_log = np.linspace(jn_xn, jn_xn + _span_p * jn_Lp, 600)
    _dnp = jn_dnp_edge * np.exp((_xp_r + jn_xp) / jn_Ln)
    _dpn = jn_dpn_edge * np.exp(-(_xn_r - jn_xn) / jn_Lp)
    _xp_um, _xn_um = _xp_r * 1.0e4, _xn_r * 1.0e4
    _ep, _en = -jn_xp * 1.0e4, jn_xn * 1.0e4

    _fig, (_ax_lin, _ax_log) = plt.subplots(
        1, 2, figsize=(13.0, 5.6), layout="constrained"
    )

    _ax_lin.plot(_xp_um, _dnp, color=BLUE, linewidth=2.4, label=r"$\delta n_p(x)$")
    _ax_lin.plot(_xn_um, _dpn, color=ORANGE, linewidth=2.4, label=r"$\delta p_n(x)$")
    _ax_lin.axvspan(_ep, _en, color=GRAY, alpha=0.15)
    _ax_lin.axhline(0.0, color=GRAY, linewidth=0.6)
    if jn_Va != 0.0:
        _yn = jn_dnp_edge / np.e
        _yp = jn_dpn_edge / np.e
        _ax_lin.annotate(
            "",
            xy=(_ep - jn_Ln * 1.0e4, _yn),
            xytext=(_ep, _yn),
            arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.5),
        )
        _ax_lin.text(
            _ep - jn_Ln * 1.0e4,
            _yn,
            rf"$L_n$ = {jn_Ln * 1.0e4:.1f} μm",
            color=BLUE,
            ha="right",
            va="bottom" if jn_Va > 0 else "top",
            fontsize=16,
        )
        _ax_lin.annotate(
            "",
            xy=(_en + jn_Lp * 1.0e4, _yp),
            xytext=(_en, _yp),
            arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.5),
        )
        _ax_lin.text(
            _en + jn_Lp * 1.0e4,
            _yp,
            rf"$L_p$ = {jn_Lp * 1.0e4:.1f} μm",
            color=ORANGE,
            ha="left",
            va="bottom" if jn_Va > 0 else "top",
            fontsize=16,
        )
    _ax_lin.set_xlabel(r"$x$ (μm)")
    _ax_lin.set_ylabel(r"Excess concentration (cm$^{-3}$)")
    _ax_lin.set_title(rf"(a)  Excess minority carriers, $V_a$ = {jn_Va:.2f} V")
    _ax_lin.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    _ax_lin.legend(fontsize=16, loc="best")
    style_ax(_ax_lin)

    _ax_log.semilogy(_xp_log * 1e4, jn_np0 + jn_dnp_edge * np.exp((_xp_log + jn_xp) / jn_Ln), color=BLUE, linewidth=2.4, label=r"$n_p(x)$")
    _ax_log.semilogy(_xn_log * 1e4, jn_pn0 + jn_dpn_edge * np.exp(-(_xn_log - jn_xn) / jn_Lp), color=ORANGE, linewidth=2.4, label=r"$p_n(x)$")
    _ax_log.axhline(
        jn_np0, color=BLUE, linestyle=":", linewidth=1.4, label=rf"$n_{{p0}} = {sci_tex(jn_np0)}$"
    )
    _ax_log.axhline(
        jn_pn0, color=ORANGE, linestyle=":", linewidth=1.4, label=rf"$p_{{n0}} = {sci_tex(jn_pn0)}$"
    )
    _ax_log.axvspan(_ep, _en, color=GRAY, alpha=0.15)
    _ax_log.set_xlabel(r"$x$ (μm)")
    _ax_log.set_ylabel(r"Concentration (cm$^{-3}$)")
    _ax_log.set_title(r"(b)  Total minority carriers")
    _ax_log.legend(fontsize=14, loc="best")
    style_ax(_ax_log)

    _table = mo.md(
        rf"""
    | \(V_{{bi}}\) | \(W\) | \(L_n\) | \(L_p\) | \(\delta n_p(-x_p)\) | \(\delta p_n(x_n)\) |
    |:--:|:--:|:--:|:--:|:--:|:--:|
    | \({jn_Vbi:.3f}\,\mathrm{{V}}\) | \({(jn_xn + jn_xp) * 1.0e4:.3f}\,\mu\mathrm{{m}}\) | \({jn_Ln * 1.0e4:.1f}\,\mu\mathrm{{m}}\) | \({jn_Lp * 1.0e4:.1f}\,\mu\mathrm{{m}}\) | \({sci_tex(jn_dnp_edge)}\,\mathrm{{cm^{{-3}}}}\) | \({sci_tex(jn_dpn_edge)}\,\mathrm{{cm^{{-3}}}}\) |
    """
    )
    _note = mo.md(
        r"The grey band is the space charge region, \(-x_p \le x \le x_n\). The excess decays as \(\exp[(x + x_p)/L_n]\) on the p side and \(\exp[-(x - x_n)/L_p]\) on the n side, with \(L = \sqrt{D\tau_0}\). Use a negative \(V_a\) to see extraction."
    )
    _blocks = [_fig, _table, _note]
    if _warn is not None:
        _blocks.insert(0, _warn)
    _out = mo.vstack(_blocks)
    _out
    return


@app.cell
def _(junction_controls, mo):
    mo.vstack(
        [
            mo.md(r"## 3. Hole and Electron Currents"),
            mo.md(
                r"Same sliders as Section 2, repeated here. Moving either copy updates both sections."
            ),
            junction_controls,
        ]
    )
    return


@app.cell
def _(
    BLUE,
    D_N,
    D_P,
    E_CHARGE,
    GRAY,
    NAVY,
    ORANGE,
    jn_J,
    jn_Jn_edge,
    jn_Jp_edge,
    jn_Ln,
    jn_Lp,
    jn_Na,
    jn_Nd,
    jn_Va,
    jn_Vbi,
    jn_dnp_edge,
    jn_dpn_edge,
    jn_hole_frac,
    jn_low_inj,
    jn_valid,
    jn_xn,
    jn_xp,
    mo,
    model_warning,
    np,
    plt,
    sci_tex,
    style_ax,
):
    _warn = model_warning(
        jn_valid, jn_low_inj, jn_Va, jn_Vbi, jn_Na, jn_Nd, jn_dnp_edge, jn_dpn_edge, sci_tex
    )
    _xp_r = np.linspace(-jn_xp - 3.0 * jn_Ln, -jn_xp, 400)
    _xd = np.linspace(-jn_xp, jn_xn, 40)
    _xn_r = np.linspace(jn_xn, jn_xn + 3.0 * jn_Lp, 400)
    # Minority diffusion currents, Eqs. (8.28) and (8.29); majority = J - minority.
    _Jn_p = E_CHARGE * D_N * jn_dnp_edge / jn_Ln * np.exp((_xp_r + jn_xp) / jn_Ln)
    _Jp_n = E_CHARGE * D_P * jn_dpn_edge / jn_Lp * np.exp(-(_xn_r - jn_xn) / jn_Lp)
    _Jp_p = jn_J - _Jn_p
    _Jn_n = jn_J - _Jp_n

    # Pick A, mA, μA, nA or pA so that |J| reads between 1 and 1000.
    _s, _J_lab = 1.0e12, "pA/cm$^2$"
    for _scale, _name in [(1.0, "A"), (1.0e3, "mA"), (1.0e6, "μA"), (1.0e9, "nA")]:
        if abs(jn_J) * _scale >= 1.0:
            _s, _J_lab = _scale, f"{_name}/cm$^2$"
            break
    _fig, _ax = plt.subplots(figsize=(12.4, 5.8), layout="constrained")
    _ax.plot(_xp_r * 1e4, _Jn_p * _s, color=BLUE, linewidth=2.4, label=r"$J_n$ (electrons)")
    _ax.plot(_xd * 1e4, np.full_like(_xd, jn_Jn_edge) * _s, color=BLUE, linewidth=2.4)
    _ax.plot(_xn_r * 1e4, _Jn_n * _s, color=BLUE, linewidth=2.4, linestyle="--")
    _ax.plot(_xp_r * 1e4, _Jp_p * _s, color=ORANGE, linewidth=2.4, linestyle="--")
    _ax.plot(_xd * 1e4, np.full_like(_xd, jn_Jp_edge) * _s, color=ORANGE, linewidth=2.4)
    _ax.plot(_xn_r * 1e4, _Jp_n * _s, color=ORANGE, linewidth=2.4, label=r"$J_p$ (holes)")
    _ax.axhline(jn_J * _s, color=NAVY, linewidth=3.0, label=r"$J = J_n + J_p$")
    _ax.axvspan(-jn_xp * 1e4, jn_xn * 1e4, color=GRAY, alpha=0.15)
    _ax.axhline(0.0, color=GRAY, linewidth=0.6)
    _ax.set_xlabel(r"$x$ (μm)")
    _ax.set_ylabel(rf"Current density ({_J_lab})")
    _ax.set_title(rf"Solid: minority diffusion. Dashed: majority current.  $V_a$ = {jn_Va:.2f} V")
    _ax.legend(fontsize=16, loc="center left" if jn_Va >= 0 else "lower left")
    style_ax(_ax)

    _eq = mo.md(rf"""
    $$
    \begin{{aligned}}
    J_p(x_n) &= \frac{{eD_p\,p_{{n0}}}}{{L_p}}\left[\exp\!\left(\frac{{eV_a}}{{kT}}\right)-1\right] = {sci_tex(jn_Jp_edge)}\ \mathrm{{A/cm^2}} \\
    J_n(-x_p) &= \frac{{eD_n\,n_{{p0}}}}{{L_n}}\left[\exp\!\left(\frac{{eV_a}}{{kT}}\right)-1\right] = {sci_tex(jn_Jn_edge)}\ \mathrm{{A/cm^2}} \\
    J &= J_p(x_n) + J_n(-x_p) = {sci_tex(jn_J)}\ \mathrm{{A/cm^2}}
    \end{{aligned}}
    $$
    """)
    _share = mo.md(
        rf"Holes carry \({100.0 * jn_hole_frac:.1f}\%\) of the current across the junction and electrons \({100.0 * (1.0 - jn_hole_frac):.1f}\%\), at any \(V_a\). In the neutral regions the minority diffusion current decays over \(L\), and the majority current makes up the rest so that \(J\) stays constant."
    )
    _blocks = [_fig, _eq, _share]
    if _warn is not None:
        _blocks.insert(0, _warn)
    _out = mo.vstack(_blocks)
    _out
    return


@app.cell
def _(mo):
    T_slider = mo.ui.slider(
        start=250,
        stop=400,
        step=5,
        value=300,
        show_value=True,
        label=r"T (K)",
    )
    Jfix_log_slider = mo.ui.slider(
        start=-3.0,
        stop=1.0,
        step=0.1,
        value=-1.0,
        show_value=True,
        label=r"log₁₀(J / A cm⁻²) for dV_a/dT",
    )
    return Jfix_log_slider, T_slider


@app.cell
def _(
    Jfix_log_slider,
    Na_slider,
    Nd_slider,
    T_slider,
    mo,
    taun_log_slider,
    taup_log_slider,
):
    mo.vstack(
        [
            mo.md(r"## 4. I–V Characteristics and Temperature"),
            mo.md(
                r"\(J_s \propto n_i^2 \propto T^3\exp(-E_g/kT)\), and \(V_t = kT/e\) also changes with \(T\). \(E_g\), \(D\), and \(\tau_0\) are held at their \(300\,\mathrm{K}\) values. The dopings and lifetimes are the Section 2 sliders, repeated here. \(V_a\) is swept."
            ),
            mo.hstack(
                [Na_slider, Nd_slider, taun_log_slider, taup_log_slider],
                justify="start",
                gap=1.0,
                wrap=True,
            ),
            mo.hstack([T_slider, Jfix_log_slider], justify="start", gap=1.0, wrap=True),
        ]
    )
    return


@app.cell
def _(
    BLUE,
    EG_EV,
    GRAY,
    GREEN,
    Jfix_log_slider,
    ORANGE,
    T_slider,
    jn_Ln,
    jn_Lp,
    jn_Na,
    jn_Nd,
    mo,
    model_limit,
    ni_at,
    np,
    plt,
    saturation_current,
    sci_tex,
    style_ax,
    thermal_voltage,
):
    _T = float(T_slider.value)
    _Vt = float(thermal_voltage(_T))
    _Js = float(saturation_current(jn_Na, jn_Nd, jn_Ln, jn_Lp, _T))
    _Js300 = float(saturation_current(jn_Na, jn_Nd, jn_Ln, jn_Lp, 300.0))
    _Vt300 = float(thermal_voltage(300.0))

    _V = np.linspace(-0.3, 0.75, 600)
    _J = _Js * np.expm1(_V / _Vt)
    _J300 = _Js300 * np.expm1(_V / _Vt300)

    # Forward voltage at fixed J versus T: V_a = V_t ln(1 + J/J_s).
    _Jfix = 10.0 ** float(Jfix_log_slider.value)
    _T_arr = np.linspace(250.0, 400.0, 301)
    _Va_fix = thermal_voltage(_T_arr) * np.log1p(_Jfix / saturation_current(jn_Na, jn_Nd, jn_Ln, jn_Lp, _T_arr))
    _dVdT = np.gradient(_Va_fix, _T_arr)
    _Va_here = float(thermal_voltage(_T) * np.log1p(_Jfix / _Js))
    _dVdT_here = float(np.interp(_T, _T_arr, _dVdT))

    # Validity: above V_lim the ideal-diode model no longer applies (low injection or V_a < V_bi fails).
    _Vlim = float(model_limit(jn_Na, jn_Nd, _T))
    _Vlim_arr = model_limit(jn_Na, jn_Nd, _T_arr)
    _ok = _Va_fix <= _Vlim_arr

    # Linear axis capped at 100 mA/cm^2 so the turn-on knee and its shift with T are visible.
    _Jtop = 0.1

    _fig, (_ax_lin, _ax_log) = plt.subplots(1, 2, figsize=(13.0, 5.6), layout="constrained")
    if _T != 300.0:
        _ax_lin.plot(_V, _J300 * 1e3, color=GRAY, linestyle="--", linewidth=1.8, label="300 K")
    _ax_lin.plot(_V, _J * 1e3, color=BLUE, linewidth=2.4, label=f"{_T:.0f} K")
    _ax_lin.axhline(0.0, color=GRAY, linewidth=0.6)
    _ax_lin.axvline(0.0, color=GRAY, linewidth=0.6)
    if _Vlim < _V[-1]:
        _ax_lin.axvspan(_Vlim, _V[-1], color=GRAY, alpha=0.18, label="outside model")
    _ax_lin.set_ylim(-0.03 * _Jtop * 1e3, 1.05 * _Jtop * 1e3)
    _ax_lin.set_xlabel(r"$V_a$ (V)")
    _ax_lin.set_ylabel(r"$J$ (mA/cm$^2$)")
    _ax_lin.set_title(r"(a)  $J$ versus $V_a$, up to 100 mA/cm$^2$")
    _ax_lin.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
    _ax_lin.legend(fontsize=16, loc="upper left")
    style_ax(_ax_lin)

    if _T != 300.0:
        _ax_log.semilogy(_V, np.abs(_J300), color=GRAY, linestyle="--", linewidth=1.8, label="300 K")
    _ax_log.semilogy(_V, np.abs(_J), color=BLUE, linewidth=2.4, label=f"{_T:.0f} K")
    if _Vlim < _V[-1]:
        _ax_log.axvspan(_Vlim, _V[-1], color=GRAY, alpha=0.18)
    _ax_log.axhline(_Js, color=ORANGE, linestyle=":", linewidth=1.6, label=rf"$J_s = {sci_tex(_Js)}$ A/cm$^2$")
    _ax_log.set_xlabel(r"$V_a$ (V)")
    _ax_log.set_ylabel(r"$|J|$ (A/cm$^2$)")
    _ax_log.set_title(rf"(b)  $|J|$, slope {1e3 * _Vt * np.log(10.0):.1f} mV/decade")
    _ax_log.legend(fontsize=14, loc="upper left")
    style_ax(_ax_log)

    _fig2, (_ax_V, _ax_d) = plt.subplots(1, 2, figsize=(13.0, 4.8), layout="constrained")
    _ax_V.plot(_T_arr, np.where(_ok, _Va_fix, np.nan), color=GREEN, linewidth=2.4)
    _ax_V.plot(_T_arr, np.where(_ok, np.nan, _Va_fix), color=GREEN, linewidth=2.0, linestyle="--")
    _ax_V.plot(_T_arr, _Vlim_arr, color=GRAY, linestyle=":", linewidth=1.8, label="model limit")
    _ax_V.legend(fontsize=14, loc="best")
    _ax_V.plot(_T, _Va_here, "o", color=ORANGE, markersize=9)
    _ax_V.set_xlabel(r"$T$ (K)")
    _ax_V.set_ylabel(r"$V_a$ (V)")
    _ax_V.set_title(rf"(c)  $V_a$ at $J = {sci_tex(_Jfix, 1)}$ A/cm$^2$")
    style_ax(_ax_V)
    _ax_d.plot(_T_arr, np.where(_ok, _dVdT * 1e3, np.nan), color=GREEN, linewidth=2.4)
    _ax_d.plot(_T_arr, np.where(_ok, np.nan, _dVdT * 1e3), color=GREEN, linewidth=2.0, linestyle="--")
    _ax_d.plot(_T, _dVdT_here * 1e3, "o", color=ORANGE, markersize=9)
    _ax_d.set_xlabel(r"$T$ (K)")
    _ax_d.set_ylabel(r"$dV_a/dT$ (mV/K)")
    _ax_d.set_title(r"(d)  $dV_a/dT$ at fixed $J$")
    style_ax(_ax_d)

    _eq = mo.md(rf"""
    $$
    \begin{{aligned}}
    n_i(T) &= {sci_tex(ni_at(_T))}\ \mathrm{{cm^{{-3}}}}, \qquad V_t = {_Vt:.4f}\ \mathrm{{V}} \\
    J_s &= e n_i^2\left(\frac{{D_n}}{{L_n N_a}} + \frac{{D_p}}{{L_p N_d}}\right) = {sci_tex(_Js)}\ \mathrm{{A/cm^2}} \\
    J &= J_s\left[\exp\!\left(\frac{{eV_a}}{{kT}}\right)-1\right]
    \end{{aligned}}
    $$
    """)
    _r = _Js / _Js300
    _Js_ratio_tex = f"{_r:.3g}" if 1.0e-2 <= _r < 1.0e3 else sci_tex(_r)
    _ratio = (
        ""
        if _T == 300.0
        else rf"At \(T = {_T:.0f}\,\mathrm{{K}}\), \(J_s\) is \({_Js_ratio_tex}\times\) its \(300\,\mathrm{{K}}\) value. "
    )
    _limit_txt = (
        rf"The grey band in (a) and (b) is outside the model: above \(V_a = {_Vlim:.3f}\,\mathrm{{V}}\) at \({_T:.0f}\,\mathrm{{K}}\), low injection fails (\(\delta/\text{{majority}} > 0.1\) on the lighter-doped side) or \(V_a\) reaches \(V_{{bi}}\). Dashed parts of (c) and (d) are outside the model too. "
    )
    _warn4 = (
        rf"**Outside the model.** \(V_a = {_Va_here:.3f}\,\mathrm{{V}}\) at this \(J\) and \(T\) is above the limit of \({_Vlim:.3f}\,\mathrm{{V}}\). Lower \(J\) or raise the doping. "
        if _Va_here > _Vlim
        else ""
    )
    _note = mo.md(
        _warn4
        + _ratio
        + rf"Holding \(J = {sci_tex(_Jfix, 1)}\,\mathrm{{A/cm^2}}\) needs \(V_a = {_Va_here:.3f}\,\mathrm{{V}}\), and \(dV_a/dT = {_dVdT_here * 1e3:.2f}\,\mathrm{{mV/K}}\). The forward voltage falls as the diode warms. The slope in (b) is \(V_t\ln 10\) per decade. \(E_g\) is fixed at \({EG_EV}\,\mathrm{{eV}}\). "
        + _limit_txt
    )
    mo.vstack([_fig, _eq, _fig2, _note])
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "What the plots are doing": mo.md(
                r"""
    **Section 1** applies the boundary conditions at the space charge edges. The equilibrium minority concentration is multiplied by \(\exp(eV_a/kT)\). The excess is that value minus equilibrium.

    **Section 2** draws the long-diode solution of the diffusion equation in each neutral region. The excess decays exponentially from its edge value over \(L_n\) or \(L_p\). The log plot shows the total concentration returning to \(n_{p0}\) and \(p_{n0}\).

    **Section 3** takes the gradient of each excess profile to get the minority diffusion current. \(J\) is the sum of the two minority currents at the edges, and \(J_n\), \(J_p\) are taken as constant across the space charge region. In each neutral region the majority current is \(J\) minus the minority diffusion current.

    **Section 4** evaluates the ideal-diode equation. Only \(n_i^2\) and \(V_t\) change with \(T\). Panel (c) solves \(J = J_s(T)[\exp(eV_a/kT)-1]\) for \(V_a\) at a fixed \(J\), and (d) is its derivative.
    """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding

    Use the sliders, then open the answers.

    1. In Section 1, raise \(V_a\) by \(0.06\,\mathrm{V}\). By what factor do \(n_p(-x_p)\) and \(p_n(x_n)\) grow?
    2. In Section 3, set \(N_a = 10^{18}\,\mathrm{cm}^{-3}\) and keep the other defaults. Which carrier now carries most of the current across the junction, and roughly what fraction?
    3. In Section 4, warm the Example 8.2 diode from \(300\) to \(310\,\mathrm{K}\). By what factor does \(J_s\) grow?
    """)
    return


@app.cell
def _(D_N, D_P, VT_300, mo, np, saturation_current):
    _f60 = float(np.exp(0.06 / VT_300))
    _Ln = float(np.sqrt(D_N * 5.0e-7))
    _Lp = float(np.sqrt(D_P * 5.0e-7))
    _jp = D_P / (_Lp * 1.0e16)
    _jn = D_N / (_Ln * 1.0e18)
    _frac_p = _jp / (_jp + _jn)
    _r10 = float(
        saturation_current(1e16, 1e16, _Ln, _Lp, 310.0) / saturation_current(1e16, 1e16, _Ln, _Lp, 300.0)
    )
    mo.accordion(
        {
            "Answers": mo.md(
                rf"""
    1. \(\exp(0.06/0.0259) = {_f60:.1f}\), so about ten times. This is the \(60\,\mathrm{{mV}}\) per decade of the Boltzmann limit.
    2. Holes, injected into the lightly doped n side: about \({100 * _frac_p:.0f}\%\). \(p_{{n0}} = n_i^2/N_d\) is now 100 times \(n_{{p0}}\), so the lightly doped side sets \(J_s\). This is a p⁺n junction.
    3. About \({_r10:.1f}\times\). \(n_i^2\) grows as \(T^3\exp(-E_g/kT)\). The exponential dominates: \(J_s\) roughly doubles every \(5\,\mathrm{{K}}\) near room temperature.
    """
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
