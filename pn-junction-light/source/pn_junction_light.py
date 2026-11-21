# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.5.3",
#     "matplotlib==3.11.2",
# ]
# ///

import marimo

__generated_with = "0.25.2"
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
    VT = 0.0259  # kT/e, V
    NI = 1.5e10  # cm^-3
    E_CHARGE = 1.6e-19  # C
    EPS_S = 11.7 * 8.85e-14  # F/cm
    D_N = 25.0  # cm^2/s, Neamen Examples 14.3 and 14.5
    D_P = 10.0

    BLUE = "#0072B2"
    ORANGE = "#D55E00"
    GREEN = "#009E73"
    NAVY = "#0f4c81"
    GRAY = "#6B6B6B"
    PINK = "#CC79A7"

    # Si absorption coefficient at 300 K (cm^-1), after M. A. Green, Sol. Energy Mater. Sol. Cells 92, 1305 (2008).
    SI_ALPHA_NM = np.array([400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100], dtype=float)
    SI_ALPHA = np.array(
        [9.52e4, 2.55e4, 1.11e4, 6.39e3, 4.14e3, 2.81e3, 1.90e3, 1.30e3, 8.50e2, 5.35e2, 3.06e2, 1.57e2, 6.40e1, 1.63e1, 3.5]
    )

    def alpha_si(lam_nm):
        """alpha(lambda) for Si, log-linear interpolation of the table, cm^-1."""
        return float(np.exp(np.interp(float(lam_nm), SI_ALPHA_NM, np.log(SI_ALPHA))))

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

    def built_in_voltage(Na, Nd):
        return float(VT * np.log(Na * Nd / NI**2))

    def depletion_edges(Na, Nd, Va):
        """x_n, x_p (cm) from the depletion approximation with barrier V_bi - V_a (>= 0)."""
        Vb = np.maximum(built_in_voltage(Na, Nd) - np.asarray(Va, dtype=float), 0.0)
        f = 2.0 * EPS_S / E_CHARGE * Vb
        xn = np.sqrt(f * Na / (Nd * (Na + Nd)))
        xp = np.sqrt(f * Nd / (Na * (Na + Nd)))
        return xn, xp

    def model_limit(Na, Nd):
        """Largest V_a in the ideal-diode model: the lower of V_bi and the low-injection
        limit (dark excess on the lighter side reaching 0.1 of the majority concentration)."""
        Nmin = min(Na, Nd)
        return min(built_in_voltage(Na, Nd), float(VT * np.log1p(0.1 * Nmin**2 / NI**2)))

    def illuminated_junction(Na, Nd, tn, tp, Va, mode, GL=0.0, Phi0=0.0, alpha=1e3, R=0.0, xj=2e-4):
        """Steady-state minority carriers and currents of an illuminated long pn junction.

        n region on the left, p region on the right (as in Neamen Fig. 14.18); in the
        'surface' mode light enters the n-type top layer at its surface (x = 0), which has
        fast surface recombination (excess = 0), and the junction is at depth x_j.
        Currents are returned as components flowing from n to p (the photocurrent direction);
        the diode-convention current is J = -(J_scr + J_p + J_n).
        Va may be an array (for the I-V curve)."""
        Va = np.asarray(Va, dtype=float)
        Ln, Lp = np.sqrt(D_N * tn), np.sqrt(D_P * tp)
        np0, pn0 = NI**2 / Na, NI**2 / Nd
        xn, xp = depletion_edges(Na, Nd, Va)
        dpb = pn0 * np.expm1(Va / VT)  # excess at the space charge edges, any V_a
        dnb = np0 * np.expm1(Va / VT)
        if mode == "uniform":
            # Neamen (14.29)-(14.42), boundary value n_p0[exp(eV_a/kT) - 1] in place of -n_p0
            J_scr = E_CHARGE * GL * (xn + xp)
            J_p = E_CHARGE * D_P / Lp * (GL * tp - dpb)
            J_n = E_CHARGE * D_N / Ln * (GL * tn - dnb)

            def dp(u):  # u = distance into the n region from its space charge edge
                return GL * tp + (dpb - GL * tp) * np.exp(-u / Lp)

            def dn(u):  # u = distance into the p region from its space charge edge
                return GL * tn + (dnb - GL * tn) * np.exp(-u / Ln)

            def gen(s):
                return GL * np.ones_like(s)

        else:
            G0 = alpha * Phi0 * (1.0 - R)
            d = xj - xn  # thickness of the neutral n layer
            # holes in n: D_p dp'' - dp/tau_p = -G0 exp(-alpha s), dp(0) = 0, dp(d) = dpb
            ap = alpha if abs(1.0 - (alpha * Lp) ** 2) > 1e-6 else alpha * (1.0 + 1e-5)
            A = G0 * tp / (1.0 - (ap * Lp) ** 2)
            P, Q = dpb - A * np.exp(-ap * d), -A

            def S(u):  # sinh(u/Lp)/sinh(d/Lp), overflow-free
                return np.exp((u - d) / Lp) * (-np.expm1(-2.0 * u / Lp)) / (-np.expm1(-2.0 * d / Lp))

            coth = 1.0 / np.tanh(d / Lp)
            csch = 2.0 * np.exp(-d / Lp) / (-np.expm1(-2.0 * d / Lp))
            J_p = -E_CHARGE * D_P * (-ap * A * np.exp(-ap * d) + (P * coth - Q * csch) / Lp)
            # electrons in the long p region beyond depth x_j + x_p
            sb = xj + xp
            an = alpha if abs(1.0 - (alpha * Ln) ** 2) > 1e-6 else alpha * (1.0 + 1e-5)
            B = G0 * np.exp(-alpha * sb) * tn / (1.0 - (an * Ln) ** 2)
            C = dnb - B
            J_n = E_CHARGE * D_N * (-an * B - C / Ln)
            J_scr = E_CHARGE * Phi0 * (1.0 - R) * (np.exp(-alpha * d) - np.exp(-alpha * sb))

            def dp(u):
                s = d - u
                return A * np.exp(-ap * s) + P * S(s) + Q * S(d - s)

            def dn(u):
                return B * np.exp(-an * u) + C * np.exp(-u / Ln)

            def gen(s):
                return G0 * np.exp(-alpha * s)

        J = -(J_scr + J_p + J_n)
        return dict(J=J, J_scr=J_scr, J_p=J_p, J_n=J_n, Ln=Ln, Lp=Lp, xn=xn, xp=xp, dp=dp, dn=dn, gen=gen, np0=np0, pn0=pn0)

    def saturation_current(Na, Nd, tn, tp):
        """J_s = e n_i^2 [D_n/(L_n N_a) + D_p/(L_p N_d)], A/cm^2 (long diode)."""
        return E_CHARGE * NI**2 * (D_N / (np.sqrt(D_N * tn) * Na) + D_P / (np.sqrt(D_P * tp) * Nd))

    return (
        BLUE,
        E_CHARGE,
        GRAY,
        GREEN,
        NAVY,
        ORANGE,
        PINK,
        VT,
        alpha_si,
        built_in_voltage,
        illuminated_junction,
        model_limit,
        plt,
        saturation_current,
        sci_tex,
        style_ax,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # pn Junction under Illumination and Solar Cells
    **ECE335 - companion to lectures**

    Light generates electron–hole pairs at the rate \(G_L\). Pairs made in the space charge region, or within about a diffusion length of it, are collected by the junction field and give the photocurrent \(J_L\). This notebook solves the steady-state minority carrier diffusion equation with \(G_L\) in each neutral region at any bias, and shows where the photocurrent comes from and how it moves the \(I\)–\(V\) curve (Section 1). Sections 2 and 3 treat the junction as a solar cell: maximum power, fill factor, efficiency, load and series resistance, concentration, and the choice of bandgap. The derivation for uniform \(G_L\) and for the ideal solar cell is presented on the slides; the surface case, series resistance, load line and bandgap panels apply the same equations.

    Silicon at \(300\,\mathrm{K}\): \(n_i = 1.5\times 10^{10}\,\mathrm{cm^{-3}}\), \(V_t = kT/e = 0.0259\,\mathrm{V}\), \(D_n = 25\), \(D_p = 10\,\mathrm{cm^2/s}\), long neutral regions. The current follows the diode convention, positive from p to n through the junction: \(J = J_s[\exp(eV_a/kT) - 1] - J_L\). A photodiode works in quadrant III, a solar cell in quadrant IV. Neamen writes the current in the reverse direction, \(I = I_L - I_F\).

    **Two generation profiles.** Uniform generation: the same generation everywhere, the long diode of Neamen Fig. 14.18, so \(J_L = e(W + L_n + L_p)G_L\). Generation from the surface: light of wavelength \(\lambda\) enters an n-type top layer at \(x = 0\), \(G_L(x) = \alpha\Phi_0(1 - R)e^{-\alpha x}\), and the junction is at depth \(x_j\), the geometry of Neamen Fig. 14.11. Here the surface has fast recombination (excess carriers zero there), the opposite limit to Fig. 14.11, which assumes no surface recombination.
    """)
    return


@app.cell
def _(mo):
    mode_radio = mo.ui.radio(
        options={"Uniform generation (long diode)": "uniform", "Generation from the surface, at wavelength λ": "surface"},
        value="Uniform generation (long diode)",
        label="Generation profile",
    )
    GL_slider = mo.ui.slider(start=18.0, stop=22.0, step=0.1, value=21.0, show_value=True, label=r"log₁₀(\(G_L\) / cm⁻³ s⁻¹)")
    Phi_slider = mo.ui.slider(start=15.0, stop=19.0, step=0.1, value=17.0, show_value=True, label=r"log₁₀(\(\Phi_0\) / cm⁻² s⁻¹)")
    lam_slider = mo.ui.slider(start=400, stop=1100, step=10, value=800, show_value=True, label=r"\(\lambda\) (nm)")
    R_slider = mo.ui.slider(start=0.0, stop=0.4, step=0.05, value=0.0, show_value=True, label=r"reflectance \(R\)")
    xj_slider = mo.ui.slider(start=0.5, stop=20.0, step=0.5, value=2.0, show_value=True, label=r"junction depth \(x_j\) (μm)")
    Na_slider = mo.ui.slider(start=15.0, stop=19.0, step=0.1, value=16.0, show_value=True, label=r"log₁₀(\(N_a\) / cm⁻³)")
    Nd_slider = mo.ui.slider(start=15.0, stop=19.0, step=0.1, value=16.0, show_value=True, label=r"log₁₀(\(N_d\) / cm⁻³)")
    _tau = [1e-9, 2e-9, 5e-9, 1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]
    taun_slider = mo.ui.slider(steps=_tau, value=5e-7, show_value=True, label=r"\(\tau_{n0}\) (s)")
    taup_slider = mo.ui.slider(steps=_tau, value=1e-7, show_value=True, label=r"\(\tau_{p0}\) (s)")
    Va_slider = mo.ui.slider(start=-5.0, stop=0.7, step=0.01, value=-5.0, show_value=True, label=r"\(V_a\) (V)")
    width_slider = mo.ui.slider(start=5, stop=300, step=5, value=140, show_value=True, label="plot width (μm)")
    return (
        GL_slider,
        Na_slider,
        Nd_slider,
        Phi_slider,
        R_slider,
        Va_slider,
        lam_slider,
        mode_radio,
        taun_slider,
        taup_slider,
        width_slider,
        xj_slider,
    )


@app.cell
def _(
    GL_slider,
    Na_slider,
    Nd_slider,
    Phi_slider,
    R_slider,
    Va_slider,
    lam_slider,
    mo,
    mode_radio,
    taun_slider,
    taup_slider,
    width_slider,
    xj_slider,
):
    _uni = mode_radio.value == "uniform"
    _light = (
        mo.vstack([mo.md("**Light (uniform)**"), GL_slider])
        if _uni
        else mo.vstack([mo.md("**Light (from the surface)**"), Phi_slider, lam_slider, R_slider, xj_slider])
    )
    mo.vstack(
        [
            mo.md(r"## 1. Minority Carriers and Photocurrent"),
            mo.md(
                r"Defaults are Neamen Example 14.5: \(N_a = N_d = 10^{16}\,\mathrm{cm^{-3}}\), \(\tau_{n0} = 5\times 10^{-7}\), \(\tau_{p0} = 10^{-7}\,\mathrm{s}\), \(V_R = 5\,\mathrm{V}\) (\(V_a = -5\,\mathrm{V}\)), \(G_L = 10^{21}\,\mathrm{cm^{-3}s^{-1}}\). Move \(V_a\) to see the operating point travel from quadrant III to quadrant IV."
            ),
            mode_radio,
            mo.hstack(
                [
                    _light,
                    mo.vstack([mo.md("**Junction**"), Na_slider, Nd_slider, taun_slider, taup_slider]),
                    mo.vstack([mo.md("**Bias and view**"), Va_slider, width_slider]),
                ],
                align="start",
                widths=[1, 1, 1],
            ),
        ]
    )
    return


@app.cell
def _(
    BLUE,
    E_CHARGE,
    GL_slider,
    GRAY,
    GREEN,
    NAVY,
    Na_slider,
    Nd_slider,
    ORANGE,
    PINK,
    Phi_slider,
    R_slider,
    Va_slider,
    alpha_si,
    built_in_voltage,
    illuminated_junction,
    lam_slider,
    mo,
    mode_radio,
    model_limit,
    np,
    plt,
    saturation_current,
    sci_tex,
    style_ax,
    taun_slider,
    taup_slider,
    width_slider,
    xj_slider,
):
    _mode = mode_radio.value
    _Na, _Nd = 10.0 ** float(Na_slider.value), 10.0 ** float(Nd_slider.value)
    _tn, _tp = float(taun_slider.value), float(taup_slider.value)
    _GL, _Phi = 10.0 ** float(GL_slider.value), 10.0 ** float(Phi_slider.value)
    _lam, _R = float(lam_slider.value), float(R_slider.value)
    _alpha = alpha_si(_lam)
    _Vbi = built_in_voltage(_Na, _Nd)
    _Vlim = model_limit(_Na, _Nd)
    _notes = []
    _Va = float(Va_slider.value)
    if _Va > _Vlim:
        _notes.append(
            rf"\(V_a\) is above the model limit \({_Vlim:.2f}\,\mathrm{{V}}\) (the lower of \(V_{{bi}}\) and the low-injection limit), so the curves are drawn at \(V_a = {_Vlim:.2f}\,\mathrm{{V}}\)."
        )
        _Va = _Vlim
    _xj = float(xj_slider.value) * 1e-4
    if _mode == "surface":
        _xn_max = float(illuminated_junction(_Na, _Nd, _tn, _tp, -5.0, "uniform")["xn"])
        if _xj < _xn_max + 0.1e-4:
            _notes.append(
                rf"\(x_j\) must exceed the n-side depletion width at \(V_a = -5\,\mathrm{{V}}\); it is raised to \({(_xn_max + 0.1e-4) * 1e4:.1f}\,\mu\mathrm{{m}}\)."
            )
            _xj = _xn_max + 0.1e-4
    _kw = dict(mode=_mode, GL=_GL, Phi0=_Phi, alpha=_alpha, R=_R, xj=_xj)
    _r = illuminated_junction(_Na, _Nd, _tn, _tp, _Va, **_kw)
    _kw_dark = {**_kw, "GL": 0.0, "Phi0": 0.0}  # same device, light off
    _d = illuminated_junction(_Na, _Nd, _tn, _tp, _Va, **_kw_dark)
    _Ln, _Lp = float(_r["Ln"]), float(_r["Lp"])
    _xn, _xp = float(_r["xn"]), float(_r["xp"])
    _W = _xn + _xp
    _Js = float(saturation_current(_Na, _Nd, _tn, _tp))
    _J = float(_r["J"])
    _Jdark = float(_d["J"])
    _JL = _Jdark - _J  # photocurrent at this bias (superposition)
    _JL_scr = float(_r["J_scr"])
    _JL_p = float(_r["J_p"] - _d["J_p"])  # photo part of the hole current from the n side
    _JL_n = float(_r["J_n"] - _d["J_n"])

    # ---------------- geometry for the plots (x in um, from the left edge)
    _X = float(width_slider.value)
    _um = 1e-4
    _x0 = _xj / _um if _mode == "surface" else 0.3 * _X  # metallurgical junction
    _left_n = _x0 - _xn / _um  # n-side space charge edge
    _right_p = _x0 + _xp / _um  # p-side space charge edge
    _xs = np.linspace(0.0, _X, 1500)
    _un = np.clip((_left_n - _xs) * _um, 0, None)
    _up = np.clip((_xs - _right_p) * _um, 0, None)
    _in_n = _xs <= _left_n
    _in_p = _xs >= _right_p
    _dp = np.where(_in_n, _r["dp"](_un), np.nan)
    _dn = np.where(_in_p, _r["dn"](_up), np.nan)
    _depth = _xs * _um if _mode == "surface" else _xs * _um
    _G = _r["gen"](_depth)

    _surf = _mode == "surface"
    _fig = plt.figure(figsize=(15, 12.0 if _surf else 8.6))
    _gs = _fig.add_gridspec(
        3 if _surf else 2,
        2,
        height_ratios=[1, 1.8, 1.6] if _surf else [1, 1.8],
        width_ratios=[1.35, 1],
        hspace=0.32 if _surf else 0.12,
        wspace=0.28,
    )
    _aG = _fig.add_subplot(_gs[0, 0])
    _aC = _fig.add_subplot(_gs[1, 0], sharex=_aG)
    _gsr = _gs[:, 1].subgridspec(2, 1, hspace=0.3)
    _aI = _fig.add_subplot(_gsr[0])
    _aZ = _fig.add_subplot(_gsr[1])

    for _a in (_aG, _aC):
        _a.axvspan(_left_n, _right_p, color="#D9D9D9", alpha=0.8, lw=0)
        _a.axvspan(max(_left_n - _Lp / _um, 0 if _mode == "surface" else -1e9), _left_n, color=ORANGE, alpha=0.08, lw=0)
        _a.axvspan(_right_p, _right_p + _Ln / _um, color=BLUE, alpha=0.08, lw=0)
        if _mode == "surface":
            _a.axvline(0, color="k", lw=2)
        style_ax(_a)
    _aG.semilogy(_xs, _G, color=GREEN, lw=2.4)
    _aG.set_ylabel(r"$G_L$ (cm$^{-3}$s$^{-1}$)")
    _gmax = float(np.nanmax(_G))
    _aG.set_ylim(max(float(np.nanmin(_G)), _gmax * 1e-4) / 2, _gmax * 3)
    plt.setp(_aG.get_xticklabels(), visible=False)
    _aG.set_title(
        rf"uniform $G_L$ = {_GL:.0e} cm$^{{-3}}$s$^{{-1}}$"
        if _mode == "uniform"
        else rf"$\lambda$ = {_lam:.0f} nm, $\alpha$ = {_alpha:.3g} cm$^{{-1}}$, 1/$\alpha$ = {1e4 / _alpha:.3g} μm"
    )

    _sc = 10.0 ** np.floor(np.log10(max(np.nanmax(np.abs(np.r_[_dp[np.isfinite(_dp)], _dn[np.isfinite(_dn)], 1.0])), 1.0)))
    _aC.plot(_xs, _dp / _sc, color=ORANGE, lw=2.6, label=r"$\delta p_n$, holes in n")
    _aC.plot(_xs, _dn / _sc, color=BLUE, lw=2.6, label=r"$\delta n_p$, electrons in p")
    if _mode == "uniform":
        _aC.axhline(_GL * _tn / _sc, color=BLUE, ls="--", lw=1.2)
        _aC.axhline(_GL * _tp / _sc, color=ORANGE, ls="--", lw=1.2)
    _aC.axhline(0, color="k", lw=0.8)
    _aC.set_xlim(0, _X)
    _aC.set_xlabel(
        r"depth from the surface (μm)" if _mode == "surface" else r"position (μm):  n side $\leftarrow$ SCR $\rightarrow$ p side"
    )
    _aC.set_ylabel(rf"excess (10$^{{{int(np.log10(_sc))}}}$ cm$^{{-3}}$)")
    _aC.legend(loc="best", frameon=False)
    _ytop = _aC.get_ylim()[1]
    if _right_p + 0.5 * _Ln / _um < 0.97 * _X:
        _aC.text(_right_p + 0.5 * _Ln / _um, 0.93 * _ytop, r"$L_n$", color=BLUE, ha="center", va="top")
    if _left_n - 0.5 * _Lp / _um > 0.03 * _X:
        _aC.text(_left_n - 0.5 * _Lp / _um, 0.93 * _ytop, r"$L_p$", color=ORANGE, ha="center", va="top")
    if _surf:
        # near-surface zoom: top layer, space charge region and a few um of the p region
        _aS = _fig.add_subplot(_gs[2, 0])
        _Xs = _right_p + max(3.0, _xj / _um)
        _xz = np.linspace(0.0, _Xs, 1200)
        _dpz = np.where(_xz <= _left_n, _r["dp"](np.clip((_left_n - _xz) * _um, 0, None)), np.nan)
        _dnz = np.where(_xz >= _right_p, _r["dn"](np.clip((_xz - _right_p) * _um, 0, None)), np.nan)
        _aS.axvspan(_left_n, _right_p, color="#D9D9D9", alpha=0.8, lw=0)
        _aS.axvline(0, color="k", lw=2)
        _aS.plot(_xz, _dpz / _sc, color=ORANGE, lw=2.6)
        _aS.plot(_xz, _dnz / _sc, color=BLUE, lw=2.6)
        _aS.axhline(0, color="k", lw=0.8)
        _aS.set_xlim(0, _Xs)
        _aS.set_xlabel(r"depth from the surface (μm), near the surface")
        _aS.set_ylabel(rf"excess (10$^{{{int(np.log10(_sc))}}}$ cm$^{{-3}}$)")
        _aS.text(0.5 * _left_n, 0.9 * _aS.get_ylim()[1], "n top layer", ha="center", va="top", color=ORANGE, fontsize=14)
        style_ax(_aS)
        plt.setp(_aC.get_xticklabels(), visible=True)

    # ---------------- I-V
    _Vmin = min(-1.0, _Va - 0.1)
    _V = np.linspace(_Vmin, _Vlim, 900)
    _Jv = illuminated_junction(_Na, _Nd, _tn, _tp, _V, **_kw)["J"]
    _Jd = illuminated_junction(_Na, _Nd, _tn, _tp, _V, **_kw_dark)["J"]
    _JLref = max(abs(float(_JL)), 1e-30)
    _mA = _JLref < 0.1
    _u = 1e3 if _mA else 1.0
    _ul = "mA/cm$^2$" if _mA else "A/cm$^2$"
    _aI.axhspan(-1.4 * _JLref * _u, 0, xmin=0, xmax=(0 - _Vmin) / (_Vlim - _Vmin), color=PINK, alpha=0.10, lw=0)
    _aI.axhspan(-1.4 * _JLref * _u, 0, xmin=(0 - _Vmin) / (_Vlim - _Vmin), xmax=1, color=GREEN, alpha=0.10, lw=0)
    _aI.plot(_V, _Jd * _u, color=GRAY, ls="--", lw=2, label="dark")
    _aI.plot(_V, _Jv * _u, color=NAVY, lw=2.6, label="illuminated")
    _aI.plot([_Va], [_J * _u], "o", color=ORANGE, ms=11, zorder=5, label="operating point")
    _aI.axhline(0, color="k", lw=0.9)
    _aI.axvline(0, color="k", lw=0.9)
    _aI.set_xlim(_Vmin, _Vlim)
    _aI.set_ylim(-1.4 * _JLref * _u, 0.6 * _JLref * _u)
    _aI.text(0.03, 0.05, "III: photodiode", transform=_aI.transAxes, color=PINK)
    _aI.set_ylabel(rf"$J$ ({_ul})")
    _aI.legend(loc="upper left", frameon=False, fontsize=14)
    style_ax(_aI)
    # zoom on quadrant IV
    _zlo = -0.05
    _Vz = np.linspace(_zlo, _Vlim, 600)
    _Jz = illuminated_junction(_Na, _Nd, _tn, _tp, _Vz, **_kw)["J"]
    _aZ.axhspan(-1.4 * _JLref * _u, 0, xmin=-_zlo / (_Vlim - _zlo), xmax=1, color=GREEN, alpha=0.10, lw=0)
    _aZ.plot(_Vz, illuminated_junction(_Na, _Nd, _tn, _tp, _Vz, **_kw_dark)["J"] * _u, color=GRAY, ls="--", lw=2)
    _aZ.plot(_Vz, _Jz * _u, color=NAVY, lw=2.6)
    if _Va >= _zlo:
        _aZ.plot([_Va], [_J * _u], "o", color=ORANGE, ms=11, zorder=5)
    _aZ.axhline(0, color="k", lw=0.9)
    _aZ.axvline(0, color="k", lw=0.9)
    _aZ.set_xlim(_zlo, _Vlim)
    _aZ.set_ylim(-1.3 * _JLref * _u, 0.5 * _JLref * _u)
    _aZ.text(0.04, 0.06, "IV: solar cell", transform=_aZ.transAxes, color=GREEN)
    _aZ.set_xlabel(r"$V_a$ (V)")
    _aZ.set_ylabel(rf"$J$ ({_ul})")
    _aZ.set_title("zoom: power quadrant", fontsize=15)
    style_ax(_aZ)
    _fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.09)

    # V_oc of the illuminated curve (if inside the model range)
    _Voc = None
    if _Jv[-1] > 0 > _Jv[0]:
        _lo, _hi = float(_V[0]), float(_V[-1])
        for _ in range(80):
            _mid = 0.5 * (_lo + _hi)
            if float(illuminated_junction(_Na, _Nd, _tn, _tp, _mid, **_kw)["J"]) < 0:
                _lo = _mid
            else:
                _hi = _mid
        _Voc = 0.5 * (_lo + _hi)

    _maj_p, _maj_n = _Na, _Nd
    _inj = max(np.nanmax(np.abs(_dn)) / _maj_p, np.nanmax(np.abs(_dp)) / _maj_n)
    if _inj >= 0.1:
        _notes.append(
            rf"Low injection fails: the largest excess is {_inj:.2f} of the majority concentration (the model assumes well below 0.1)."
        )

    def _cur(x):
        return rf"\({x * _u:.4g}\,\mathrm{{{'mA' if _mA else 'A'}/cm^2}}\)"

    _P = -_J * _Va
    if _mode == "uniform":
        _js_label, _Js_dev = r"\(J_s\)", _Js
    else:
        # this device's saturation current: dark current at a small reverse bias, J -> -J_s
        _js_label = r"\(J_s\) (thin n layer)"
        _Js_dev = -float(illuminated_junction(_Na, _Nd, _tn, _tp, -0.5, **_kw_dark)["J"])
    _quad = (
        "quadrant III (photodiode)"
        if (_Va < 0 and _J < 0)
        else (
            "quadrant IV (power delivered)"
            if (_Va > 0 and _J < 0)
            else ("on an axis" if _Va == 0 or _J == 0 else ("quadrant I" if _Va > 0 else "quadrant II"))
        )
    )
    _JL_formula = (
        rf"| \(e(W + L_n + L_p)G_L\) | {_cur(E_CHARGE * (_W + _Ln + _Lp) * _GL)} | | |"
        if _mode == "uniform"
        else rf"| \(\alpha\) | \({_alpha:.3g}\,\mathrm{{cm^{{-1}}}}\) | \(x_j\) | \({_xj * 1e4:.2f}\,\mu\mathrm{{m}}\) |"
    )
    _table = mo.md(
        rf"""
    | Quantity | Value | Quantity | Value |
    |:--|:--|:--|:--|
    | \(V_{{bi}}\) | \({_Vbi:.3f}\,\mathrm{{V}}\) | \(W = x_n + x_p\) | \({_W * 1e4:.3f}\,\mu\mathrm{{m}}\) |
    | \(L_n = \sqrt{{D_n\tau_{{n0}}}}\) | \({_Ln * 1e4:.1f}\,\mu\mathrm{{m}}\) | \(L_p = \sqrt{{D_p\tau_{{p0}}}}\) | \({_Lp * 1e4:.1f}\,\mu\mathrm{{m}}\) |
    | prompt, from the SCR | {_cur(_JL_scr)} | delayed, holes from n | {_cur(_JL_p)} |
    | delayed, electrons from p | {_cur(_JL_n)} | \(J_L\) (total) | {_cur(_JL)} |
    | prompt fraction | \({100 * _JL_scr / _JLref:.1f}\%\) | {_js_label} | \({sci_tex(_Js_dev)}\,\mathrm{{A/cm^2}}\) |
    {_JL_formula}
    | \(J\) at \(V_a = {_Va:.2f}\,\mathrm{{V}}\) | {_cur(_J)} | \(V_{{oc}}\) | {rf"\({_Voc:.3f}\,\mathrm{{V}}\)" if _Voc is not None else "outside the model range"} |
    | operating point | {_quad} | power out, \(-JV_a\) | {rf"\({_P * 1e3:.4g}\,\mathrm{{mW/cm^2}}\)" if _P > 0 else "none (absorbed)"} |
    """
    )
    _cap = mo.md(
        r"Grey: space charge region. Shaded bands: one diffusion length \(L_p\) (orange) and \(L_n\) (blue) beyond each edge, where most collected pairs come from. "
        + (r"Dashed lines: \(G_L\tau_{n0}\) and \(G_L\tau_{p0}\), the uniform result far from the junction." if _mode == "uniform" else r"Black line: the illuminated surface.")
        + r" The photocurrent components are the light-induced parts of the currents at the edges, at this bias; \(J_L\) is the dark current minus \(J\)."
    )
    mo.vstack([_fig, _cap, _table] + [mo.callout(mo.md(n), kind="warn") for n in _notes])
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Equations used in Section 1": mo.md(
                r"""
    In the neutral p region (\(\mathcal{E} = 0\), steady state), with \(L_n^2 = D_n\tau_{n0}\):

    $$
    D_n\frac{d^2(\delta n_p)}{dx^2} - \frac{\delta n_p}{\tau_{n0}} = -G_L(x)
    $$

    and the same for holes in the n region. At the space charge edges the boundary value at any bias is \(\delta n_p = n_{p0}[\exp(eV_a/kT) - 1]\) and \(\delta p_n = p_{n0}[\exp(eV_a/kT) - 1]\); in reverse bias this is \(-n_{p0}\), Neamen Eq. (14.36).

    **Uniform \(G_L\), long regions.** With \(x\) measured from each edge into the neutral region:

    $$
    \delta n_p(x) = G_L\tau_{n0} + \left\{n_{p0}\left[e^{eV_a/kT} - 1\right] - G_L\tau_{n0}\right\}e^{-x/L_n}
    $$

    which is Eq. (14.37) in reverse bias. Adding the space charge region gives Eq. (14.42), \(J_L = e(W + L_n + L_p)G_L\), and \(J = J_s[\exp(eV_a/kT) - 1] - J_L\), with the prompt part \(eG_LW\).

    **\(G_L(x) = \alpha\Phi_0(1 - R)e^{-\alpha x}\) from the surface.** The particular solution is \(\delta = G_L(x)\tau/(1 - \alpha^2L^2)\). In the n top layer the excess is zero at the surface and takes the boundary value at the space charge edge; in the long p region it decays to the particular solution. The prompt part is \(e\Phi_0(1 - R)[e^{-\alpha x_1} - e^{-\alpha x_2}]\), the photons absorbed between the space charge edges \(x_1\) and \(x_2\).

    **Currents.** \(J_p = -eD_p\,d(\delta p_n)/dx\) and \(J_n = eD_n\,d(\delta n_p)/dx\) at the edges, plus the generation in the space charge region; the depletion widths come from the depletion approximation with barrier \(V_{bi} - V_a\), so the prompt part grows with reverse bias. Superposition holds because the equations are linear, so \(J_L\) is the dark current of the same device minus the illuminated current at the same \(V_a\). In surface mode the dark current of the thin n layer is larger than the long-diode value (the surface removes holes), so the dashed curve and \(J_s\) in the table are those of the thin-layer device.

    Si absorption coefficient at \(300\,\mathrm{K}\) after M. A. Green (2008), interpolated in \(\log\alpha\); it agrees with Neamen Fig. 14.4 at short wavelengths (about \(10^4\) at \(0.5\), \(4\times 10^3\) at \(0.6\) and \(10^3\,\mathrm{cm^{-1}}\) at \(0.8\,\mu\mathrm{m}\)), but gives \(64\,\mathrm{cm^{-1}}\) at \(1.0\,\mu\mathrm{m}\), against about \(10^2\) read from the figure, so 90% absorption takes \(360\) rather than \(230\,\mu\mathrm{m}\) there.
    """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Check your understanding: illuminated junction

    Use the sliders, then open the answers.

    1. With the defaults (Neamen Example 14.5), what fraction of \(J_L\) is prompt? Now change \(V_a\) from \(-5\) to \(-1\,\mathrm{V}\). Which part of \(J_L\) changes, and why?
    2. Switch to generation from the surface with \(\Phi_0 = 10^{17}\,\mathrm{cm^{-2}s^{-1}}\), \(R = 0\), \(x_j = 2\,\mu\mathrm{m}\) and \(V_a = 0\), keeping the other defaults. Compare \(J_L\) at \(\lambda = 450\) and \(1000\,\mathrm{nm}\). Which region collects most of the current in each case?
    3. Back to uniform generation with the defaults, but set \(G_L = 10^{20}\,\mathrm{cm^{-3}s^{-1}}\) (so low injection still holds) and \(V_a = 0\). Raise \(\tau_{n0}\) from \(5\times 10^{-7}\) to \(5\times 10^{-6}\,\mathrm{s}\). How do \(J_L\) and \(V_{oc}\) change?
    """)
    return


@app.cell
def _(
    E_CHARGE,
    VT,
    alpha_si,
    illuminated_junction,
    mo,
    np,
    saturation_current,
):
    _r5 = illuminated_junction(1e16, 1e16, 5e-7, 1e-7, -5.0, "uniform", GL=1e21)
    _JL5 = E_CHARGE * float(_r5["xn"] + _r5["xp"] + _r5["Ln"] + _r5["Lp"]) * 1e21
    _f5 = E_CHARGE * float(_r5["xn"] + _r5["xp"]) * 1e21 / _JL5
    _r1 = illuminated_junction(1e16, 1e16, 5e-7, 1e-7, -1.0, "uniform", GL=1e21)
    _W5, _W1 = float(_r5["xn"] + _r5["xp"]) * 1e4, float(_r1["xn"] + _r1["xp"]) * 1e4
    _JL1 = E_CHARGE * float(_r1["xn"] + _r1["xp"] + _r1["Ln"] + _r1["Lp"]) * 1e21

    _out = {}
    for _lam in (450, 1000):
        _kw = dict(Phi0=1e17, alpha=alpha_si(_lam), R=0.0, xj=2e-4)
        _r = illuminated_junction(1e16, 1e16, 5e-7, 1e-7, 0.0, "surface", **_kw)
        _out[_lam] = (-float(_r["J"]), float(_r["J_p"]), float(_r["J_scr"]), float(_r["J_n"]))
    _eP = 1.6e-19 * 1e17

    _rA = illuminated_junction(1e16, 1e16, 5e-7, 1e-7, 0.0, "uniform", GL=1e20)
    _rB = illuminated_junction(1e16, 1e16, 5e-6, 1e-7, 0.0, "uniform", GL=1e20)
    _JLA, _JLB = -float(_rA["J"]), -float(_rB["J"])
    _JsA, _JsB = float(saturation_current(1e16, 1e16, 5e-7, 1e-7)), float(saturation_current(1e16, 1e16, 5e-6, 1e-7))
    _VocA, _VocB = VT * np.log1p(_JLA / _JsA), VT * np.log1p(_JLB / _JsB)

    mo.accordion(
        {
            "Answers 1–3": mo.md(
                rf"""
    1. \(J_L = {_JL5:.3f}\,\mathrm{{A/cm^2}}\) and the prompt part \(eG_LW\) is \({100 * _f5:.1f}\%\) of it: most pairs are collected by diffusion from within \(L_n = 35.4\) and \(L_p = 10\,\mu\mathrm{{m}}\), far more than \(W\). At \(V_a = -1\,\mathrm{{V}}\), \(W\) shrinks from \({_W5:.2f}\) to \({_W1:.2f}\,\mu\mathrm{{m}}\), so only the prompt part falls; \(J_L\) becomes \({_JL1:.3f}\,\mathrm{{A/cm^2}}\). The diffusion parts do not depend on bias.
    2. At \(450\,\mathrm{{nm}}\) (\(1/\alpha = {1e4 / alpha_si(450):.2f}\,\mu\mathrm{{m}}\)) nearly all photons are absorbed in the n top layer: \(J_L = {_out[450][0] * 1e3:.2f}\,\mathrm{{mA/cm^2}}\), of which \({_out[450][1] * 1e3:.2f}\) comes from holes in the n layer, \({_out[450][2] * 1e3:.2f}\) from the space charge region and \({_out[450][3] * 1e3:.2f}\) from electrons in the p region. Most pairs are made closer to the surface than to the junction and recombine at the surface, which is why solar cells use shallow junctions and passivated surfaces. At \(1000\,\mathrm{{nm}}\) (\(1/\alpha = {1e4 / alpha_si(1000):.0f}\,\mu\mathrm{{m}}\)) most photons pass the junction: \(J_L = {_out[1000][0] * 1e3:.2f}\,\mathrm{{mA/cm^2}}\), nearly all from electrons in the p region within about \(L_n\) of the junction; the photons absorbed deeper than that are lost. The maximum, one pair per photon, is \(e\Phi_0 = {_eP * 1e3:.0f}\,\mathrm{{mA/cm^2}}\).
    3. \(L_n\) grows by \(\sqrt{{10}}\), from \(35.4\) to \(112\,\mu\mathrm{{m}}\), so \(J_L\) rises from \({_JLA:.3f}\) to \({_JLB:.3f}\,\mathrm{{A/cm^2}}\). \(J_s\) falls from \({_JsA:.3g}\) to \({_JsB:.3g}\,\mathrm{{A/cm^2}}\), so \(V_{{oc}}\) rises from \({_VocA:.3f}\) to \({_VocB:.3f}\,\mathrm{{V}}\). Long lifetimes help both, which is why solar cells need clean material. (At \(G_L = 10^{{21}}\) the far-field excess \(G_L\tau_{{n0}} = 5\times 10^{{15}}\,\mathrm{{cm^{{-3}}}}\) would be half of \(N_a\), outside low injection.)
    """
            )
        }
    )
    return


@app.cell
def _(E_CHARGE, VT, np):
    # Solar-cell additions; Si constants, colours and helpers come from the first setup cell.
    P_SUN = 0.100  # W/cm^2 incident at one sun

    def ideal_max_power(JL, Js, Vt=VT):
        """V_oc, and V_m from (1 + V_m/V_t) exp(V_m/V_t) = 1 + J_L/J_s (Neamen 14.13), by bisection."""
        Voc = Vt * np.log1p(JL / Js)
        lo, hi = 0.0, Voc
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if (1.0 + mid / Vt) * np.exp(mid / Vt) < 1.0 + JL / Js:
                lo = mid
            else:
                hi = mid
        Vm = 0.5 * (lo + hi)
        Jm = JL - Js * np.expm1(Vm / Vt)
        return Voc, Vm, Jm, Vm * Jm, Vm * Jm / (JL * Voc)

    def cell_current(V, JL, Js, rs):
        """Terminal current density (diode convention) at terminal voltage V with series
        resistance rs (ohm cm^2): J = J_s{exp[(V - J r_s)/V_t] - 1} - J_L, by bisection."""
        V = np.asarray(V, dtype=float)
        lo = np.full_like(V, -JL - 1e-15)
        hi = np.maximum(Js * np.expm1(np.minimum(V, 1.5) / VT) - JL, 0.0) + 1e-15
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            f = Js * np.expm1(np.minimum((V - mid * rs) / VT, 700.0)) - JL - mid
            lo = np.where(f > 0, mid, lo)
            hi = np.where(f > 0, hi, mid)
        return 0.5 * (lo + hi)

    # ------------- spectrum for the efficiency-versus-bandgap panel
    H, C_LIGHT, K_B = 6.626e-34, 3.0e8, 1.381e-23
    T_SUN, T_CELL = 5778.0, 300.0
    EPH = np.linspace(0.05, 8.0, 16000)  # photon energy grid, eV

    def bb_photon_flux(Ev, T):
        """Hemispherical blackbody photon flux per unit energy, photons/(s cm^2 eV)."""
        Ej = Ev * E_CHARGE
        return 2.0 * np.pi / (H**3 * C_LIGHT**2) * Ej**2 / np.expm1(Ej / (K_B * T)) * E_CHARGE * 1e-4

    _sun = bb_photon_flux(EPH, T_SUN)
    SUN_FLUX = _sun * P_SUN / np.trapezoid(_sun * EPH * E_CHARGE, EPH)  # scaled to 100 mW/cm^2
    _dE = np.diff(EPH)
    # photons per (s cm^2) with energy above each grid point, by cumulative trapezoid from the top
    _seg = 0.5 * (SUN_FLUX[1:] + SUN_FLUX[:-1]) * _dE
    N_ABOVE = np.r_[np.cumsum(_seg[::-1])[::-1], 0.0]
    _cell = bb_photon_flux(EPH, T_CELL)
    _segc = 0.5 * (_cell[1:] + _cell[:-1]) * _dE
    NC_ABOVE = np.r_[np.cumsum(_segc[::-1])[::-1], 0.0]

    def photons_above(Eg):
        return float(np.interp(Eg, EPH, N_ABOVE))

    def efficiency_vs_gap(Eg, conc=1.0, model="sze", band_top=None):
        """Ideal efficiency of a cell of gap Eg (eV) under the scaled 5778 K blackbody sun.
        J_L = e x (photons with Eg < hv < band_top) x conc, one pair per photon.
        model 'sze': J_s = 1.5e5 exp(-Eg/kT) A/cm^2 (empirical); 'radiative': J_s from
        300 K blackbody emission above Eg (detailed balance)."""
        n = photons_above(Eg) - (photons_above(band_top) if band_top is not None else 0.0)
        JL = E_CHARGE * n * conc
        if model == "sze":
            Js = 1.5e5 * np.exp(-Eg / VT)
        else:
            Js = E_CHARGE * float(np.interp(Eg, EPH, NC_ABOVE))
        if JL <= 0:
            return 0.0, JL, 0.0
        Voc, Vm, Jm, Pm, FF = ideal_max_power(JL, Js)
        return Pm / (P_SUN * conc), JL, Voc

    EG_GRID = np.linspace(0.6, 2.6, 201)
    EFF_SZE_1 = np.array([efficiency_vs_gap(g, 1.0)[0] for g in EG_GRID])
    EFF_SZE_1000 = np.array([efficiency_vs_gap(g, 1000.0)[0] for g in EG_GRID])
    EFF_RAD_1 = np.array([efficiency_vs_gap(g, 1.0, "radiative")[0] for g in EG_GRID])
    EG_SI = 1.12
    TOP_GRID = np.linspace(1.2, 2.6, 141)
    EFF_TANDEM = np.array(
        [efficiency_vs_gap(g, 1.0)[0] + efficiency_vs_gap(EG_SI, 1.0, band_top=g)[0] for g in TOP_GRID]
    )
    return (
        EFF_RAD_1,
        EFF_SZE_1,
        EFF_SZE_1000,
        EFF_TANDEM,
        EG_GRID,
        EG_SI,
        P_SUN,
        TOP_GRID,
        cell_current,
        efficiency_vs_gap,
        ideal_max_power,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Solar Cell: Maximum Power, Load Line and Concentration

    The same diode with \(J_L\) set directly: \(J = J_s[\exp(eV_a/kT) - 1] - J_L\) in quadrant IV, with the long-diode \(J_s\).
    """)
    return


@app.cell
def _(mo):
    JL1_slider = mo.ui.slider(start=5.0, stop=45.0, step=1.0, value=15.0, show_value=True, label=r"\(J_L\) at one sun (mA/cm²)")
    conc_slider = mo.ui.slider(start=0.0, stop=3.0, step=0.1, value=0.0, show_value=True, label=r"log₁₀(concentration \(C\), suns)")
    cell_Na_slider = mo.ui.slider(start=15.0, stop=19.0, step=0.1, value=18.7, show_value=True, label=r"log₁₀(\(N_a\) / cm⁻³)")
    cell_Nd_slider = mo.ui.slider(start=15.0, stop=19.0, step=0.1, value=16.0, show_value=True, label=r"log₁₀(\(N_d\) / cm⁻³)")
    _tau = [1e-9, 2e-9, 5e-9, 1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]
    cell_taun_slider = mo.ui.slider(steps=_tau, value=5e-7, show_value=True, label=r"\(\tau_{n0}\) (s)")
    cell_taup_slider = mo.ui.slider(steps=_tau, value=1e-7, show_value=True, label=r"\(\tau_{p0}\) (s)")
    rs_slider = mo.ui.slider(start=0.0, stop=5.0, step=0.1, value=0.0, show_value=True, label=r"series resistance \(r_s\) (Ω·cm²)")
    RL_slider = mo.ui.slider(start=0.0, stop=3.0, step=0.01, value=1.49, show_value=True, label=r"log₁₀(load \(R_L\) / Ω·cm²)")
    return (
        JL1_slider,
        RL_slider,
        cell_Na_slider,
        cell_Nd_slider,
        cell_taun_slider,
        cell_taup_slider,
        conc_slider,
        rs_slider,
    )


@app.cell
def _(
    JL1_slider,
    RL_slider,
    cell_Na_slider,
    cell_Nd_slider,
    cell_taun_slider,
    cell_taup_slider,
    conc_slider,
    mo,
    rs_slider,
):
    mo.vstack(
        [
            mo.md(
                r"Defaults are Neamen Example 14.3: \(N_a = 5\times 10^{18}\) (log₁₀ = 18.7), \(N_d = 10^{16}\,\mathrm{cm^{-3}}\), \(\tau_{n0} = 5\times 10^{-7}\), \(\tau_{p0} = 10^{-7}\,\mathrm{s}\), \(J_L = 15\,\mathrm{mA/cm^2}\). Resistances are per unit area (Ω·cm²), so \(V = -JR_L\) for the load. The incident power is taken as \(100\,\mathrm{mW/cm^2}\) per sun."
            ),
            mo.hstack(
                [
                    mo.vstack([mo.md("**Light**"), JL1_slider, conc_slider]),
                    mo.vstack([mo.md("**Junction (sets \\(J_s\\))**"), cell_Na_slider, cell_Nd_slider, cell_taun_slider, cell_taup_slider]),
                    mo.vstack([mo.md("**Circuit**"), rs_slider, RL_slider]),
                ],
                align="start",
                widths=[1, 1, 1],
            ),
        ]
    )
    return


@app.cell
def _(
    BLUE,
    GRAY,
    GREEN,
    JL1_slider,
    NAVY,
    ORANGE,
    P_SUN,
    RL_slider,
    VT,
    cell_Na_slider,
    cell_Nd_slider,
    cell_current,
    cell_taun_slider,
    cell_taup_slider,
    conc_slider,
    ideal_max_power,
    mo,
    np,
    plt,
    rs_slider,
    saturation_current,
    sci_tex,
    style_ax,
):
    _C = 10.0 ** float(conc_slider.value)
    _JL1 = float(JL1_slider.value) * 1e-3
    _JL = _JL1 * _C
    _Na, _Nd = 10.0 ** float(cell_Na_slider.value), 10.0 ** float(cell_Nd_slider.value)
    _tn, _tp = float(cell_taun_slider.value), float(cell_taup_slider.value)
    _rs = float(rs_slider.value)
    _RL = 10.0 ** float(RL_slider.value)
    _Js = float(saturation_current(_Na, _Nd, _tn, _tp))
    _Voc, _Vm0, _Jm0, _Pm0, _FF0 = ideal_max_power(_JL, _Js)

    _V = np.linspace(0.0, _Voc, 1500)
    _J = cell_current(_V, _JL, _Js, _rs)
    _P = -_J * _V
    # maximum power with r_s: refine on a fine grid around the discrete maximum
    _i = int(np.argmax(_P))
    _Vf = np.linspace(_V[max(_i - 2, 0)], _V[min(_i + 2, _V.size - 1)], 2001)
    _Pf = -cell_current(_Vf, _JL, _Js, _rs) * _Vf
    _Vm = float(_Vf[np.argmax(_Pf)])
    _Pm = float(_Pf.max())
    _Jm = _Pm / _Vm
    _eta = _Pm / (P_SUN * _C)
    _Jsc = -float(cell_current(0.0, _JL, _Js, _rs))

    # operating point with the load: V = -J R_L (bisection on V in [0, V_oc])
    _lo, _hi = 0.0, _Voc
    for _ in range(100):
        _mid = 0.5 * (_lo + _hi)
        if -float(cell_current(_mid, _JL, _Js, _rs)) * _RL > _mid:
            _lo = _mid
        else:
            _hi = _mid
    _Vop = 0.5 * (_lo + _hi)
    _Jop = float(cell_current(_Vop, _JL, _Js, _rs))
    _Pop = -_Jop * _Vop
    _Ropt = _Vm / _Jm

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(15, 5.6))
    _a1.fill_between([0, _Vm], [-_Jm * 1e3] * 2, [0, 0], color=NAVY, alpha=0.12, lw=0)
    _a1.plot(_V, _J * 1e3, color=NAVY, lw=2.8, label="cell")
    _VL = np.linspace(0, _Voc * 1.05, 50)
    _a1.plot(_VL, -_VL / _RL * 1e3, color=GREEN, lw=2, ls="--", label=rf"load line, $R_L$ = {_RL:.3g} Ω·cm²")
    _a1.plot([_Vm], [-_Jm * 1e3], "s", color=NAVY, ms=10, label="maximum power")
    _a1.plot([_Vop], [_Jop * 1e3], "o", color=ORANGE, ms=11, zorder=5, label="operating point")
    _a1.axhline(0, color="k", lw=0.9)
    _a1.set_xlim(0, _Voc * 1.08)
    _a1.set_ylim(-1.15 * _JL * 1e3, 0.12 * _JL * 1e3)
    _a1.text(0.8 * _Vm, -0.12 * _Jm * 1e3, r"$P_m$", color=NAVY, ha="center", va="center", fontsize=18)
    _a1.text(_Voc, 0.03 * _JL * 1e3, r"$V_{oc}$", ha="center", va="bottom")
    _a1.set_xlabel(r"$V_a$ (V)")
    _a1.set_ylabel(r"$J$ (mA/cm$^2$)")
    _a1.legend(loc="center left", bbox_to_anchor=(0.0, 0.5), frameon=True, framealpha=0.9, edgecolor="none", fontsize=14)
    style_ax(_a1)

    _a2.plot(_V, _P * 1e3, color=BLUE, lw=2.8)
    _a2.plot([_Vm], [_Pm * 1e3], "s", color=NAVY, ms=10)
    _a2.plot([_Vop], [_Pop * 1e3], "o", color=ORANGE, ms=11, zorder=5)
    _a2.axvline(_Vm, color=NAVY, ls=":", lw=1.4)
    _a2.set_xlim(0, _Voc * 1.08)
    _a2.set_ylim(0, 1.2 * _Pm * 1e3)
    _a2.set_xlabel(r"$V_a$ (V)")
    _a2.set_ylabel(r"$P_{out} = -JV_a$ (mW/cm$^2$)")
    style_ax(_a2)
    _fig.tight_layout(w_pad=3)

    # V_oc and efficiency versus concentration
    _Cs = np.logspace(0, 3, 61)
    _Voc_c = VT * np.log1p(_JL1 * _Cs / _Js)
    # low injection at open circuit: excess minority carriers on the lighter side vs its majority
    _Nmin = min(_Na, _Nd)
    _inj_c = (1.5e10**2 / _Nmin) * np.expm1(_Voc_c / VT) / _Nmin
    _inj = float((1.5e10**2 / _Nmin) * np.expm1(_Voc / VT) / _Nmin)
    _bad = _inj_c >= 0.1
    _eta_c = []
    for _c, _vo in zip(_Cs, _Voc_c):
        _vv = np.linspace(0.0, _vo, 600)
        _eta_c.append(float(np.max(-cell_current(_vv, _JL1 * _c, _Js, _rs) * _vv)) / (P_SUN * _c))
    _eta_c = np.array(_eta_c)
    _fig2, (_b1, _b2) = plt.subplots(1, 2, figsize=(15, 4.6))
    for _bx in (_b1, _b2):
        if _bad.any():
            _bx.axvspan(_Cs[_bad][0], _Cs[-1], color=GRAY, alpha=0.15, lw=0)
    _b1.semilogx(_Cs, _Voc_c, color=NAVY, lw=2.8)
    _b1.plot([_C], [_Voc], "o", color=ORANGE, ms=11, zorder=5)
    _b1.set_xlabel(r"concentration $C$ (suns)")
    _b1.set_ylabel(r"$V_{oc}$ (V)")
    _b1.set_title(rf"$V_{{oc}}$ rises by $V_t\ln 10$ = {1e3 * VT * np.log(10):.1f} mV per decade", fontsize=15)
    style_ax(_b1)
    _b2.semilogx(_Cs, 100 * _eta_c, color=BLUE, lw=2.8)
    _b2.plot([_C], [100 * _eta], "o", color=ORANGE, ms=11, zorder=5)
    _b2.set_xlabel(r"concentration $C$ (suns)")
    _b2.set_ylabel(r"$\eta$ (%)")
    _b2.set_title(rf"efficiency with $r_s$ = {_rs:.1f} Ω·cm²", fontsize=15)
    style_ax(_b2)
    _fig2.tight_layout()

    _rsnote = (
        r"With \(r_s = 0\) these are the closed-form results, \(V_m\) from Eq. (14.13)."
        if _rs == 0
        else rf"With \(r_s > 0\) the maximum is found numerically. Without \(r_s\): \(V_m = {_Vm0:.3f}\,\mathrm{{V}}\), \(P_m = {_Pm0 * 1e3:.3f}\,\mathrm{{mW/cm^2}}\), \(FF = {_FF0:.3f}\)."
    )
    _table = mo.md(
        rf"""
    | Quantity | Value | Quantity | Value |
    |:--|:--|:--|:--|
    | \(J_s\) | \({sci_tex(_Js)}\,\mathrm{{A/cm^2}}\) | \(J_L = CJ_L(1\,\text{{sun}})\) | \({_JL * 1e3:.4g}\,\mathrm{{mA/cm^2}}\) |
    | \(V_{{oc}} = V_t\ln(1 + J_L/J_s)\) | \({_Voc:.3f}\,\mathrm{{V}}\) | \(J_{{sc}}\) | \({_Jsc * 1e3:.4g}\,\mathrm{{mA/cm^2}}\) |
    | \(V_m\) | \({_Vm:.3f}\,\mathrm{{V}}\) | \(J_m\) | \({_Jm * 1e3:.4g}\,\mathrm{{mA/cm^2}}\) |
    | \(P_m = V_mJ_m\) | \({_Pm * 1e3:.3f}\,\mathrm{{mW/cm^2}}\) | \(FF = V_mJ_m/(V_{{oc}}J_{{sc}})\) | \({_Pm / (_Voc * _Jsc):.3f}\) |
    | incident power | \({P_SUN * _C * 1e3:.4g}\,\mathrm{{mW/cm^2}}\) | \(\eta = P_m/P_{{in}}\) | \({100 * _eta:.2f}\%\) |
    | operating point \(V_a\) | \({_Vop:.3f}\,\mathrm{{V}}\) | operating point \(J\) | \({_Jop * 1e3:.4g}\,\mathrm{{mA/cm^2}}\) |
    | power into \(R_L\) | \({_Pop * 1e3:.3f}\,\mathrm{{mW/cm^2}}\) (\({100 * _Pop / _Pm:.1f}\%\) of \(P_m\)) | best load \(V_m/J_m\) | \({_Ropt:.3g}\,\Omega\,\mathrm{{cm^2}}\) |
    """
    )
    _ff_note = (
        ""
        if abs(_Jsc - _JL) < 1e-6 * _JL
        else rf" Series resistance lowers \(J_{{sc}}\) to {100 * _Jsc / _JL:.1f}% of \(J_L\): at short circuit the junction is forward biased by \(J_{{sc}}r_s\)."
    )
    mo.vstack(
        [
            _fig,
            mo.md(
                r"Left: the cell in quadrant IV, the largest rectangle \(P_m\) under the curve, and the load line \(J = -V_a/R_L\); the operating point is where they cross. Right: the delivered power. "
                + _rsnote
                + _ff_note
            ),
            _table,
            _fig2,
            mo.md(
                r"Grey band: outside low injection at open circuit (excess minority carriers on the lighter-doped side reach 0.1 of its majority concentration), so the ideal-diode results there are only indicative."
                if _bad.any()
                else ""
            ),
        ]
        + (
            [
                mo.callout(
                    mo.md(
                        rf"At open circuit the excess minority carriers on the lighter-doped side are {_inj:.2f} of its majority concentration: low injection fails, and the ideal-diode model overestimates \(V_{{oc}}\) and \(\eta\)."
                    ),
                    kind="warn",
                )
            ]
            if _inj >= 0.1
            else []
        )
    )
    return


@app.cell
def _(mo):
    Eg_slider = mo.ui.slider(start=0.6, stop=2.6, step=0.02, value=1.12, show_value=True, label=r"\(E_g\) (eV)")
    return (Eg_slider,)


@app.cell
def _(Eg_slider, mo):
    mo.vstack(
        [
            mo.md(r"## 3. Efficiency and Bandgap"),
            mo.md(
                r"""
    Photons with \(h\nu < E_g\) are lost, and each absorbed photon gives one pair whose energy above \(E_g\) becomes heat. A small \(E_g\) gives a large \(J_L\) but a small \(V_{oc}\); a large \(E_g\) the reverse. The sun is modelled as a \(5778\,\mathrm{K}\) blackbody scaled to \(100\,\mathrm{mW/cm^2}\), with one pair per photon above \(E_g\) and no reflection.

    Solid and dashed curves use the empirical saturation current \(J_s \approx 1.5\times 10^5\exp(-E_g/kT)\,\mathrm{A/cm^2}\) (after Sze), at one sun and at \(1000\) suns, an estimate for a real cell. The thin grey curve is the radiative (detailed-balance) limit, where the only recombination is the light the cell itself emits; it is an upper bound. Neamen Fig. 14.10 (from Sze) is a calculation of this type: its peaks, about 31% at one sun and 37% at \(1000\) suns, are close to the grey curve (and to its \(1000\)-sun value, 36.5%). The Si limit of about 28% quoted on the slides lies between the two curves here (24% and 30% at \(1.12\,\mathrm{eV}\)); each number depends on the assumed \(J_s\) and solar spectrum.
    """
            ),
            Eg_slider,
        ]
    )
    return


@app.cell
def _(
    EFF_RAD_1,
    EFF_SZE_1,
    EFF_SZE_1000,
    EFF_TANDEM,
    EG_GRID,
    EG_SI,
    Eg_slider,
    GRAY,
    NAVY,
    ORANGE,
    PINK,
    TOP_GRID,
    efficiency_vs_gap,
    mo,
    np,
    plt,
    style_ax,
):
    _Eg = float(Eg_slider.value)
    _eta, _JLg, _Vocg = efficiency_vs_gap(_Eg, 1.0)
    _eta1000 = efficiency_vs_gap(_Eg, 1000.0)[0]
    _etar = efficiency_vs_gap(_Eg, 1.0, "radiative")[0]
    _ipk = int(np.argmax(EFF_SZE_1))
    _ipk3 = int(np.argmax(EFF_SZE_1000))
    _iT = int(np.argmax(EFF_TANDEM))

    _fig, (_a, _t) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    _a.plot(EG_GRID, 100 * EFF_SZE_1, color=NAVY, lw=2.8, label="1 sun")
    _a.plot(EG_GRID, 100 * EFF_SZE_1000, color=NAVY, lw=2.4, ls="--", label="1000 suns")
    _a.plot(EG_GRID, 100 * EFF_RAD_1, color=GRAY, lw=1.5, label="radiative limit, 1 sun")
    _a.plot([_Eg], [100 * _eta], "o", color=ORANGE, ms=11, zorder=5)
    for _g, _n in ((1.12, "Si"), (1.42, "GaAs"), (0.66, "Ge"), (1.7, "a-Si")):
        _a.axvline(_g, color=PINK, lw=1, ls=":")
        _a.text(_g, 1.5, _n, color=PINK, ha="center", fontsize=14, bbox=dict(fc="white", ec="none", pad=1))
    _a.set_xlim(EG_GRID[0], EG_GRID[-1])
    _a.set_ylim(0, 40)
    _a.set_xlabel(r"$E_g$ (eV)")
    _a.set_ylabel(r"ideal efficiency $\eta$ (%)")
    _a.legend(loc="upper right", frameon=False, fontsize=14)
    style_ax(_a)

    _t.plot(TOP_GRID, 100 * EFF_TANDEM, color=ORANGE, lw=2.8)
    _t.axhline(100 * efficiency_vs_gap(EG_SI, 1.0)[0], color=NAVY, lw=1.6, ls="--")
    _t.text(TOP_GRID[-1], 100 * efficiency_vs_gap(EG_SI, 1.0)[0] + 0.6, "Si alone", color=NAVY, ha="right", fontsize=14)
    if TOP_GRID[0] <= _Eg <= TOP_GRID[-1]:
        _t.plot([_Eg], [100 * float(np.interp(_Eg, TOP_GRID, EFF_TANDEM))], "o", color=ORANGE, ms=11, zorder=5, mfc="white", mew=2.5)
    _t.set_xlim(TOP_GRID[0], TOP_GRID[-1])
    _t.set_ylim(20, 45)
    _t.set_xlabel(r"top-cell $E_g$ on a Si bottom cell (eV)")
    _t.set_ylabel(r"two-junction $\eta$ (%), 1 sun")
    style_ax(_t)
    _fig.tight_layout(w_pad=3)

    _tand = (
        rf"\({100 * float(np.interp(_Eg, TOP_GRID, EFF_TANDEM)):.1f}\%\)"
        if TOP_GRID[0] <= _Eg <= TOP_GRID[-1]
        else rf"top gap must exceed {TOP_GRID[0]:.1f} eV"
    )
    _table = mo.md(
        rf"""
    | Quantity | Value | Quantity | Value |
    |:--|:--|:--|:--|
    | \(J_L\) at \(E_g = {_Eg:.2f}\,\mathrm{{eV}}\), 1 sun | \({_JLg * 1e3:.1f}\,\mathrm{{mA/cm^2}}\) | \(V_{{oc}}\), 1 sun | \({_Vocg:.3f}\,\mathrm{{V}}\) |
    | \(\eta\), 1 sun | \({100 * _eta:.1f}\%\) | \(\eta\), 1000 suns | \({100 * _eta1000:.1f}\%\) |
    | radiative limit, 1 sun | \({100 * _etar:.1f}\%\) | two-junction on Si, top gap \({_Eg:.2f}\,\mathrm{{eV}}\) | {_tand} |
    | best single gap, 1 sun | \({EG_GRID[_ipk]:.2f}\,\mathrm{{eV}}\), \({100 * EFF_SZE_1[_ipk]:.1f}\%\) | best single gap, 1000 suns | \({EG_GRID[_ipk3]:.2f}\,\mathrm{{eV}}\), \({100 * EFF_SZE_1000[_ipk3]:.1f}\%\) |
    | best top gap on Si | \({TOP_GRID[_iT]:.2f}\,\mathrm{{eV}}\), \({100 * EFF_TANDEM[_iT]:.1f}\%\) | | |
    """
    )
    mo.vstack(
        [
            _fig,
            mo.md(
                r"Right: a two-junction stack, a top cell of gap \(E_g\) over a Si bottom cell. The top cell absorbs \(h\nu > E_g\); the bottom cell absorbs \(1.12\,\mathrm{eV} < h\nu < E_g\). Each cell runs at its own maximum power (separate terminals), with the same empirical \(J_s\)."
            ),
            _table,
        ]
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Equations used in Sections 2 and 3": mo.md(
                r"""
    Saturation current of the long diode and the illuminated cell:

    $$
    J_s = en_i^2\left(\frac{D_n}{L_nN_a} + \frac{D_p}{L_pN_d}\right), \qquad J = J_s\left[\exp\left(\frac{eV_a}{kT}\right) - 1\right] - J_L
    $$

    Open circuit (\(J = 0\)) and the maximum-power condition \(dP/dV_a = 0\), Neamen Eqs. (14.10) and (14.13):

    $$
    V_{oc} = V_t\ln\left(1 + \frac{J_L}{J_s}\right), \qquad \left(1 + \frac{V_m}{V_t}\right)\exp\left(\frac{V_m}{V_t}\right) = 1 + \frac{J_L}{J_s}
    $$

    with \(J_m = J_L - J_s[\exp(V_m/V_t) - 1]\), \(P_m = V_mJ_m\), \(FF = V_mJ_m/(V_{oc}J_{sc})\) and \(\eta = P_m/P_{in}\), Eq. (14.14). \(V_m\) is found by bisection.

    **Series resistance and load.** With \(r_s\) the junction sees \(V_a - Jr_s\): \(J = J_s\{\exp[e(V_a - Jr_s)/kT] - 1\} - J_L\), solved for \(J\) by bisection, and the maximum of \(-JV_a\) is found numerically. The load sets \(V_a = -JR_L\); the operating point is the intersection with the cell curve.

    **Concentration.** \(J_L\) is proportional to the light, \(J_L = CJ_L(1\,\text{sun})\), so \(V_{oc}\) rises by \(V_t\ln 10\) per decade when \(J_L \gg J_s\).

    **Efficiency versus \(E_g\).** \(J_L = e\int_{E_g}^{\infty}\Phi(h\nu)\,d(h\nu)\), with \(\Phi\) the photon flux of a \(5778\,\mathrm{K}\) blackbody scaled to \(100\,\mathrm{mW/cm^2}\) (one sun), times \(C\). The efficiency is \(P_m/(C\times 100\,\mathrm{mW/cm^2})\) with \(P_m\) from Eq. (14.13). The empirical \(J_s = 1.5\times 10^5\exp(-E_g/kT)\,\mathrm{A/cm^2}\) stands in for a real cell; the radiative limit uses the \(300\,\mathrm{K}\) blackbody emission above \(E_g\), \(J_s = e\int_{E_g}^{\infty}\Phi_{300\,\mathrm{K}}\,d(h\nu)\). The real AM1.5 spectrum has absorption bands, so curves for the measured spectrum differ in detail; in every version the efficiency is flat near its peak, between about \(1.1\) and \(1.5\,\mathrm{eV}\).
    """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Check your understanding: solar cell

    4. With the defaults (Neamen Example 14.3), raise the light tenfold (\(\log_{10}C = 1\)). How much do \(V_{oc}\), \(P_m\) and \(\eta\) change?
    5. Back at one sun, set \(r_s = 2\,\Omega\,\mathrm{cm^2}\). What happens to \(FF\) and \(P_m\)? Then try \(r_s = 2\,\Omega\,\mathrm{cm^2}\) at \(100\) suns.
    6. With \(r_s = 0\) and one sun, set \(R_L\) to twice and to half its best value. What fraction of \(P_m\) reaches the load in each case?
    """)
    return


@app.cell
def _(P_SUN, cell_current, ideal_max_power, mo, np, saturation_current):
    _Js = float(saturation_current(5e18, 1e16, 5e-7, 1e-7))
    _a = ideal_max_power(15e-3, _Js)
    _b = ideal_max_power(150e-3, _Js)

    def _pmax(JL, rs):
        V = np.linspace(0, ideal_max_power(JL, _Js)[0], 20001)
        P = -cell_current(V, JL, _Js, rs) * V
        i = int(np.argmax(P))
        jsc = -float(cell_current(0.0, JL, _Js, rs))
        return float(P[i]), float(P[i] / (jsc * ideal_max_power(JL, _Js)[0])), jsc

    _p1, _ff1, _ = _pmax(15e-3, 2.0)
    _p100, _ff100, _jsc100 = _pmax(1.5, 2.0)
    _a100 = ideal_max_power(1.5, _Js)

    def _pload(RL):
        lo, hi = 0.0, _a[0]
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if -float(cell_current(mid, 15e-3, _Js, 0.0)) * RL > mid:
                lo = mid
            else:
                hi = mid
        v = 0.5 * (lo + hi)
        return -float(cell_current(v, 15e-3, _Js, 0.0)) * v

    _Ropt = _a[1] / _a[2]
    _f2, _fh = _pload(2 * _Ropt) / _a[3], _pload(0.5 * _Ropt) / _a[3]
    mo.accordion(
        {
            "Answers 4–6": mo.md(
                rf"""
    4. \(V_{{oc}}\) rises from \({_a[0]:.3f}\) to \({_b[0]:.3f}\,\mathrm{{V}}\), by \(V_t\ln 10 = 59.6\,\mathrm{{mV}}\). \(P_m\) rises from \({_a[3] * 1e3:.2f}\) to \({_b[3] * 1e3:.1f}\,\mathrm{{mW/cm^2}}\), more than tenfold, because both \(J\) and \(V\) grow: \(\eta\) goes from \({100 * _a[3] / P_SUN:.2f}\) to \({100 * _b[3] / (10 * P_SUN):.2f}\%\). This is why concentrators work. (\(\eta\) is low here because \(J_L = 15\,\mathrm{{mA/cm^2}}\) is well below what a good Si cell collects, about \(40\,\mathrm{{mA/cm^2}}\).)
    5. At one sun, \(FF\) drops from \({_a[4]:.3f}\) to \({_ff1:.3f}\) and \(P_m\) from \({_a[3] * 1e3:.2f}\) to \({_p1 * 1e3:.2f}\,\mathrm{{mW/cm^2}}\): the drop \(J_mr_s \approx {_a[2] * 2.0 * 1e3:.0f}\,\mathrm{{mV}}\) is small next to \(V_m\). At \(100\) suns \(J_Lr_s = 3\,\mathrm{{V}}\), several times \(V_{{oc}}\): the cell is limited by the resistor, \(J_{{sc}}\) falls to \({_jsc100 * 1e3:.0f}\,\mathrm{{mA/cm^2}}\) (from \(1500\)), \(FF\) collapses from \({_a100[4]:.3f}\) to \({_ff100:.3f}\), and \(P_m = {_p100 * 1e3:.0f}\) instead of \({_a100[3] * 1e3:.0f}\,\mathrm{{mW/cm^2}}\). (At \(100\) suns the cell is also near the edge of low injection.) Concentrator cells need very low series resistance.
    6. The best load is \(V_m/J_m = {_Ropt:.1f}\,\Omega\,\mathrm{{cm^2}}\). At twice that, \({100 * _f2:.1f}\%\) of \(P_m\) reaches the load; at half, \({100 * _fh:.1f}\%\). The power peak is broad, but too small a load is worse: it pulls the cell toward short circuit, where \(V_a \to 0\).
    """
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
