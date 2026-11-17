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

    # n-channel Si MOSFET at 300 K: the body-effect example on the slides (after Neamen Figs. 10.50, 10.51).
    E_CHARGE = 1.6e-19
    EPS0 = 8.85e-14
    EPS_S = 11.7 * EPS0
    EPS_OX = 3.9 * EPS0
    VT_KT = 0.0259
    NI = 1.5e10
    NA = 3e16  # cm^-3
    TOX = 20e-7  # cm
    VT0 = 0.64  # V, threshold voltage at V_SB = 0 (from the slides)
    MU_N = 400.0  # cm^2/V-s
    W_OVER_L = 10.0
    COX = EPS_OX / TOX
    PHI_FP = VT_KT * np.log(NA / NI)
    GAMMA = np.sqrt(2.0 * E_CHARGE * EPS_S * NA) / COX
    BETA = MU_N * COX * W_OVER_L  # A/V^2, W mu_n C_ox / L (twice the slides' K_n)
    VGS_LIST = (1.0, 1.5, 2.0, 2.5, 3.0)
    VDS_MAX = 4.0

    COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    GRAY = "#6B6B6B"
    NAVY = "#0f4c81"

    def sci_tex(x, digits=2):
        x = float(x)
        if not np.isfinite(x) or x == 0.0:
            return "0"
        exp = int(np.floor(np.log10(abs(x))))
        man = x / 10.0**exp
        if round(abs(man), digits) >= 10.0:
            man /= 10.0
            exp += 1
        return rf"{man:.{digits}f}\times 10^{{{exp}}}"

    def style_ax(ax):
        ax.tick_params(direction="out", length=4, width=1.0)
        ax.grid(True, alpha=0.28, color=GRAY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def threshold(VSB):
        """Body effect, Neamen (10.82)-(10.83): V_T = V_T0 + gamma [sqrt(2 phi_fp + V_SB) - sqrt(2 phi_fp)]."""
        return VT0 + GAMMA * (np.sqrt(2.0 * PHI_FP + VSB) - np.sqrt(2.0 * PHI_FP))

    def m_estimate(VSB):
        """Bulk-charge factor estimate from the slides, m = 1 + eps_s t_ox / (eps_ox x_dT),
        with x_dT the space charge width at threshold, sqrt(2 eps_s (2 phi_fp + V_SB) / (e N_a))."""
        xd = np.sqrt(2.0 * EPS_S * (2.0 * PHI_FP + VSB) / (E_CHARGE * NA))
        return 1.0 + EPS_S * TOX / (EPS_OX * xd)

    def intrinsic_current(Vgs, Vds, VT, m, Ec, lam):
        """Drain current of the intrinsic channel (no series resistance), arrays allowed.

        Bulk charge: I_D = (W mu_n C_ox / L)[(V_GS - V_T) V_DS - (m/2) V_DS^2].
        Velocity saturation: divided by (1 + V_DS/(E_sat L)), with E_sat = v_sat/mu_n; V_DS(sat) from dI_D/dV_DS = 0.
        Channel length modulation: times [1 + lambda (V_DS - V_DS(sat))] above V_DS(sat) only, so Delta L = 0 at pinch-off.
        """
        Vgs = np.asarray(Vgs, dtype=float)
        Vds = np.maximum(np.asarray(Vds, dtype=float), 0.0)
        vov = np.maximum(Vgs - VT, 0.0)
        # E_sat L [sqrt(1 + 2 vov/(m E_sat L)) - 1], rationalized: no cancellation, and vov/m when E_sat L = inf.
        vdsat = (2.0 * vov / m) / (np.sqrt(1.0 + 2.0 * vov / (m * Ec)) + 1.0)
        V = np.minimum(Vds, vdsat)
        I = BETA * (vov * V - 0.5 * m * V**2) / (1.0 + V / Ec)
        clm = 1.0 + lam * np.maximum(Vds - vdsat, 0.0)
        return np.maximum(I, 0.0) * clm, vdsat

    def drain_current(Vgs, Vds, VT, m, Ec, lam, rs, rd):
        """Terminal current with r_s, r_d: solve I = I_int(V_GS - I r_s, V_DS - I (r_s + r_d)) by bisection.
        The right side falls as I rises, so the root is unique and lies in [0, I_int(V_GS, V_DS)]."""
        Vgs = np.broadcast_to(np.asarray(Vgs, dtype=float), np.broadcast(Vgs, Vds).shape).copy()
        Vds = np.broadcast_to(np.asarray(Vds, dtype=float), Vgs.shape).copy()
        hi, _ = intrinsic_current(Vgs, Vds, VT, m, Ec, lam)
        if rs == 0 and rd == 0:
            return hi
        lo = np.zeros_like(hi)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            rhs, _ = intrinsic_current(Vgs - mid * rs, Vds - mid * (rs + rd), VT, m, Ec, lam)
            up = rhs > mid
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        return 0.5 * (lo + hi)

    def ideal_current(Vgs, Vds):
        """Ideal MOSFET at V_SB = 0 (slide recap): square law, V_DS(sat) = V_GS - V_T0."""
        vov = np.maximum(np.asarray(Vgs, dtype=float) - VT0, 0.0)
        V = np.minimum(np.asarray(Vds, dtype=float), vov)
        return 0.5 * BETA * (2.0 * vov * V - V**2)

    return (
        COLORS,
        COX,
        GAMMA,
        GRAY,
        BETA,
        MU_N,
        NAVY,
        PHI_FP,
        VDS_MAX,
        VGS_LIST,
        VT0,
        W_OVER_L,
        drain_current,
        ideal_current,
        intrinsic_current,
        m_estimate,
        plt,
        sci_tex,
        style_ax,
        threshold,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # MOSFET I–V
    **ECE335 - companion to lectures**

    An n-channel Si MOSFET at \(300\,\mathrm{K}\). The interactive below compares the ideal \(I_D\)–\(V_{DS}\) characteristics with the modified ones: the body effect (\(V_{SB}\)), the bulk charge effect (\(m\)), velocity saturation, source and drain resistance, and channel length modulation. The derivation is presented on the slides.

    Device: the body-effect example on the slides, \(N_a = 3\times 10^{16}\,\mathrm{cm^{-3}}\), \(t_{ox} = 20\,\mathrm{nm}\), \(V_{T0} = 0.64\,\mathrm{V}\); with \(\mu_n = 400\,\mathrm{cm^2/V\text{-}s}\) and \(W/L = 10\).
    """)
    return


@app.cell
def _(mo):
    VSB_slider = mo.ui.slider(start=0.0, stop=5.0, step=0.05, value=0.0, show_value=True, label=r"$V_{SB}$ (V)")
    m_slider = mo.ui.slider(start=1.0, stop=1.5, step=0.01, value=1.0, show_value=True, label=r"$m$")
    vsat_on = mo.ui.checkbox(value=False, label="velocity saturation")
    L_slider = mo.ui.slider(start=0.1, stop=10.0, step=0.05, value=5.0, show_value=True, label=r"$L$ (μm)")
    vsat_log_slider = mo.ui.slider(start=6.0, stop=8.0, step=0.05, value=7.0, show_value=True, label=r"$\log_{10}(v_{sat}/\mathrm{cm\,s^{-1}})$")
    rs_slider = mo.ui.slider(start=0.0, stop=200.0, step=5.0, value=0.0, show_value=True, label=r"$r_s$ (Ω)")
    rd_slider = mo.ui.slider(start=0.0, stop=200.0, step=5.0, value=0.0, show_value=True, label=r"$r_d$ (Ω)")
    lam_slider = mo.ui.slider(start=0.0, stop=0.3, step=0.01, value=0.0, show_value=True, label=r"$\lambda$ (V$^{-1}$)")
    mo.vstack(
        [
            mo.md(r"## Interactive: \(I_D\) versus \(V_{DS}\) with modifications"),
            mo.md(
                r"Start with \(V_{SB}\) and \(m\): \(V_{SB}\) raises \(V_T\) (body effect), and \(m\) lowers \(V_{DS}(\mathrm{sat})\) and \(I_D(\mathrm{sat})\) (bulk charge effect). Velocity saturation is off by default; with it on, \(W/L\) is held at 10, so \(L\) changes only how strongly velocity saturation acts. Dashed: ideal MOSFET at \(V_{SB} = 0\). Solid: with the modifications. Dots: \(V_{DS}(\mathrm{sat})\)."
            ),
            mo.hstack([VSB_slider, m_slider], justify="start", gap=1.0, wrap=True),
            mo.hstack([vsat_on, L_slider, vsat_log_slider], justify="start", gap=1.0, wrap=True),
            mo.hstack([rs_slider, rd_slider, lam_slider], justify="start", gap=1.0, wrap=True),
        ]
    )
    return (
        L_slider,
        VSB_slider,
        lam_slider,
        m_slider,
        rd_slider,
        rs_slider,
        vsat_log_slider,
        vsat_on,
    )


@app.cell
def _(
    COLORS,
    COX,
    GAMMA,
    GRAY,
    L_slider,
    MU_N,
    NAVY,
    PHI_FP,
    VDS_MAX,
    VGS_LIST,
    VSB_slider,
    VT0,
    W_OVER_L,
    drain_current,
    ideal_current,
    intrinsic_current,
    lam_slider,
    m_estimate,
    m_slider,
    mo,
    np,
    plt,
    rd_slider,
    rs_slider,
    sci_tex,
    style_ax,
    threshold,
    vsat_log_slider,
    vsat_on,
):
    _VSB = float(VSB_slider.value)
    _m = float(m_slider.value)
    _L = float(L_slider.value) * 1e-4  # cm
    _vsat = 10.0 ** float(vsat_log_slider.value)
    _rs, _rd = float(rs_slider.value), float(rd_slider.value)
    _lam = float(lam_slider.value)
    _VT = float(threshold(_VSB))
    _Esat = _vsat / MU_N  # V/cm
    # With velocity saturation off, E_sat L is effectively infinite (constant mobility).
    _Ec = _Esat * _L if vsat_on.value else np.inf  # V

    _Vds = np.linspace(0.0, VDS_MAX, 600)
    _fig, (_ax, _axq) = plt.subplots(
        1, 2, figsize=(14.5, 6.0), layout="constrained", gridspec_kw={"width_ratios": [1.55, 1.0]}
    )
    _rows_vgs = []
    for _c, _vgs in zip(COLORS, VGS_LIST):
        _Iid = ideal_current(_vgs, _Vds)
        _Imod = drain_current(_vgs, _Vds, _VT, _m, _Ec, _lam, _rs, _rd)
        _ax.plot(_Vds, _Iid * 1e3, color=_c, lw=1.6, ls="--", alpha=0.6)
        _ax.plot(_Vds, _Imod * 1e3, color=_c, lw=2.4, label=rf"$V_{{GS}}$ = {_vgs:.1f} V")
        _vds_id = max(_vgs - VT0, 0.0)
        if _vds_id > 0:
            _ax.plot([_vds_id], [float(ideal_current(_vgs, _vds_id)) * 1e3], "o", mfc="white", mec=_c, mew=1.6, ms=7)
        # V_DS(sat) of the intrinsic channel, mapped to the terminals (V_DS = V_DS,int + I_D (r_s + r_d)).
        _isat_int = None
        if _vgs > _VT:
            # Self-consistent intrinsic gate drive with r_s: find I with I = I_int at the saturation point.
            # Channel length modulation is zero at pinch-off, so lambda does not enter here.
            _lo, _hi = 0.0, float(intrinsic_current(_vgs, 1e3, _VT, _m, _Ec, 0.0)[0])
            for _ in range(60):
                _mid = 0.5 * (_lo + _hi)
                _vgs_i = _vgs - _mid * _rs
                _ival = float(intrinsic_current(_vgs_i, 1e3, _VT, _m, _Ec, 0.0)[0])
                if _ival > _mid:
                    _lo = _mid
                else:
                    _hi = _mid
            _isat_int = 0.5 * (_lo + _hi)
            _vgs_i = _vgs - _isat_int * _rs
            _vdsat_int = float(intrinsic_current(_vgs_i, 1e3, _VT, _m, _Ec, 0.0)[1])
            _vdsat_term = _vdsat_int + _isat_int * (_rs + _rd)
            if _vdsat_term <= VDS_MAX:
                _ax.plot([_vdsat_term], [_isat_int * 1e3], "o", color=_c, ms=8, zorder=5)
            _rows_vgs.append(
                (
                    _vgs,
                    _vds_id,
                    float(ideal_current(_vgs, _vds_id)),
                    _vdsat_term,
                    _isat_int,
                )
            )
        else:
            _rows_vgs.append((_vgs, _vds_id, float(ideal_current(_vgs, _vds_id)), None, None))
    _ax.set_xlim(0.0, VDS_MAX)
    _ax.set_ylim(bottom=0.0)
    _ax.set_xlabel(r"$V_{DS}$ (V)")
    _ax.set_ylabel(r"$I_D$ (mA)")
    _ax.set_title(
        rf"$V_{{SB}}$ = {_VSB:.2f} V, $m$ = {_m:.2f}, $L$ = {L_slider.value:.2f} μm"
        if vsat_on.value
        else rf"$V_{{SB}}$ = {_VSB:.2f} V, $m$ = {_m:.2f} (velocity saturation off)"
    )
    _ax.legend(fontsize=14, loc="upper left")
    style_ax(_ax)

    # sqrt(I_D) versus V_GS in saturation: the intercept moves from V_T0 to V_T (cf. the body-effect slide).
    _Vg = np.linspace(0.0, max(VGS_LIST) + 0.5, 400)
    _Iq_id = ideal_current(_Vg, VDS_MAX)
    _Iq_mod = drain_current(_Vg, VDS_MAX, _VT, _m, _Ec, _lam, _rs, _rd)
    _axq.plot(_Vg, np.sqrt(_Iq_id * 1e3), color=GRAY, lw=1.6, ls="--", label="ideal, $V_{SB}$ = 0")
    _axq.plot(_Vg, np.sqrt(_Iq_mod * 1e3), color=NAVY, lw=2.4, label="modified")
    _axq.axvline(VT0, color=GRAY, ls=":", lw=1.2)
    _axq.axvline(_VT, color=NAVY, ls=":", lw=1.2)
    _ymax = _axq.get_ylim()[1]
    _axq.text(VT0, 0.97 * _ymax, r" $V_{T0}$", color=GRAY, va="top", ha="right" if _VT > VT0 + 0.05 else "left", fontsize=15)
    if _VT > VT0 + 0.05:
        _axq.text(_VT, 0.97 * _ymax, r" $V_T$", color=NAVY, va="top", ha="left", fontsize=15)
    _axq.set_xlabel(r"$V_{GS}$ (V)")
    _axq.set_ylabel(r"$\sqrt{I_D}$ (mA$^{1/2}$)")
    _axq.set_title(rf"Saturation, $V_{{DS}}$ = {VDS_MAX:.0f} V")
    _axq.legend(fontsize=13, loc="lower right")
    style_ax(_axq)

    _dVT = _VT - VT0
    _params = mo.md(
        rf"""
    | Quantity | Value | Quantity | Value |
    |:--|:--|:--|:--|
    | \(\phi_{{fp}}\) | \({PHI_FP:.3f}\,\mathrm{{V}}\) | \(C_{{ox}}\) | \({COX * 1e9:.1f}\,\mathrm{{nF/cm^2}}\) |
    | \(\gamma\) | \({GAMMA:.3f}\,\mathrm{{V^{{1/2}}}}\) | \(V_{{T0}}\) | \({VT0:.3f}\,\mathrm{{V}}\) |
    | \(\Delta V_T\) | \({_dVT:.3f}\,\mathrm{{V}}\) | \(V_T\) | \({_VT:.3f}\,\mathrm{{V}}\) |
    | \(m\) (slider) | \({_m:.2f}\) | \(m\) estimate, \(1 + \varepsilon_s t_{{ox}}/(\varepsilon_{{ox}} x_{{dT}})\) | \({m_estimate(_VSB):.2f}\) |
    | \(v_{{sat}}\) | \({sci_tex(_vsat)}\,\mathrm{{cm/s}}\) | \(\mathcal{{E}}_{{sat}} = v_{{sat}}/\mu_n\) | \({sci_tex(_Esat)}\,\mathrm{{V/cm}}\) |
    | \(\mathcal{{E}}_{{sat}}L\) | {(rf"\({_Ec:.2f}\,\mathrm{{V}}\)" if vsat_on.value else "off (constant mobility)")} | \(W\mu_nC_{{ox}}/L\) | \({MU_N * COX * W_OVER_L * 1e3:.3f}\,\mathrm{{mA/V^2}}\) |
    """
    )

    _lines = ["| \\(V_{GS}\\) | ideal \\(V_{DS}(\\mathrm{sat})\\) | ideal \\(I_D(\\mathrm{sat})\\) | modified \\(V_{DS}(\\mathrm{sat})\\) | modified \\(I_D(\\mathrm{sat})\\) |", "|:--|:--|:--|:--|:--|"]
    for _vgs, _v0, _i0, _v1, _i1 in _rows_vgs:
        _m1 = "below \\(V_T\\)" if _v1 is None else f"\\({_v1:.2f}\\,\\mathrm{{V}}\\)"
        _m2 = "0" if _i1 is None else f"\\({_i1 * 1e3:.3f}\\,\\mathrm{{mA}}\\)"
        _lines.append(f"| \\({_vgs:.1f}\\,\\mathrm{{V}}\\) | \\({_v0:.2f}\\,\\mathrm{{V}}\\) | \\({_i0 * 1e3:.3f}\\,\\mathrm{{mA}}\\) | {_m1} | {_m2} |")
    _sat_table = mo.md("\n".join(_lines))
    _note = mo.md(
        r"The modified \(V_{DS}(\mathrm{sat})\) and \(I_D(\mathrm{sat})\) mark the onset of saturation at the terminals: \(V_{DS}(\mathrm{sat}) = V_{DS,\mathrm{int}}(\mathrm{sat}) + I_D(\mathrm{sat})(r_s + r_d)\), with \(V_{DS,\mathrm{int}}(\mathrm{sat})\) evaluated at the intrinsic gate voltage \(V_{GS,\mathrm{int}} = V_{GS} - I_D(\mathrm{sat})r_s\). They do not depend on \(\lambda\); above \(V_{DS}(\mathrm{sat})\), \(\lambda > 0\) gives the curves a positive slope."
    )
    mo.vstack([_fig, _params, _sat_table, _note])
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Equations used in the plot": mo.vstack(
                [
                    mo.md(r"Body effect, and the estimate of the bulk-charge factor shown in the table:"),
                    mo.md(r"""
    $$
    V_T = V_{T0} + \gamma\left[\sqrt{2\phi_{fp} + V_{SB}} - \sqrt{2\phi_{fp}}\right], \qquad \gamma = \frac{\sqrt{2e\varepsilon_s N_a}}{C_{ox}}, \qquad \phi_{fp} = \frac{kT}{e}\ln\frac{N_a}{n_i}, \qquad C_{ox} = \frac{\varepsilon_{ox}}{t_{ox}}
    $$
    """),
                    mo.md(r"""
    $$
    m \approx 1 + \frac{\varepsilon_s t_{ox}}{\varepsilon_{ox} x_{dT}}, \qquad x_{dT} = \left[\frac{2\varepsilon_s(2\phi_{fp} + V_{SB})}{eN_a}\right]^{1/2}
    $$
    """),
                    mo.md(r"Intrinsic channel (no series resistance), for \(V_{GS} > V_T\); \(I_D = 0\) for \(V_{GS} \le V_T\). Bulk charge effect, velocity saturation, and channel length modulation:"),
                    mo.md(r"""
    $$
    I_D = \begin{cases}
    \dfrac{\dfrac{W\mu_n C_{ox}}{L}\left[(V_{GS} - V_T)V_{DS} - \dfrac{m}{2}V_{DS}^2\right]}{1 + \dfrac{V_{DS}}{\mathcal{E}_{sat}L}}, & 0 \le V_{DS} \le V_{DS}(\mathrm{sat}) \\[14pt]
    I_D(\mathrm{sat})\left[1 + \lambda\left(V_{DS} - V_{DS}(\mathrm{sat})\right)\right], & V_{DS} > V_{DS}(\mathrm{sat})
    \end{cases} \qquad \mathcal{E}_{sat} = \frac{v_{sat}}{\mu_n}
    $$
    """),
                    mo.md(r"Saturation begins where \(\partial I_D/\partial V_{DS} = 0\) for the nonsaturation expression:"),
                    mo.md(r"""
    $$
    V_{DS}(\mathrm{sat}) = \mathcal{E}_{sat}L\left[\sqrt{1 + \frac{2(V_{GS} - V_T)}{m\,\mathcal{E}_{sat}L}} - 1\right] \;\to\; \frac{V_{GS} - V_T}{m} \quad [\mathcal{E}_{sat}L \gg (V_{GS} - V_T)/m]
    $$
    """),
                    mo.md(r"""
    $$
    I_D(\mathrm{sat}) = WC_{ox}\left[V_{GS} - V_T - mV_{DS}(\mathrm{sat})\right]v_{sat} \;\to\; \begin{cases} \dfrac{W\mu_n C_{ox}}{2mL}(V_{GS} - V_T)^2, & \mathcal{E}_{sat}L \gg (V_{GS} - V_T)/m \\[8pt] WC_{ox}(V_{GS} - V_T)v_{sat}, & \mathcal{E}_{sat}L \ll (V_{GS} - V_T)/m \end{cases}
    $$
    """),
                    mo.md(
                        r"Source and drain resistance: the intrinsic voltages \(V_{GS,\mathrm{int}} = V_{GS} - I_Dr_s\) and \(V_{DS,\mathrm{int}} = V_{DS} - I_D(r_s + r_d)\) replace \(V_{GS}\) and \(V_{DS}\) above, and \(I_D\) is solved self-consistently. The body effect and the series resistance are applied as separate corrections: \(V_T\) is calculated from the selected \(V_{SB}\) and held fixed, so the small extra body bias \(I_Dr_s\) on the intrinsic source is neglected, as on the slides."
                    ),
                    mo.md(
                        r"With \(m = 1\), \(V_{SB} = 0\), \(\lambda = 0\), \(r_s = r_d = 0\) and velocity saturation off, this is the ideal MOSFET of the slides."
                    ),
                    mo.md(
                        r"**Model notes.** Velocity saturation uses the field-dependent mobility \(\mu = \mu_n/(1 + \mathcal{E}/\mathcal{E}_{sat})\), which integrates in closed form and has the same low- and high-field limits as the slide's \(\mu = \mu_{\mathrm{eff}}/[1 + (\mu_{\mathrm{eff}}\mathcal{E}/v_{sat})^2]^{1/2}\) (Eq. 11.18). It falls off faster at moderate fields, so it overstates the effect at intermediate \(L\): at \(L = 5\,\mu\mathrm{m}\) and \(V_{GS} = 3\,\mathrm{V}\) it lowers \(I_D(\mathrm{sat})\) by about 15%, against about 3% for the slide's law. "
                        r"Channel length modulation uses \(V_{DS} - V_{DS}(\mathrm{sat}) = \Delta V_{DS}\), so \(\Delta L = 0\) and the factor is 1 at pinch-off, and the curves stay continuous there. The output slope is \(\lambda I_D(\mathrm{sat})\), the same as for Neamen's empirical \(I'_D = (k'_n/2)(W/L)(V_{GS} - V_T)^2(1 + \lambda V_{DS})\) (Eq. 11.12), which also scales the current at pinch-off by \(1 + \lambda V_{DS}(\mathrm{sat})\)."
                    ),
                ]
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding

    Use the sliders, then open the answers.

    1. Set \(V_{SB} = 1\,\mathrm{V}\) with everything else ideal. By how much does \(V_T\) rise, and what happens to \(I_D(\mathrm{sat})\) at \(V_{GS} = 2\,\mathrm{V}\)?
    2. Return to \(V_{SB} = 0\) and set \(m\) to the estimate in the table. By what factor do \(V_{DS}(\mathrm{sat})\) and \(I_D(\mathrm{sat})\) change?
    3. Reset \(m = 1\), with \(V_{SB} = 0\), \(r_s = r_d = 0\) and \(\lambda = 0\). Turn on velocity saturation with \(v_{sat} = 10^7\,\mathrm{cm/s}\) and shorten \(L\) to \(0.2\,\mu\mathrm{m}\) (keeping \(W/L = 10\)). How does \(I_D(\mathrm{sat})\) now depend on \(V_{GS} - V_T\)?
    """)
    return


@app.cell
def _(
    COX,
    BETA,
    MU_N,
    VT0,
    W_OVER_L,
    ideal_current,
    intrinsic_current,
    m_estimate,
    mo,
    threshold,
):
    _VT1 = float(threshold(1.0))
    _i0 = float(ideal_current(2.0, 4.0))
    _i1 = 0.5 * BETA * (2.0 - _VT1) ** 2
    _m0 = float(m_estimate(0.0))
    _L = 0.2e-4
    _Ec = 1e7 / MU_N * _L
    _isat = [float(intrinsic_current(v, 10.0, VT0, 1.0, _Ec, 0.0)[0]) for v in (2.0, 3.0)]
    _W = W_OVER_L * _L
    _lim = [_W * COX * (v - VT0) * 1e7 for v in (2.0, 3.0)]
    mo.accordion(
        {
            "Answers": mo.md(
                rf"""
    1. \(\Delta V_T = \gamma[\sqrt{{2\phi_{{fp}} + 1}} - \sqrt{{2\phi_{{fp}}}}] = {_VT1 - VT0:.3f}\,\mathrm{{V}}\), so \(V_T = {_VT1:.3f}\,\mathrm{{V}}\), as on the body-effect slide. At \(V_{{GS}} = 2\,\mathrm{{V}}\), \(I_D(\mathrm{{sat}})\) falls from \({_i0 * 1e3:.2f}\) to \({_i1 * 1e3:.2f}\,\mathrm{{mA}}\), since \((V_{{GS}} - V_T)^2\) is smaller. Every curve drops, and its \(V_{{DS}}(\mathrm{{sat}})\) moves left by \(\Delta V_T\).
    2. With \(m = {_m0:.2f}\), both \(V_{{DS}}(\mathrm{{sat}}) = (V_{{GS}} - V_T)/m\) and \(I_D(\mathrm{{sat}}) = (W\mu_nC_{{ox}}/2mL)(V_{{GS}} - V_T)^2\) are divided by \(m\): about \({100 * (1 - 1 / _m0):.0f}\%\) lower. The curves still start with the same slope \((W\mu_nC_{{ox}}/L)(V_{{GS}} - V_T)\) at \(V_{{DS}} = 0\), because the \(m\) term is second order in \(V_{{DS}}\).
    3. Much closer to linearly. With \(\mathcal{{E}}_{{sat}}L = {_Ec:.2f}\,\mathrm{{V}}\), \(I_D(\mathrm{{sat}}) = {_isat[0] * 1e3:.2f}\) and \({_isat[1] * 1e3:.2f}\,\mathrm{{mA}}\) at \(V_{{GS}} = 2\) and \(3\,\mathrm{{V}}\): a ratio of \({_isat[1] / _isat[0]:.2f}\), against \({(3 - VT0) ** 2 / (2 - VT0) ** 2:.2f}\) for the square law and \({(3 - VT0) / (2 - VT0):.2f}\) for a current proportional to \(V_{{GS}} - V_T\). The fully saturated limit \(WC_{{ox}}(V_{{GS}} - V_T)v_{{sat}}\) (\({_lim[0] * 1e3:.2f}\) and \({_lim[1] * 1e3:.2f}\,\mathrm{{mA}}\)) is reached only when \(\mathcal{{E}}_{{sat}}L \ll V_{{GS}} - V_T\); try \(L = 0.1\,\mu\mathrm{{m}}\) and a lower \(v_{{sat}}\).
    """
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
