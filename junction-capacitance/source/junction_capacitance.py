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

    # Silicon at 300 K. Same constants as the pn-junction companion and the slides.
    KT_E = 0.0259
    NI_SI = 1.5e10
    E_CHARGE = 1.6e-19
    EPS_S = 11.7 * 8.85e-14
    VR_MAX = 10.0
    AREA_CM2 = 1.0e-4

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
        return rf"{sign}{man:.{digits}f}\times 10^{{{exp}}}"

    def style_ax(ax):
        ax.tick_params(direction="out", length=4, width=1.0)
        ax.grid(True, alpha=0.28, color=GRAY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def built_in_voltage(Na, Nd):
        """V_bi = (kT/e) ln(N_a N_d / n_i^2), Eq. (7.10), volts."""
        return float(KT_E * np.log(float(Na) * float(Nd) / NI_SI**2))

    def depletion_width(Na, Nd, Vbarrier):
        """W (cm) for a positive barrier V_bi + V_R, Eq. (7.34)."""
        Na, Nd = float(Na), float(Nd)
        return np.sqrt(2.0 * EPS_S * Vbarrier / E_CHARGE * (Na + Nd) / (Na * Nd))

    def junction_cap(Na, Nd, Vbarrier):
        """C' (F/cm^2) for a positive barrier V_bi + V_R, Eq. (7.42)."""
        Na, Nd = float(Na), float(Nd)
        return np.sqrt(E_CHARGE * EPS_S * Na * Nd / (2.0 * Vbarrier * (Na + Nd)))

    def inv_cap2_slope(Na, Nd):
        """d(1/C'^2)/dV_R in (F/cm^2)^-2 V^-1."""
        Na, Nd = float(Na), float(Nd)
        return 2.0 * (Na + Nd) / (E_CHARGE * EPS_S * Na * Nd)

    return (
        AREA_CM2,
        BLUE,
        GRAY,
        GREEN,
        NAVY,
        ORANGE,
        VR_MAX,
        built_in_voltage,
        depletion_width,
        inv_cap2_slope,
        junction_cap,
        plt,
        sci_tex,
        style_ax,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Junction Capacitance
    **ECE335 - companion to lectures**

    An abrupt silicon pn junction under reverse bias \(V_R\). This computes the junction capacitance per unit area \(C'\) and \((1/C')^2\) versus \(V_R\) from the depletion approximation. The derivation stays on the slides.

    Silicon at \(300\,\mathrm{K}\), with \(n_i = 1.5\times 10^{10}\,\mathrm{cm}^{-3}\), \(kT/e = 0.0259\,\mathrm{V}\), and \(\varepsilon_s = 11.7\varepsilon_0\).
    """)
    return


@app.cell
def _(VR_MAX, mo):
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
    VR_slider = mo.ui.slider(
        start=0.0,
        stop=VR_MAX,
        step=0.1,
        value=5.0,
        show_value=True,
        label=r"V_R (V)",
    )
    _controls = mo.vstack(
        [
            mo.md(r"## Interactive: Junction Capacitance"),
            mo.hstack(
                [Na_log_slider, Nd_log_slider, VR_slider],
                justify="start",
                gap=1.0,
                wrap=True,
            ),
        ]
    )
    _controls
    return Na_log_slider, Nd_log_slider, VR_slider


@app.cell
def _(
    AREA_CM2,
    BLUE,
    GRAY,
    GREEN,
    NAVY,
    Na_log_slider,
    Nd_log_slider,
    ORANGE,
    VR_MAX,
    VR_slider,
    built_in_voltage,
    depletion_width,
    inv_cap2_slope,
    junction_cap,
    mo,
    np,
    plt,
    sci_tex,
    style_ax,
):
    _Na = 10.0 ** float(Na_log_slider.value)
    _Nd = 10.0 ** float(Nd_log_slider.value)
    _VR = float(VR_slider.value)
    _Vbi = built_in_voltage(_Na, _Nd)

    _VR_arr = np.linspace(0.0, VR_MAX, 500)
    _C_arr = junction_cap(_Na, _Nd, _Vbi + _VR_arr)
    _C0 = float(junction_cap(_Na, _Nd, _Vbi))
    _C_op = float(junction_cap(_Na, _Nd, _Vbi + _VR))
    _W_op = float(depletion_width(_Na, _Nd, _Vbi + _VR))
    _slope = inv_cap2_slope(_Na, _Nd)

    # (1/C')^2 is linear in V_R, so the extension to the intercept at -V_bi is exact
    # within the depletion approximation. It is dashed because forward bias adds C_d.
    _VR_ext = np.linspace(-_Vbi, 0.0, 50)
    _inv2_arr = _slope * (_Vbi + _VR_arr)
    _inv2_ext = _slope * (_Vbi + _VR_ext)
    _inv2_op = _slope * (_Vbi + _VR)
    _inv2_exp = int(np.floor(np.log10(_slope * (_Vbi + VR_MAX))))
    _inv2_scale = 10.0**_inv2_exp

    _fig, (_ax_C, _ax_inv) = plt.subplots(1, 2, figsize=(12.4, 5.2), layout="constrained")

    _ax_C.plot(_VR_arr, _C_arr * 1.0e9, color=BLUE, linewidth=2.2)
    _ax_C.plot(
        0.0,
        _C0 * 1.0e9,
        "o",
        color=GRAY,
        markersize=8,
        clip_on=False,
        label=rf"$C'_0 = {_C0 * 1.0e9:.2f}$ nF/cm$^2$",
    )
    _ax_C.plot(
        _VR,
        _C_op * 1.0e9,
        "o",
        color=ORANGE,
        markersize=9,
        label=rf"$C' = {_C_op * 1.0e9:.2f}$ nF/cm$^2$",
    )
    _ax_C.set_xlim(0.0, VR_MAX)
    _ax_C.set_ylim(0.0, 1.08 * _C0 * 1.0e9)
    _ax_C.set_xlabel(r"$V_R$ (V)")
    _ax_C.set_ylabel(r"$C'$ (nF/cm$^2$)")
    _ax_C.set_title(r"(a)  $C'$ versus $V_R$")
    _ax_C.legend(fontsize=16, loc="upper right")
    style_ax(_ax_C)

    _ax_inv.plot(_VR_arr, _inv2_arr / _inv2_scale, color=GREEN, linewidth=2.2)
    _ax_inv.plot(
        _VR_ext, _inv2_ext / _inv2_scale, color=GREEN, linewidth=1.6, linestyle="--"
    )
    _ax_inv.plot(
        -_Vbi,
        0.0,
        "o",
        color=NAVY,
        markersize=8,
        clip_on=False,
        label=rf"$-V_{{bi}} = {-_Vbi:.3f}$ V",
    )
    _ax_inv.plot(_VR, _inv2_op / _inv2_scale, "o", color=ORANGE, markersize=9)
    _ax_inv.axhline(0.0, color=GRAY, linewidth=0.6)
    _ax_inv.axvline(0.0, color=GRAY, linewidth=0.6, linestyle=":")
    _ax_inv.set_xlim(-1.0, VR_MAX)
    _ax_inv.set_ylim(bottom=-0.03 * _slope * (_Vbi + VR_MAX) / _inv2_scale)
    _ax_inv.set_xlabel(r"$V_R$ (V)")
    _ax_inv.set_ylabel(
        rf"$(1/C')^2$ ($10^{{{_inv2_exp}}}$ cm$^4$/F$^2$)"
    )
    _ax_inv.set_title(r"(b)  $(1/C')^2$ versus $V_R$")
    _ax_inv.legend(fontsize=16, loc="upper left")
    style_ax(_ax_inv)

    _calc = mo.md(rf"""
    $$
    \begin{{aligned}}
    V_{{bi}} &= \frac{{kT}}{{e}}\ln\frac{{N_aN_d}}{{n_i^2}} = {_Vbi:.3f}\ \mathrm{{V}} \\
    W &= \left[\frac{{2\varepsilon_s(V_{{bi}}+V_R)}}{{e}}\,\frac{{N_a+N_d}}{{N_aN_d}}\right]^{{1/2}} = {_W_op * 1.0e4:.3f}\ \mu\mathrm{{m}} \\
    C' &= \left[\frac{{e\varepsilon_s N_aN_d}}{{2(V_{{bi}}+V_R)(N_a+N_d)}}\right]^{{1/2}} = \frac{{\varepsilon_s}}{{W}} = {sci_tex(_C_op)}\ \mathrm{{F/cm^2}} \\
    \frac{{d(1/C')^2}}{{dV_R}} &= \frac{{2(N_a+N_d)}}{{e\varepsilon_s N_aN_d}} = {sci_tex(_slope)}\ \mathrm{{(F/cm^2)^{{-2}}\,V^{{-1}}}}
    \end{{aligned}}
    $$
    """)
    _note = mo.md(
        rf"""
    At \(V_R = {_VR:.1f}\,\mathrm{{V}}\): \(N_a = {sci_tex(_Na)}\,\mathrm{{cm}}^{{-3}}\), \(N_d = {sci_tex(_Nd)}\,\mathrm{{cm}}^{{-3}}\). For a junction area \(A = 10^{{-4}}\,\mathrm{{cm}}^2\), \(C = C'A = {_C_op * AREA_CM2 * 1.0e12:.3f}\,\mathrm{{pF}}\). The dashed part of (b) is the straight line extended to its intercept at \(V_R = -V_{{bi}}\).
    """
    )
    mo.vstack([_fig, _calc, _note])
    return


@app.cell
def _(mo):
    _explain = mo.accordion(
        {
            "What the plots are doing": mo.md(
                r"""
    Both panels use the same \(C'\), Eq. (7.42). \(C'\) falls as \(1/\sqrt{V_{bi}+V_R}\) because \(W\) grows as \(\sqrt{V_{bi}+V_R}\), and \(C' = \varepsilon_s/W\).

    Squaring the inverse gives a straight line in \(V_R\). Its slope depends only on \(N_aN_d/(N_a+N_d)\), and it reaches zero at \(V_R = -V_{bi}\). When one side is much more heavily doped, the slope is set by the lighter side alone.

    The grey dot in (a) is the zero-bias value \(C'_0\). The orange dots mark the \(V_R\) slider in both panels.

    The line in (b) is dashed for \(V_R < 0\), which is forward bias. The depletion formula still gives a number there, but the diffusion capacitance \(C_d\) of the stored minority carriers is added and soon dominates. A C–V measurement is done in reverse bias and extrapolated to the intercept.
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

    Use the sliders and the numbers under the plot, then open the answers.

    1. At the default dopings, what is \(C'\) at \(V_R = 5\,\mathrm{V}\)? By what factor does \(C'\) fall from zero bias to \(V_R = 5\,\mathrm{V}\)?
    2. Keep \(N_d = 10^{15}\,\mathrm{cm}^{-3}\) and raise \(N_a\) from \(10^{16}\) to \(10^{18}\,\mathrm{cm}^{-3}\). Does the slope of \((1/C')^2\) change much? Why?
    3. Swap the two dopings, so \(N_a = 10^{15}\) and \(N_d = 10^{16}\,\mathrm{cm}^{-3}\). What changes on either plot?
    """)
    return


@app.cell
def _(built_in_voltage, inv_cap2_slope, junction_cap, mo, sci_tex):
    _Na = 1.0e16
    _Nd = 1.0e15
    _Vbi = built_in_voltage(_Na, _Nd)
    _C0 = float(junction_cap(_Na, _Nd, _Vbi))
    _C5 = float(junction_cap(_Na, _Nd, _Vbi + 5.0))
    _s16 = inv_cap2_slope(1.0e16, _Nd)
    _s18 = inv_cap2_slope(1.0e18, _Nd)
    mo.accordion(
        {
            "Answers": mo.md(
                rf"""
    1. \(C' = {sci_tex(_C5)}\,\mathrm{{F/cm^2}}\) at \(V_R = 5\,\mathrm{{V}}\), down from \(C'_0 = {sci_tex(_C0)}\,\mathrm{{F/cm^2}}\). The ratio is \(\sqrt{{(V_{{bi}}+5)/V_{{bi}}}} = {_C0 / _C5:.2f}\), because \(C' \propto 1/\sqrt{{V_{{bi}}+V_R}}\).
    2. Only a little. The slope goes from \({sci_tex(_s16)}\) to \({sci_tex(_s18)}\,\mathrm{{(F/cm^2)^{{-2}}\,V^{{-1}}}}\), a change of \({(1 - _s18 / _s16) * 100:.0f}\%\). \((N_a+N_d)/(N_aN_d) = 1/N_a + 1/N_d\) is already dominated by \(1/N_d\) at \(N_a = 10N_d\). The intercept moves a little too, because \(V_{{bi}}\) grows with \(\ln N_a\).
    3. Nothing on the plots. \(C'\), \(V_{{bi}}\), and the slope depend only on the product and sum of \(N_a\) and \(N_d\). Only the side that holds most of \(W\) changes. A C–V measurement cannot tell which side is which.
    """
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
