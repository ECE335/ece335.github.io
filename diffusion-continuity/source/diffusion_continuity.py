# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.5.3",
#     "matplotlib==3.11.2",
#     "plotly==7.0.0",
#     "scipy==1.15.1",
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
async def _():
    import sys as _sys

    if "pyodide" in _sys.modules:
        import micropip

        _ = await micropip.install("plotly")
        _ = await micropip.install("scipy")
    solver_ready = True
    return (solver_ready,)


@app.cell
def _(solver_ready):
    import plotly.graph_objects as go
    from scipy.integrate import solve_ivp

    _ = solver_ready
    return go, solve_ivp


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

    # kT/e at 300 K, same value as the other companions.
    KT_E = 0.0259
    T_MAX_NS = 10.0
    N_FRAMES = 41
    N_POINTS = 800
    SIGMA0_UM = 1.0
    DELTA_N0 = 1.0e10
    W_LP_MIN = 0.5
    W_LP_MAX = 5.0

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

    def plain_tex(x):
        x = float(x)
        if not np.isfinite(x) or x == 0.0:
            return "0"
        ax = abs(x)
        if ax >= 1.0 and abs(x - round(x)) < 1e-6 * max(ax, 1.0):
            return str(int(round(x)))
        if ax >= 0.01:
            return f"{x:.2f}"
        return sci_tex(x, 2)

    def style_ax(ax):
        ax.tick_params(direction="out", length=4, width=1.0)
        ax.grid(True, alpha=0.28, color=GRAY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def dp_over_dp0(x_over_W, W_over_L):
        """δp(x)/δp(0) for δp(W) = 0 and no applied field."""
        ratio = float(W_over_L)
        z = ratio * (1.0 - np.asarray(x_over_W, dtype=float))
        return np.sinh(z) / np.sinh(ratio)

    def jp_norm(x_over_W, W_over_L):
        """Jp Lp / (e Dp δp(0))."""
        ratio = float(W_over_L)
        z = ratio * (1.0 - np.asarray(x_over_W, dtype=float))
        return np.cosh(z) / np.sinh(ratio)

    return (
        BLUE,
        DELTA_N0,
        GRAY,
        GREEN,
        KT_E,
        N_FRAMES,
        N_POINTS,
        ORANGE,
        SIGMA0_UM,
        T_MAX_NS,
        W_LP_MAX,
        W_LP_MIN,
        dp_over_dp0,
        jp_norm,
        plain_tex,
        plt,
        style_ax,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Carrier Diffusion
    **ECE335 - companion to lectures**

    Two calculations for a homogeneous bar. The steady-state section is excess holes. The video computes the continuity equation for the excess electrons.

    Symbols follow the slides: \(e\) for the elementary charge, \(\mathcal E \) for the electric field, \(\delta p\) for the excess hole concentration, and \(\tau_{p0}\) for the minority-hole lifetime. The video computes the continuity equation for the excess electrons \(\delta n\).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Steady-state diffusion across a finite region

    n-type region of width \(W\), with \(\mathcal E = 0\) and no generation inside. Excess holes are injected at \(x = 0\) and held at \(\delta p(0)\). A contact at \(x = W\) removes these excess holes, so \(\delta p(W) = 0\).

    The shape depends only on \(W/L_p\), with \(L_p = \sqrt{D_p \tau_{p0}}\).
    """)
    return


@app.cell
def _(W_LP_MAX, W_LP_MIN, mo):
    W_Lp_slider = mo.ui.slider(
        start=W_LP_MIN,
        stop=W_LP_MAX,
        step=0.1,
        value=2.0,
        show_value=True,
        label=r"W / L_p",
    )
    _controls = mo.vstack(
        [
            mo.md(r"### Width compared with the diffusion length"),
            W_Lp_slider,
        ]
    )
    _controls
    return (W_Lp_slider,)


@app.cell
def _(
    BLUE,
    GRAY,
    ORANGE,
    W_LP_MIN,
    W_Lp_slider,
    dp_over_dp0,
    jp_norm,
    mo,
    np,
    plain_tex,
    plt,
    style_ax,
):
    _ratio = round(float(W_Lp_slider.value), 1)
    _x = np.linspace(0.0, 1.0, 400)
    _dp = dp_over_dp0(_x, _ratio)
    _jp = jp_norm(_x, _ratio)
    _exp_ref = np.exp(-_ratio * _x)
    _mid = float(dp_over_dp0(0.5, _ratio))
    _j_at_0 = float(jp_norm(0.0, _ratio))
    _j_at_W = float(jp_norm(1.0, _ratio))
    _j_top = float(np.cosh(W_LP_MIN) / np.sinh(W_LP_MIN))

    _fig, (_ax_p, _ax_j) = plt.subplots(
        1, 2, figsize=(10.6, 4.8), layout="constrained"
    )
    _ax_p.plot(_x, _dp, color=BLUE, lw=2.2, label=r"$\delta p(x)/\delta p(0)$")
    _ax_p.plot(
        _x,
        _exp_ref,
        color=GRAY,
        lw=1.6,
        ls="--",
        label=r"$\mathrm{e}^{-x/L_p}$",
    )
    _ax_p.plot(0.0, 1.0, "o", color=BLUE, ms=7)
    _ax_p.plot(1.0, 0.0, "o", color=ORANGE, ms=7)
    _ax_p.set_xlim(0.0, 1.0)
    _ax_p.set_ylim(0.0, 1.15)
    _ax_p.set_xlabel(r"Position $x/W$")
    _ax_p.set_ylabel(r"$\delta p(x)/\delta p(0)$")
    _ax_p.set_title(rf"$W/L_p = {plain_tex(_ratio)}$")
    _ax_p.legend(frameon=False, loc="upper right")
    style_ax(_ax_p)

    _ax_j.plot(_x, _jp, color=ORANGE, lw=2.2)
    _ax_j.set_xlim(0.0, 1.0)
    _ax_j.set_ylim(0.0, _j_top * 1.08)
    _ax_j.set_xlabel(r"Position $x/W$")
    _ax_j.set_ylabel(r"$J_p L_p / (e D_p \delta p(0))$")
    _ax_j.set_title(r"Hole diffusion current")
    style_ax(_ax_j)

    _calc = mo.vstack(
        [
            mo.md(r"### With this ratio"),
            mo.md(
                rf"""
    $$
    \frac{{\delta p(x)}}{{\delta p(0)}}
    = \frac{{\sinh[(W-x)/L_p]}}{{\sinh(W/L_p)}}
    $$
    """
            ),
            mo.md(
                rf"""
    $$
    \frac{{\delta p(W/2)}}{{\delta p(0)}} = {plain_tex(_mid)}
    $$
    """
            ),
            mo.md(
                rf"""
    $$
    \frac{{J_p(x) L_p}}{{e D_p \delta p(0)}}
    = \frac{{\cosh[(W-x)/L_p]}}{{\sinh(W/L_p)}}
    $$
    """
            ),

        ]
    )
    _explain = mo.accordion(
        {
            "What the plot is doing": mo.md(
                r"""
    \(L_p = \sqrt{D_p \tau_{p0}}\) is the effective distance a hole diffuses before it recombines. The dashed curve is the semi-infinite profile \(\delta p(0)\,e^{-x/L_p}\), which does not meet \(\delta p(W) = 0\).

    When \(W \gg L_p\), the solid curve is similar to the exponential and \(J_p(W)\) is nearly zero: almost every injected hole recombines before the far contact. When \(W \ll L_p\), \(\delta p(x)\) is nearly a straight line and \(J_p\) is large and almost constant: holes cross the region without recombining. Keep this result in mind -- we will encounter it again in the bipolar junction transistor.

    The same square root for electrons in p-type material is \(L_n = \sqrt{D_n \tau_{n0}}\).
    """
            )
        }
    )
    mo.vstack([_fig, _calc, _explain])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Spatio-temporal evolution of a carrier pulse

    A uniform field \(\mathcal{E}\) points along \(+x\). A narrow pulse of excess electrons starts at \(x = 0\). For \(t > 0\) there is no generation. Press **Play**. This computes the continuity equation for the excess electrons \(\delta n\). The axes stay fixed for the whole clip.

    \(D_n = (kT/e)\,\mu_n\) at \(300\,\mathrm{K}\). Electrons drift toward \(-\mathcal{E}\).
    """)
    return


@app.cell
def _(mo):
    mu_n_slider = mo.ui.slider(
        start=200,
        stop=1500,
        step=20,
        value=1350,
        show_value=True,
        label=r"μ_n (cm²/V·s)",
    )
    E_slider = mo.ui.slider(
        start=-40,
        stop=40,
        step=2,
        value=20,
        show_value=True,
        label=r"ℰ (V/cm)",
    )
    tau_n0_slider = mo.ui.slider(
        start=5,
        stop=100,
        step=5,
        value=20,
        show_value=True,
        label=r"τ_n0 (ns)",
    )
    _controls = mo.vstack(
        [
            mo.md(r"### Pulse parameters"),
            mo.hstack(
                [mu_n_slider, E_slider, tau_n0_slider],
                justify="start",
                gap=1.0,
                wrap=True,
            ),
        ]
    )
    _controls
    return E_slider, mu_n_slider, tau_n0_slider


@app.cell
def _(
    BLUE,
    DELTA_N0,
    E_slider,
    GREEN,
    KT_E,
    N_FRAMES,
    N_POINTS,
    SIGMA0_UM,
    T_MAX_NS,
    go,
    mo,
    mu_n_slider,
    np,
    plain_tex,
    solve_ivp,
    tau_n0_slider,
):
    _mu = float(mu_n_slider.value)
    _E = float(E_slider.value)
    _tau_ns = float(tau_n0_slider.value)
    _Dn = KT_E * _mu
    _tau_s = _tau_ns * 1.0e-9
    _t_max_s = T_MAX_NS * 1.0e-9
    # Electrons drift opposite the field.
    _v = -_mu * _E
    _sigma0_cm = SIGMA0_UM * 1.0e-4
    _sigma_final_cm = float(np.sqrt(_sigma0_cm**2 + 2.0 * _Dn * _t_max_s))
    _x_final_cm = _v * _t_max_s
    _pad_cm = 4.5 * _sigma_final_cm
    _x_lo_cm = min(-4.0 * _sigma0_cm, _x_final_cm) - _pad_cm
    _x_hi_cm = max(4.0 * _sigma0_cm, _x_final_cm) + _pad_cm
    _x_cm = np.linspace(_x_lo_cm, _x_hi_cm, N_POINTS)
    _dx_cm = float(_x_cm[1] - _x_cm[0])
    _x_um = _x_cm * 1.0e4
    _n_initial = DELTA_N0 * np.exp(-0.5 * (_x_cm / _sigma0_cm) ** 2)
    _times_ns = np.linspace(0.0, T_MAX_NS, N_FRAMES)
    _times_s = _times_ns * 1.0e-9

    def _continuity_rhs(_t, _n):
        _dndt = np.zeros_like(_n)
        _diffusion = _Dn * (_n[2:] - 2.0 * _n[1:-1] + _n[:-2]) / _dx_cm**2
        if _v >= 0.0:
            _drift = _v * (_n[1:-1] - _n[:-2]) / _dx_cm
        else:
            _drift = _v * (_n[2:] - _n[1:-1]) / _dx_cm
        _dndt[1:-1] = _diffusion - _drift - _n[1:-1] / _tau_s
        _diff_left = _Dn * (_n[2] - 2.0 * _n[1] + _n[0]) / _dx_cm**2
        _drift_left = _v * (_n[1] - _n[0]) / _dx_cm
        _dndt[0] = _diff_left - _drift_left - _n[0] / _tau_s
        _diff_right = _Dn * (_n[-1] - 2.0 * _n[-2] + _n[-3]) / _dx_cm**2
        _drift_right = _v * (_n[-1] - _n[-2]) / _dx_cm
        _dndt[-1] = _diff_right - _drift_right - _n[-1] / _tau_s
        return _dndt

    _sol = solve_ivp(
        _continuity_rhs,
        (0.0, _t_max_s),
        _n_initial,
        method="RK45",
        t_eval=_times_s,
        max_step=_t_max_s / 400.0,
    )
    _profiles = _sol.y
    _area0 = float(np.trapezoid(_n_initial, _x_cm))
    _y_top = float(max(DELTA_N0, np.max(_profiles))) * 1.12

    def _moments(_profile):
        _area = float(np.trapezoid(_profile, _x_cm))
        _peak = float(np.max(_profile))
        if _area > 1.0e-10 * _area0:
            _center_cm = float(np.trapezoid(_x_cm * _profile, _x_cm) / _area)
            _var = float(
                np.trapezoid((_x_cm - _center_cm) ** 2 * _profile, _x_cm) / _area
            )
            _sigma_cm = float(np.sqrt(max(_var, 0.0)))
        else:
            _center_cm = 0.0
            _sigma_cm = _sigma0_cm
        return _peak, _center_cm * 1.0e4, _sigma_cm * 1.0e4, _area

    def _readout(_t_ns, _profile):
        _peak, _center_um, _sigma_um, _area = _moments(_profile)
        return [
            dict(
                x=0.98,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                text=(
                    f"t = {_t_ns:.2f} ns<br>"
                    f"δn peak = {_peak:.2e} cm⁻³<br>"
                    f"center = {_center_um:.2f} μm<br>"
                    f"σ = {_sigma_um:.2f} μm<br>"
                    f"area = {_area:.2e} cm⁻²"
                ),
                align="right",
                xanchor="right",
                yanchor="top",
                font=dict(size=16),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#E5E7EB",
                borderwidth=1,
            ),
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                text=(
                    f"D<sub>n</sub> = {_Dn:.2f} cm²/s<br>"
                    f"v = −μ<sub>n</sub>ℰ = {_v * 1.0e-5:.3f} μm/ns"
                ),
                align="left",
                xanchor="left",
                yanchor="top",
                font=dict(size=16),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#E5E7EB",
                borderwidth=1,
            ),
        ]

    def _traces(_profile):
        _center_um = _moments(_profile)[1]
        return [
            go.Scatter(
                x=_x_um,
                y=_profile,
                mode="lines",
                fill="tozeroy",
                line=dict(color=BLUE, width=2.5),
                fillcolor="rgba(0,114,178,0.28)",
                hoverinfo="skip",
            ),
            go.Scatter(
                x=[_center_um, _center_um],
                y=[0.0, _y_top],
                mode="lines",
                line=dict(color=GREEN, width=2, dash="dot"),
                hoverinfo="skip",
            ),
        ]

    _frames = []
    for _i, _t_ns in enumerate(_times_ns):
        _frames.append(
            go.Frame(
                data=_traces(_profiles[:, _i]),
                name=str(_i),
                layout=go.Layout(annotations=_readout(_t_ns, _profiles[:, _i])),
            )
        )
    _pulse = go.Figure(
        data=_traces(_profiles[:, 0]),
        frames=_frames,
        layout=go.Layout(
            title=dict(
                text=(
                    "Continuity equation for excess electrons,  "
                    f"ℰ = {plain_tex(_E)} V/cm"
                ),
                font=dict(size=16, color="#0f4c81"),
            ),
            xaxis=dict(
                title=dict(text="Position x (μm)", font=dict(size=16)),
                range=[float(_x_um[0]), float(_x_um[-1])],
                tickfont=dict(size=16),
                showgrid=True,
                gridcolor="rgba(107,107,107,0.28)",
                zeroline=False,
            ),
            yaxis=dict(
                title=dict(text="δn (cm⁻³)", font=dict(size=16)),
                range=[0.0, _y_top],
                tickfont=dict(size=16),
                exponentformat="e",
                showgrid=True,
                gridcolor="rgba(107,107,107,0.28)",
                zeroline=False,
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            height=520,
            margin=dict(l=80, r=30, t=60, b=130),
            annotations=_readout(float(_times_ns[0]), _profiles[:, 0]),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.0,
                    y=-0.22,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=70, redraw=True),
                                    fromcurrent=True,
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    pad=dict(t=40),
                    x=0.12,
                    len=0.86,
                    currentvalue=dict(
                        prefix="t = ",
                        suffix=" ns",
                        font=dict(size=16),
                        visible=True,
                    ),
                    steps=[
                        dict(
                            args=[
                                [str(_i)],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                            label=f"{_times_ns[_i]:.2f}" if _i % 4 == 0 else "",
                            method="animate",
                        )
                        for _i in range(N_FRAMES)
                    ],
                )
            ],
        ),
    )

    _calc = mo.vstack(
        [
            mo.md(r"### What the clip is computing"),
            mo.md(
                r"""
    The video computes the continuity equation for the excess electrons:
    $$
    \frac{\partial \delta n}{\partial t} = D_n\frac{\partial^2 \delta n}{\partial x^2} + \mu_n\mathcal{E}\frac{\partial \delta n}{\partial x}- \frac{\delta n}{\tau_{n0}}
    $$
    """
            ),
            mo.md(
                rf"""
    $$
    D_n = (kT/e)\,\mu_n = {plain_tex(KT_E)}\times {plain_tex(_mu)} = {plain_tex(_Dn)}\ \mathrm{{cm}}^2/\mathrm{{s}}
    $$
    """
            ),
            mo.md(
                rf"""
    $$
    v = -\mu_n\mathcal{{E}} = {plain_tex(_v * 1.0e-5)}\ \mu\mathrm{{m}}/\mathrm{{ns}}
    $$
    """
            ),
        ]
    )
    _explain = mo.accordion(
        {
            "What the video is doing": mo.md(
                r"""
    Each frame is \(\delta n(x, t)\) from that continuity equation. The dotted line is the centroid of the computed profile.

    The \(\mu_n\mathcal{E}\) term carries the packet toward \(-\mathcal{E}\). The \(D_n\) term spreads it. The \(-\delta n/\tau_{n0}\) term removes electrons, so the area on the readout falls.

    Set \(\mathcal{E} = 0\) and play again. The centroid stays near \(x = 0\) while the packet spreads and decays. Raise \(\mathcal{E}\) and the packet walks toward \(-x\).

    The pulse starts narrow and centered at \(x = 0\). After that, the shape is the solution of the continuity equation. The calculation assumes low injection and a constant \(\mathcal{E}\), \(D_n\), and \(\tau_{n0}\).
    """
            )
        }
    )
    mo.vstack([_pulse, _calc, _explain])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding

    Use the sliders and the video, then open the answers.

    1. Compare \(W/L_p = 0.5\) and \(W/L_p = 5\). Where do the injected holes recombine, and what happens to \(J_p\) at \(x = W\)?
    2. Play the clip with \(\mathcal{E} > 0\). Which way does the \(\delta n\) peak move, and what sets that speed?
    3. Set \(\mathcal{E} = 0\), make \(\tau_{n0}\) long, and play again. Why does the peak still fall as \(t\) increases?
    4. Pause at one time and note the area. Change \(\mu_n\) or \(\mathcal{E}\) and return to that same time. What happens to the area?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers and suggested checks": mo.md(
                r"""
    1. At \(W/L_p = 5\) the profile is close to \(e^{-x/L_p}\) and \(J_p(W)\) is almost zero: holes recombine inside the region. At \(W/L_p = 0.5\) the profile is nearly a straight line and \(J_p\) is large and almost flat: holes reach \(x = W\) before they recombine. Read both panels at those two slider values.
    2. Toward \(-x\). The drift speed is \(v = -\mu_n\mathcal{E}\). The dotted line is the centroid of the profile from the continuity equation. Flip the sign of \(\mathcal{E}\) and the packet walks the other way.
    3. Diffusion. The \(D_n\) term spreads the same electrons over a wider packet, so the peak falls even when recombination is slow. A long \(\tau_{n0}\) only weakens the \(-\delta n/\tau_{n0}\) term.
    4. The area is \(\int \delta n\,dx\) of the computed profile. It falls as electrons recombine, close to the initial area times \(e^{-t/\tau_{n0}}\). At one fixed time, changing \(\mu_n\) or \(\mathcal{E}\) moves the packet and does not change that area.
    """
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
