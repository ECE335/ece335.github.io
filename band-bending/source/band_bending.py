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
    from matplotlib.patches import Rectangle

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

    # Lecture example: uniform Si bar, Neamen Ch. 5. Eg from Table 4.1.
    EG_EV = 1.12
    L_UM = 10.0
    VA_MIN = -1.0
    VA_MAX = 1.0

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
        """Decimal for slider-scale numbers; scientific notation below 0.01."""
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

    return (
        BLUE,
        EG_EV,
        GRAY,
        GREEN,
        L_UM,
        NAVY,
        ORANGE,
        Rectangle,
        VA_MAX,
        VA_MIN,
        plain_tex,
        plt,
        sci_tex,
        style_ax,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Band Bending
    **ECE335 - companion to lectures**

    A uniform silicon bar has voltage \(V_a\) applied at \(x=0\) and is grounded at \(x=L\).
    This activity computes \(\phi(x)\), \(E_c(x)\), \(E_v(x)\), and \(\mathcal{E}\) from that boundary condition.

    The default is the lecture example: \(V_a=0.7\,\mathrm{V}\), \(L=10\,\mu\mathrm{m}\).
    Move \(V_a\) to tilt the bands. This is not a pn-junction depletion diagram.
    The charge density inside the bar is taken to be zero, so the field is uniform.
    """)
    return


@app.cell
def _(VA_MAX, VA_MIN, mo):
    Va_slider = mo.ui.slider(
        start=VA_MIN,
        stop=VA_MAX,
        step=0.05,
        value=0.7,
        show_value=True,
        label=r"V_a (V) at x = 0",
    )
    _controls = mo.vstack(
        [
            mo.md(
                r"""
    ### Applied voltage

    Positive \(V_a\) puts the positive terminal at \(x=0\). The end at \(x=L\) stays at \(0\).
    """
            ),
            Va_slider,
        ]
    )
    _controls
    return (Va_slider,)


@app.cell
def _(
    BLUE,
    EG_EV,
    GRAY,
    GREEN,
    L_UM,
    NAVY,
    ORANGE,
    Rectangle,
    VA_MAX,
    VA_MIN,
    Va_slider,
    mo,
    np,
    plain_tex,
    plt,
    sci_tex,
    style_ax,
):
    _Va = round(float(Va_slider.value), 2)
    _L_um = float(L_UM)
    _L_cm = _L_um * 1.0e-4
    _x_um = np.linspace(0.0, _L_um, 400)

    # φ(0) = Va, φ(L) = 0. Energy in eV with φ in volts, so the factor e is 1.
    _phi = _Va * (1.0 - _x_um / _L_um)
    _E_field = _Va / _L_cm
    _C = EG_EV / 2.0
    _Ec = _C - _phi
    _Ev = _Ec - EG_EV
    _EFi = 0.5 * (_Ec + _Ev)
    _dEc = float(_Ec[-1] - _Ec[0])

    # Fixed limits over the whole slider, so the axes do not jump.
    _phi_lo, _phi_hi = VA_MIN - 0.15, VA_MAX + 0.15
    _Ec_hi = _C - VA_MIN
    _Ec_lo = _C - VA_MAX
    _E_top = _Ec_hi + 0.28
    _E_bot = _Ec_lo - EG_EV - 0.28
    _field_max = VA_MAX / _L_cm

    _fig = plt.figure(figsize=(8.6, 11.4), layout="constrained")
    _gs = _fig.add_gridspec(4, 1, height_ratios=[0.82, 1.0, 1.22, 1.0])
    _ax_setup = _fig.add_subplot(_gs[0, 0])
    _ax_phi = _fig.add_subplot(_gs[1, 0])
    _ax_band = _fig.add_subplot(_gs[2, 0], sharex=_ax_phi)
    _ax_field = _fig.add_subplot(_gs[3, 0], sharex=_ax_phi)

    # Panel (a) of the lecture example: battery across a uniform bar.
    # The long battery plate is the positive terminal. It follows the sign of Va.
    _ax_setup.set_axis_off()
    _ax_setup.add_patch(
        Rectangle(
            (0.0, 0.30),
            1.0,
            0.24,
            facecolor="#D6EAF8",
            edgecolor=NAVY,
            linewidth=1.6,
            transform=_ax_setup.transAxes,
            zorder=2,
        )
    )
    _ax_setup.text(
        0.5,
        0.42,
        "Si",
        transform=_ax_setup.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        color=NAVY,
        zorder=3,
    )
    _ax_setup.plot(
        [0.0, 0.0, 0.44],
        [0.54, 0.78, 0.78],
        color="black",
        lw=1.6,
        transform=_ax_setup.transAxes,
        clip_on=False,
        zorder=3,
    )
    _ax_setup.plot(
        [1.0, 1.0, 0.56],
        [0.54, 0.78, 0.78],
        color="black",
        lw=1.6,
        transform=_ax_setup.transAxes,
        clip_on=False,
        zorder=3,
    )
    _plus_on_left = _Va >= 0.0
    if _Va == 0.0:
        _plate_xs = (0.455, 0.545)
        _plate_ys = ((0.72, 0.84), (0.72, 0.84))
        _vlabel = r"$V_a=0$"
        _left_sign, _right_sign = "", ""
    else:
        _long_x = 0.455 if _plus_on_left else 0.545
        _short_x = 0.545 if _plus_on_left else 0.455
        _plate_xs = (_long_x, _short_x)
        _plate_ys = ((0.68, 0.88), (0.73, 0.83))
        _mag = plain_tex(abs(_Va))
        _vlabel = (
            rf"$+\,{_mag}\,\mathrm{{V}}\,-$"
            if _plus_on_left
            else rf"$-\,{_mag}\,\mathrm{{V}}\,+$"
        )
        _left_sign = "+" if _plus_on_left else r"$-$"
        _right_sign = r"$-$" if _plus_on_left else "+"
    for _px, (_y0, _y1) in zip(_plate_xs, _plate_ys):
        _ax_setup.plot(
            [_px, _px],
            [_y0, _y1],
            color="black",
            lw=2.6,
            transform=_ax_setup.transAxes,
            clip_on=False,
            zorder=4,
        )
    _ax_setup.text(
        0.5,
        0.96,
        _vlabel,
        transform=_ax_setup.transAxes,
        ha="center",
        va="top",
        fontsize=16,
        color="black",
    )
    _ax_setup.text(
        0.045,
        0.62,
        _left_sign,
        transform=_ax_setup.transAxes,
        ha="center",
        va="bottom",
        fontsize=16,
        color="black",
        zorder=4,
    )
    _ax_setup.text(
        0.955,
        0.62,
        _right_sign,
        transform=_ax_setup.transAxes,
        ha="center",
        va="bottom",
        fontsize=16,
        color="black",
        zorder=4,
    )
    _ax_setup.text(
        0.0,
        0.24,
        r"$0$",
        transform=_ax_setup.transAxes,
        ha="center",
        va="top",
        fontsize=16,
    )
    _ax_setup.text(
        1.0,
        0.24,
        r"$L$",
        transform=_ax_setup.transAxes,
        ha="center",
        va="top",
        fontsize=16,
    )
    if _E_field > 0.0:
        _ax_setup.annotate(
            "",
            xy=(0.72, 0.12),
            xytext=(0.28, 0.12),
            xycoords=_ax_setup.transAxes,
            textcoords=_ax_setup.transAxes,
            arrowprops=dict(
                arrowstyle="-|>", color=ORANGE, lw=2.0, mutation_scale=16
            ),
        )
    elif _E_field < 0.0:
        _ax_setup.annotate(
            "",
            xy=(0.28, 0.12),
            xytext=(0.72, 0.12),
            xycoords=_ax_setup.transAxes,
            textcoords=_ax_setup.transAxes,
            arrowprops=dict(
                arrowstyle="-|>", color=ORANGE, lw=2.0, mutation_scale=16
            ),
        )
    _efield_label = (
        r"$\mathcal{E}$" if _Va != 0.0 else r"$\mathcal{E}=0$"
    )
    _ax_setup.text(
        0.50,
        0.20,
        _efield_label,
        transform=_ax_setup.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        color=ORANGE,
    )

    _ax_phi.plot(_x_um, _phi, color=NAVY, linewidth=2.2)
    _ax_phi.axhline(0.0, color=GRAY, linestyle="--", linewidth=1.0)
    _ax_phi.set_ylabel(r"$\phi$ (V)")
    _ax_phi.set_title(
        rf"$\phi(0)=V_a={plain_tex(_Va)}\ \mathrm{{V}}$"
    )
    _ax_phi.set_xlim(0.0, _L_um)
    _ax_phi.set_ylim(_phi_lo, _phi_hi)
    _ax_phi.text(
        0.98,
        0.95,
        r"$\phi(x)=V_a\left(1-x/L\right)$",
        transform=_ax_phi.transAxes,
        fontsize=16,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="#F7F7F7", edgecolor="#E5E7EB"),
    )
    style_ax(_ax_phi)

    _ax_band.plot(_x_um, _Ec, color=BLUE, linewidth=2.2, label=r"$E_c$")
    _ax_band.plot(_x_um, _Ev, color=ORANGE, linewidth=2.2, label=r"$E_v$")
    _ax_band.plot(
        _x_um,
        _EFi,
        color=GRAY,
        linestyle=":",
        linewidth=2.0,
        label=r"$E_{Fi}$",
    )
    _ax_band.fill_between(_x_um, _Ec, _E_top, color=BLUE, alpha=0.12, linewidth=0)
    _ax_band.fill_between(_x_um, _E_bot, _Ev, color=ORANGE, alpha=0.12, linewidth=0)
    _ax_band.set_ylabel("Energy (eV)")
    _ax_band.set_title(
        rf"$E_g={plain_tex(EG_EV)}\ \mathrm{{eV}},\ "
        rf"E_c(L)-E_c(0)={plain_tex(_dEc)}\ \mathrm{{eV}}$"
    )
    _ax_band.set_ylim(_E_bot, _E_top)
    _ax_band.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )
    style_ax(_ax_band)

    _ax_field.plot(_x_um, np.full_like(_x_um, _E_field), color=GREEN, linewidth=2.2)
    _ax_field.axhline(0.0, color=GRAY, linestyle="--", linewidth=1.0)
    _ax_field.set_xlabel(r"Position $x$ ($\mu$m)")
    _ax_field.set_ylabel(r"$\mathcal{E}$ (V/cm)")
    _ax_field.set_title(
        rf"$\mathcal{{E}}={plain_tex(_E_field)}\ \mathrm{{V/cm}}$"
    )
    _ax_field.set_ylim(-_field_max - 120.0, _field_max + 120.0)
    # Keep the formula off the field line: the line sits in the top half when Va > 0.
    _field_box_y = 0.06 if _E_field > 0.0 else 0.95
    _field_box_va = "bottom" if _E_field > 0.0 else "top"
    _ax_field.text(
        0.98,
        _field_box_y,
        r"$\mathcal{E}=-d\phi/dx=V_a/L$",
        transform=_ax_field.transAxes,
        fontsize=16,
        va=_field_box_va,
        ha="right",
        bbox=dict(boxstyle="round", facecolor="#F7F7F7", edgecolor="#E5E7EB"),
    )
    style_ax(_ax_field)

    if _Va > 0:
        _drift = mo.md(
            r"""
    Electrons roll **downhill** in \(E_c\), toward \(x=0\) and the positive terminal.
    Holes float **uphill** in \(E_v\), toward \(x=L\).
    Both give conventional current along \(+\hat{x}\), the direction of \(\mathcal{E}\).
    """
        )
    elif _Va < 0:
        _drift = mo.md(
            r"""
    Electrons roll **downhill** in \(E_c\), toward \(x=L\), where \(\phi\) is higher.
    Holes float **uphill** in \(E_v\), toward \(x=0\).
    Both give conventional current along \(-\hat{x}\), the direction of \(\mathcal{E}\).
    """
        )
    else:
        _drift = mo.md(
            r"""
    The bands are flat. \(\mathcal{E}=0\), so this field drives no drift.
    """
        )

    _calc = mo.vstack(
        [
            mo.md(r"### With these numbers"),
            mo.md(
                rf"""
    $$
    \phi(x)=V_a\left(1-\frac{{x}}{{L}}\right)
    $$
    """
            ),
            mo.md(
                rf"""
    $$
    \begin{{aligned}}
    \phi(0) &= {plain_tex(_Va)}\ \mathrm{{V}} \\
    \phi(L) &= 0
    \end{{aligned}}
    $$
    """
            ),
            mo.md(
                rf"""
    $$
    \mathcal{{E}}=-\frac{{d\phi}}{{dx}}=\frac{{V_a}}{{L}}
    =\frac{{{plain_tex(_Va)}\ \mathrm{{V}}}}{{{sci_tex(_L_cm)}\ \mathrm{{cm}}}}
    ={plain_tex(_E_field)}\ \mathrm{{V/cm}}
    $$
    """
            ),
            mo.md(
                rf"""
    $$
    \begin{{aligned}}
    E_c(x) - E_{{ref}}&= -e\phi(x) \\
    \end{{aligned}}
    $$
    """
            ),
            mo.md(
                rf"""
    $$
    E_c(L)-E_c(0)={plain_tex(_dEc)}\ \mathrm{{eV}}
    $$
    """
            ),
            mo.md(
                rf"""
    $$
    E_g=E_c-E_v={plain_tex(EG_EV)}\ \mathrm{{eV}}
    $$
    """
            ),
            mo.md(
                r"""
    The rise of \(E_c\) from \(x=0\) to \(x=L\) is \(eV_a\). It is not the bandgap.
    \(E_g\) is the vertical spacing between \(E_c\) and \(E_v\), and it is the same at every \(x\).
    \(E_{Fi}\) is midgap. It bends with the bands. It is not the Fermi level \(E_F\). 
    """
            ),
            _drift,
            mo.md(
                r"""
    The Poisson relation relates the electric field to the charge density \( \rho \):
                """
            ),
            mo.md(
                r"""
    $$
    \frac{d\mathcal{E}}{dx}=\frac{\rho}{\epsilon_s}=0
    $$
    """
            ),
        ]
    )
    _explain = mo.accordion(
        {
            "What the plot is doing": mo.md(
                r"""
    \(\phi\) falls linearly from \(V_a\) at \(x=0\) to \(0\) at \(x=L\).
    An electron's potential energy is \(-e\phi\), so \(E_c\) tilts opposite to \(\phi\).
    \(E_v\) is \(E_c-E_g\), and \(E_{Fi}\approx (E_c+E_v)/2\). All three have the same slope.

    $$
    \mathcal{E}=-\frac{d\phi}{dx}=\frac{1}{e}\frac{dE_c}{dx}
    =\frac{1}{e}\frac{dE_v}{dx}=\frac{1}{e}\frac{dE_{Fi}}{dx}.
    $$

    Bands that rise along \(+x\) mean \(\mathcal{E}\) points along \(+x\).
    The constant E_{{ref}} sets the zero of energy so that \(E_{Fi}(L)=0\). It does not change \(\mathcal{E}\) or \(E_g\).

    Inside this bar \(\rho=0\), so \(\mathcal{E}\) cannot vary with \(x\).

    With current flowing, \(E_F\) is not drawn. Quasi-Fermi levels will be introduced later in the course.
    """
            )
        }
    )
    mo.vstack([_fig, _calc, _explain])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding

    Use the slider, then open the answers.

    1. At \(V_a=0.7\,\mathrm{V}\), which end has the higher \(E_c\), and by how many eV?
    2. \(\phi\) falls along \(+x\). Why is \(\mathcal{E}\) nevertheless positive?
    3. The plot draws \(E_{Fi}\) and does not draw \(E_F\). What is the difference here?
    4. Flip \(V_a\) from \(+0.7\,\mathrm{V}\) to \(-0.7\,\mathrm{V}\). What happens to the sign of \(\mathcal{E}\), the direction electrons drift, and \(E_c(L)-E_c(0)\)?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers and suggested checks": mo.md(
                r"""
    1. \(E_c\) is higher at \(x=L\), by \(0.70\,\mathrm{eV}\). That difference is \(eV_a\), read from the band panel. \(E_g\) stays \(1.12\,\mathrm{eV}\).
    2. \(\mathcal{E}=-d\phi/dx\). From \(0.70\,\mathrm{V}\) down to \(0\) over \(10^{-3}\,\mathrm{cm}\), \(d\phi/dx=-700\,\mathrm{V/cm}\), so \(\mathcal{E}=+700\,\mathrm{V/cm}\), along \(+x\).
    3. \(E_{Fi}\) is the intrinsic level, halfway between \(E_c\) and \(E_v\). It tracks the bands. \(E_F\) is defined in thermal equilibrium. Current is flowing, so \(E_F\) is left off the diagram.
    4. \(\mathcal{E}\) becomes \(-700\,\mathrm{V/cm}\). Electrons still roll downhill, now toward \(x=L\). \(E_c(L)-E_c(0)\) becomes \(-0.70\,\mathrm{eV}\): the bands fall along \(+x\). \(E_g\) does not change.
    """
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
