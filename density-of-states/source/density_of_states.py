# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.5.3",
#     "matplotlib==3.11.2",
#     "plotly==7.0.0",
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
    return


@app.cell
def _():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots


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

    H = 6.62607015e-34
    M_E = 9.1093837015e-31
    Q = 1.602176634e-19
    K_EV = 8.617333262e-5
    MN_SI = 1.08
    MP_SI = 0.56

    BLUE = "#0072B2"
    ORANGE = "#D55E00"
    GREEN = "#009E73"
    GOLD = "#E69F00"
    NAVY = "#0f4c81"
    GRAY = "#6B6B6B"

    def style_ax(ax):
        ax.tick_params(direction="out", length=4, width=1.0)
        ax.grid(True, alpha=0.28, color=GRAY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def g_prefactor(m_over_m0):
        """4π(2m*)^{3/2}/h³ in SI (states J⁻¹ m⁻³ / sqrt(J))."""
        m = m_over_m0 * M_E
        return 4.0 * np.pi * (2.0 * m) ** 1.5 / H**3

    def g_c_cm3_eV(E_eV, Ec_eV, mn_over_m0):
        dE = np.asarray(E_eV, dtype=float) - Ec_eV
        out = np.zeros_like(dE, dtype=float)
        mask = dE > 0.0
        pre = g_prefactor(mn_over_m0)
        out[mask] = pre * np.sqrt(dE[mask] * Q) * Q / 1e6
        return out

    def g_v_cm3_eV(E_eV, Ev_eV, mp_over_m0):
        dE = Ev_eV - np.asarray(E_eV, dtype=float)
        out = np.zeros_like(dE, dtype=float)
        mask = dE > 0.0
        pre = g_prefactor(mp_over_m0)
        out[mask] = pre * np.sqrt(dE[mask] * Q) * Q / 1e6
        return out

    def n_states_band_edge(m_over_m0, dE_eV):
        """∫_0^{ΔE} g(E) dE, cm⁻³ (Neamen Example 3.3)."""
        pre = g_prefactor(m_over_m0)
        return pre * (2.0 / 3.0) * (dE_eV * Q) ** 1.5 / 1e6

    def Ncv_cm3(m_over_m0, T):
        """Effective density of states N_c or N_v, cm⁻³."""
        if T <= 0.0:
            return 0.0
        m = m_over_m0 * M_E
        kT_J = K_EV * T * Q
        return 2.0 * ((2.0 * np.pi * m * kT_J) / H**2) ** 1.5 / 1e6

    def fermi_dirac(E, EF, T):
        E = np.asarray(E, dtype=float)
        if T <= 0.0:
            return np.where(E < EF, 1.0, np.where(E > EF, 0.0, 0.5))
        x = (E - EF) / (K_EV * T)
        out = np.empty_like(E, dtype=float)
        hi = x > 40.0
        lo = x < -40.0
        mid = ~hi & ~lo
        out[hi] = 0.0
        out[lo] = 1.0
        out[mid] = 1.0 / (1.0 + np.exp(x[mid]))
        return out

    return (
        BLUE,
        GOLD,
        GRAY,
        GREEN,
        K_EV,
        MN_SI,
        MP_SI,
        NAVY,
        Ncv_cm3,
        ORANGE,
        fermi_dirac,
        g_c_cm3_eV,
        g_v_cm3_eV,
        n_states_band_edge,
        plt,
        style_ax,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Density of States and Fermi–Dirac Statistics
    **ECE335 - companion to lectures**

    Use these activities alongside lectures and Neamen §§3.4–3.5.
    Before moving a slider, predict what will change.

    An allowed energy is not the same thing as an occupied energy.
    The **density of states** $g(E)$ counts available quantum states per
    volume and per energy. The **Fermi–Dirac function** $f(E)$
    is the probability that an existing state is occupied. Their product is
    the electron distribution (lecture: $dn$):

    $$
    dn = g(E)\,f(E)\,dE.
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Density of states

    The **density of states** $g(E)$ (Neamen) tells us how many quantum states
    are available per unit energy per unit volume:

    $$
    g(E)=\frac{\text{number of available states}}{\text{unit volume}\times\text{unit energy}}
    \quad[\mathrm{eV}^{-1}\,\mathrm{cm}^{-3}].
    $$

    Then $g(E)\,dE$ is the number of states per unit volume between $E$ and
    $E+dE$. The volume in the denominator is there because a larger crystal
    has more atoms and therefore more states.

    ### How to think about this?

    Consider a **single atom** with states $1s$, $2s$, $2p$, $3s$, $3p$,
    $3d$, $\ldots$ which increase in energy. As energy increases, there are
    more orbitals available within a given window $\Delta E$. In a crystal
    the same idea applies, and $g(E)$ generally increases as one moves away
    from a band edge.

    **Predict:** Move $\Delta E$ from the $1s$ level up toward $3d$ and $4p$.
    Does the number of states in the window stay the same?
    """)
    return


@app.cell
def _(mo):
    E_center = mo.ui.slider(
        start=1.0,
        stop=3.0,
        step=0.1,
        value=1.0,
        show_value=True,
        label="Energy window center",
    )
    dos_atom_controls = mo.vstack(
        [
            mo.md("### Adjust the energy window to count states of a single atom:"),
            E_center,
        ]
    )
    dos_atom_controls
    return (E_center,)


@app.cell
def _(BLUE, E_center, GOLD, GREEN, NAVY, ORANGE, mo, plt):
    _levels = [
        (1.00, "1s", NAVY, 1),
        (1.50, "2s", BLUE, 1),
        (1.75, "2p", BLUE, 3),
        (2.25, "3s", GREEN, 1),
        (2.50, "3p", GREEN, 3),
        (2.75, "4s", ORANGE, 1),
        (2.90, "3d", GREEN, 5),
        (3.10, "4p", ORANGE, 3),
    ]
    _dE = 0.35
    _center = float(E_center.value)
    _lo = _center - 0.5 * _dE
    _hi = _center + 0.5 * _dE

    _fig, _ax = plt.subplots(figsize=(8.0, 4.2))
    _ax.annotate(
        "",
        xy=(0.0, 3.5),
        xytext=(0.0, 0.5),
        arrowprops=dict(arrowstyle="->", lw=2.0, color="black"),
    )
    _ax.text(-0.32, 3.05, r"$E$", fontsize=16)
    for _idot in range(3):
        _ax.plot(0.45, 3.28 + _idot * 0.07, "o", color="black", ms=4)

    _x0 = 0.30
    _lw = 0.30
    _n_in = 0
    _in_levels = []
    for _energy, _label, _color, _deg in _levels:
        for _j in range(_deg):
            _off = (_j - (_deg - 1) / 2.0) * 0.08
            _ax.hlines(_energy, _x0 + _off, _x0 + _off + _lw, colors=_color, lw=3.0)
        _ax.text(
            _x0 + _lw + 0.18,
            _energy,
            _label,
            va="center",
            color=_color,
            fontsize=16,
        )
        if _lo <= _energy <= _hi:
            _n_in += 2 * _deg
            _in_levels.append(f"{_label} ({_deg}×2 = {2 * _deg} states)")

    _ax.fill_between(
        [-0.10, 1.20],
        _lo,
        _hi,
        alpha=0.28,
        color=GOLD,
        edgecolor=ORANGE,
        linewidth=2.0,
        zorder=0,
    )
    _ax.annotate(
        "",
        xy=(1.00, _hi),
        xytext=(1.00, _lo),
        arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2.0),
    )
    _ax.text(1.10, _center, r"$\Delta E$", color=ORANGE, va="center", fontsize=16)

    _info_lines = [f"States in window: {_n_in}"]
    if _in_levels:
        _info_lines.extend(_in_levels)
    else:
        _info_lines.append("(no levels in window)")
    _ax.text(
        1.50,
        2.50,
        "\n".join(_info_lines),
        fontsize=14,
        va="center",
        bbox=dict(boxstyle="round", facecolor="#FFF6D5", edgecolor=GOLD, alpha=0.95),
    )
    _ax.set_xlim(-0.50, 3.00)
    _ax.set_ylim(0.50, 3.50)
    _ax.set_aspect("equal")
    _ax.axis("off")
    _ax.set_title(r"DoS of an atom: counting states in $\Delta E$", fontsize=16, pad=4)

    mo.vstack(
        [
            _fig,
            mo.accordion(
                {
                    "Explanation": mo.md(
                        r"""
    Each tick is one spatial orbital. Degenerate orbitals ($2p$, $3p$, $3d$)
    are drawn as several ticks at the same energy. The count includes a
    factor of two for spin. $g(E)\,dE$ is this kind of count, per unit
    volume, in a window $dE$. A crystal has a near-continuum of levels, so
    the count becomes a smooth function of $E$ (Neamen §3.4).
    """
                    )
                }
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Density of states in $k$-space

    **Predict:** Why does the count use only the **positive** octant of the
    $k$-sphere?

    For the infinite cube, $k_i=n_i\pi/a$ with $n_i>0$ (lecture: $\pi/L$), so
    allowed $\mathbf{k}$ lie in $k_x,k_y,k_z>0$. One spatial state occupies a
    $k$-space cell $(\pi/a)^3$. A spherical shell of radius $k$ and thickness
    $dk$ has volume $4\pi k^2\,dk$. Take one-eighth of that shell, double for
    spin, and divide by the cell volume (Neamen Eqs. 3.62–3.64):

    $$
    dN=2\cdot\frac18\cdot\frac{4\pi k^2\,dk}{(\pi/a)^3}
    =\frac{a^3 k^2}{\pi^2}\,dk.
    $$

    Per unit volume, $dN/V=k^2\,dk/\pi^2$. Periodic boundaries with spacing
    $2\pi/a$ give the same bulk $g(E)$ if used consistently; do not mix the two
    conventions. Near a parabolic edge, $E=\hbar^2k^2/(2m)$ converts $k\to E$.
    """)
    return


@app.cell
def _(mo):
    k_shell = mo.ui.slider(
        start=1.5,
        stop=4.5,
        value=2.5,
        step=0.1,
        show_value=True,
        label="shell radius k (units of π/a)",
    )
    k_controls = mo.vstack([k_shell])
    k_controls
    return (k_shell,)


@app.cell
def _(BLUE, ORANGE, go, k_shell, mo, np):
    _n_max = 5
    _ns = np.arange(1, _n_max + 1)
    _NX, _NY, _NZ = np.meshgrid(_ns, _ns, _ns)
    _k0 = float(k_shell.value)
    _dk = 0.45
    _r = np.sqrt(_NX**2 + _NY**2 + _NZ**2)
    _in_shell = (_r >= _k0) & (_r <= _k0 + _dk)

    _figk = go.Figure()
    _figk.add_trace(
        go.Scatter3d(
            x=_NX.ravel(),
            y=_NY.ravel(),
            z=_NZ.ravel(),
            mode="markers",
            marker=dict(size=3.2, color=BLUE, opacity=0.55),
            name="allowed k (n>0)",
            hovertemplate="n<sub>x</sub>=%{x}<br>n<sub>y</sub>=%{y}<br>n<sub>z</sub>=%{z}<extra></extra>",
        )
    )
    _figk.add_trace(
        go.Scatter3d(
            x=_NX[_in_shell],
            y=_NY[_in_shell],
            z=_NZ[_in_shell],
            mode="markers",
            marker=dict(size=6, color=ORANGE),
            name="in shell k → k+dk",
        )
    )
    _u = np.linspace(0.0, 0.5 * np.pi, 24)
    _v = np.linspace(0.0, 0.5 * np.pi, 18)
    _uu, _vv = np.meshgrid(_u, _v)
    for _rad, _col, _op in ((_k0, BLUE, 0.18), (_k0 + _dk, ORANGE, 0.12)):
        _figk.add_trace(
            go.Surface(
                x=_rad * np.sin(_vv) * np.cos(_uu),
                y=_rad * np.sin(_vv) * np.sin(_uu),
                z=_rad * np.cos(_vv),
                opacity=_op,
                showscale=False,
                colorscale=[[0, _col], [1, _col]],
                hoverinfo="skip",
            )
        )
    _figk.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.02, y=0.98, font=dict(size=14)),
        scene=dict(
            xaxis_title="kₓ (π/a)",
            yaxis_title="kᵧ (π/a)",
            zaxis_title="k_z (π/a)",
            xaxis=dict(range=[0, _n_max + 0.5]),
            yaxis=dict(range=[0, _n_max + 0.5]),
            zaxis=dict(range=[0, _n_max + 0.5]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.55, y=1.45, z=1.15)),
        ),
        title=dict(
            text=f"Positive octant: {int(_in_shell.sum())} orbitals in the shell",
            font=dict(size=16),
        ),
    )
    _n_shell = int(_in_shell.sum())
    _info = mo.md(
        rf"""
    Shell ${ _k0:.2f}\le k a/\pi \le {_k0+_dk:.2f}$.
    Orbitals in the shell: {_n_shell}. Electron states: ${2*_n_shell}$ (spin).

    Drag to rotate. Only $n_i\ge 1$ appear: negative $n_i$ repeat the same
    infinite-well $\psi$ up to a sign.
    """
    )
    mo.vstack(
        [
            _figk,
            _info,
            mo.accordion(
                {
                    "Explanation": mo.md(
                        r"""
    Negative $n_i$ repeats the same standing wave up to a sign, so the
    derivation uses the positive octant and $\Delta k=\pi/a$ (lecture: $\pi/L$).
    After including spin,

    $$
    g(E)=\frac{4\pi(2m)^{3/2}}{h^3}\sqrt{E}
    =\frac{1}{2\pi^2}\left(\frac{2m}{\hbar^2}\right)^{3/2}\sqrt{E}
    $$

    (Neamen Eq. 3.69). That is the free-electron result used at the band edges.
    """
                    )
                }
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## $g_c(E)$ and $g_v(E)$ in a semiconductor

    **Predict:** If $m_n^*$ increases, does $g_c(E)$ at a fixed $E>E_c$ rise or
    fall? Why is $g(E)=0$ for $E_v<E<E_c$?

    Replace $m$ by the density-of-states effective mass and shift the zero of
    kinetic energy to the band edge (Neamen Eqs. 3.72 and 3.75):

    $$
    g_c(E)=\frac{4\pi(2m_n^*)^{3/2}}{h^3}\sqrt{E-E_c},\qquad E\ge E_c,
    $$

    $$
    g_v(E)=\frac{4\pi(2m_p^*)^{3/2}}{h^3}\sqrt{E_v-E},\qquad E\le E_v.
    $$

    Silicon density-of-states masses (Neamen Example 3.4): $m_n^*=1.08m_0$,
    $m_p^*=0.56m_0$. Integrating $g$ over an energy window counts **available**
    states (Neamen Example 3.3).
    """)
    return


@app.cell
def _(mo):
    mn_s = mo.ui.slider(
        start=0.05,
        stop=1.20,
        value=1.08,
        step=0.01,
        show_value=True,
        label="mₙ*/m₀",
    )
    mp_s = mo.ui.slider(
        start=0.05,
        stop=1.20,
        value=0.56,
        step=0.01,
        show_value=True,
        label="mₚ*/m₀",
    )
    dE_s = mo.ui.slider(
        start=0.20,
        stop=2.00,
        value=1.00,
        step=0.05,
        show_value=True,
        label="ΔE from the band edge (eV)",
    )
    dos_controls = mo.hstack([mn_s, mp_s, dE_s], justify="start", gap=1.2)
    dos_controls
    return dE_s, mn_s, mp_s


@app.cell
def _(
    BLUE,
    GOLD,
    GREEN,
    NAVY,
    ORANGE,
    dE_s,
    g_c_cm3_eV,
    g_v_cm3_eV,
    mn_s,
    mo,
    mp_s,
    n_states_band_edge,
    np,
    plt,
    style_ax,
):
    _Ec, _Ev = 1.12, 0.0
    _mn = float(mn_s.value)
    _mp = float(mp_s.value)
    _dE = float(dE_s.value)
    _E_cb = max(_dE, 0.50)
    _E_vb = float(np.clip(_E_cb * (_mn / _mp) ** 3, 3.0, 2.0))
    _E = np.linspace(_Ev - _E_vb, _Ec + _E_cb + 0.05, 1000)
    _gc = g_c_cm3_eV(_E, _Ec, _mn)
    _gv = g_v_cm3_eV(_E, _Ev, _mp)
    _Nc_win = n_states_band_edge(_mn, _dE)
    _Nv_win = n_states_band_edge(_mp, _dE)
    _N_ex33 = n_states_band_edge(1.0, 1.0)

    _fig, _ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    _ax.plot(_E, _gv / 1e21, color=ORANGE, lw=2.6, label=r"$g_v(E)$")
    _ax.plot(_E, _gc / 1e21, color=BLUE, lw=2.6, label=r"$g_c(E)$")
    _ax.axvline(_Ev, color=GOLD, ls="--", lw=1.3)
    _ax.axvline(_Ec, color=GREEN, ls="--", lw=1.3)
    _ax.set_xlabel(r"$E$ (eV)")
    _ax.set_ylabel(r"$g(E)$  ($10^{21}$ cm$^{-3}$ eV$^{-1}$)")
    _gmax = max(float(np.max(_gc)), float(np.max(_gv)), 1.0) / 1e21
    _ax.set_xlim(_Ev - _E_vb, _Ec + _E_cb + 0.05)
    _ax.set_ylim(0.0, 1.15 * _gmax)
    _ax.legend(frameon=False, loc="upper center")
    style_ax(_ax)
    _ax.text(_Ec + 0.03, 0.12 * _gmax, r"$E_c$", color=GREEN, fontsize=16)
    _ax.text(_Ev - 0.03, 0.12 * _gmax, r"$E_v$", color=GOLD, fontsize=16, ha="right")
    _ax.text(0.5 * (_Ec + _Ev), 0.08 * _gmax, r"$g=0$", color=NAVY, fontsize=16, ha="center")

    _info = mo.md(
        rf"""
    $\Delta E={_dE:.2f}$ eV from each band edge.

    Available states per cm$^3$ from $E_c$ to $E_c+\Delta E$:
    ${_Nc_win:.2e}$. From $E_v-\Delta E$ to $E_v$: ${_Nv_win:.2e}$.

    Neamen Example 3.3 (free electrons, $m=m_0$, $\Delta E=1$ eV):
    $N/V={_N_ex33:.2e}$ cm$^{{-3}}$ (text: $4.5\times 10^{{21}}$).
    """
    )
    mo.vstack(
        [
            _fig,
            _info,
            mo.accordion(
                {
                    "Explanation": mo.md(
                        r"""
    There are no crystal eigenstates in the gap, so $g(E)=0$ there. Larger
    $m^*$ flattens $E(k)$, packing more $k$-states into a given $dE$, and
    $g\propto(m^*)^{3/2}$. Integrating the free-electron $g(E)$ gives
    $N/V=8\pi(2m)^{3/2}(\Delta E)^{3/2}/(3h^3)$ (Example 3.3). That count is
    **available** states. Occupied electrons also need $f(E)$.
    """
                    )
                }
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Fermi–Dirac distribution

    **Predict:** Does $f(E_F)$ depend on temperature? If you raise $T$, do
    states **below** $E_F$ become more or less occupied?

    Electrons are indistinguishable fermions: at most one electron per complete
    quantum state (spin included). Neamen Eq. (3.79); lecture slides use
    $k_BT$ for the same $kT$:

    $$
    f(E)=\frac{1}{1+\exp[(E-E_F)/kT]}.
    $$

    At $T=0$, $f=1$ for $E<E_F$ and $0$ for $E>E_F$. For all $T>0$,
    $f(E_F)=\tfrac12$, and $f(E_F+\Delta E)=1-f(E_F-\Delta E)$.
    Maxwell–Boltzmann $e^{-(E-E_F)/kT}$ is the dilute tail; Bose–Einstein
    $1/(e^{(E-E_F)/kT}-1)$ describes bosons such as photons.
    """)
    return


@app.cell
def _(mo):
    Tfd = mo.ui.slider(
        start=0,
        stop=800,
        value=300,
        step=25,
        show_value=True,
        label="T (K)",
    )
    fd_controls = mo.vstack([Tfd])
    fd_controls
    return (Tfd,)


@app.cell
def _(
    BLUE,
    GOLD,
    GRAY,
    GREEN,
    K_EV,
    NAVY,
    ORANGE,
    Tfd,
    fermi_dirac,
    mo,
    np,
    plt,
    style_ax,
):
    _T = float(Tfd.value)
    _EF = 0.0
    _kT = K_EV * _T if _T > 0 else 0.02585
    _xmax = max(0.30, 5.6 * _kT) if _T > 0 else 0.30
    _E = np.linspace(-_xmax, _xmax, 900)
    _f = fermi_dirac(_E, _EF, _T)
    _fB = np.exp(-(_E - _EF) / _kT) if _T > 0 else np.full_like(_E, np.nan)
    _xBE = (_E - _EF) / _kT if _T > 0 else np.full_like(_E, np.nan)
    _fBE = np.full_like(_E, np.nan)
    if _T > 0:
        _mask_be = _xBE > 0.04
        _fBE[_mask_be] = 1.0 / (np.exp(_xBE[_mask_be]) - 1.0)

    _fig, _ax = plt.subplots(figsize=(8.2, 4.6), layout="constrained")
    _ax.plot(_E, _f, color=BLUE, lw=2.6, label=r"$f(E)$")
    if _T > 0:
        _ax.plot(_E, _fB, color=ORANGE, ls="--", lw=2.0, label="Maxwell–Boltzmann")
        _ax.plot(_E, _fBE, color=GREEN, ls=":", lw=2.2, label="Bose–Einstein")
        _ax.axvline(_EF + 3.0 * _kT, color=GOLD, ls="-.", lw=1.2)
        _ax.axvline(_EF + 5.0 * _kT, color=GRAY, ls="-.", lw=1.2)
        _ax.text(_EF + 3.0 * _kT, 1.05, r"$3kT$", color=GOLD, ha="center", fontsize=13)
        _ax.text(_EF + 5.0 * _kT, 1.05, r"$5kT$", color=GRAY, ha="center", fontsize=13)
    _ax.axvline(_EF, color=NAVY, ls="--", lw=1.4)
    _ax.axhline(0.5, color=GRAY, ls=":", lw=1.2)
    _ax.set_xlabel(r"$E-E_F$ (eV)")
    _ax.set_ylabel(r"$f(E)$")
    _ax.set_xlim(-_xmax, _xmax)
    _ax.set_ylim(0.0, 1.15)
    _ax.legend(frameon=False, loc="center right", fontsize=13)
    style_ax(_ax)

    if _T == 0:
        _msg = r"At $T=0$, $f$ is a step: filled below $E_F$, empty above."
    else:
        _f3 = float(fermi_dirac(np.array([3 * _kT]), _EF, _T)[0])
        _f5 = float(fermi_dirac(np.array([5 * _kT]), _EF, _T)[0])
        _b3 = float(np.exp(-3.0))
        _b5 = float(np.exp(-5.0))
        _rel3 = (_b3 - _f3) / _f3 * 100.0
        _rel5 = (_b5 - _f5) / _f5 * 100.0
        _msg = (
            rf"$T={_T:.0f}$ K, $kT={_kT:.4f}$ eV (300 K: $0.02585$ eV). "
            rf"At $E=E_F+3kT$, $f={_f3:.4f}$ (lecture: $0.0474$ at 300 K) "
            rf"and the Boltzmann overestimate is {_rel3:.1f}%. "
            rf"At $5kT$, $f={_f5:.4f}$ and the relative error is {_rel5:.2f}%."
        )
    _info = mo.md(_msg)
    mo.vstack(
        [
            _fig,
            _info,
            mo.accordion(
                {
                    "Explanation": mo.md(
                        r"""
    $f(E_F)=1/2$ at every $T>0$. Raising $T$ at fixed $E_F$ **empties**
    states below $E_F$ and **fills** states above; the step smears over a few
    $kT$. Maxwell–Boltzmann exceeds 1 below $E_F$ and is not a probability
    there; use it only in the dilute tail. At $3kT$ the relative error is
    about 5%; at $5kT$ it is below 1%. Bose–Einstein diverges as $E\to E_F$
    from above and is for bosons, not electrons.
    """
                    )
                }
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Carrier density $N(E)=g(E)\,f(E)$

    **Predict:** If $E_F$ moves toward $E_c$ at fixed $T$, does the electron
    distribution $g_c f$ grow or shrink? What happens to $g_v(1-f)$?
    """)
    return


@app.cell
def _(mo):
    material_select = mo.ui.dropdown(
        options=["Silicon", "GaAs", "Germanium", "InP"],
        value="Silicon",
        label="Select material:",
    )
    Tcar = mo.ui.slider(
        start=50,
        stop=500,
        step=25,
        value=300,
        show_value=True,
        label="Temperature (K):",
    )
    EFpos = mo.ui.slider(
        start=-0.56,
        stop=0.56,
        step=0.02,
        value=0.0,
        show_value=True,
        label="Fermi level position (eV from midgap):",
    )
    occ_controls = mo.vstack(
        [
            mo.md("### Interactive Carrier Density vs. Fermi Level Position"),
            material_select,
            Tcar,
            EFpos,
        ]
    )
    occ_controls
    return EFpos, Tcar, material_select


@app.cell
def _(
    BLUE,
    EFpos,
    GREEN,
    K_EV,
    MN_SI,
    MP_SI,
    NAVY,
    Ncv_cm3,
    ORANGE,
    Tcar,
    fermi_dirac,
    g_c_cm3_eV,
    g_v_cm3_eV,
    go,
    make_subplots,
    material_select,
    mo,
    np,
):
    _materials = {
        "Silicon": {"mn": MN_SI, "mp": MP_SI, "Eg": 1.12},
        "GaAs": {"mn": 0.067, "mp": 0.45, "Eg": 1.42},
        "Germanium": {"mn": 0.55, "mp": 0.37, "Eg": 0.66},
        "InP": {"mn": 0.08, "mp": 0.60, "Eg": 1.34},
    }
    _par = _materials[material_select.value]
    _T = float(Tcar.value)
    _EF = float(EFpos.value)
    _Eg = _par["Eg"]
    _mn = _par["mn"]
    _mp = _par["mp"]
    _Ec = 0.5 * _Eg
    _Ev = -0.5 * _Eg
    _Emax = 1.25
    _E = np.linspace(-_Emax, _Emax, 1000)
    _gc = g_c_cm3_eV(_E, _Ec, _mn)
    _gv = g_v_cm3_eV(_E, _Ev, _mp)
    _f = fermi_dirac(_E, _EF, _T)
    _nE = _gc * _f
    _pE = _gv * (1.0 - _f)
    _cb = _E >= _Ec
    _vb = _E <= _Ev
    _Nc = Ncv_cm3(_mn, _T)
    _Nv = Ncv_cm3(_mp, _T)
    _kT = K_EV * _T
    _n0 = _Nc * np.exp(-(_Ec - _EF) / _kT)
    _p0 = _Nv * np.exp(-(_EF - _Ev) / _kT)
    _ni = np.sqrt(_Nc * _Nv) * np.exp(-_Eg / (2.0 * _kT))

    _figc = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=(
            "Energy bands",
            "g(E)",
            "f(E)",
            "g(E) f(E)",
        ),
        horizontal_spacing=0.08,
    )
    _figc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[_Ec, _Ec],
            mode="lines",
            line=dict(color=BLUE, width=3),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    _figc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[_Ev, _Ev],
            mode="lines",
            line=dict(color=ORANGE, width=3),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    _figc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[_EF, _EF],
            mode="lines",
            line=dict(color=NAVY, width=3, dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    _figc.add_trace(
        go.Scatter(
            x=[0, 1, 1, 0, 0],
            y=[_Ec, _Ec, _Emax, _Emax, _Ec],
            fill="toself",
            fillcolor="rgba(0,114,178,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    _figc.add_trace(
        go.Scatter(
            x=[0, 1, 1, 0, 0],
            y=[_Ev, _Ev, -_Emax, -_Emax, _Ev],
            fill="toself",
            fillcolor="rgba(213,94,0,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    _figc.add_annotation(
        x=1.08,
        y=_Ec,
        text="E<sub>c</sub>",
        showarrow=False,
        font=dict(size=14, color=BLUE),
        xref="x1",
        yref="y1",
    )
    _figc.add_annotation(
        x=1.08,
        y=_Ev,
        text="E<sub>v</sub>",
        showarrow=False,
        font=dict(size=14, color=ORANGE),
        xref="x1",
        yref="y1",
    )
    _figc.add_annotation(
        x=1.08,
        y=_EF,
        text="E<sub>F</sub>",
        showarrow=False,
        font=dict(size=14, color=NAVY),
        xref="x1",
        yref="y1",
    )

    _figc.add_trace(
        go.Scatter(
            x=_gc[_cb],
            y=_E[_cb],
            mode="lines",
            line=dict(color=BLUE, width=2),
            name="g<sub>c</sub>(E)",
            hovertemplate="g<sub>c</sub> = %{x:.2e} cm⁻³eV⁻¹<br>E = %{y:.3f} eV<extra></extra>",
        ),
        row=1,
        col=2,
    )
    _figc.add_trace(
        go.Scatter(
            x=_gv[_vb],
            y=_E[_vb],
            mode="lines",
            line=dict(color=ORANGE, width=2),
            name="g<sub>v</sub>(E)",
            hovertemplate="g<sub>v</sub> = %{x:.2e} cm⁻³eV⁻¹<br>E = %{y:.3f} eV<extra></extra>",
        ),
        row=1,
        col=2,
    )
    _figc.add_hline(y=_Ec, line=dict(color="#6B6B6B", width=1, dash="dot"), row=1, col=2)
    _figc.add_hline(y=_Ev, line=dict(color="#6B6B6B", width=1, dash="dot"), row=1, col=2)
    _figc.add_hline(y=_EF, line=dict(color=NAVY, width=1, dash="dash"), row=1, col=2)

    _figc.add_trace(
        go.Scatter(
            x=_f,
            y=_E,
            mode="lines",
            line=dict(color=GREEN, width=2),
            name="f(E)",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    _figc.add_hline(y=_EF, line=dict(color=NAVY, width=1, dash="dash"), row=1, col=3)
    _figc.add_vline(x=0.5, line=dict(color="#6B6B6B", width=1, dash="dot"), row=1, col=3)

    _figc.add_trace(
        go.Scatter(
            x=_nE[_cb],
            y=_E[_cb],
            mode="lines",
            line=dict(color=BLUE, width=2),
            name="electrons g<sub>c</sub> f",
            hovertemplate="n(E) = %{x:.2e} cm⁻³eV⁻¹<br>E = %{y:.3f} eV<extra></extra>",
        ),
        row=1,
        col=4,
    )
    _figc.add_trace(
        go.Scatter(
            x=_pE[_vb],
            y=_E[_vb],
            mode="lines",
            line=dict(color=ORANGE, width=2),
            name="holes g<sub>v</sub>(1−f)",
            hovertemplate="p(E) = %{x:.2e} cm⁻³eV⁻¹<br>E = %{y:.3f} eV<extra></extra>",
        ),
        row=1,
        col=4,
    )
    _figc.add_hline(y=_Ec, line=dict(color="#6B6B6B", width=1, dash="dot"), row=1, col=4)
    _figc.add_hline(y=_Ev, line=dict(color="#6B6B6B", width=1, dash="dot"), row=1, col=4)
    _figc.add_hline(y=_EF, line=dict(color=NAVY, width=1, dash="dash"), row=1, col=4)

    _figc.update_xaxes(showticklabels=False, row=1, col=1)
    _figc.update_xaxes(
        title_text="g(E) (cm<sup>−3</sup> eV<sup>−1</sup>)",
        range=[0, None],
        exponentformat="power",
        row=1,
        col=2,
    )
    _figc.update_xaxes(title_text="f(E)", range=[0, 1], row=1, col=3)
    _figc.update_xaxes(
        title_text="carrier density (cm<sup>−3</sup> eV<sup>−1</sup>)",
        range=[0, None],
        exponentformat="power",
        row=1,
        col=4,
    )
    for _col in (1, 2, 3, 4):
        _figc.update_yaxes(range=[-_Emax, _Emax], row=1, col=_col)
    _figc.update_yaxes(title_text="Energy (eV), from midgap", row=1, col=1)
    _figc.update_layout(
        height=520,
        font=dict(size=16),
        title_text=(
            f"{material_select.value}: T = {_T:.0f} K, "
            f"(E<sub>F</sub> − E<sub>midgap</sub>) = {_EF:.2f} eV"
        ),
        legend=dict(orientation="h", y=1.14, x=0.35, font=dict(size=14)),
        margin=dict(t=90, l=60, r=20, b=50),
        hovermode="closest",
    )

    _info = mo.md(
        rf"""
    $m_n^*={_mn:.3f}\,m_0$, $m_p^*={_mp:.3f}\,m_0$, $E_g={_Eg:.2f}$ eV.
    Boltzmann preview (Neamen §4.1):
    $n_0={_n0:.2e}$ cm$^{{-3}}$, $p_0={_p0:.2e}$ cm$^{{-3}}$,
    $n_i={_ni:.2e}$ cm$^{{-3}}$.
    Both $g_c$ and $g_v$ are plotted $\ge 0$.
    """
    )
    mo.vstack(
        [
            _figc,
            _info,
            mo.accordion(
                {
                    "Explanation": mo.md(
                        r"""
    Energy is measured from midgap in this figure. $g_c$ and $g_v$ are both
    positive: $g_c$ lives above $E_c$, $g_v$ below $E_v$. $f(E)$ is the
    occupation probability of an existing state; holes use $1-f$. The
    product $g f$ is appreciable only where **both** factors are nonzero, so
    electrons sit in a thin tail above $E_c$ and holes in a thin tail below
    $E_v$. Raising $E_F$ toward $E_c$ increases $n_0$ and decreases $p_0$.
    The Boltzmann $n_0=N_c e^{-(E_c-E_F)/kT}$ (and the analogous $p_0$) is
    a §4.1 preview, valid when $E_F$ is several $kT$ from the edge.
    """
                    )
                }
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. What does $g(E)\,dE$ count, and what does $f(E)$ describe?
    2. Where do the spin factor and the positive-octant factor enter the
       $k$-space count?
    3. How does doubling the effective mass change the three-dimensional DOS?
    4. Can the Fermi level lie where no states are available?
    5. When is the Boltzmann approximation accurate to about 1%?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers and suggested checks": mo.md(
                r"""
    1. $g(E)\,dE$ is available states per volume in $dE$. $f$ is the
       occupation probability of an existing state. Their product is occupied
       electrons per volume in $dE$.
    2. Factor two is spin. One-eighth is the positive octant of standing-wave
       indices ($n_i>0$, cell $(\pi/a)^3$). Confirm on the $k$-space plot.
    3. $g\propto(m^*)^{3/2}$, so doubling mass multiplies $g$ by
       $2^{3/2}\approx 2.83$ at fixed energy from the edge. Move $m_n^*$ from
       $0.54$ to $1.08$.
    4. Yes. $E_F$ can sit in the gap. Then $f(E_F)=1/2$ but $g(E_F)=0$, so
       $g f=0$ there. Occupied electrons appear only for $E\ge E_c$.
    5. At $E_F+5kT$ the relative Boltzmann error is below 1%
       ($f\approx 0.0067$). At $3kT$ it is about 5%. Use the $T$ slider.
    """
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
