# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.24.0"
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
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.linewidth": 1.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "mathtext.fontset": "dejavusans",
        }
    )

    H = 6.62607015e-34
    HBAR = H / (2.0 * np.pi)
    M_E = 9.1093837015e-31
    Q = 1.602176634e-19
    C = 2.99792458e8
    HC_EV_NM = H * C / Q * 1e9
    RY_EV = 13.605693122994
    A0_NM = 5.29177210903e-2
    KT_300K = 8.617333262e-5 * 300.0

    BLUE = "#0072B2"
    ORANGE = "#D55E00"
    GREEN = "#009E73"
    GOLD = "#E69F00"
    NAVY = "#0f4c81"
    DARK = "#2D2D2D"
    GRAY = "#6B6B6B"

    def style_ax(ax):
        ax.tick_params(direction="out", length=4, width=1.0)
        ax.grid(True, alpha=0.28, color=GRAY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def photon_energy_eV(lambda_nm):
        return HC_EV_NM / lambda_nm

    def debroglie_nm(E_eV):
        return H / np.sqrt(2.0 * M_E * E_eV * Q) * 1e9

    def infinite_well_E_eV(n, a_nm):
        a = a_nm * 1e-9
        return (n**2 * np.pi**2 * HBAR**2) / (2.0 * M_E * a**2) / Q

    def barrier_coefficients(E_eV, U0_eV, L_nm):
        if abs(E_eV - U0_eV) < 1e-4:
            E_eV = U0_eV + 1e-4
        E = E_eV * Q
        U0 = U0_eV * Q
        L = L_nm * 1e-9
        k = np.sqrt(2.0 * M_E * E) / HBAR
        k2 = np.lib.scimath.sqrt(2.0 * M_E * (E - U0)) / HBAR
        e_k2L = np.exp(1j * k2 * L)
        e_mk2L = np.exp(-1j * k2 * L)
        e_kL = np.exp(1j * k * L)
        mat = np.array(
            [
                [1.0, -1.0, -1.0, 0.0],
                [-k, -k2, k2, 0.0],
                [0.0, e_k2L, e_mk2L, -e_kL],
                [0.0, k2 * e_k2L, -k2 * e_mk2L, -k * e_kL],
            ],
            dtype=complex,
        )
        rhs = np.array([-1.0, -k, 0.0, 0.0], dtype=complex)
        r, A, B, t = np.linalg.solve(mat, rhs)
        T = float(np.abs(t) ** 2)
        R = float(np.abs(r) ** 2)
        return r, A, B, t, k, k2, T, R

    def psi_barrier(x, r, A, B, t, k, k2, L_m):
        x = np.asarray(x)
        out = np.zeros_like(x, dtype=complex)
        left = x < 0.0
        mid = (x >= 0.0) & (x <= L_m)
        right = x > L_m
        out[left] = np.exp(1j * k * x[left]) + r * np.exp(-1j * k * x[left])
        out[mid] = A * np.exp(1j * k2 * x[mid]) + B * np.exp(-1j * k2 * x[mid])
        out[right] = t * np.exp(1j * k * x[right])
        return out

    return (
        A0_NM,
        BLUE,
        C,
        GOLD,
        GREEN,
        H,
        HBAR,
        HC_EV_NM,
        KT_300K,
        M_E,
        NAVY,
        ORANGE,
        Q,
        RY_EV,
        barrier_coefficients,
        debroglie_nm,
        infinite_well_E_eV,
        photon_energy_eV,
        plt,
        psi_barrier,
        style_ax,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Quantum Physics
    **ECE335 - companion to lectures**

    Electronic devices are nanometre-scale. Electrons there are not billiard balls:
    energy comes in packets, matter has a wavelength, and a confined electron
    has only certain allowed energies. Use this notebook **with the slides**.
    Every curve below is computed from the formula — move a slider and watch
    the physics change.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Photons — the photoelectric effect

    Light absorbed by an electron acts as a stream of photons,

    $$E = hf = \frac{hc}{\lambda}, \qquad hc \approx 1240\ \mathrm{eV\cdot nm}.$$

    An electron leaves a metal only if $hf$ exceeds the work function $\varphi$.
    The leftover energy is kinetic:

    $$K_{\max} = hf - \varphi.$$

    Intensity sets **how many** photons arrive per second, not the energy of
    each one. Brighter light $\Rightarrow$ more electrons, same $K_{\max}$.
    """)
    return


@app.cell
def _(mo):
    pe_lambda = mo.ui.slider(
        start=200,
        stop=800,
        value=400,
        step=5,
        show_value=True,
        label="wavelength λ (nm)",
    )
    pe_metal = mo.ui.dropdown(
        options={
            "Cs  (φ = 2.14 eV)": 2.14,
            "Na  (φ = 2.36 eV)": 2.36,
            "Al  (φ = 4.28 eV)": 4.28,
            "Cu  (φ = 4.65 eV)": 4.65,
        },
        value="Cs  (φ = 2.14 eV)",
        label="cathode metal",
    )
    pe_intensity = mo.ui.slider(
        start=0.25,
        stop=4.0,
        value=1.0,
        step=0.25,
        show_value=True,
        label="relative intensity",
    )
    pe_controls = mo.vstack(
        [
            mo.md("### Interactive: photoelectric $K_{\\max}$"),
            mo.hstack([pe_lambda, pe_metal, pe_intensity], justify="start", gap=1.5),
        ]
    )
    pe_controls
    return pe_controls, pe_intensity, pe_lambda, pe_metal


@app.cell
def _(
    BLUE,
    C,
    GOLD,
    H,
    NAVY,
    ORANGE,
    Q,
    mo,
    np,
    pe_controls,
    pe_intensity,
    pe_lambda,
    pe_metal,
    photon_energy_eV,
    plt,
    style_ax,
):
    _lam = float(pe_lambda.value)
    _phi = float(pe_metal.value)
    _I = float(pe_intensity.value)
    _Eph = photon_energy_eV(_lam)
    _f = C / (_lam * 1e-9)
    _K = _Eph - _phi
    _ejected = _K > 0.0
    _Kshow = _K if _ejected else 0.0
    _f0 = _phi * Q / H

    _f_plot = np.linspace(0.0, 1.6e15, 500)
    _K_line = H * _f_plot / Q - _phi
    _K_line = np.where(_K_line > 0.0, _K_line, np.nan)

    _fig, _ax = plt.subplots(figsize=(8.2, 4.4), layout="constrained")
    _ax.plot(_f_plot / 1e14, _K_line, color=BLUE, lw=2.6, label=rf"$\varphi = {_phi:.2f}$ eV")
    _ax.axvline(_f0 / 1e14, color=ORANGE, ls="--", lw=1.4, label=r"threshold $f_0 = \varphi/h$")
    _ax.plot([_f0 / 1e14], [0.0], "o", color=ORANGE, ms=8)
    _ax.plot(
        [_f / 1e14],
        [_Kshow],
        "o",
        color=GOLD,
        ms=11,
        zorder=5,
        label="this photon",
    )
    if _ejected:
        _ax.vlines(_f / 1e14, 0.0, _Kshow, color=GOLD, lw=1.6, alpha=0.7)
    _ax.axhline(0.0, color=NAVY, lw=0.8)
    _ax.set_xlim(0.0, 16.0)
    _ax.set_ylim(-0.25, 5.4)
    _ax.set_xlabel(r"frequency $f$  ($10^{14}$ Hz)")
    _ax.set_ylabel(r"$K_{\mathrm{max}}$ (eV)")
    _ax.legend(frameon=False, loc="upper left")
    style_ax(_ax)

    if _ejected:
        _status = (
            f"Electrons are ejected.  $K_{{\\max}} = {_K:.3f}$ eV.  "
            f"Relative electron rate $\\propto$ intensity $= {_I:.2f}$ "
            r"(the kinetic energy does **not** change with intensity)."
        )
    else:
        _status = (
            f"Below threshold: $hf = {_Eph:.3f}$ eV $< \\varphi = {_phi:.2f}$ eV.  "
            "No electrons, no matter how bright the lamp or how long you wait."
        )

    _info = mo.md(
        rf"""
    Photon: $\lambda = {_lam:.0f}$ nm,  $f = {_f/1e14:.2f}\times 10^{{14}}$ Hz,  $E = {_Eph:.3f}$ eV.

    {_status}

    $$K_{{\max}} = hf - \varphi = {_Eph:.3f} - {_phi:.2f} = {_K:.3f}\ \mathrm{{eV}}$$
    """
    )
    mo.vstack([pe_controls, _fig, _info])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Matter waves — de Broglie

    If light (a wave) can act as a particle, an electron (a particle) can act as a wave:

    $$\lambda = \frac{h}{p} = \frac{h}{\sqrt{2mE}}.$$

    Wave effects appear when $\lambda$ is comparable to a length in the problem
    (a slit, an atom, a gate). A baseball's $\lambda$ is $\sim 10^{-34}$ m — Newtonian
    mechanics is enough. A 100 eV electron has $\lambda \approx 0.12$ nm, about a
    Si–Si bond.
    """)
    return


@app.cell
def _(mo):
    db_logE = mo.ui.slider(
        start=-2.0,
        stop=3.0,
        value=2.0,
        step=0.05,
        show_value=True,
        label="log10(E / eV)  for an electron",
    )
    db_controls = mo.vstack(
        [
            mo.md("### Interactive: electron de Broglie wavelength"),
            db_logE,
        ]
    )
    db_controls
    return db_controls, db_logE


@app.cell
def _(
    BLUE,
    GOLD,
    GREEN,
    ORANGE,
    db_controls,
    db_logE,
    debroglie_nm,
    mo,
    np,
    plt,
    style_ax,
):
    _E = 10.0 ** float(db_logE.value)
    _lam_e = debroglie_nm(_E)
    _E_plot = np.logspace(-2.0, 3.0, 400)
    _lam_plot = debroglie_nm(_E_plot)
    _a_Si = 0.543
    _bond = 0.235

    _fig, _ax = plt.subplots(figsize=(8.2, 4.4), layout="constrained")
    _ax.loglog(_E_plot, _lam_plot, color=BLUE, lw=2.6, label=r"$\lambda = h/\sqrt{2mE}$")
    _ax.axhline(_a_Si, color=GREEN, ls="--", lw=1.5, label=rf"Si lattice $a = {_a_Si}$ nm")
    _ax.axhline(_bond, color=ORANGE, ls=":", lw=1.8, label=rf"Si–Si bond $\approx {_bond}$ nm")
    _ax.plot([_E], [_lam_e], "o", color=GOLD, ms=11, zorder=5)
    _ax.set_xlabel("electron kinetic energy $E$ (eV)")
    _ax.set_ylabel(r"de Broglie $\lambda$ (nm)")
    _ax.set_xlim(1e-2, 1e3)
    _ax.set_ylim(3e-2, 20)
    _ax.legend(frameon=False, loc="upper right")
    style_ax(_ax)

    _thermal = debroglie_nm(0.02585)
    _info = mo.md(
        rf"""
    At $E = {_E:.3g}$ eV,  $\lambda = {_lam_e:.3g}$ nm.

    Room-temperature thermal electron ($E \approx kT = 0.026$ eV):
    $\lambda \approx {_thermal:.2f}$ nm — several nanometres, so a modern transistor
    gate is a wave-mechanical object.
    """
    )
    mo.vstack([db_controls, _fig, _info])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Two slits: a wave of probability

    A wave that can take two paths interferes:

    $$I(\theta) \propto \cos^2\!\left(\frac{\pi d \sin\theta}{\lambda}\right).$$

    Electrons, sent **one at a time**, still fill in this pattern. Each hit is a
    particle (one spot, charge $-q$). The pattern of many hits is a wave. You
    cannot say "which slit" and keep the fringes.
    """)
    return


@app.cell
def _(mo):
    slit_lambda = mo.ui.slider(
        start=0.05,
        stop=0.40,
        value=0.12,
        step=0.01,
        show_value=True,
        label="λ (nm)",
    )
    slit_d = mo.ui.slider(
        start=0.4,
        stop=4.0,
        value=1.2,
        step=0.1,
        show_value=True,
        label="slit spacing d (nm)",
    )
    slit_controls = mo.vstack(
        [
            mo.md("### Interactive: two-slit intensity"),
            mo.hstack([slit_lambda, slit_d], justify="start", gap=2),
        ]
    )
    slit_controls
    return slit_controls, slit_d, slit_lambda


@app.cell
def _(BLUE, NAVY, mo, np, plt, slit_controls, slit_d, slit_lambda, style_ax):
    _lam = float(slit_lambda.value)
    _d = float(slit_d.value)
    _theta = np.linspace(-0.45, 0.45, 1200)
    _I = np.cos(np.pi * _d * np.sin(_theta) / _lam) ** 2
    _fig, _ax = plt.subplots(figsize=(8.2, 4.2), layout="constrained")
    _ax.plot(np.degrees(_theta), _I, color=NAVY, lw=1.6)
    _ax.fill_between(np.degrees(_theta), 0.0, _I, color=BLUE, alpha=0.55)
    _ax.set_xlim(-25.0, 25.0)
    _ax.set_ylim(0.0, 1.15)
    _ax.set_xlabel(r"angle $\theta$ (degrees)")
    _ax.set_ylabel("intensity  (arrival probability)")
    style_ax(_ax)

    _fringe = np.degrees(np.arcsin(min(1.0, _lam / _d))) if _d > 0 else np.nan
    _info = mo.md(
        rf"""
    $\lambda = {_lam:.2f}$ nm,  $d = {_d:.2f}$ nm.
    Adjacent bright fringes sit near $\Delta\theta \approx \lambda/d = {_fringe:.2f}^\circ$
    (small-angle). Stretch $\lambda$ or squeeze $d$ and the fringes spread.
    """
    )
    mo.vstack([slit_controls, _fig, _info])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Bohr hydrogen — keep discrete energies, drop the orbits

    Bohr's working rules (1913): only certain orbits, $mvr = n\hbar$, and light
    is emitted in jumps $hf = |E_i - E_f|$. What survives in modern quantum
    mechanics is the **menu of energies**, not the little circles:

    $$E_n = -\frac{13.6\ \mathrm{eV}}{n^2}, \qquad r_n = n^2 a_0, \quad a_0 = 0.0529\ \mathrm{nm}.$$

    Bound electrons have discrete energies. That is why atoms have spectral
    lines — and why a semiconductor has a band gap.
    """)
    return


@app.cell
def _(mo):
    bohr_n = mo.ui.slider(
        start=1,
        stop=6,
        value=3,
        step=1,
        show_value=True,
        label="n (occupied level)",
    )
    bohr_nf = mo.ui.slider(
        start=1,
        stop=5,
        value=2,
        step=1,
        show_value=True,
        label="n_f (lower level for the photon)",
    )
    bohr_controls = mo.vstack(
        [
            mo.md("### Interactive: hydrogen levels and a photon jump"),
            mo.hstack([bohr_n, bohr_nf], justify="start", gap=2),
        ]
    )
    bohr_controls
    return bohr_controls, bohr_n, bohr_nf


@app.cell
def _(
    A0_NM,
    BLUE,
    GOLD,
    GREEN,
    HC_EV_NM,
    NAVY,
    ORANGE,
    RY_EV,
    bohr_controls,
    bohr_n,
    bohr_nf,
    mo,
    np,
    plt,
    style_ax,
):
    _n = int(bohr_n.value)
    _nf = int(bohr_nf.value)
    _En = -RY_EV / _n**2
    _rn = (_n**2) * A0_NM
    _levels = np.arange(1, 7)
    _E_all = -RY_EV / _levels**2

    _fig, (_axL, _axR) = plt.subplots(1, 2, figsize=(9.4, 4.6), layout="constrained")
    _colors = [ORANGE, BLUE, GREEN, GOLD, NAVY, "#CC79A7"]
    for _k, _nn in enumerate(_levels[:4]):
        _r = (_nn**2) * A0_NM
        _circ = plt.Circle(
            (0.0, 0.0),
            _r,
            fill=False,
            color=_colors[_k],
            lw=2.4 if _nn == _n else 1.3,
            alpha=1.0 if _nn == _n else 0.45,
        )
        _axL.add_patch(_circ)
        _axL.text(
            0.05,
            _r + 0.02,
            rf"$n={_nn}$",
            color=_colors[_k],
            fontsize=13,
            alpha=1.0 if _nn == _n else 0.55,
        )
    _axL.plot([0], [0], "o", color=GOLD, ms=10)
    _axL.set_aspect("equal")
    _axL.set_xlim(-1.05, 1.05)
    _axL.set_ylim(-1.05, 1.15)
    _axL.axis("off")
    _axL.set_title(r"Bohr radii (to scale): $r_n = n^2 a_0$")

    for _nn, _EE, _col in zip(_levels, _E_all, _colors):
        _axR.hlines(_EE, 0.15, 0.85, color=_col, lw=2.4 if _nn == _n else 1.6)
        _axR.text(0.90, _EE, rf"$n={_nn}$", va="center", fontsize=13, color=_col)
    _axR.plot([0.5], [_En], "o", color=GOLD, ms=10, zorder=5)
    if _nf < _n:
        _Ef = -RY_EV / _nf**2
        _axR.annotate(
            "",
            xy=(0.50, _Ef),
            xytext=(0.50, _En),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.0),
        )
    _axR.set_xlim(0.0, 1.25)
    _axR.set_ylim(-15.0, 1.0)
    _axR.axhline(0.0, color=NAVY, lw=0.8, ls="--")
    _axR.set_ylabel("$E$ (eV)")
    _axR.set_xticks([])
    _axR.set_title(r"$E_n = -13.6~\mathrm{eV}/n^2$")
    style_ax(_axR)
    _axR.spines["bottom"].set_visible(False)

    if _nf >= _n:
        _jump = (
            "Choose $n_f < n$ to emit a photon. "
            "If $n_f \\ge n$ the electron would have to absorb energy."
        )
    else:
        _dE = RY_EV * (1.0 / _nf**2 - 1.0 / _n**2)
        _lam_ph = HC_EV_NM / _dE
        _jump = (
            rf"Jump $n={_n}\to n_f={_nf}$:  $hf = {_dE:.3f}$ eV,  "
            rf"$\lambda = hc/E = {_lam_ph:.1f}$ nm."
        )

    _info = mo.md(
        rf"""
    Occupied level $n = {_n}$:  $E_n = {_En:.3f}$ eV,  $r_n = {_rn:.3f}$ nm
    ($a_0 = {A0_NM:.4f}$ nm).

    {_jump}

    Keep discrete $E_n$ and $hf = |\Delta E|$. Drop planetary orbits — the electron
    is a cloud $|\psi|^2$, not a marble on a circle.
    """
    )
    mo.vstack([bohr_controls, _fig, _info])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Uncertainty, from wave packets

    A pure sine wave has one $k$ (so one $p = \hbar k$) but is spread over all $x$.
    To localize the particle, add a range of $k$. They pile up in a bump — a
    **wave packet** — and cancel elsewhere. A narrower bump needs a wider mix of
    $k$:

    $$\Delta x\,\Delta p \ge \frac{\hbar}{2}.$$

    That is a fact about Fourier transforms, not a statement about equipment.
    """)
    return


@app.cell
def _(mo):
    pkt_sig = mo.ui.slider(
        start=0.4,
        stop=3.5,
        value=1.0,
        step=0.1,
        show_value=True,
        label="packet width σx (arb. units)",
    )
    pkt_controls = mo.vstack(
        [
            mo.md("### Interactive: a Gaussian packet in $x$ and in $k$"),
            pkt_sig,
        ]
    )
    pkt_controls
    return pkt_controls, pkt_sig


@app.cell
def _(BLUE, ORANGE, mo, np, pkt_controls, pkt_sig, plt, style_ax):
    _sx = float(pkt_sig.value)
    _sk = 1.0 / (2.0 * _sx)
    _k0 = 6.0
    _x = np.linspace(-12.0, 12.0, 800)
    _k = np.linspace(_k0 - 8.0, _k0 + 8.0, 800)
    _psi2 = np.exp(-(_x**2) / (2.0 * _sx**2)) / (_sx * np.sqrt(2.0 * np.pi))
    _phi2 = np.exp(-((_k - _k0) ** 2) / (2.0 * _sk**2)) / (_sk * np.sqrt(2.0 * np.pi))

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9.4, 4.0), layout="constrained")
    _ax1.fill_between(_x, 0.0, _psi2, color=BLUE, alpha=0.45)
    _ax1.plot(_x, _psi2, color=BLUE, lw=2.2)
    _ax1.set_xlim(-12.0, 12.0)
    _ax1.set_ylim(0.0, 1.05 * _psi2.max())
    _ax1.set_xlabel(r"$x$ (arb.)")
    _ax1.set_ylabel(r"$|\psi(x)|^2$")
    _ax1.set_title(rf"$\Delta x = \sigma_x = {_sx:.2f}$")
    style_ax(_ax1)

    _ax2.fill_between(_k, 0.0, _phi2, color=ORANGE, alpha=0.45)
    _ax2.plot(_k, _phi2, color=ORANGE, lw=2.2)
    _ax2.set_xlim(_k0 - 8.0, _k0 + 8.0)
    _ax2.set_ylim(0.0, 1.05 * _phi2.max())
    _ax2.set_xlabel(r"$k$ (arb.)")
    _ax2.set_ylabel(r"$|\phi(k)|^2$")
    _ax2.set_title(rf"$\Delta k = 1/(2\sigma_x) = {_sk:.2f}$")
    style_ax(_ax2)

    _info = mo.md(
        rf"""
    Minimum-uncertainty Gaussian: $\Delta x\,\Delta k = 1/2$, so
    $\Delta x\,\Delta p = \hbar/2$.

    Squeeze the packet in $x$ and it **must** spread in $k$ — that is the
    uncertainty principle as a fact about waves.
    """
    )
    mo.vstack([pkt_controls, _fig, _info])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Schrödinger's equation, in one line

    A particle of mass $m$ in a potential $U(x)$, in a state of definite energy $E$,
    obeys

    $$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + U(x)\,\psi = E\,\psi.$$

    $|\psi(x)|^2\,dx$ is the probability of finding the electron between $x$ and
    $x+dx$. Curvier $\psi$ means more kinetic energy. Only those $E$ for which
    $\psi$ can meet the boundary conditions are allowed — that is quantization.

    Recipe: write $U$ in each region $\to$ sines where $E>U$, exponentials where
    $E<U$ $\to$ match $\psi$ and $d\psi/dx$ $\to$ normalize.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Infinite well: confinement makes discrete energies

    $U = 0$ for $0 < x < a$, and $U = \infty$ outside. Then $\psi = 0$ at the walls,
    so an integer number of half-wavelengths must fit:

    $$\psi_n(x) = \sqrt{\frac{2}{a}}\sin\left(\frac{n\pi x}{a}\right),
    \qquad
    E_n = \frac{n^2 \pi^2 \hbar^2}{2ma^2}, \quad n = 1,2,3,\ldots$$

    Squeeze the well and the levels move apart ($E \propto 1/a^2$). A quantum well,
    a nanowire, and a MOS inversion layer are all boxes for electrons.
    """)
    return


@app.cell
def _(mo):
    well_a = mo.ui.slider(
        start=0.5,
        stop=8.0,
        value=2.0,
        step=0.1,
        show_value=True,
        label="well width a (nm)",
    )
    well_n = mo.ui.slider(
        start=1,
        stop=6,
        value=1,
        step=1,
        show_value=True,
        label="quantum number n",
    )
    well_controls = mo.vstack(
        [
            mo.md("### Interactive: particle in an infinite well"),
            mo.hstack([well_a, well_n], justify="start", gap=2),
        ]
    )
    well_controls
    return well_a, well_controls, well_n


@app.cell
def _(
    BLUE,
    GOLD,
    GREEN,
    KT_300K,
    NAVY,
    ORANGE,
    infinite_well_E_eV,
    mo,
    np,
    plt,
    style_ax,
    well_a,
    well_controls,
    well_n,
):
    _a = float(well_a.value)
    _n = int(well_n.value)
    _En = infinite_well_E_eV(_n, _a)
    _E1 = infinite_well_E_eV(1, _a)
    _x = np.linspace(0.0, _a, 600)
    _psi = np.sqrt(2.0 / _a) * np.sin(_n * np.pi * _x / _a)
    _prob = _psi**2

    _fig, _axes = plt.subplots(1, 2, figsize=(9.6, 4.4), layout="constrained")
    _ax, _axE = _axes
    _ax.plot(_x, _psi, color=BLUE, lw=2.4, label=r"$\psi_n(x)$")
    _ax.plot(_x, _prob, color=ORANGE, lw=2.2, label=r"$|\psi_n|^2$")
    _ax.axhline(0.0, color=NAVY, lw=0.8)
    _ax.axvline(0.0, color=NAVY, lw=2.5)
    _ax.axvline(_a, color=NAVY, lw=2.5)
    _ax.set_xlim(-0.08 * _a, 1.08 * _a)
    _ymax = 1.25 * max(_prob.max(), np.abs(_psi).max())
    _ax.set_ylim(-_ymax, _ymax)
    _ax.set_xlabel(r"$x$ (nm)")
    _ax.set_ylabel(r"$\psi$  and  $|\psi|^2$")
    _ax.legend(frameon=False, loc="upper right")
    _ax.set_title(rf"$n = {_n}$,  $a = {_a:.1f}$ nm")
    style_ax(_ax)

    _ns = np.arange(1, 7)
    _Es = infinite_well_E_eV(_ns, _a)
    for _nn, _EE in zip(_ns, _Es):
        _col = GOLD if _nn == _n else BLUE
        _axE.hlines(_EE, 0.2, 0.8, color=_col, lw=2.6 if _nn == _n else 1.6)
        _axE.text(0.85, _EE, rf"$n={_nn}$", va="center", fontsize=13, color=_col)
    _axE.axhline(KT_300K, color=GREEN, ls="--", lw=1.5, label=rf"$kT(300\,\mathrm{{K}}) = {KT_300K:.3f}$ eV")
    _axE.set_xlim(0.0, 1.35)
    _axE.set_ylim(-0.05 * _Es[-1], 1.15 * _Es[-1])
    _axE.set_xticks([])
    _axE.set_ylabel("$E_n$ (eV)")
    _axE.set_title(r"$E_n \propto n^2/a^2$")
    _axE.legend(frameon=False, loc="upper left", fontsize=12)
    style_ax(_axE)
    _axE.spines["bottom"].set_visible(False)

    _info = mo.md(
        rf"""
    $$E_n = \frac{{n^2\pi^2\hbar^2}}{{2ma^2}} = {_En:.3f}\ \mathrm{{eV}}
    \quad (E_1 = {_E1:.3f}\ \mathrm{{eV}}).$$

    $E_n/E_1 = 1,4,9,16,\ldots$ — **not** equally spaced. $n=0$ is forbidden
    ($\psi$ would vanish). Shrink $a$ by $2$ and every energy **quadruples**.
    In a crystal, replace $m$ by the effective mass $m^*$.
    """
    )
    mo.vstack([well_controls, _fig, _info])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Tunneling: a thin barrier is leaky

    Classically, if $E < U_0$ the particle turns around. Quantum: $\psi$ is an
    exponential $e^{-\kappa x}$ inside the barrier, with
    $\kappa = \sqrt{2m(U_0-E)}/\hbar$. If the barrier is thin, $\psi$ is still
    nonzero on the far side — a transmitted wave. The electron keeps the same
    energy $E$; it did not "lose energy climbing."

    Transmission for a rectangular barrier drops roughly as $e^{-2\kappa L}$.
    That exponential is why a slightly thinner gate oxide leaks a lot more
    current, and why an STM tip is so sensitive to distance.
    """)
    return


@app.cell
def _(mo):
    tun_E = mo.ui.slider(
        start=0.20,
        stop=4.50,
        value=1.00,
        step=0.05,
        show_value=True,
        label="electron energy E (eV)",
    )
    tun_U = mo.ui.slider(
        start=0.50,
        stop=6.00,
        value=3.00,
        step=0.10,
        show_value=True,
        label="barrier height U0 (eV)",
    )
    tun_L = mo.ui.slider(
        start=0.20,
        stop=2.50,
        value=0.80,
        step=0.05,
        show_value=True,
        label="barrier width L (nm)",
    )
    tun_controls = mo.vstack(
        [
            mo.md("### Interactive: rectangular barrier (exact matching of $\\psi$)"),
            mo.hstack([tun_E, tun_U, tun_L], justify="start", gap=1.2),
        ]
    )
    tun_controls
    return tun_E, tun_L, tun_U, tun_controls


@app.cell
def _(
    BLUE,
    GOLD,
    GREEN,
    HBAR,
    M_E,
    NAVY,
    ORANGE,
    Q,
    barrier_coefficients,
    mo,
    np,
    plt,
    psi_barrier,
    style_ax,
    tun_E,
    tun_L,
    tun_U,
    tun_controls,
):
    _E = float(tun_E.value)
    _U0 = float(tun_U.value)
    _L = float(tun_L.value)
    _r, _A, _B, _t, _k, _k2, _T, _R = barrier_coefficients(_E, _U0, _L)
    _Lm = _L * 1e-9
    _x = np.linspace(-2.0 * _L, 3.0 * _L, 1400)
    _psi = psi_barrier(_x * 1e-9, _r, _A, _B, _t, _k, _k2, _Lm)
    _prob = np.abs(_psi) ** 2
    _U = np.where((_x >= 0.0) & (_x <= _L), _U0, 0.0)

    _L_scan = np.linspace(0.05, 2.50, 250)
    _T_scan = np.array([barrier_coefficients(_E, _U0, _Ls)[6] for _Ls in _L_scan])

    _fig, (_axW, _axT) = plt.subplots(1, 2, figsize=(9.6, 4.4), layout="constrained")
    _axW.fill_between(_x, 0.0, _U, color=ORANGE, alpha=0.22, step=None)
    _axW.plot(_x, _U, color=ORANGE, lw=2.2, label="$U(x)$")
    _axW.axhline(_E, color=GREEN, ls="--", lw=1.6, label=rf"$E = {_E:.2f}$ eV")
    _axP = _axW.twinx()
    _axP.plot(_x, _prob, color=BLUE, lw=2.0, label=r"$|\psi|^2$")
    _axW.set_xlim(_x.min(), _x.max())
    _axW.set_ylim(-0.05 * max(_U0, _E), 1.25 * max(_U0, _E))
    _axP.set_ylim(0.0, 1.15 * max(_prob.max(), 1e-12))
    _axW.set_xlabel(r"$x$ (nm)")
    _axW.set_ylabel("$U$ (eV)")
    _axP.set_ylabel(r"$|\psi|^2$")
    _axW.set_title("barrier and probability")
    style_ax(_axW)
    _axW.spines["right"].set_visible(True)
    _h1, _l1 = _axW.get_legend_handles_labels()
    _h2, _l2 = _axP.get_legend_handles_labels()
    _axW.legend(_h1 + _h2, _l1 + _l2, frameon=False, loc="upper right", fontsize=12)

    _axT.semilogy(_L_scan, np.clip(_T_scan, 1e-16, 1.0), color=NAVY, lw=2.4)
    _axT.plot([_L], [max(_T, 1e-16)], "o", color=GOLD, ms=10, zorder=5)
    _axT.set_xlim(0.0, 2.55)
    _axT.set_ylim(1e-10, 1.5)
    _axT.set_xlabel(r"barrier width $L$ (nm)")
    _axT.set_ylabel("transmission $T$")
    _axT.set_title(r"$T(L)$ at this $E$, $U_0$")
    style_ax(_axT)

    if _E < _U0:
        _kappa = np.sqrt(2.0 * M_E * (_U0 - _E) * Q) / HBAR
        _exp_approx = float(np.exp(-2.0 * _kappa * _Lm))
        _regime = (
            rf"$E < U_0$ (classically forbidden). "
            rf"$\kappa = {_kappa*1e-9:.2f}$ nm$^{{-1}}$,  "
            rf"crude $e^{{-2\kappa L}} = {_exp_approx:.3e}$."
        )
    else:
        _regime = (
            r"$E > U_0$: part of the wave still reflects at the step. "
            r"$T$ oscillates with $L$ (resonances when an integer number of "
            r"half-waves fit in the barrier)."
        )

    _info = mo.md(
        rf"""
    $E = {_E:.2f}$ eV,  $U_0 = {_U0:.2f}$ eV,  $L = {_L:.2f}$ nm.

    Exact matching:  $T = {_T:.3e}$,  $R = {_R:.3e}$,  $R+T = {_R+_T:.4f}$.

    {_regime}

    Device uses: flash memory, STM, MOSFET gate leakage, tunnel diodes.
    Thinner or lower barrier $\Rightarrow$ exponentially more current.
    """
    )
    mo.vstack([tun_controls, _fig, _info])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From one atom to a crystal (looking ahead)

    Hydrogen recovers Bohr's energies, $E_n = -13.6~\mathrm{eV}/n^2$, from
    Schrödinger's equation with $U(r) \propto -1/r$. Each electron in an atom is
    labelled by four quantum numbers $(n,\ell,m,s)$. Pauli: **no two electrons
    share the same set**. Silicon ($Z=14$) is $[\mathrm{Ne}]\,3s^2 3p^2$ — four
    valence electrons, four tetrahedral bonds.

    Bring $N$ atoms together and each atomic level splits into $N$ closely spaced
    levels. In a wafer that is a continuous **band**, with a possible gap — the
    band gap of the semiconductor. That is the next chapter.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaways

    - Photons: $E = hf$. Photoelectric $K_{\max} = hf - \varphi$. Intensity is a rate.
    - Electrons have $\lambda = h/p$. They interfere. Devices are wave-mechanical.
    - Bound energies are discrete. Bohr: $E_n = -13.6~\mathrm{eV}/n^2$ (keep this; drop orbits).
    - A localized particle is a packet $\Rightarrow \Delta x\,\Delta p \ge \hbar/2$.
    - Schrödinger: $|\psi|^2$ is probability. Infinite well: $E_n \propto n^2/a^2$.
    - Finite walls leak. Thin barriers **tunnel**. Exponential in width and height.
    - Pauli filling + many atoms $\to$ energy bands (next).
    """)
    return


if __name__ == "__main__":
    app.run()
