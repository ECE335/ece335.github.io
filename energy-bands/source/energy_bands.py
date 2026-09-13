# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "plotly",
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
    HBAR = H / (2.0 * np.pi)
    M_E = 9.1093837015e-31
    Q = 1.602176634e-19
    HBAR2_2M_EVNM2 = HBAR**2 / (2.0 * M_E) / Q * 1e18

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

    def kp_lhs(E_eV, V0_eV, a_nm, b_nm):
        """Left-hand side of Neamen Eqs. (3.19) and (3.21).

        Region I (0 < x < a): V = 0, α = sqrt(2mE)/ħ.
        Region II (−b < x < 0): V = V0.
        For E ≥ V0, β = sqrt(2m(E−V0))/ħ.
        For E < V0, β = jγ with γ = sqrt(2m(V0−E))/ħ.
        Allowed E satisfy |LHS| ≤ 1 = |cos k(a+b)|.
        """
        scalar = np.ndim(E_eV) == 0
        E = np.atleast_1d(np.asarray(E_eV, dtype=float)) * Q
        V0 = float(V0_eV) * Q
        a = float(a_nm) * 1e-9
        b = float(b_nm) * 1e-9
        E = np.maximum(E, 1e-24)
        alpha = np.sqrt(2.0 * M_E * E) / HBAR
        out = np.empty(E.shape, dtype=float)
        above = E >= V0
        below = ~above

        if np.any(above):
            al = alpha[above]
            beta = np.sqrt(np.maximum(2.0 * M_E * (E[above] - V0), 0.0)) / HBAR
            small = beta < 1e-12
            lhs = np.empty_like(al)
            if np.any(~small):
                als = al[~small]
                bts = beta[~small]
                lhs[~small] = (
                    -(als**2 + bts**2)
                    / (2.0 * als * bts)
                    * np.sin(als * a)
                    * np.sin(bts * b)
                    + np.cos(als * a) * np.cos(bts * b)
                )
            if np.any(small):
                als = al[small]
                lhs[small] = np.cos(als * a) - 0.5 * als * b * np.sin(als * a)
            out[above] = lhs

        if np.any(below):
            al = alpha[below]
            gamma = np.sqrt(np.maximum(2.0 * M_E * (V0 - E[below]), 0.0)) / HBAR
            small = gamma < 1e-12
            lhs = np.empty_like(al)
            if np.any(~small):
                als = al[~small]
                gs = gamma[~small]
                gb = np.clip(gs * b, -80.0, 80.0)
                with np.errstate(over="ignore", invalid="ignore"):
                    lhs[~small] = (
                        (gs**2 - als**2)
                        / (2.0 * als * gs)
                        * np.sin(als * a)
                        * np.sinh(gb)
                        + np.cos(als * a) * np.cosh(gb)
                    )
            if np.any(small):
                als = al[small]
                lhs[small] = np.cos(als * a) - 0.5 * als * b * np.sin(als * a)
            out[below] = lhs

        out = np.where(np.isfinite(out), out, np.inf)
        return float(out[0]) if scalar else out

    def kp_band_segments(E, lhs):
        allowed = np.abs(lhs) <= 1.0
        if not np.any(allowed):
            return []
        padded = np.concatenate(([False], allowed, [False]))
        d = np.diff(padded.astype(int))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0] - 1
        return list(zip(starts.tolist(), ends.tolist()))

    def kp_dispersion(V0_eV, a_nm, b_nm, E_max=12.0, nE=2500):
        E = np.linspace(0.02, E_max, nE)
        lhs = kp_lhs(E, V0_eV, a_nm, b_nm)
        segments = kp_band_segments(E, lhs)
        reduced = []
        extended = []
        for n, (s, e) in enumerate(segments):
            Es = E[s : e + 1]
            theta = np.arccos(np.clip(lhs[s : e + 1], -1.0, 1.0))
            kL_red = np.concatenate((-theta[::-1], theta))
            E_red = np.concatenate((Es[::-1], Es))
            reduced.append((kL_red, E_red))
            if n % 2 == 0:
                kL_pos = n * np.pi + theta
            else:
                kL_pos = n * np.pi + (np.pi - theta)
            kL_ext = np.concatenate((-kL_pos[::-1], kL_pos))
            E_ext = np.concatenate((Es[::-1], Es))
            extended.append((kL_ext, E_ext))
        gaps = []
        for i in range(len(segments) - 1):
            e_hi = E[segments[i][1]]
            e_lo = E[segments[i + 1][0]]
            gaps.append((e_hi, e_lo, e_lo - e_hi))
        return E, lhs, segments, reduced, extended, gaps

    def kp_Ek_bands(V0_eV, a_nm, b_nm, kL_max, nk=250, nE=2000, E_max=12.5):
        """Allowed E(k) from LHS(E) = cos(k(a+b)), same construction as ECE350."""
        E = np.linspace(0.01, E_max, nE)
        lhs = kp_lhs(E, V0_eV, a_nm, b_nm)
        kL = np.linspace(-kL_max, kL_max, nk)
        diff = lhs[:, None] - np.cos(kL)[None, :]
        sign = np.where(np.isfinite(diff), np.sign(diff), 0.0)
        i_at, j_at = np.nonzero((sign[:-1] * sign[1:]) < 0)
        if i_at.size == 0:
            return []
        d1 = diff[i_at, j_at]
        d2 = diff[i_at + 1, j_at]
        denom = d2 - d1
        ok = np.abs(denom) > 1e-30
        E_c = np.empty(i_at.shape, dtype=float)
        E_c[ok] = (
            E[i_at[ok]]
            - d1[ok] * (E[i_at[ok] + 1] - E[i_at[ok]]) / denom[ok]
        )
        E_c[~ok] = E[i_at[~ok]]
        order = np.argsort(j_at * (nE + 1) + i_at, kind="mergesort")
        j_sorted = j_at[order]
        new_group = np.ones(i_at.size, dtype=bool)
        new_group[1:] = j_sorted[1:] != j_sorted[:-1]
        idx = np.arange(i_at.size)
        rank_sorted = idx - np.maximum.accumulate(np.where(new_group, idx, 0))
        rank = np.empty(i_at.size, dtype=int)
        rank[order] = rank_sorted
        bands = []
        for b in range(min(int(rank.max()) + 1, 7)):
            sel = rank == b
            kk = kL[j_at[sel]]
            ee = E_c[sel]
            srt = np.argsort(kk)
            bands.append((kk[srt], ee[srt]))
        return bands

    def bloch_components(k_pi, sigma_frac, n_cells=8, n_pts=1200):
        """ψ(x) = u(x) exp(jkx) with a lattice-periodic Gaussian u(x)."""
        a = 1.0
        k = k_pi * np.pi / a
        x = np.linspace(0.0, n_cells * a, n_pts)
        plane = np.exp(1j * k * x)
        sigma = sigma_frac * a
        u = np.zeros_like(x)
        for site in range(-1, n_cells + 2):
            u += np.exp(-((x - site * a) ** 2) / (2.0 * sigma**2))
        u = u / np.max(u)
        psi = plane * u
        return x, u, plane.real, psi.real, psi.imag, np.abs(psi) ** 2

    def parabolic_bands(mn_over_m, mp_over_m, k_max=2.0, n_pts=400):
        """Parabolic edges E=Ec+ħ²k²/(2m_n*) and E=Ev−ħ²k²/(2m_p*), k in nm⁻¹."""
        k = np.linspace(-k_max, k_max, n_pts)
        Ec, Ev = 1.12, 0.0
        Ec_k = Ec + HBAR2_2M_EVNM2 * k**2 / mn_over_m
        Ev_k = Ev - HBAR2_2M_EVNM2 * k**2 / mp_over_m
        return k, Ec_k, Ev_k, Ec, Ev, k_max

    def cosine_bands(ka):
        """Model CB minimum and VB maximum at k = 0 (Neamen Fig. 3.16)."""
        Ec = 1.2 + 0.45 * (1.0 - np.cos(ka))
        Ev = 0.0 - 0.30 * (1.0 - np.cos(ka))
        vg_c = 0.45 * np.sin(ka)
        vg_v = -0.30 * np.sin(ka)
        return Ec, Ev, vg_c, vg_v

    def occupy_and_current(N, n_e, n_h, n_shift):
        n = np.arange(-N // 2, N // 2)
        ka = 2.0 * np.pi * n / N
        Ec, Ev, vg_c, vg_v = cosine_bands(ka)
        cb_filled = np.zeros(N, dtype=bool)
        vb_filled = np.ones(N, dtype=bool)
        cb_order = np.argsort(Ec)
        vb_order = np.argsort(-Ev)
        cb_filled[cb_order[:n_e]] = True
        vb_filled[vb_order[:n_h]] = False
        cb_filled = np.roll(cb_filled, n_shift)
        vb_filled = np.roll(vb_filled, n_shift)
        I_e = -np.sum(vg_c[cb_filled])
        I_h = np.sum(vg_v[~vb_filled])
        return ka, Ec, Ev, vg_c, vg_v, cb_filled, vb_filled, I_e, I_h

    return (
        BLUE,
        GOLD,
        GREEN,
        GRAY,
        HBAR2_2M_EVNM2,
        NAVY,
        ORANGE,
        bloch_components,
        cosine_bands,
        kp_Ek_bands,
        kp_dispersion,
        kp_lhs,
        occupy_and_current,
        parabolic_bands,
        plt,
        style_ax,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Energy Bands
    **ECE335 - companion to lectures**

    Use these activities alongside lectures and Neamen Chapter 3, §§3.1–3.3.
    Before moving a slider, predict what will change. Then open the explanation.

    Notation follows Neamen: $V(x)$ is potential energy, $j^2=-1$, and
    $\psi(x)=u(x)e^{jkx}$. The electron and hole effective masses are $m_n^*$ and
    $m_p^*$. Band-edge energies are $E_c$ and $E_v$; the gap is $E_g=E_c-E_v$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Periodic potential in a crystal

    Isolated atoms have discrete energies. In a crystal the ion-core potential is
    periodic, $V(x+a)=V(x)$. Overlapping wells split those levels into **allowed
    bands** separated by **forbidden gaps** (Neamen §3.1.1).
    """)
    return


@app.cell
def _(BLUE, GOLD, NAVY, np, plt, style_ax):
    _a = 1.0
    _n_per = 5
    _x = np.linspace(-0.5 * _a, (_n_per - 0.5) * _a, 2500)
    _V = np.zeros_like(_x)
    for _n in range(-12, _n_per + 12):
        _d = np.maximum(np.abs(_x - _n * _a), 0.06)
        _V += -2.2 / _d
    _V = np.clip(_V, -32.0, 0.0)

    _fig, _ax = plt.subplots(figsize=(8.4, 4.2), layout="constrained")
    _ax.plot(_x, _V, color=BLUE, lw=2.4, label=r"$V(x)$")
    _ax.fill_between(_x, _V, -32.0, color=BLUE, alpha=0.12)
    for _n in range(_n_per):
        _ax.axvline(_n * _a, color=GOLD, ls="--", lw=1.1, alpha=0.7)
        _ax.plot(_n * _a, -26.0, "o", color=GOLD, ms=10, zorder=5)
    _ax.annotate(
        "",
        xy=(_a, -16.0),
        xytext=(0.0, -16.0),
        arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.8),
    )
    _ax.text(0.5 * _a, -14.6, r"$a$", ha="center", color=NAVY)
    _ax.set_xlim(_x[0], _x[-1])
    _ax.set_ylim(-30.0, -6.0)
    _ax.set_xlabel(r"position $x$  (units of $a$)")
    _ax.set_ylabel(r"$V(x)$")
    _ax.set_yticklabels([])
    _ax.legend(frameon=False, loc="upper right")
    style_ax(_ax)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Bloch wavefunction

    **Predict:** If $k=0$, how do $\operatorname{Re}\psi$, $\operatorname{Im}\psi$,
    and $|\psi|^2$ compare? If you make $u(x)$ more localized on each ion, does
    $|\psi|^2$ stay periodic with $a$?

    Bloch's theorem (Neamen Eq. 3.1): if $V(x+a)=V(x)$, every one-electron
    eigenfunction can be written

    $$
    \psi(x)=u(x)\,e^{jkx},\qquad u(x+a)=u(x).
    $$

    Equivalently, $\psi(x+a)=e^{jka}\psi(x)$. Here $k$ is the wave number;
    $\hbar k$ is the **crystal momentum**, a constant of the motion that includes
    the lattice interaction. In this illustration $u(x)$ is a sum of Gaussians
    on the lattice sites and is held independent of $k$.
    """)
    return


@app.cell
def _(mo):
    bloch_k = mo.ui.slider(
        start=-1.0,
        stop=1.0,
        value=0.35,
        step=0.05,
        show_value=True,
        label="wave number k (units of π/a)",
    )
    bloch_sigma = mo.ui.slider(
        start=0.08,
        stop=0.40,
        value=0.16,
        step=0.02,
        show_value=True,
        label="u(x) width σ/a",
    )
    bloch_controls = mo.hstack([bloch_k, bloch_sigma], justify="start", gap=1.5)
    bloch_controls
    return bloch_k, bloch_sigma


@app.cell
def _(
    BLUE,
    GOLD,
    GREEN,
    NAVY,
    ORANGE,
    bloch_components,
    bloch_k,
    bloch_sigma,
    mo,
    plt,
    style_ax,
):
    _k_pi = float(bloch_k.value)
    _sig = float(bloch_sigma.value)
    _x, _u, _coskx, _re, _im, _prob = bloch_components(_k_pi, _sig)

    _fig, _axs = plt.subplots(3, 1, figsize=(8.4, 8.2), layout="constrained", sharex=True)
    _axs[0].plot(_x, _coskx, color=BLUE, lw=2.0, ls="--", label=r"$\cos(kx)$")
    _axs[0].plot(_x, _u, color=GREEN, lw=2.4, label=r"$u(x)$")
    _axs[0].set_ylabel("amplitude")
    _axs[0].set_ylim(-1.25, 1.35)
    _axs[0].legend(frameon=False, loc="upper right", ncol=2)

    _axs[1].plot(_x, _re, color=ORANGE, lw=2.4, label=r"$\mathrm{Re}\,\psi$")
    _axs[1].plot(_x, _im, color=GOLD, lw=2.2, ls=":", label=r"$\mathrm{Im}\,\psi$")
    _axs[1].set_ylabel(r"$\psi(x)$")
    _axs[1].set_ylim(-1.35, 1.35)
    _axs[1].legend(frameon=False, loc="upper right", ncol=2)

    _axs[2].fill_between(_x, _prob, color=NAVY, alpha=0.25)
    _axs[2].plot(_x, _prob, color=NAVY, lw=2.4, label=r"$|\psi|^2$")
    _axs[2].set_ylabel(r"$|\psi(x)|^2$")
    _axs[2].set_xlabel(r"position $x$  (units of $a$)")
    _axs[2].set_ylim(0.0, 1.35)
    _axs[2].legend(frameon=False, loc="upper right")

    for _ax, _y0, _y1 in ((_axs[0], -1.25, 1.35), (_axs[1], -1.35, 1.35), (_axs[2], 0.0, 1.35)):
        for _n in range(9):
            _ax.axvline(_n, color="#bbbbbb", ls=":", lw=1.0)
        style_ax(_ax)
        _ax.set_xlim(0.0, 8.0)

    _info = mo.md(
        rf"""
    $k={_k_pi:.2f}\,\pi/a$. One lattice step multiplies $\psi$ by
    $e^{{jka}}=\cos(ka)+j\sin(ka)$ with $ka={_k_pi:.2f}\pi$.

    $|\psi|^2=|u|^2$ is independent of $k$ in this model, because $u(x)$ was
    not allowed to change with $k$. The probability still repeats every $a$.
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
    At $k=0$, $e^{jkx}=1$, so $\psi=u(x)$ is real and $\operatorname{Im}\psi=0$.
    Finite $k$ wraps a travelling-wave phase onto the same periodic $u(x)$.
    Narrower Gaussians (smaller $\sigma/a$) concentrate $|\psi|^2$ on the ions,
    as for a tightly bound electron. Broader $u(x)$ approaches a plane wave.
    The lattice period of $|\psi|^2$ is required by Bloch's theorem, not by the
    particular Gaussian shape used here.
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
    ## Kronig–Penney model

    **Predict:** If you raise the barrier $V_0$, do the forbidden gaps get wider
    or narrower? If $V_0\to 0$, should $E(k)$ approach the free-electron parabola
    $E=\hbar^2k^2/2m$?

    Neamen §3.1.2 replaces the crystal potential by a periodic square wave:
    $V=0$ on $0<x<a$ (region I) and $V=V_0$ on $-b<x<0$ (region II). The period
    is $a+b$. Matching $\psi$ and $d\psi/dx$, together with Bloch's condition
    on $u(x)$, gives a nontrivial solution only when (Neamen Eqs. 3.19 and 3.21)

    $$
    E\ge V_0:\quad
    -\frac{\alpha^2+\beta^2}{2\alpha\beta}\sin(\alpha a)\sin(\beta b)
    +\cos(\alpha a)\cos(\beta b)
    =\cos k(a+b),
    $$

    $$
    E<V_0:\quad
    \frac{\gamma^2-\alpha^2}{2\alpha\gamma}\sin(\alpha a)\sinh(\gamma b)
    +\cos(\alpha a)\cosh(\gamma b)
    =\cos k(a+b),
    $$

    with $\alpha=\sqrt{2mE}/\hbar$, $\beta=\sqrt{2m(E-V_0)}/\hbar$, and
    $\gamma=\sqrt{2m(V_0-E)}/\hbar$. Because $|\cos k(a+b)|\le 1$, only energies
    whose left-hand side lies in $[-1,1]$ are allowed. The first Brillouin zone
    is $-\pi/(a+b)\le k\le\pi/(a+b)$ (reduced-zone plot).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualization with Interactive Parameters
    """)
    return


@app.cell
def _(mo):
    kp_V0 = mo.ui.slider(
        start=0.05,
        stop=8.0,
        value=3.0,
        step=0.1,
        show_value=True,
        label="Barrier Height V₀ (eV)",
    )
    kp_a = mo.ui.slider(
        start=0.1,
        stop=1.0,
        value=0.4,
        step=0.01,
        show_value=True,
        label="Well Width a (nm)",
    )
    kp_b = mo.ui.slider(
        start=0.1,
        stop=1.0,
        value=0.1,
        step=0.01,
        show_value=True,
        label="Barrier Width b (nm)",
    )
    return kp_V0, kp_a, kp_b


@app.cell
def _(
    BLUE,
    GOLD,
    GREEN,
    NAVY,
    ORANGE,
    go,
    kp_Ek_bands,
    kp_V0,
    kp_a,
    kp_b,
    kp_lhs,
    make_subplots,
    mo,
    np,
):
    _V0 = float(kp_V0.value)
    _a = float(kp_a.value)
    _b = float(kp_b.value)
    _period = _a + _b
    _energy_res = 2000
    _E_max_lhs = max(3.0 * _V0, 4.0)
    _E_max_bands = 12.5
    _band_colors = [BLUE, ORANGE, GREEN, GOLD, NAVY, "#56B4E9", "#CC79A7"]

    _fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=(
            "Periodic Potential",
            "Dispersion Function LHS",
            "E-k Diagram (Extended Zone) <br> k(a+b) ∈ [−2π, +2π]",
            "E-k Diagram (Reduced Zone) <br> k(a+b) ∈ [−π, +π]",
        ),
        horizontal_spacing=0.08,
        column_widths=[0.25, 0.25, 0.25, 0.25],
    )

    _num_periods = 5
    _x_total = _num_periods * _period
    _x = np.linspace(-_b, _x_total, 2000)
    _V = np.zeros_like(_x)
    for _i in range(_num_periods + 1):
        _x_start = _i * _period - _b
        _x_end = _i * _period
        _mask = (_x >= _x_start) & (_x < _x_end)
        _V[_mask] = _V0

    _fig.add_trace(
        go.Scatter(
            x=_x,
            y=_V,
            mode="lines",
            name="Potential V(x)",
            line=dict(color=BLUE, width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 114, 178, 0.20)",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    _fig.add_annotation(
        x=_a / 2,
        y=_V0 * 0.5,
        text=f"Region I<br>V=0<br>a={_a:.2f} nm",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        row=1,
        col=1,
    )
    _fig.add_annotation(
        x=-_b / 2 + 2 * _period,
        y=_V0 * 1.15,
        text=f"Region II<br>V₀={_V0:.1f} eV<br>b={_b:.2f} nm",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        row=1,
        col=1,
    )

    _E_values = np.linspace(0.01, _E_max_lhs, _energy_res)
    _LHS_values = kp_lhs(_E_values, _V0, _a, _b)
    _LHS_plot = np.array(_LHS_values, dtype=float)
    _LHS_plot[np.abs(_LHS_plot) > 8] = np.nan
    _fig.add_trace(
        go.Scatter(
            x=_E_values,
            y=_LHS_plot,
            mode="lines",
            name="LHS",
            line=dict(color=BLUE, width=2),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    _fig.add_hline(y=1, line_dash="dash", line_color=ORANGE, row=1, col=2)
    _fig.add_hline(y=-1, line_dash="dash", line_color=ORANGE, row=1, col=2)
    _fig.add_vline(x=_V0, line_dash="dot", line_color="#CC79A7", row=1, col=2)

    _allowed_mask = np.abs(_LHS_values) <= 1.0
    _bands_found = []
    _in_band = False
    _band_start = None
    for _i, _allowed in enumerate(_allowed_mask):
        if _allowed and not _in_band:
            _band_start = _E_values[_i]
            _in_band = True
        elif (not _allowed) and _in_band:
            _bands_found.append((_band_start, _E_values[_i - 1]))
            _in_band = False
    if _in_band:
        _bands_found.append((_band_start, _E_values[-1]))
    _shade = [
        "rgba(0,158,115,0.25)",
        "rgba(0,114,178,0.25)",
        "rgba(230,159,0,0.25)",
        "rgba(213,94,0,0.25)",
        "rgba(15,76,129,0.25)",
    ]
    for _i, (_E_start, _E_end) in enumerate(_bands_found[:5]):
        _fig.add_vrect(
            x0=_E_start,
            x1=_E_end,
            fillcolor=_shade[_i % len(_shade)],
            opacity=0.5,
            layer="below",
            line_width=0,
            row=1,
            col=2,
        )

    _ext_bands = kp_Ek_bands(_V0, _a, _b, 2.0 * np.pi, nk=400, nE=2000, E_max=_E_max_bands)
    for _i, (_k_band, _E_band) in enumerate(_ext_bands):
        _fig.add_trace(
            go.Scatter(
                x=_k_band,
                y=_E_band,
                mode="lines",
                name=f"Band {_i + 1}",
                line=dict(width=2, color=_band_colors[_i % len(_band_colors)]),
                showlegend=False,
            ),
            row=1,
            col=3,
        )
    _fig.add_hline(y=_V0, line_dash="dash", line_color=ORANGE, row=1, col=3)
    for _tick in (-2 * np.pi, -np.pi, 0.0, np.pi, 2 * np.pi):
        _fig.add_vline(x=_tick, line_dash="dot", line_color="gray", row=1, col=3)

    _red_bands = kp_Ek_bands(_V0, _a, _b, np.pi, nk=200, nE=2000, E_max=_E_max_bands)
    for _i, (_k_band, _E_band) in enumerate(_red_bands):
        _fig.add_trace(
            go.Scatter(
                x=_k_band,
                y=_E_band,
                mode="lines",
                name=f"Band {_i + 1}",
                line=dict(width=2, color=_band_colors[_i % len(_band_colors)]),
                showlegend=False,
            ),
            row=1,
            col=4,
        )
    _fig.add_hline(y=_V0, line_dash="dash", line_color=ORANGE, row=1, col=4)
    _fig.add_vline(x=-np.pi, line_dash="dot", line_color="gray", row=1, col=4)
    _fig.add_vline(x=np.pi, line_dash="dot", line_color="gray", row=1, col=4)

    _fig.update_xaxes(title_text="Position x (nm)", row=1, col=1)
    _fig.update_yaxes(title_text="V(x) (eV)", row=1, col=1)
    _fig.update_xaxes(title_text="Energy E (eV)", row=1, col=2)
    _fig.update_yaxes(title_text="LHS", range=[-3, 3], row=1, col=2)
    _fig.update_xaxes(title_text="k(a+b)", row=1, col=3)
    _fig.update_yaxes(title_text="Energy E (eV)", range=[0, 12], row=1, col=3)
    _fig.update_xaxes(title_text="k(a+b)", row=1, col=4)
    _fig.update_yaxes(title_text="Energy E (eV)", range=[0, 12], row=1, col=4)
    _fig.update_layout(
        height=480,
        font=dict(size=14),
        margin=dict(t=80, b=50, l=50, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    mo.vstack(
        [
            mo.hstack([kp_V0, kp_a, kp_b], justify="start", gap=1.2),
            mo.ui.plotly(_fig),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Explorations

    - How do the bandgaps change with barrier height $V_0$? Does a higher barrier
      lead to a larger or smaller gap?
    - Does reducing the barrier height and/or width recover the free-electron
      parabola $E=\hbar^2k^2/2m$?
    - Do the allowed energy ranges get wider or narrower as $E$ increases?
      Is that reasonable?
    - How do the bands change with well width $a$ and barrier width $b$?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Effective mass

    **Predict:** Which $E(k)$ curve — the flatter one or the more curved one —
    belongs to the **heavier** carrier? At the top of the valence band, is the
    electron $m^*$ positive or negative?

    Near a band edge Neamen approximates $E(k)$ by a parabola (Fig. 3.16):

    $$
    E-E_c=C_1 k^2,\qquad E-E_v=-C_2 k^2,
    $$

    with $C_1,C_2>0$. Comparing with the free-electron result
    $E=\hbar^2k^2/2m$ gives

    $$
    \frac{1}{m^*}=\frac{1}{\hbar^2}\frac{d^2E}{dk^2}.
    $$

    At the conduction-band minimum, $d^2E/dk^2>0$ so $m_n^*>0$ and
    $a=-q\mathcal{E}/m_n^*$ (slides write $q$ for Neamen's $e$). At the
    valence-band maximum, $d^2E/dk^2<0$, so the **electron** effective mass is
    negative. Empty valence states are described instead as **holes** with
    charge $+q$ and $m_p^*=|m^*|>0$.
    """)
    return


@app.cell
def _(mo):
    mn_slider = mo.ui.slider(
        start=0.05,
        stop=1.50,
        value=0.26,
        step=0.01,
        show_value=True,
        label="mₙ*/m₀  (electron)",
    )
    mp_slider = mo.ui.slider(
        start=0.08,
        stop=2.00,
        value=0.49,
        step=0.01,
        show_value=True,
        label="mₚ*/m₀  (hole)",
    )
    mass_controls = mo.hstack([mn_slider, mp_slider], justify="start", gap=1.5)
    mass_controls
    return mn_slider, mp_slider


@app.cell
def _(
    BLUE,
    GOLD,
    GREEN,
    NAVY,
    ORANGE,
    mn_slider,
    mo,
    mp_slider,
    parabolic_bands,
    plt,
    style_ax,
):
    _mn = float(mn_slider.value)
    _mp = float(mp_slider.value)
    _k, _Ec_k, _Ev_k, _Ec, _Ev, _kmax = parabolic_bands(_mn, _mp)
    _k_heavy, _Ec_h, _Ev_h, _, _, _ = parabolic_bands(1.2, 1.2)
    _k_light, _Ec_l, _Ev_l, _, _, _ = parabolic_bands(0.08, 0.16)

    _fig, _axs = plt.subplots(1, 2, figsize=(10.4, 4.6), layout="constrained", sharey=True)

    _axs[0].plot(_k_heavy, _Ec_h, color=BLUE, lw=2.0, alpha=0.35)
    _axs[0].plot(_k_heavy, _Ev_h, color=ORANGE, lw=2.0, alpha=0.35)
    _axs[0].plot(_k, _Ec_k, color=BLUE, lw=2.6, label=rf"$E_c(k)$, $m_n^*={_mn:.2f}m_0$")
    _axs[0].plot(_k, _Ev_k, color=ORANGE, lw=2.6, label=rf"$E_v(k)$, $m_p^*={_mp:.2f}m_0$")
    _axs[0].axhline(_Ec, color=GREEN, ls=":", lw=1.3)
    _axs[0].axhline(_Ev, color=GOLD, ls=":", lw=1.3)
    _axs[0].set_title("selected masses")
    _axs[0].legend(frameon=False, loc="upper right", fontsize=13)

    _axs[1].plot(_k_light, _Ec_l, color=BLUE, lw=2.5, label=r"light $m_n^*=0.08m_0$")
    _axs[1].plot(_k_light, _Ev_l, color=ORANGE, lw=2.5, label=r"light $m_p^*=0.16m_0$")
    _axs[1].plot(_k_heavy, _Ec_h, color=NAVY, lw=2.5, ls="--", label=r"heavy $m_n^*=1.2m_0$")
    _axs[1].plot(_k_heavy, _Ev_h, color=GOLD, lw=2.5, ls="--", label=r"heavy $m_p^*=1.2m_0$")
    _axs[1].set_title("curvature comparison")
    _axs[1].legend(frameon=False, loc="upper right", fontsize=13)

    for _ax in _axs:
        _ax.set_xlabel(r"$k$ (nm$^{-1}$)")
        _ax.set_ylabel(r"$E$ (eV)")
        _ax.set_xlim(-_kmax, _kmax)
        _ax.set_ylim(-3.3, 3.6)
        _ax.axvline(0.0, color="#cccccc", lw=1.0)
        style_ax(_ax)

    _Eg = _Ec - _Ev
    _info = mo.md(
        rf"""
    Silicon-like defaults: $m_n^*\approx 0.26m_0$, $m_p^*\approx 0.49m_0$
    (density-of-states masses, order of magnitude).

    $$
    E(k)=E_c+\frac{{\hbar^2 k^2}}{{2m_n^*}}
    ={_Ec:.2f}+\frac{{\hbar^2 k^2}}{{2({_mn:.2f}\,m_0)}}\ \mathrm{{eV}},
    \qquad
    E(k)=E_v-\frac{{\hbar^2 k^2}}{{2m_p^*}}.
    $$

    $E_g={_Eg:.2f}$ eV in this sketch.  Larger curvature $\Leftrightarrow$
    smaller $|m^*|$.
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
    $m^*\propto 1/(d^2E/dk^2)$. A flatter band (small curvature) is a heavy
    carrier; a sharply curved band is a light carrier. That is why the left
    panel gets steeper when you lower $m_n^*$ or $m_p^*$. Holes live near
    $E_v$, where the electron mass would be negative; assigning charge $+q$
    and $m_p^*=-m^*_{\mathrm{electron}}>0$ restores Newton's law
    $a=+q\mathcal{E}/m_p^*$ for the empty states.
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
    ## Band filling and conduction

    **Predict:** Can a completely full valence band carry current? If you shift
    the occupied $k$ states to one side (as an electric field does), which way
    does the electron current go?

    Each $k$ point in a 1D crystal of $N$ cells is $k=2\pi n/(Na)$ with
    $ka\in(-\pi,+\pi]$ (periodic boundary conditions). Two spin states belong
    to each $k$. The group velocity is Neamen Eq. (3.39),

    $$
    v=\frac{1}{\hbar}\frac{dE}{dk}.
    $$

    A full or $k$-symmetric occupation has $\sum v_i=0$. For a nearly full
    band Neamen Eq. (3.52) rewrites the current as a sum over **empty** states
    with charge $+q$: those are holes. An applied field makes the occupation
    asymmetric in $k$ (Neamen Fig. 3.15).
    """)
    return


@app.cell
def _(mo):
    fill_e = mo.ui.slider(
        start=0,
        stop=8,
        value=3,
        step=1,
        show_value=True,
        label="conduction electrons N_e",
    )
    fill_h = mo.ui.slider(
        start=0,
        stop=8,
        value=3,
        step=1,
        show_value=True,
        label="valence holes N_h",
    )
    fill_shift = mo.ui.slider(
        start=-4,
        stop=4,
        value=0,
        step=1,
        show_value=True,
        label="occupation shift Δn (field)",
    )
    fill_controls = mo.hstack([fill_e, fill_h, fill_shift], justify="start", gap=1.2)
    fill_controls
    return fill_e, fill_h, fill_shift


@app.cell
def _(
    BLUE,
    GOLD,
    GREEN,
    NAVY,
    ORANGE,
    cosine_bands,
    fill_e,
    fill_h,
    fill_shift,
    mo,
    np,
    occupy_and_current,
    plt,
    style_ax,
):
    _N = 16
    _ne = int(fill_e.value)
    _nh = int(fill_h.value)
    _dn = int(fill_shift.value)
    _ka, _Ec, _Ev, _vgc, _vgv, _cbf, _vbf, _Ie, _Ih = occupy_and_current(
        _N, _ne, _nh, _dn
    )
    _ka_c = np.linspace(-np.pi, np.pi, 400)
    _Ec_c, _Ev_c, _, _ = cosine_bands(_ka_c)
    _Itot = _Ie + _Ih

    _fig, _axs = plt.subplots(1, 2, figsize=(10.6, 4.8), layout="constrained")

    _axs[0].plot(_ka_c / np.pi, _Ec_c, color=BLUE, lw=2.2)
    _axs[0].plot(_ka_c / np.pi, _Ev_c, color=ORANGE, lw=2.2)
    _axs[0].plot(_ka[_cbf] / np.pi, _Ec[_cbf], "o", color=BLUE, ms=9, label="filled CB")
    _axs[0].plot(
        _ka[~_cbf] / np.pi,
        _Ec[~_cbf],
        "o",
        ms=9,
        markerfacecolor="white",
        markeredgecolor=BLUE,
        markeredgewidth=1.6,
        label="empty CB",
    )
    _axs[0].plot(_ka[_vbf] / np.pi, _Ev[_vbf], "o", color=ORANGE, ms=9, label="filled VB")
    _axs[0].plot(
        _ka[~_vbf] / np.pi,
        _Ev[~_vbf],
        "o",
        ms=9,
        markerfacecolor="white",
        markeredgecolor=ORANGE,
        markeredgewidth=1.6,
        label="empty VB (holes)",
    )
    _axs[0].set_xlabel(r"$k$  (units of $\pi/a$)")
    _axs[0].set_ylabel(r"$E$ (arb. units)")
    _axs[0].set_title("occupation of $E(k)$")
    _axs[0].set_xlim(-1.15, 1.15)
    _axs[0].set_ylim(-1.0, 2.5)
    _axs[0].legend(frameon=False, loc="upper right", fontsize=13)

    _w = 0.22
    _axs[1].bar([-0.6], [_Ie], width=_w, color=BLUE, label=r"electrons $I\propto -q\sum v$")
    _axs[1].bar([0.0], [_Ih], width=_w, color=ORANGE, label=r"holes $I\propto +q\sum_{\mathrm{empty}} v$")
    _axs[1].bar([0.6], [_Itot], width=_w, color=GREEN, label="net")
    _axs[1].axhline(0.0, color=NAVY, lw=1.0)
    _axs[1].set_xticks([-0.6, 0.0, 0.6], [r"$I_n$", r"$I_p$", r"$I$"])
    _axs[1].set_ylabel("current (arb. units)")
    _axs[1].set_title("drift current")
    _axs[1].set_xlim(-1.1, 1.1)
    _axs[1].set_ylim(-4.6, 4.6)
    _axs[1].legend(frameon=False, loc="upper right", fontsize=13)

    if _dn != 0:
        _axs[0].annotate(
            r"field shifts occupation",
            xy=(0.55 if _dn > 0 else -0.55, 1.55),
            fontsize=14,
            color=GOLD,
            ha="center",
        )

    for _ax in _axs:
        style_ax(_ax)

    if abs(_Itot) < 1e-12:
        _status = "Net current is zero: every $+k$ contribution is cancelled by $-k$."
    elif _Itot < 0:
        _status = (
            "Net current is negative: more occupied CB weight at $k>0$ "
            r"($v>0$ electrons contribute $I_n<0$), and/or holes on the $k>0$ side."
        )
    else:
        _status = "Net current is positive: the $k<0$ side carries more electron occupation."

    _info = mo.md(
        rf"""
    $N={_N}$ cells, $N_e={_ne}$ electrons in the conduction band,
    $N_h={_nh}$ holes in the valence band, shift $\Delta n={_dn}$.

    Dimensionless currents from $v\propto dE/d(ka)$:
    $I_n={_Ie:.2f}$, $I_p={_Ih:.2f}$, $I={_Itot:.2f}$.

    {_status}
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
    Set $N_e=N_h=0$: both bands are full or empty, so $I=0$ for any shift
    (a full band cannot rearrange). Restore some carriers and keep
    $\Delta n=0$: electrons sit at the CB minimum ($k=0$) and holes at the
    VB maximum ($k=0$), still symmetric, so $I=0$. A nonzero shift models
    Neamen Fig. 3.15. Electrons with $v=(1/\hbar)dE/dk$ contribute
    $I_n=-q\sum v_i$. Empty valence states contribute as holes,
    $I_p=+q\sum_{\mathrm{empty}}v_i$. A completely filled or symmetric
    band cannot carry current; a metal has a partly filled band that a
    field can displace.
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

    Explain each result before opening the answers.

    1. Why is $|\psi(x)|^2$ periodic with $a$ even when $k\neq 0$?
    2. In Kronig–Penney, where do the gaps open, and what happens as $V_0\to 0$?
    3. Which valence band has the larger hole mass $m_p^*$: a flat maximum or a
       sharply curved one?
    4. Why does a completely full band give $J=0$ even if every electron is moving?
    5. An applied field shifts electrons toward $+k$ in the conduction band.
       What is the sign of the electron current $I_n$?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers and suggested checks": mo.md(
                r"""
    1. $\psi=u(x)e^{jkx}$ with $u(x+a)=u(x)$, so
       $|\psi|^2=|u|^2$ is lattice-periodic. Set $k=0$ in the Bloch activity:
       $\operatorname{Im}\psi=0$ and $\psi=u$.
    2. Gaps open at the zone boundaries $k(a+b)=n\pi$, where Bragg scattering
       is strongest. Lower $V_0$ in the Kronig–Penney activity until the
       bands hug the dashed free-electron parabola.
    3. The flatter maximum. $m_p^*\propto 1/|d^2E/dk^2|$. Compare the two
       $m_p^*$ values in the effective-mass activity.
    4. $E(k)=E(-k)$ implies $v(-k)=-v(k)$. Every state is occupied, so the
       sum of velocities is identically zero (Neamen Eq. 3.51). Fill the
       valence band ($N_h=0$) and move the field slider: $I_p$ stays zero.
    5. For the conduction-band minimum, $v$ has the sign of $k$, and
       $I_n=-q\sum v_i$, so extra occupation at $+k$ gives $I_n<0$.
       Set $N_e>0$, $N_h=0$, and $\Delta n>0$ to see this.
    """
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
