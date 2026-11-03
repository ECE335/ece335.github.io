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
    from matplotlib.gridspec import GridSpec

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

    # Silicon and SiO2 at 300 K, as on the slides and in Neamen Ch. 10.
    VT = 0.0259
    NI = 1.5e10
    E_CHARGE = 1.6e-19
    EPS0 = 8.85e-14
    EPS_S = 11.7 * EPS0
    EPS_OX = 3.9 * EPS0
    EG = 1.12
    CHI_P = 3.25  # modified electron affinity chi' for Si-SiO2 (V)
    OX_DRAW = 1.0  # oxide drawn to scale, so the oxide field shows as a steeper E_c (oxide)

    BLUE = "#0072B2"
    ORANGE = "#D55E00"
    GREEN = "#009E73"
    NAVY = "#0f4c81"
    GRAY = "#6B6B6B"
    PINK = "#CC79A7"
    METAL_FILL = "#b9c4d2"
    OX_FILL = "#d9dee6"

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

    def energy_arrow(ax, x, E_lo, E_hi, label, color, side="right", fontsize=15):
        if abs(E_hi - E_lo) < 1e-3:
            return
        ax.annotate(
            "",
            xy=(x, E_hi),
            xytext=(x, E_lo),
            arrowprops=dict(arrowstyle="<->", color=color, lw=1.5, shrinkA=0, shrinkB=0),
        )
        _dx = 0.012 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        ax.text(
            x + (_dx if side == "right" else -_dx),
            0.5 * (E_lo + E_hi),
            label,
            color=color,
            fontsize=fontsize,
            ha="left" if side == "right" else "right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
        )

    # ---------------- Physics: exact 1D Poisson-Boltzmann, both carriers ----------------
    # phi(x) is the electrostatic potential relative to the neutral bulk (V); bands are E(x) = E(bulk) - e phi(x).
    # p = p0 exp(-phi/V_t), n = n0 exp(+phi/V_t).

    def bulk_carriers(body, N):
        maj = 0.5 * (N + np.sqrt(N**2 + 4.0 * NI**2))
        return (maj, NI**2 / maj) if body == "p" else (NI**2 / maj, maj)

    def field_of_phi(phi, p0, n0):
        """Electric field E(phi) (V/cm) from the first integral of Poisson's equation; sign(E) = sign(phi)."""
        u = np.asarray(phi, dtype=float) / VT
        g = p0 * (np.expm1(-u) + u) + n0 * (np.expm1(u) - u)
        return np.sign(u) * np.sqrt(2.0 * E_CHARGE * VT / EPS_S * np.maximum(g, 0.0))

    def cs_low_freq(phi, p0, n0):
        """Semiconductor capacitance per area dQ'_s/dphi_s when all carriers follow (low frequency)."""
        phi = np.asarray(phi, dtype=float)
        u = phi / VT
        LD = np.sqrt(EPS_S * VT / (E_CHARGE * (p0 + n0)))
        with np.errstate(divide="ignore", invalid="ignore"):
            c = E_CHARGE * (p0 * (-np.expm1(-u)) + n0 * np.expm1(u)) / field_of_phi(phi, p0, n0)
        return np.where(np.abs(u) < 1e-6, EPS_S / LD, c)

    def cs_high_freq(phi_end, p0, n0, body, n=1500):
        """High-frequency capacitance along 0 -> phi_end: majority carriers follow the signal, minority carriers do not.

        Small-signal Poisson equation with a frozen minority profile, as a Riccati equation for
        y = (d delta-phi/dx)/delta-phi along the dc profile, written in phi:
        dy/dphi = (k^2 - y^2)/(-E(phi)), k^2 = e p_maj(phi)/(eps_s V_t). Then C'_s = -eps_s y.
        """
        maj0, s_maj = (p0, -1.0) if body == "p" else (n0, 1.0)
        LD = np.sqrt(EPS_S * VT / (E_CHARGE * maj0))
        s = np.linspace(np.log(1e-6), np.log(abs(phi_end)), n)
        sgn = np.sign(phi_end)
        phis = sgn * np.exp(s)

        def rhs(ph, yy):
            return (np.exp(s_maj * ph / VT) / LD**2 - yy * yy) / (-field_of_phi(ph, p0, n0)) * ph

        y = -1.0 / LD
        out = np.empty(n)
        out[0] = y
        for i in range(n - 1):
            h = s[i + 1] - s[i]
            ph_m = sgn * np.exp(s[i] + 0.5 * h)
            k1 = rhs(phis[i], y)
            k2 = rhs(ph_m, y + 0.5 * h * k1)
            k3 = rhs(ph_m, y + 0.5 * h * k2)
            k4 = rhs(phis[i + 1], y + h * k3)
            y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            out[i + 1] = y
        return phis, -EPS_S * out

    def potential_profile(phi_s, p0, n0, x_cm):
        """phi(x) for surface potential phi_s: x(phi) = int_phi^phi_s dphi'/E(phi'), with an exponential tail."""
        x_cm = np.asarray(x_cm, dtype=float)
        if abs(phi_s) < 1e-7:
            return np.zeros_like(x_cm)
        LD = np.sqrt(EPS_S * VT / (E_CHARGE * (p0 + n0)))
        sgn = np.sign(phi_s)
        a_min = min(1e-5, 0.5 * abs(phi_s))
        s = np.linspace(np.log(abs(phi_s)), np.log(a_min), 6000)
        a = np.exp(s)
        integrand = a / np.abs(field_of_phi(sgn * a, p0, n0))
        x_of = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * -np.diff(s))])
        mag = np.where(
            x_cm <= x_of[-1],
            np.interp(x_cm, x_of, a),
            a_min * np.exp(-(x_cm - x_of[-1]) / LD),
        )
        return sgn * mag

    def mos_params(body, N, tox_nm, phim_p, Qss_e):
        """Neamen Ch. 10: phi_f, x_dT, C_ox, phi_ms (10.15; n type 10.33a), V_FB (10.25), V_TN (10.31c), V_TP (10.32)."""
        p0, n0 = bulk_carriers(body, N)
        Cox = EPS_OX / (tox_nm * 1e-7)
        phif = VT * np.log(N / NI)
        xdT = np.sqrt(4.0 * EPS_S * phif / (E_CHARGE * N))
        QSDmax = E_CHARGE * N * xdT
        # Semiconductor term: energy from E_Fs to the oxide conduction band (V).
        phims_semi = CHI_P + EG / 2.0 + phif if body == "p" else CHI_P + EG / 2.0 - phif
        phims = phim_p - phims_semi
        Qss = E_CHARGE * Qss_e
        VFB = phims - Qss / Cox
        VTh = VFB + 2.0 * phif + QSDmax / Cox if body == "p" else VFB - 2.0 * phif - QSDmax / Cox
        LD = np.sqrt(EPS_S * VT / (E_CHARGE * N))
        return dict(
            p0=p0, n0=n0, Cox=Cox, phif=phif, xdT=xdT, QSDmax=QSDmax, phims=phims, phims_semi=phims_semi,
            Qss=Qss, VFB=VFB, VTh=VTh, LD=LD,
        )

    def surface_potential(VG, prm):
        """Solve V_G = V_FB + phi_s - Q'_s(phi_s)/C_ox for phi_s (monotonic in phi_s)."""
        grid = np.concatenate([-np.logspace(0.3, -7, 1500), [0.0], np.logspace(-7, np.log10(1.6), 1500)])
        def vg_of(ph):
            return prm["VFB"] + ph + EPS_S * field_of_phi(ph, prm["p0"], prm["n0"]) / prm["Cox"]

        vg = vg_of(grid)
        k = int(np.clip(np.searchsorted(vg, VG), 1, len(grid) - 1))
        lo, hi = grid[k - 1], grid[k]
        # Bisection between the bracketing grid points (V_G is monotonic in phi_s).
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if vg_of(mid) < VG:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    return (
        BLUE,
        CHI_P,
        EG,
        EPS_OX,
        EPS_S,
        E_CHARGE,
        GRAY,
        GREEN,
        GridSpec,
        METAL_FILL,
        NAVY,
        ORANGE,
        OX_DRAW,
        OX_FILL,
        PINK,
        VT,
        cs_high_freq,
        cs_low_freq,
        energy_arrow,
        field_of_phi,
        mos_params,
        plt,
        potential_profile,
        sci_tex,
        style_ax,
        surface_potential,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # MOS Capacitor
    **ECE335 - companion to lectures**

    An MOS capacitor on uniformly doped silicon, with an SiO\(_2\) gate oxide. Set \(V_G\), the doping, \(t_{ox}\), the gate work function \(\phi'_m\), and the fixed oxide charge \(Q'_{ss}\). The notebook computes \(\phi_{ms}\), the flat-band voltage \(V_{FB}\), the threshold voltage, the energy-band diagram, \(\rho(x)\), \(\mathcal{E}(x)\), \(\phi(x)\), and the C–V curves. The derivation is presented on the slides.

    Silicon at \(300\,\mathrm{K}\): \(n_i = 1.5\times 10^{10}\,\mathrm{cm^{-3}}\), \(kT/e = 0.0259\,\mathrm{V}\), \(E_g = 1.12\,\mathrm{eV}\), \(\varepsilon_s = 11.7\varepsilon_0\), \(\varepsilon_{ox} = 3.9\varepsilon_0\), \(\chi' = 3.25\,\mathrm{V}\). The band bending, charge, field, potential and C–V curves are computed from Poisson's equation with Boltzmann electrons and holes (no depletion approximation); \(x_d\), \(x_{dT}\), \(V_{FB}\) and \(V_T\) in the tables use the formulas from the slides.

    **Fermi levels under bias.** At \(V_G = 0\) the gate and the semiconductor share one Fermi level \(E_F\). With \(V_G \ne 0\) they are not in equilibrium with each other, so the diagrams label two quasi-Fermi levels: \(E_{Fm}\) in the gate and \(E_{Fs}\) in the semiconductor, with \(E_{Fs} - E_{Fm} = eV_G\). (Neamen keeps writing \(E_F\) under bias.)
    """)
    return


@app.cell
def _(
    BLUE,
    CHI_P,
    EG,
    EPS_OX,
    EPS_S,
    E_CHARGE,
    GRAY,
    GREEN,
    GridSpec,
    METAL_FILL,
    NAVY,
    ORANGE,
    OX_DRAW,
    OX_FILL,
    PINK,
    VT,
    energy_arrow,
    field_of_phi,
    mos_params,
    np,
    plt,
    potential_profile,
    sci_tex,
    style_ax,
    surface_potential,
):
    def mos_explorer(body, N, tox_nm, phim_p, Qss_e, VG):
        """Energy bands (top) and rho, E, phi (bottom) for one MOS capacitor; returns fig, table rows, notes."""
        prm = mos_params(body, N, tox_nm, phim_p, Qss_e)
        p0, n0, Cox, phif = prm["p0"], prm["n0"], prm["Cox"], prm["phif"]
        phis = surface_potential(VG, prm)
        Es = float(field_of_phi(phis, p0, n0))
        Qs = -EPS_S * Es
        Qm = -(Qs + prm["Qss"])
        Vox = Qm / Cox
        s_dep = 1.0 if body == "p" else -1.0  # sign of phi_s in depletion and inversion

        # Regime from phi_s (Neamen 10.1.2).
        a = s_dep * phis
        if abs(phis) < 1e-3:
            regime = "flat band"
        elif a < 0:
            regime = "accumulation"
        elif a < phif:
            regime = "depletion"
        elif a < 2.0 * phif:
            regime = "weak inversion"
        else:
            regime = "inversion"

        # Depletion-approximation width, Neamen (10.5) and (10.6).
        xd = 0.0 if a <= 0 else float(np.sqrt(2.0 * EPS_S * min(a, 2.0 * phif) / (E_CHARGE * N)))

        xmax = max(1.6 * prm["xdT"], 8.0 * prm["LD"])
        x = np.linspace(0.0, xmax, 1500)
        phi = potential_profile(phis, p0, n0, x)
        p = p0 * np.exp(-phi / VT)
        n = n0 * np.exp(phi / VT)
        rho_e = (p - p0) - (n - n0)
        Ex = field_of_phi(phi, p0, n0)
        xnm = x * 1e7
        tox_d = OX_DRAW * tox_nm
        n_surf, p_surf = float(n[0]), float(p[0])

        # Energies relative to E_Fs = 0. E_Fi in the bulk: +phi_fp (p type), -phi_fn (n type).
        Ei_b = phif if body == "p" else -phif
        Ei = Ei_b - phi
        Ec, Ev = Ei + EG / 2.0, Ei - EG / 2.0
        EFm = -VG
        biased = abs(VG) > 1e-9
        lab_s = r"$E_{Fs}$" if biased else r"$E_F$"
        lab_m = r"$E_{Fm}$" if biased else r"$E_F$"

        fig = plt.figure(figsize=(14.0, 12.5), layout="constrained")
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1.6, 1.0])
        axb = fig.add_subplot(gs[0, :])
        axr, axe, axp = (fig.add_subplot(gs[1, k]) for k in range(3))

        # ---- Energy-band diagram ----
        xl_m = -tox_d - 0.22 * xmax * 1e7
        # Oxide conduction-band edge: e chi' above E_c(0) at the interface, e phi'_m above E_Fm at the gate.
        # It is linear in the oxide; its drop is e V_ox, since V_G = V_ox + phi_s + phi_ms.
        Ecox_s = Ec[0] + CHI_P
        Ecox_g = EFm + phim_p
        ylo = min(Ev.min(), EFm, 0.0) - 0.35
        yhi = max(Ec.max(), EFm, 0.0, Ecox_s, Ecox_g) + 0.55
        axb.set_xlim(xl_m, xmax * 1e7 * 1.08)
        axb.set_ylim(ylo, yhi)
        axb.fill_between([xl_m, -tox_d], ylo, EFm, color=METAL_FILL, edgecolor=NAVY, lw=1.0)
        axb.plot([xl_m, -tox_d], [EFm, EFm], color=BLUE, lw=2.2, ls="-.")
        axb.add_patch(plt.Rectangle((-tox_d, ylo), tox_d, yhi - ylo, color=OX_FILL, zorder=0))
        axb.plot(xnm, Ec, color="black", lw=2.2)
        axb.plot(xnm, Ev, color="black", lw=2.2)
        axb.plot(xnm, Ei, color="black", lw=1.4, ls="--")
        axb.plot([0, xnm[-1]], [0.0, 0.0], color=BLUE, lw=2.2, ls="-.")
        xr = xnm[-1] * 1.01
        # Right-edge labels, pushed apart so close levels (heavy doping) do not overlap.
        _labs = sorted([(Ev[-1], r"$E_v$", "black"), (0.0, lab_s, BLUE), (Ei[-1], r"$E_{Fi}$", "black"), (Ec[-1], r"$E_c$", "black")])
        _gap = 0.04 * (yhi - ylo)
        _ys = [_labs[0][0]]
        for _E, _l, _c in _labs[1:]:
            _ys.append(max(_E, _ys[-1] + _gap))
        for (_E, _lab, _c), _y in zip(_labs, _ys):
            axb.text(xr, _y, _lab, color=_c, ha="left", va="center", fontsize=16)
        axb.text(xl_m + 0.07 * (xmax * 1e7), EFm - 0.05, lab_m, color=BLUE, ha="left", va="top", fontsize=16)
        # Threshold level: E_Fi(0) reaches it at phi_s = 2 phi_f.
        thr = -phif if body == "p" else phif
        axb.plot([0, 0.35 * xmax * 1e7], [thr, thr], color=PINK, ls=":", lw=1.6)
        # e phi_s: E_Fi in the bulk (dotted) to E_Fi at the surface.
        if abs(phis) > 0.01:
            axb.plot([0, 0.5 * xmax * 1e7], [Ei_b, Ei_b], color=ORANGE, ls=":", lw=1.2)
            energy_arrow(axb, 0.03 * xmax * 1e7, min(Ei[0], Ei_b), max(Ei[0], Ei_b), r"$e\phi_s$", ORANGE)
        fsym = r"\phi_{fp}" if body == "p" else r"\phi_{fn}"
        energy_arrow(axb, 0.9 * xmax * 1e7, min(0.0, Ei_b), max(0.0, Ei_b), rf"$e{fsym}$", PINK, side="left")
        if biased:
            axb.plot([-tox_d, 0], [0.0, 0.0], color=BLUE, ls=":", lw=1.1)
            axb.plot([xl_m, -tox_d], [0.0, 0.0], color=BLUE, ls=":", lw=1.1)
            energy_arrow(axb, -tox_d - 0.05 * xmax * 1e7, min(0.0, EFm), max(0.0, EFm), r"$eV_G$", NAVY, side="left")
        if xd > 0:
            _yxd = Ec.max() + 0.3
            axb.annotate("", xy=(xd * 1e7, _yxd), xytext=(0, _yxd),
                         arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.2, shrinkA=0, shrinkB=0))
            axb.text(0.6 * xd * 1e7, _yxd + 0.03, r"$x_d$", color=NAVY, ha="center", va="bottom", fontsize=16)
            axb.axvline(xd * 1e7, color=GRAY, ls="--", lw=0.9)
        axb.text(0.5 * (xl_m - tox_d), yhi - 0.1, "gate", ha="center", va="top", fontsize=16, color=NAVY)
        if tox_d > 0.025 * xmax * 1e7:
            axb.text(-0.5 * tox_d, ylo + 0.08, "ox", ha="center", va="bottom", fontsize=14, color=GRAY)
        axb.plot([-tox_d, 0.0], [Ecox_g, Ecox_s], color="black", lw=2.2)
        axb.plot([-tox_d, -tox_d], [EFm, Ecox_g], color="black", lw=0.8)
        axb.plot([0.0, 0.0], [Ec[0], Ecox_s], color="black", lw=0.8)
        axb.text(-0.5 * tox_d, max(Ecox_g, Ecox_s) + 0.08, r"$E_c$ (oxide)", ha="center", va="bottom", fontsize=14)
        energy_arrow(axb, 0.02 * xmax * 1e7, Ec[0], Ecox_s, r"$e\chi'$", "black")
        energy_arrow(axb, xl_m + 0.03 * xmax * 1e7, EFm, Ecox_g, r"$e\phi'_m$", "black")
        axb.text(0.6 * xmax * 1e7, ylo + 0.08, f"{body}-type Si", ha="center", va="bottom", fontsize=16)
        axb.set_xlabel(r"$x$ (nm), oxide to scale")
        axb.set_ylabel(r"Energy (eV), $E_{Fs}$ = 0" if biased else r"Energy (eV), $E_F$ = 0")
        axb.set_title(
            rf"$V_G$ = {VG:+.2f} V    $\phi_s$ = {phis:+.3f} V    $V_{{ox}}$ = {Vox:+.3f} V    $x_d$ = {xd * 1e4:.3f} μm    ({regime})"
        )
        axb.spines["top"].set_visible(False)
        axb.spines["right"].set_visible(False)

        # ---- Charge density ----
        lim = 3.0 * N
        axr.axvspan(-tox_d, 0, color=OX_FILL, zorder=0)
        axr.plot(xnm, np.clip(rho_e, -lim, lim), color=NAVY, lw=2.0)
        axr.fill_between(xnm, np.clip(rho_e, -lim, lim), 0.0, color=NAVY, alpha=0.18)
        axr.axhline(0.0, color=GRAY, lw=0.6)
        _qscale = 0.8 * lim
        if abs(Qm) > 1e-12:
            axr.annotate("", xy=(-tox_d, np.sign(Qm) * _qscale), xytext=(-tox_d, 0.0),
                         arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2))
            axr.text(-tox_d + 0.015 * xmax * 1e7, np.sign(Qm) * _qscale, r"$Q'_m$", color=ORANGE, ha="left", va="center", fontsize=15)
        if prm["Qss"] > 0:
            axr.annotate("", xy=(-0.04 * tox_d, 0.5 * _qscale), xytext=(-0.04 * tox_d, 0.0),
                         arrowprops=dict(arrowstyle="-|>", color=PINK, lw=2.0))
            axr.text(-0.3 * tox_d, 0.55 * _qscale, r"$Q'_{ss}$", color=PINK, ha="right", va="bottom", fontsize=14)
        if np.max(np.abs(rho_e)) > lim:
            axr.text(0.97, 0.05 if rho_e[0] < 0 else 0.92, rf"peak $|\rho|/e$ = ${sci_tex(np.max(np.abs(rho_e)))}$ cm$^{{-3}}$",
                     transform=axr.transAxes, ha="right", va="bottom" if rho_e[0] < 0 else "top", fontsize=13, color=GRAY)
        axr.set_ylim(-1.05 * lim, 1.05 * lim)
        axr.set_yticks([-N, 0, N])
        axr.set_yticklabels([rf"$-{sci_tex(N, 0)}$".replace("1\\times ", ""), "0", rf"${sci_tex(N, 0)}$".replace("1\\times ", "")])
        axr.set_title(r"$\rho(x)/e$ (cm$^{-3}$)")
        axr.set_xlabel(r"$x$ (nm)")
        style_ax(axr)

        # ---- Electric field (oxide: D continuous with the sheet Q'_ss at the interface) ----
        Eox = (EPS_S * Es - prm["Qss"]) / EPS_OX
        axe.axvspan(-tox_d, 0, color=OX_FILL, zorder=0)
        axe.plot([-tox_d, 0, 0], [Eox, Eox, Es], color=BLUE, lw=2.0)
        axe.plot(xnm, Ex, color=BLUE, lw=2.0)
        axe.axhline(0.0, color=GRAY, lw=0.6)
        axe.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        axe.set_title(r"$\mathcal{E}(x)$ (V/cm)")
        axe.set_xlabel(r"$x$ (nm)")
        style_ax(axe)

        # ---- Potential: phi = 0 in the bulk, linear in the oxide ----
        axp.axvspan(-tox_d, 0, color=OX_FILL, zorder=0)
        axp.plot([-tox_d, 0], [phis + Vox, phis], color=GREEN, lw=2.0)
        axp.plot(xnm, phi, color=GREEN, lw=2.0)
        axp.axhline(0.0, color=GRAY, lw=0.6)
        axp.set_title(r"$\phi(x)$ (V), $\phi = 0$ in the bulk")
        axp.set_xlabel(r"$x$ (nm)")
        style_ax(axp)
        for _ax in (axr, axe, axp):
            _ax.set_xlim(-max(1.3 * tox_d, 0.08 * xmax * 1e7), xmax * 1e7)
            if xd > 0:
                _ax.axvline(xd * 1e7, color=GRAY, ls="--", lw=0.9)

        # ---- Table rows (Neamen symbols) ----
        Nsym, fsub = ("N_a", "fp") if body == "p" else ("N_d", "fn")
        VTsym = "V_{TN}" if body == "p" else "V_{TP}"
        surf_min = (r"n_s", n_surf) if body == "p" else (r"p_s", p_surf)
        rows = [
            ("Regime", regime),
            (r"\(V_G\)", rf"\({VG:+.2f}\,\mathrm{{V}}\)"),
            (r"\(\phi_s\)", rf"\({phis:+.3f}\,\mathrm{{V}}\)"),
            (r"\(V_{ox}\)", rf"\({Vox:+.3f}\,\mathrm{{V}}\)"),
            (rf"\(\phi_{{{fsub}}}\)", rf"\({phif:.3f}\,\mathrm{{V}}\)"),
            (rf"\(2\phi_{{{fsub}}}\)", rf"\({2 * phif:.3f}\,\mathrm{{V}}\)"),
            (r"\(x_d\)", rf"\({xd * 1e4:.3f}\,\mu\mathrm{{m}}\)"),
            (r"\(x_{dT}\)", rf"\({prm['xdT'] * 1e4:.3f}\,\mu\mathrm{{m}}\)"),
            (rf"\({surf_min[0]}\) (surface)", rf"\({sci_tex(surf_min[1])}\,\mathrm{{cm^{{-3}}}}\)"),
            (r"\(\lvert\mathcal{E}\rvert\) at the surface", rf"\({sci_tex(abs(Es))}\,\mathrm{{V/cm}}\)"),
            (r"\(C_{ox}\)", rf"\({Cox * 1e9:.1f}\,\mathrm{{nF/cm^2}}\)"),
            (r"\(Q'_m\)", rf"\({sci_tex(Qm)}\,\mathrm{{C/cm^2}}\)"),
            (r"\(\phi'_m\)", rf"\({phim_p:.2f}\,\mathrm{{V}}\)"),
            (r"\(\chi'\)", rf"\({CHI_P:.2f}\,\mathrm{{V}}\)"),
            (rf"\(\chi' + E_g/2e {'+' if body == 'p' else '-'} \phi_{{{fsub}}}\)", rf"\({prm['phims_semi']:.3f}\,\mathrm{{V}}\)"),
            (r"\(\phi_{ms}\)", rf"\({prm['phims']:+.3f}\,\mathrm{{V}}\)"),
            (r"\(Q'_{ss}/e\)", rf"\({sci_tex(Qss_e) if Qss_e else '0'}\,\mathrm{{cm^{{-2}}}}\)"),
            (r"\(V_{FB}\)", rf"\({prm['VFB']:+.3f}\,\mathrm{{V}}\)"),
            (r"\(\lvert Q'_{SD}(\max)\rvert\)", rf"\({sci_tex(prm['QSDmax'])}\,\mathrm{{C/cm^2}}\)"),
            (rf"\({VTsym}\)", rf"\({prm['VTh']:+.3f}\,\mathrm{{V}}\)"),
        ]

        notes = []
        NC_SI, NV_SI = 2.8e19, 1.04e19
        if n_surf > NC_SI or p_surf > NV_SI:
            notes.append(
                r"**Outside the model at the surface.** The surface carrier density is above \(N_c\) (electrons) or \(N_v\) (holes): the surface is degenerate, and the Boltzmann results for the surface charge, field and \(\phi_s\) are unreliable."
            )
        elif n_surf > 0.05 * NC_SI or p_surf > 0.05 * NV_SI:
            notes.append(
                r"**Note.** The surface carrier density is within a factor of about 20 of \(N_c\) (electrons) or \(N_v\) (holes), so \(E_{Fs}\) is within \(3kT\) of the band edge at the surface: Boltzmann statistics overestimate the density, and the surface charge and field are approximate."
            )
        return fig, rows, notes, prm

    def table_4col(rows):
        """Two quantity/value pairs per row, so the table is half as long."""
        if len(rows) % 2:
            rows = rows + [("", "")]
        half = len(rows) // 2
        lines = ["| Quantity | Value | Quantity | Value |", "|:--|:--|:--|:--|"]
        for (q1, v1), (q2, v2) in zip(rows[:half], rows[half:]):
            lines.append(f"| {q1} | {v1} | {q2} | {v2} |")
        return "\n".join(lines)

    return mos_explorer, table_4col


@app.cell
def _(mo):
    mo.md(r"""
    Gate work functions for reference, from the slides: aluminum \(\phi'_m = 3.20\,\mathrm{V}\); n\(^+\) polysilicon \(\phi'_m \approx \chi' = 3.25\,\mathrm{V}\); p\(^+\) polysilicon \(\phi'_m \approx \chi' + E_g/e = 4.37\,\mathrm{V}\). The defaults below give \(\phi_{ms} \approx 0\), the ideal MOS capacitor of the slides.
    """)
    return


@app.cell
def _(mo):
    VG_p_slider = mo.ui.slider(start=-3.0, stop=3.0, step=0.01, value=0.45, show_value=True, label=r"$V_G$ (V)")
    Na_log_slider = mo.ui.slider(start=14.0, stop=18.0, step=0.1, value=16.0, show_value=True, label=r"$\log_{10}(N_a/\mathrm{cm^{-3}})$")
    tox_p_slider = mo.ui.slider(start=2.0, stop=50.0, step=0.5, value=18.0, show_value=True, label=r"$t_{ox}$ (nm)")
    phim_p_slider = mo.ui.slider(start=2.5, stop=5.5, step=0.01, value=4.16, show_value=True, label=r"$\phi'_m$ (V)")
    Qss_p_slider = mo.ui.slider(start=0.0, stop=100.0, step=1.0, value=0.0, show_value=True, label=r"$Q'_{ss}/e$ ($10^{10}$ cm$^{-2}$)")
    p_controls = mo.vstack(
        [
            mo.hstack([VG_p_slider, Na_log_slider, tox_p_slider], justify="start", gap=1.0, wrap=True),
            mo.hstack([phim_p_slider, Qss_p_slider], justify="start", gap=1.0, wrap=True),
        ]
    )
    mo.vstack(
        [
            mo.md(r"## P-body MOS capacitor"),
            mo.md(r"Defaults: \(N_a = 10^{16}\,\mathrm{cm^{-3}}\), \(t_{ox} = 18\,\mathrm{nm}\), \(V_G = 0.45\,\mathrm{V}\), as on the slides. Accumulation for \(V_G < V_{FB}\), inversion for \(V_G > V_{TN}\)."),
            p_controls,
            mo.md(
                r"\(\phi'_m\) and \(\chi'\) are the modified work function and modified electron affinity: they are measured to the **oxide conduction band**, not to the vacuum level. \(e\phi'_m\) is the energy from \(E_{Fm}\) up to the oxide conduction band at the gate, and \(e\chi' = 3.25\,\mathrm{eV}\) is the energy from \(E_c\) up to the oxide conduction band at the Si surface. \(\phi_{ms} = \phi'_m - (\chi' + E_g/2e + \phi_{fp})\) is the metal–semiconductor work-function difference with both work functions measured from the oxide conduction band; \(e(\chi' + E_g/2e + \phi_{fp})\) is the energy from \(E_{Fs}\) up to the oxide conduction band with the bands flat, the semiconductor's modified work function (at the interface it is lower by \(e\phi_s\), as the diagram shows)."
            ),
        ]
    )
    return (
        Na_log_slider,
        Qss_p_slider,
        VG_p_slider,
        p_controls,
        phim_p_slider,
        tox_p_slider,
    )


@app.cell
def _(
    Na_log_slider,
    Qss_p_slider,
    VG_p_slider,
    mo,
    mos_explorer,
    phim_p_slider,
    table_4col,
    tox_p_slider,
):
    _fig, _rows, _notes, _prm = mos_explorer(
        "p",
        10.0 ** float(Na_log_slider.value),
        float(tox_p_slider.value),
        float(phim_p_slider.value),
        float(Qss_p_slider.value) * 1e10,
        float(VG_p_slider.value),
    )
    mo.vstack([mo.md(_n) for _n in _notes] + [_fig, mo.md(table_4col(_rows))])
    return


@app.cell
def _(mo):
    VG_n_slider = mo.ui.slider(start=-3.0, stop=3.0, step=0.01, value=-0.45, show_value=True, label=r"$V_G$ (V)")
    Nd_log_slider = mo.ui.slider(start=14.0, stop=18.0, step=0.1, value=16.0, show_value=True, label=r"$\log_{10}(N_d/\mathrm{cm^{-3}})$")
    tox_n_slider = mo.ui.slider(start=2.0, stop=50.0, step=0.5, value=18.0, show_value=True, label=r"$t_{ox}$ (nm)")
    phim_n_slider = mo.ui.slider(start=2.5, stop=5.5, step=0.01, value=3.46, show_value=True, label=r"$\phi'_m$ (V)")
    Qss_n_slider = mo.ui.slider(start=0.0, stop=100.0, step=1.0, value=0.0, show_value=True, label=r"$Q'_{ss}/e$ ($10^{10}$ cm$^{-2}$)")
    n_controls = mo.vstack(
        [
            mo.hstack([VG_n_slider, Nd_log_slider, tox_n_slider], justify="start", gap=1.0, wrap=True),
            mo.hstack([phim_n_slider, Qss_n_slider], justify="start", gap=1.0, wrap=True),
        ]
    )
    mo.vstack(
        [
            mo.md(r"## N-body MOS capacitor"),
            mo.md(r"Defaults: \(N_d = 10^{16}\,\mathrm{cm^{-3}}\), \(t_{ox} = 18\,\mathrm{nm}\), \(V_G = -0.45\,\mathrm{V}\). Accumulation for \(V_G > V_{FB}\), inversion for \(V_G < V_{TP}\). The signs of \(\phi_s\) are reversed from the p body: \(\phi_s < 0\) in depletion and inversion."),
            n_controls,
            mo.md(
                r"\(\phi'_m\) and \(\chi'\) are the modified work function and modified electron affinity: they are measured to the **oxide conduction band**, not to the vacuum level. \(e\phi'_m\) is the energy from \(E_{Fm}\) up to the oxide conduction band at the gate, and \(e\chi' = 3.25\,\mathrm{eV}\) is the energy from \(E_c\) up to the oxide conduction band at the Si surface. \(\phi_{ms} = \phi'_m - (\chi' + E_g/2e - \phi_{fn})\) is the metal–semiconductor work-function difference with both work functions measured from the oxide conduction band; \(e(\chi' + E_g/2e - \phi_{fn})\) is the energy from \(E_{Fs}\) up to the oxide conduction band with the bands flat, the semiconductor's modified work function (at the interface it is lower by \(e\phi_s\), as the diagram shows)."
            ),
        ]
    )
    return (
        Nd_log_slider,
        Qss_n_slider,
        VG_n_slider,
        n_controls,
        phim_n_slider,
        tox_n_slider,
    )


@app.cell
def _(
    Nd_log_slider,
    Qss_n_slider,
    VG_n_slider,
    mo,
    mos_explorer,
    phim_n_slider,
    table_4col,
    tox_n_slider,
):
    _fig, _rows, _notes, _prm = mos_explorer(
        "n",
        10.0 ** float(Nd_log_slider.value),
        float(tox_n_slider.value),
        float(phim_n_slider.value),
        float(Qss_n_slider.value) * 1e10,
        float(VG_n_slider.value),
    )
    mo.vstack([mo.md(_n) for _n in _notes] + [_fig, mo.md(table_4col(_rows))])
    return


@app.cell
def _(
    EPS_S,
    E_CHARGE,
    cs_high_freq,
    cs_low_freq,
    field_of_phi,
    mos_params,
    np,
):
    def cv_curves(body, N, tox_nm, phim_p, Qss_e):
        """Low- and high-frequency C/C_ox versus V_G, parametrized by phi_s (exact, no depletion approximation)."""
        prm = mos_params(body, N, tox_nm, phim_p, Qss_e)
        p0, n0, Cox, phif = prm["p0"], prm["n0"], prm["Cox"], prm["phif"]
        s_dep = 1.0 if body == "p" else -1.0
        VFB, VTh = prm["VFB"], prm["VTh"]
        lo = min(-3.0, min(VFB, VTh) - 1.5)
        hi = max(3.0, max(VFB, VTh) + 1.5)

        def vg_of(ph):
            return VFB + ph + EPS_S * field_of_phi(ph, p0, n0) / Cox

        def end_beyond(target, sign):
            """|phi_s| at which V_G reaches target on the branch of the given sign (bisection; V_G monotonic)."""
            a, b = 1e-6, 2.5
            for _ in range(60):
                m = 0.5 * (a + b)
                if (vg_of(sign * m) - target) * sign < 0:
                    a = m
                else:
                    b = m
            return b

        # Each branch ends just past the plotted window.
        inv_end = max(end_beyond(hi + 0.3 if body == "p" else lo - 0.3, s_dep), 2.0 * phif + 6.0 * 0.0259 + 0.01)
        acc_end = end_beyond(lo - 0.3 if body == "p" else hi + 0.3, -s_dep)
        ph_inv, c_hf_inv = cs_high_freq(s_dep * inv_end, p0, n0, body)
        ph_acc, c_hf_acc = cs_high_freq(-s_dep * acc_end, p0, n0, body)
        LD = np.sqrt(EPS_S * 0.0259 / (E_CHARGE * (p0 + n0)))
        # Fine linear phi_s grid, so consecutive V_G points stay close even in strong inversion and accumulation.
        a_inv = np.linspace(0.0, inv_end, 2500)
        a_acc = np.linspace(0.0, acc_end, 1500)[1:]
        phis = np.concatenate([(-s_dep * a_acc)[::-1], s_dep * a_inv])
        a_br_inv = np.concatenate([[0.0], np.abs(ph_inv)])
        a_br_acc = np.concatenate([[0.0], np.abs(ph_acc)])
        c_hf = np.concatenate([
            np.interp(a_acc, a_br_acc, np.concatenate([[EPS_S / LD], c_hf_acc]))[::-1],
            np.interp(a_inv, a_br_inv, np.concatenate([[EPS_S / LD], c_hf_inv])),
        ])
        c_lf = cs_low_freq(phis, p0, n0)
        # HF cannot exceed LF; this removes ~1e-4 interpolation round-off in accumulation, where they are equal.
        c_hf = np.minimum(c_hf, c_lf)
        VG = vg_of(phis)
        order = np.argsort(VG)
        VG, phis, c_lf, c_hf = VG[order], phis[order], c_lf[order], c_hf[order]
        C_lf = 1.0 / (1.0 + Cox / c_lf)
        C_hf = 1.0 / (1.0 + Cox / c_hf)
        CFB = 1.0 / (1.0 + Cox * LD / EPS_S)
        Cmin_da = 1.0 / (1.0 + Cox * prm["xdT"] / EPS_S)
        # High-frequency C in strong inversion at a fixed point, |phi_s| = 2 phi_f + 6 kT/e (independent of the window).
        _o = np.argsort(phis)
        ph_strong = s_dep * (2.0 * phif + 6.0 * 0.0259)
        C_hf_strong = float(np.interp(ph_strong, phis[_o], C_hf[_o]))
        return dict(
            VG=VG, phis=phis, C_lf=C_lf, C_hf=C_hf, CFB=CFB, Cmin_da=Cmin_da, prm=prm,
            lo=lo, hi=hi, C_hf_strong=C_hf_strong,
        )

    return (cv_curves,)


@app.cell
def _(BLUE, GRAY, GREEN, ORANGE, PINK, np, plt, style_ax):
    def cv_figure(cv, body, VG_now):
        prm = cv["prm"]
        VFB, VTh = prm["VFB"], prm["VTh"]
        lo, hi = cv["lo"], cv["hi"]
        # Keep one point beyond each edge so the curves reach the frame.
        _i = np.flatnonzero((cv["VG"] >= lo) & (cv["VG"] <= hi))
        m = np.zeros_like(cv["VG"], dtype=bool)
        m[max(_i[0] - 1, 0): min(_i[-1] + 2, len(m))] = True
        fig, ax = plt.subplots(figsize=(12.0, 5.6), layout="constrained")
        ax.plot(cv["VG"][m], cv["C_lf"][m], color=BLUE, lw=2.4, label="low frequency")
        ax.plot(cv["VG"][m], cv["C_hf"][m], color=ORANGE, lw=2.4, ls="--", label="high frequency")
        ax.axhline(1.0, color=GRAY, lw=1.0, ls=":")
        ax.text(lo + 0.05, 1.01, r"$C_{ox}$", color=GRAY, va="bottom", fontsize=15)
        _dx = 0.01 * (hi - lo)
        _close = abs(VFB - VTh) < 0.12 * (hi - lo)
        for _k, (_v, _lab, _c) in enumerate(((VFB, r"$V_{FB}$", GREEN), (VTh, r"$V_{TN}$" if body == "p" else r"$V_{TP}$", PINK))):
            ax.axvline(_v, color=_c, ls=":", lw=1.6)
            ax.text(_v + _dx, 1.1 if (_k == 0 or not _close) else 1.03, _lab, color=_c, ha="left", va="bottom", fontsize=15)
        ax.plot([VFB], [cv["CFB"]], "s", color=GREEN, ms=8, label=rf"$C_{{FB}}/C_{{ox}}$ = {cv['CFB']:.3f}")
        _clf = float(np.interp(VG_now, cv["VG"], cv["C_lf"]))
        _chf = float(np.interp(VG_now, cv["VG"], cv["C_hf"]))
        ax.plot([VG_now], [_chf], "o", mfc="white", mec=ORANGE, mew=2.2, ms=13)
        ax.plot([VG_now], [_clf], "o", color=BLUE, ms=7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0.0, 1.17)
        ax.set_xlabel(r"$V_G$ (V)")
        ax.set_ylabel(r"$C'/C_{ox}$")
        ax.set_title(f"{'P' if body == 'p' else 'N'}-body C–V; dots at the explorer's $V_G$ = {VG_now:+.2f} V")
        ax.legend(fontsize=14, loc="center right" if body == "p" else "center left")
        style_ax(ax)
        rows = [
            (r"\(C_{ox}\)", rf"\({prm['Cox'] * 1e9:.1f}\,\mathrm{{nF/cm^2}}\)"),
            (r"\(C_{FB}/C_{ox}\)", rf"\({cv['CFB']:.3f}\)"),
            (r"\(C'_{min}/C_{ox}\) (depletion approx.)", rf"\({cv['Cmin_da']:.3f}\)"),
            (
                r"High freq., strong inversion (\(\lvert\phi_s\rvert = 2\phi_{fp} + 6kT/e\))" if body == "p"
                else r"High freq., strong inversion (\(\lvert\phi_s\rvert = 2\phi_{fn} + 6kT/e\))",
                rf"\({cv['C_hf_strong']:.3f}\)",
            ),
            (r"\(V_{FB}\)", rf"\({VFB:+.3f}\,\mathrm{{V}}\)"),
            (r"\(V_{TN}\)" if body == "p" else r"\(V_{TP}\)", rf"\({VTh:+.3f}\,\mathrm{{V}}\)"),
            (r"\(C'/C_{ox}\) at \(V_G\), low freq.", rf"\({_clf:.3f}\)"),
            (r"\(C'/C_{ox}\) at \(V_G\), high freq.", rf"\({_chf:.3f}\)"),
        ]
        return fig, rows

    return (cv_figure,)


@app.cell
def _(mo):
    mo.md(r"""
    ## C–V characteristics

    Each curve uses the same sliders as its explorer above, repeated here. The low-frequency curve lets every carrier follow the small signal. The high-frequency curve holds the minority carriers (the inversion layer) fixed and lets only the majority carriers respond, so it stays near \(C'_{min}\) in inversion. Freezing the inversion layer completely is an approximation: a model that lets the inversion charge redistribute at fixed total charge gives a high-frequency \(C'\) about 2–3% higher in strong inversion. The filled (low-frequency) and open (high-frequency) dots mark the explorer's \(V_G\).
    """)
    return


@app.cell
def _(Na_log_slider, Qss_p_slider, cv_curves, phim_p_slider, tox_p_slider):
    cv_p = cv_curves(
        "p",
        10.0 ** float(Na_log_slider.value),
        float(tox_p_slider.value),
        float(phim_p_slider.value),
        float(Qss_p_slider.value) * 1e10,
    )
    return (cv_p,)


@app.cell
def _(VG_p_slider, cv_figure, cv_p, mo, p_controls, table_4col):
    _fig, _rows = cv_figure(cv_p, "p", float(VG_p_slider.value))
    mo.vstack([mo.md(r"### P-body C–V"), p_controls, _fig, mo.md(table_4col(_rows))])
    return


@app.cell
def _(Nd_log_slider, Qss_n_slider, cv_curves, phim_n_slider, tox_n_slider):
    cv_n = cv_curves(
        "n",
        10.0 ** float(Nd_log_slider.value),
        float(tox_n_slider.value),
        float(phim_n_slider.value),
        float(Qss_n_slider.value) * 1e10,
    )
    return (cv_n,)


@app.cell
def _(VG_n_slider, cv_figure, cv_n, mo, n_controls, table_4col):
    _fig, _rows = cv_figure(cv_n, "n", float(VG_n_slider.value))
    mo.vstack([mo.md(r"### N-body C–V"), n_controls, _fig, mo.md(table_4col(_rows))])
    return


if __name__ == "__main__":
    app.run()
