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

    # Silicon at 300 K, Neamen Appendix B / Tables 9.1-9.2 and the slides.
    VT = 0.0259
    E_CHARGE = 1.6e-19
    EPS_S = 11.7 * 8.85e-14
    EG = 1.12
    CHI_SI = 4.01
    NC = 2.8e19
    NV = 1.04e19

    # Neamen Table 9.1, work functions phi_m (V).
    METALS = {
        "Ag": 4.26,
        "Al": 4.28,
        "Au": 5.1,
        "Cr": 4.5,
        "Mo": 4.6,
        "Ni": 5.15,
        "Pd": 5.12,
        "Pt": 5.65,
        "Ti": 4.33,
        "W": 4.55,
    }

    BLUE = "#0072B2"
    ORANGE = "#D55E00"
    GREEN = "#009E73"
    NAVY = "#0f4c81"
    GRAY = "#6B6B6B"
    PINK = "#CC79A7"
    METAL_FILL = "#c9d6e8"

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

    def energy_arrow(ax, x, E_lo, E_hi, label, color, side="right", fontsize=16, label_y=None):
        """Double-headed vertical arrow between two energies, labelled on one side."""
        if abs(E_hi - E_lo) < 1e-6:
            return
        ax.annotate(
            "",
            xy=(x, E_hi),
            xytext=(x, E_lo),
            arrowprops=dict(arrowstyle="<->", color=color, lw=1.6, shrinkA=0, shrinkB=0),
        )
        _dx = 0.012 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        ax.text(
            x + (_dx if side == "right" else -_dx),
            0.5 * (E_lo + E_hi) if label_y is None else label_y,
            label,
            color=color,
            fontsize=fontsize,
            ha="left" if side == "right" else "right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
        )

    def contact_params(phi_m, sc_type, N):
        """Ideal (Schottky-Mott) barriers and V_bi, Neamen Eqs. (9.1)-(9.2) and the p-type forms.

        n type: e phi_n = kT ln(Nc/Nd), phi_s = chi + phi_n, phi_B0 = phi_m - chi, V_bi = phi_B0 - phi_n.
        p type: e phi_p = kT ln(Nv/Na), phi_s = chi + Eg/e - phi_p, e phi_Bp = Eg - e(phi_m - chi),
                V_bi = phi_Bp - phi_p.
        V_bi > 0 means a depletion (rectifying) contact; V_bi < 0 means accumulation (ohmic).
        """
        if sc_type == "n":
            phi_bulk = VT * np.log(NC / N)
            phi_s = CHI_SI + phi_bulk
            barrier = phi_m - CHI_SI
        else:
            phi_bulk = VT * np.log(NV / N)
            phi_s = CHI_SI + EG - phi_bulk
            barrier = EG - (phi_m - CHI_SI)
        V_bi = barrier - phi_bulk
        return dict(phi_bulk=phi_bulk, phi_s=phi_s, barrier=barrier, V_bi=V_bi)

    def depletion_profile(N, Vbarrier, x_cm, sc_type):
        """Depletion approximation, x = 0 at the interface, phi = 0 in the neutral semiconductor.

        n type (Eqs. 9.3-9.7): rho = +eNd, E = -(eNd/eps)(W - x), phi = -(eNd/2eps)(W - x)^2.
        p type: signs reversed. Returns W (cm), rho/e, E (V/cm), phi (V).
        """
        W = float(np.sqrt(2.0 * EPS_S * Vbarrier / (E_CHARGE * N)))
        inside = x_cm < W
        s = 1.0 if sc_type == "n" else -1.0
        rho_e = np.where(inside, s * N, 0.0)
        field = np.where(inside, -s * E_CHARGE * N / EPS_S * (W - x_cm), 0.0)
        phi = np.where(inside, -s * E_CHARGE * N / (2.0 * EPS_S) * (W - x_cm) ** 2, 0.0)
        return W, rho_e, field, phi

    def accumulation_profile(N, psi0, x_cm, sc_type):
        """Exact Poisson-Boltzmann accumulation layer for an ohmic contact (majority carriers only).

        u = |phi|/V_t, majority density N exp(u). First integral:
        |E| = sqrt(2 e N V_t / eps) sqrt(exp(u) - 1 - u); x(u) = int_u^u0 V_t du' / |E|.
        n type: phi > 0, bands bend down, rho = e(N - n) < 0. p type: phi < 0, rho > 0.
        """
        u0 = abs(psi0) / VT
        LD = float(np.sqrt(EPS_S * VT / (E_CHARGE * N)))
        u_min = min(1.0e-4, 0.5 * u0)
        s_grid = np.linspace(np.log(u0), np.log(u_min), 20000)
        u_grid = np.exp(s_grid)
        Emag = np.sqrt(2.0 * E_CHARGE * N * VT / EPS_S) * np.sqrt(np.expm1(u_grid) - u_grid)
        integrand = VT * u_grid / Emag  # dx/ds with s = ln u (s decreasing)
        x_of_u = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * -np.diff(s_grid))])
        x_end = x_of_u[-1]
        u = np.where(
            x_cm <= x_end,
            np.interp(x_cm, x_of_u, u_grid),
            u_min * np.exp(-(x_cm - x_end) / LD),
        )
        Emag_x = np.sqrt(2.0 * E_CHARGE * N * VT / EPS_S) * np.sqrt(np.maximum(np.expm1(u) - u, 0.0))
        if sc_type == "n":
            return LD, N * (1.0 - np.exp(u)), Emag_x, VT * u
        return LD, N * (np.exp(u) - 1.0), -Emag_x, -VT * u

    return (
        BLUE,
        CHI_SI,
        EG,
        EPS_S,
        E_CHARGE,
        GRAY,
        GREEN,
        METALS,
        METAL_FILL,
        NAVY,
        NC,
        NV,
        ORANGE,
        PINK,
        VT,
        accumulation_profile,
        contact_params,
        depletion_profile,
        energy_arrow,
        plt,
        sci_tex,
        style_ax,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Metal–Semiconductor Junction
    **ECE335 - companion to lectures**

    An ideal metal contact on uniformly doped silicon. Pick a metal, the semiconductor type, the doping, and the applied voltage. The energy bands before and after contact, and \(\rho(x)\), \(\mathcal{E}(x)\), and \(\phi(x)\), are computed from the Schottky–Mott barrier and Poisson's equation. The derivation is presented on the slides.

    Silicon at \(300\,\mathrm{K}\): \(\chi = 4.01\,\mathrm{V}\), \(E_g = 1.12\,\mathrm{eV}\), \(N_c = 2.8\times 10^{19}\), \(N_v = 1.04\times 10^{19}\,\mathrm{cm^{-3}}\), \(\varepsilon_s = 11.7\varepsilon_0\), \(kT/e = 0.0259\,\mathrm{V}\). Work functions \(\phi_m\) from Neamen Table 9.1.
    """)
    return


@app.cell
def _(NAVY, mo, plt):
    # Schottky diode orientation: V is applied to the anode, the other terminal is grounded.
    _fig, _axs = plt.subplots(1, 2, figsize=(12.0, 2.9), layout="constrained")
    _cols = {"metal": "#8c8c8c", "n": "#cfe3ec", "p": "#f4d3cc"}
    for _ax, _sc in zip(_axs, ("n", "p")):
        _ax.set_xlim(0.0, 10.0)
        _ax.set_ylim(0.3, 4.0)
        _ax.axis("off")
        _blocks = [("metal", 2.0, 2.8), (_sc, 2.8, 6.4)] if _sc == "n" else [(_sc, 2.0, 5.6), ("metal", 5.6, 6.4)]
        for _name, _x0, _x1 in _blocks:
            _ax.add_patch(plt.Rectangle((_x0, 1.3), _x1 - _x0, 1.6, facecolor=_cols[_name], edgecolor="black", lw=1.2))
            if _name == "metal":
                _ax.text(0.5 * (_x0 + _x1), 2.1, "metal", rotation=90, ha="center", va="center", fontsize=16)
            else:
                _ax.text(0.5 * (_x0 + _x1), 2.1, _name, ha="center", va="center", fontsize=22)
        # Anode wire (V) on the left, cathode wire to ground on the right.
        _ax.plot([0.6, 2.0], [2.1, 2.1], color="black", lw=1.6)
        _ax.text(0.6, 2.3, r"$V$", fontsize=20, ha="left", va="bottom")
        _ax.plot([6.4, 8.2, 8.2], [2.1, 2.1, 1.25], color="black", lw=1.6)
        for _k, _w in enumerate((0.45, 0.3, 0.15)):
            _ax.plot([8.2 - _w, 8.2 + _w], [1.25 - 0.15 * _k] * 2, color="black", lw=1.6)
        _anode = "anode: metal" if _sc == "n" else "anode: p side"
        _ax.text(2.0, 1.05, _anode, ha="left", va="top", fontsize=15, color=NAVY)
        _ax.set_title(f"{_sc} type: $V$ on the {'metal' if _sc == 'n' else 'p side'}", fontsize=16)
    mo.vstack(
        [
            mo.md(r"## Schottky diode orientation"),
            _fig,
            mo.md(
                r"The anode is the metal on n type and the semiconductor on p type; the other terminal is grounded. With \(V\) applied to the anode, \(V > 0\) is forward bias in both cases (\(V_a = V\)) and \(V < 0\) is reverse bias (\(V_R = -V\))."
            ),
        ]
    )
    return


@app.cell
def _(METALS, mo):
    _options = {f"{_name} (φm = {_v:.2f} V)": _name for _name, _v in sorted(METALS.items())}
    _options["Custom φm (slider)"] = "custom"
    metal_select = mo.ui.dropdown(options=_options, value="W (φm = 4.55 V)", label="Metal")
    phim_slider = mo.ui.slider(
        start=3.6,
        stop=5.8,
        step=0.01,
        value=4.10,
        show_value=True,
        label=r"Custom φm (V)",
    )
    type_select = mo.ui.dropdown(
        options={"n type": "n", "p type": "p"}, value="n type", label="Semiconductor"
    )
    logN_slider = mo.ui.slider(
        start=14.0,
        stop=19.0,
        step=0.1,
        value=16.0,
        show_value=True,
        label=r"log₁₀(N_d or N_a / cm⁻³)",
    )
    V_slider = mo.ui.slider(
        start=-3.0,
        stop=1.0,
        step=0.01,
        value=0.0,
        show_value=True,
        label=r"V on the anode (V)",
    )
    mo.vstack(
        [
            mo.md(r"## Interactive Explorer: Energy Bands and Electrostatics"),
            mo.md(
                r"The defaults are W on n-type Si, \(N_d = 10^{16}\,\mathrm{cm}^{-3}\). \(V\) is applied to the anode, as in the orientation diagram: the metal on n type, the p side on p type. \(V > 0\) is forward bias (\(V_a = V\)) and \(V < 0\) is reverse bias (\(V_R = -V\)) for both types. The custom \(\phi_m\) slider is used only when “Custom” is selected."
            ),
            mo.hstack([metal_select, type_select], justify="start", gap=1.0, wrap=True),
            mo.hstack([logN_slider, V_slider, phim_slider], justify="start", gap=1.0, wrap=True),
        ]
    )
    return V_slider, logN_slider, metal_select, phim_slider, type_select


@app.cell
def _(
    BLUE,
    CHI_SI,
    EG,
    EPS_S,
    E_CHARGE,
    GRAY,
    GREEN,
    METALS,
    METAL_FILL,
    NAVY,
    NC,
    NV,
    ORANGE,
    PINK,
    VT,
    V_slider,
    accumulation_profile,
    contact_params,
    depletion_profile,
    energy_arrow,
    logN_slider,
    metal_select,
    mo,
    np,
    phim_slider,
    plt,
    sci_tex,
    style_ax,
    type_select,
):
    _metal = metal_select.value
    _phim = float(phim_slider.value) if _metal == "custom" else METALS[_metal]
    _metal_name = "custom metal" if _metal == "custom" else _metal
    _metal_tag = "custom" if _metal == "custom" else _metal
    _sc = type_select.value
    _N = 10.0 ** float(logN_slider.value)
    _V = float(V_slider.value)
    # V is applied to the anode (metal on n type, semiconductor on p type), so V_a = V for both;
    # negative V is reverse bias, V_R = -V.
    _Va = _V
    _p = contact_params(_phim, _sc, _N)
    _phi_bulk, _phi_s, _barrier, _Vbi = _p["phi_bulk"], _p["phi_s"], _p["barrier"], _p["V_bi"]
    _rect = _Vbi > 0.0
    _Nsym = "N_d" if _sc == "n" else "N_a"
    _xsym = "x_n" if _sc == "n" else "x_p"
    _bsym = r"\phi_{B0}" if _sc == "n" else r"\phi_{Bp}"
    _bulksym = r"\phi_n" if _sc == "n" else r"\phi_p"

    # ---- Electrostatics and band profile (x = 0 at the interface, metal at x < 0) ----
    _notes = []
    _valid = True
    if _rect:
        _Vb = _Vbi - _Va
        if _Vb <= 0.0:
            _valid = False
            _notes.append(
                mo.md(
                    rf"**Outside the model.** \(V_a = {_Va:.2f}\,\mathrm{{V}}\) is not below \(V_{{bi}} = {_Vbi:.3f}\,\mathrm{{V}}\), so there is no space charge region left in the depletion approximation. The bands are drawn flat. Reduce the forward bias."
                )
            )
        elif _Vb < 3.0 * VT:
            _notes.append(
                mo.md(
                    rf"**Near flat band.** The barrier from the semiconductor is only \({_Vb * 1e3:.0f}\,\mathrm{{mV}}\), less than \(3kT/e\). The depletion approximation is poor here: the space charge edge is not abrupt, so \(W\) and \(|\mathcal{{E}}_{{max}}|\) are rough."
                )
            )
        _W_eq = float(np.sqrt(2.0 * EPS_S * _Vbi / (E_CHARGE * _N)))
        _W_b = float(np.sqrt(2.0 * EPS_S * max(_Vb, 0.0) / (E_CHARGE * _N)))
        _xmax = 2.0 * max(_W_b, 0.35 * _W_eq)
        _x = np.linspace(0.0, _xmax, 800)
        if _valid:
            _W, _rho_e, _field, _phi = depletion_profile(_N, _Vb, _x, _sc)
        else:
            _W, _rho_e, _field, _phi = 0.0, np.zeros_like(_x), np.zeros_like(_x), np.zeros_like(_x)
        # Metal positive (n type) raises the semiconductor E_F by eV_a; semiconductor positive (p type) lowers it.
        _EFs = _Va if _sc == "n" else -_Va
        _V_shown = _Va
    else:
        # Ohmic: accumulation at equilibrium. The applied voltage drops across the neutral bulk,
        # so the contact itself is drawn at V_a = 0.
        _LD = float(np.sqrt(EPS_S * VT / (E_CHARGE * _N)))
        _xmax = 8.0 * _LD
        _x = np.linspace(0.0, _xmax, 800)
        _LD, _rho_e, _field, _phi = accumulation_profile(_N, _Vbi, _x, _sc)
        _W = 0.0
        _EFs = 0.0
        _V_shown = 0.0
        if _Va != 0.0:
            _notes.append(
                mo.md(
                    r"**Ohmic contact.** There is no barrier to bias. The applied voltage drops across the neutral semiconductor, not the contact, so the bands near the contact are drawn at equilibrium."
                )
            )

    if _sc == "n":
        _Ec_bulk = _EFs + _phi_bulk
        _Ec = _Ec_bulk - _phi
        _Ev = _Ec - EG
    else:
        _Ev_bulk = _EFs - _phi_bulk
        _Ev = _Ev_bulk - _phi
        _Ec = _Ev + EG
    _bform = r"kT\ln(N_c/N_d)" if _sc == "n" else r"kT\ln(N_v/N_a)"
    if _N > np.exp(-3.0) * (NC if _sc == "n" else NV):
        _notes.append(
            mo.md(
                rf"**Near degenerate.** \({_bulksym} = {_phi_bulk:.3f}\,\mathrm{{V}} < 3kT/e\): \(E_F\) is within \(3kT\) of the band edge, and the Boltzmann form \({_bform}\) is no longer accurate."
            )
        )
    # Equilibrium minority density at the interface: holes N_v exp[-(E_g - e phi_B0)/kT] on n type,
    # electrons N_c exp[-(E_g - e phi_Bp)/kT] on p type.
    _min_surf = (NV if _sc == "n" else NC) * np.exp(-(EG - _barrier) / VT)
    if _rect and _min_surf > _N:
        _carrier = "hole" if _sc == "n" else "electron"
        _Nmin = "N_v" if _sc == "n" else "N_c"
        if _barrier > EG - 3.0 * VT:
            # The opposite band edge is at or above E_F at the interface: a Boltzmann density is meaningless.
            _inv_density = rf"the {'valence' if _sc == 'n' else 'conduction'} band edge is within \(3kT\) of \(E_F\) or crosses it at the interface, so the surface {_carrier} density is degenerate and far larger than \({_Nsym}\)"
        else:
            _inv_density = rf"at equilibrium the {_carrier} density at the interface, \({_Nmin}\exp[-(E_g - e{_bsym})/kT] = {sci_tex(_min_surf)}\,\mathrm{{cm^{{-3}}}}\), is larger than \({_Nsym}\)"
        _notes.append(
            mo.md(
                rf"**Inversion at the interface.** \({_bsym} = {_barrier:.2f}\,\mathrm{{V}}\): "
                + _inv_density
                + r". The ideal model and the depletion approximation ignore these minority carriers, so \(|\mathcal{E}_{max}|\) is underestimated."
            )
        )
    # Item 1: degenerate surface for an ohmic contact (barrier within 3kT of zero, or negative).
    _deg_surface = (not _rect) and _barrier < 3.0 * VT
    if _deg_surface and _barrier < 0.0:
        _notes.append(
            mo.md(
                rf"**Degenerate surface.** \({_bsym} = {_barrier:.2f}\,\mathrm{{V}}\) is negative: the {'conduction' if _sc == 'n' else 'valence'} band edge crosses \(E_F\) at the interface. Boltzmann statistics badly overestimate the carrier density and field at the surface, so the surface values in (c) and (d) are not meaningful and \(|\mathcal{{E}}(0)|\) is not quoted. The band shape away from the surface is still indicative."
            )
        )
    elif _deg_surface:
        _notes.append(
            mo.md(
                rf"**Note.** \(e{_bsym} = {_barrier:.3f}\,\mathrm{{eV}}\) is within \(3kT\) of zero, so the surface is close to degenerate and the Boltzmann accumulation layer is approximate right at the interface."
            )
        )

    _xum = _x * 1.0e4
    _W_um = _W * 1.0e4
    _xmax_um = _xmax * 1.0e4

    # ---- Band diagrams ----
    _fig, (_axb, _axa) = plt.subplots(
        1, 2, figsize=(13.5, 6.0), layout="constrained", gridspec_kw={"width_ratios": [1.0, 1.35]}
    )

    # Before contact: vacuum levels aligned, E_F in the metal = 0.
    _Evac = _phim
    _EFs0 = _phim - _phi_s
    _Ec0 = _phim - CHI_SI
    _Ev0 = _Ec0 - EG
    _axb.set_xlim(-1.45, 1.45)
    _axb.fill_between([-1.1, -0.15], min(_Ev0, 0.0) - 0.6, 0.0, color=METAL_FILL, edgecolor=NAVY, lw=1.2)
    _axb.plot([-1.1, 1.3], [_Evac, _Evac], color="black", lw=1.4)
    _axb.text(1.3, _Evac + 0.12, "vacuum level", ha="right", fontsize=14)
    _axb.plot([-1.1, -0.15], [0.0, 0.0], color=BLUE, lw=2.0, ls="-.")
    _axb.plot([0.15, 1.3], [_Ec0, _Ec0], color="black", lw=2.2)
    _axb.plot([0.15, 1.3], [_Ev0, _Ev0], color="black", lw=2.2)
    _axb.plot([0.15, 1.3], [_EFs0, _EFs0], color=BLUE, lw=2.0, ls="-.")
    _axb.text(-1.12, 0.0, r"$E_F$", color=BLUE, ha="right", va="center", fontsize=16)
    # Keep the E_F label clear of whichever band edge it sits near.
    _near_c = abs(_Ec0 - _EFs0) < abs(_EFs0 - _Ev0)
    _axb.text(1.32, _EFs0, r"$E_F$", color=BLUE, ha="left", va="top" if _near_c else "bottom", fontsize=16)
    _axb.text(1.32, _Ec0, r"$E_c$", ha="left", va="bottom" if _near_c else "center", fontsize=16)
    _axb.text(1.32, _Ev0, r"$E_v$", ha="left", va="center" if _near_c else "top", fontsize=16)
    energy_arrow(_axb, -0.7, 0.0, _Evac, r"$e\phi_m$", "black")
    energy_arrow(_axb, 0.35, _Ec0, _Evac, r"$e\chi$", "black")
    energy_arrow(_axb, 0.95, _EFs0, _Evac, r"$e\phi_s$", "black", side="left")
    _axb.text(-0.62, min(_Ev0, 0.0) - 0.4, _metal_tag, ha="center", fontsize=16, color=NAVY, fontweight="bold")
    _axb.text(0.72, _Ev0 - 0.35, f"{_sc}-type Si", ha="center", fontsize=16)
    _axb.set_ylim(min(_Ev0, 0.0) - 0.7, _Evac + 0.45)
    _axb.set_xticks([])
    _axb.set_ylabel(r"Energy (eV), $E_F$ in metal = 0")
    _axb.set_title("(a)  Before contact")
    _axb.spines["top"].set_visible(False)
    _axb.spines["right"].set_visible(False)

    # After contact.
    _ylo = min(_Ev.min(), 0.0, _EFs) - 0.45
    _yhi = max(_Ec.max(), 0.0, _EFs) + 0.45
    _xl = -0.28 * _xmax_um
    _axa.set_xlim(_xl, _xmax_um * 1.02)
    _axa.set_ylim(_ylo, _yhi)
    _axa.fill_between([_xl, 0.0], _ylo, 0.0, color=METAL_FILL, edgecolor=NAVY, lw=1.2)
    _axa.plot([_xl, 0.0], [0.0, 0.0], color=BLUE, lw=2.0, ls="-.")
    _axa.plot(_xum, _Ec, color="black", lw=2.4)
    _axa.plot(_xum, _Ev, color="black", lw=2.4)
    _xF0 = max(_W_um, 0.0) if _rect else 0.0
    _axa.plot([_xF0, _xmax_um], [_EFs, _EFs], color=BLUE, lw=2.0, ls="-.")
    _axa.text(_xl, 0.04, r" $E_F$", color=BLUE, ha="left", va="bottom", fontsize=16)
    _axa.text(_xmax_um * 1.02, _EFs, r"$E_F$", color=BLUE, ha="left", va="center", fontsize=16)
    _axa.text(_xmax_um * 1.02, _Ec[-1], r"$E_c$", ha="left", va="center", fontsize=16)
    _axa.text(_xmax_um * 1.02, _Ev[-1], r"$E_v$", ha="left", va="center", fontsize=16)
    _axa.text(0.5 * _xl, _ylo + 0.15, _metal_tag, ha="center", va="bottom", fontsize=16, color=NAVY, fontweight="bold")
    if _rect and _W_um > 0:
        _axa.axvline(_W_um, color=GRAY, ls="--", lw=1.0)
        _axa.annotate(
            "",
            xy=(_W_um, _ylo + 0.1),
            xytext=(0.0, _ylo + 0.1),
            arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.2, shrinkA=0, shrinkB=0),
        )
        _axa.text(0.5 * _W_um, _ylo + 0.13, rf"${_xsym}$", color=GRAY, ha="center", va="bottom", fontsize=16)

    _xb = 0.04 * _xmax_um
    _show_barrier = _valid and _barrier >= 0.0
    if _sc == "n":
        if _show_barrier:
            energy_arrow(_axa, _xb, 0.0, _Ec[0], rf"$e{_bsym}$", ORANGE, side="left")
        _edge, _bulk_edge = _Ec[0], _Ec[-1]
        if _V_shown == 0.0:
            energy_arrow(_axa, 0.97 * _xmax_um, _EFs, _Ec[-1], rf"$e{_bulksym}$", "black", side="left")
    else:
        if _show_barrier:
            energy_arrow(_axa, _xb, _Ev[0], 0.0, rf"$e{_bsym}$", ORANGE, side="left")
        _edge, _bulk_edge = _Ev[0], _Ev[-1]
        if _V_shown == 0.0:
            energy_arrow(_axa, 0.97 * _xmax_um, _Ev[-1], _EFs, rf"$e{_bulksym}$", "black", side="left")
    if _rect and _valid:
        # The barrier from the semiconductor: the band-edge step from the interface to the neutral region.
        _xs = _W_um + 0.1 * (_xmax_um - _W_um)
        _axa.plot([0.0, _xs], [_edge, _edge], color=ORANGE, ls=":", lw=1.4)
        if _Va > 0:
            _lab = r"$e(V_{bi}-V_a)$"
        elif _Va < 0:
            _lab = r"$e(V_{bi}+V_R)$"
        else:
            _lab = r"$eV_{bi}$"
        energy_arrow(_axa, _xs, min(_edge, _bulk_edge), max(_edge, _bulk_edge), _lab, ORANGE)
    if _V_shown != 0.0:
        _xv = 0.96 * _xmax_um
        _axa.plot([0.0, _xmax_um], [0.0, 0.0], color=BLUE, ls=":", lw=1.2)
        _vlab = r"$eV_a$" if _V_shown > 0 else r"$eV_R$"
        _lo_v, _hi_v = min(0.0, _EFs), max(0.0, _EFs)
        _cands = [_lo_v + f * (_hi_v - _lo_v) for f in (0.5, 0.25, 0.75)]
        _ly = max(_cands, key=lambda y: min(abs(y - _Ec[-1]), abs(y - _Ev[-1])))
        energy_arrow(_axa, _xv, _lo_v, _hi_v, _vlab, PINK, side="left", label_y=_ly)
    _axa.set_xlabel(r"$x$ (μm)")
    _axa.set_ylabel("Energy (eV)")
    if not _rect:
        _state = "ohmic, equilibrium"
    elif _Va > 0:
        _state = rf"forward, $V_a$ = {_Va:.2f} V"
    elif _Va < 0:
        _state = rf"reverse, $V_R$ = {-_Va:.2f} V"
    else:
        _state = "equilibrium"
    _axa.set_title(f"(b)  After contact ({_state})")
    _axa.spines["top"].set_visible(False)
    _axa.spines["right"].set_visible(False)

    # ---- Electrostatics ----
    _fig2, (_ar, _ae, _ap) = plt.subplots(1, 3, figsize=(15.5, 4.6), layout="constrained")
    if _rect:
        _ar.plot(_xum, _rho_e, color=ORANGE, lw=2.2)
        _ar.fill_between(_xum, _rho_e, 0.0, color=ORANGE, alpha=0.18)
        _ar.set_ylabel(r"$\rho/e$ (cm$^{-3}$)")
        _ar.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        if _valid:
            _ar.plot([0.0, 0.0], [0.0, -0.8 * np.sign(_rho_e[0]) * _N], color=NAVY, lw=4)
            _ar.text(0.03 * _xmax_um, -0.75 * np.sign(_rho_e[0]) * _N, "metal surface charge", color=NAVY, fontsize=13, va="center")
        _ar.set_title(r"(c)  Charge density")
    else:
        _ar.semilogy(_xum, np.abs(_rho_e), color=ORANGE, lw=2.2)
        _ar.set_ylabel(r"$|\rho|/e$ (cm$^{-3}$)")
        _ar.set_title(r"(c)  $|\rho|$: accumulated " + ("electrons" if _sc == "n" else "holes"))
    _ar.axhline(0.0, color=GRAY, lw=0.6)
    _ar.set_xlabel(r"$x$ (μm)")
    style_ax(_ar)

    _ae.plot(_xum, _field, color=BLUE, lw=2.2)
    _ae.fill_between(_xum, _field, 0.0, color=BLUE, alpha=0.15)
    _ae.axhline(0.0, color=GRAY, lw=0.6)
    _ae.set_ylabel(r"$\mathcal{E}$ (V/cm)")
    _ae.set_xlabel(r"$x$ (μm)")
    _ae.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    _Emax = float(np.max(np.abs(_field)))
    _Etitle = "" if (_deg_surface and _barrier < 0.0) or not _valid else rf", $|\mathcal{{E}}_{{max}}| = {sci_tex(_Emax)}$ V/cm"
    _ae.set_title(r"(d)  $\mathcal{E}$" + _Etitle)
    style_ax(_ae)

    _ap.plot(_xum, _phi, color=GREEN, lw=2.2)
    _ap.axhline(0.0, color=GRAY, lw=0.6)
    _ap.set_ylabel(r"$\phi(x)$ (V)")
    _ap.set_xlabel(r"$x$ (μm)")
    _ap.set_title(r"(e)  Potential, $\phi = 0$ in the bulk")
    style_ax(_ap)
    for _axx in (_ar, _ae, _ap):
        if _rect and _W_um > 0:
            _axx.axvline(_W_um, color=GRAY, ls="--", lw=1.0)
        _axx.axvline(0.0, color="black", lw=1.0)

    # ---- Numbers ----
    _kind = "rectifying (Schottky barrier)" if _rect else "ohmic (accumulation, no barrier)"
    _head = mo.md(
        rf"**{_metal_name} on {_sc}-type Si, \({_Nsym} = {sci_tex(_N)}\,\mathrm{{cm^{{-3}}}}\): {_kind}.** \(\phi_m = {_phim:.2f}\,\mathrm{{V}}\), \(\phi_s = {_phi_s:.3f}\,\mathrm{{V}}\)."
    )
    if _sc == "n":
        _eq_bulk = rf"e\phi_n &= kT\ln\frac{{N_c}}{{N_d}} = 0.0259\ln\frac{{{sci_tex(NC, 1)}}}{{{sci_tex(_N)}}} = {_phi_bulk:.3f}\ \mathrm{{eV}}, \qquad \phi_s = \chi + \phi_n = {_phi_s:.3f}\ \mathrm{{V}}"
        _eq_bar = rf"\phi_{{B0}} &= \phi_m - \chi = {_phim:.2f} - {CHI_SI:.2f} = {_barrier:.3f}\ \mathrm{{V}}"
        _eq_vbi = rf"V_{{bi}} &= \phi_{{B0}} - \phi_n = {_Vbi:.3f}\ \mathrm{{V}}"
    else:
        _eq_bulk = rf"e\phi_p &= kT\ln\frac{{N_v}}{{N_a}} = 0.0259\ln\frac{{{sci_tex(NV, 2)}}}{{{sci_tex(_N)}}} = {_phi_bulk:.3f}\ \mathrm{{eV}}, \qquad \phi_s = \chi + \frac{{E_g}}{{e}} - \phi_p = {_phi_s:.3f}\ \mathrm{{V}}"
        _eq_bar = rf"e\phi_{{Bp}} &= E_g - e(\phi_m - \chi) = {EG:.2f} - ({_phim:.2f} - {CHI_SI:.2f}) = {_barrier:.3f}\ \mathrm{{eV}}"
        _eq_vbi = rf"V_{{bi}} &= \phi_{{Bp}} - \phi_p = {_Vbi:.3f}\ \mathrm{{V}}"
    _lines = [_eq_bulk, _eq_bar, _eq_vbi]
    if _rect and _valid:
        if _Va < 0:
            _vb_sym = r"V_{bi}+V_R"
            _lines.append(rf"V_{{bi}}+V_R &= {_Vbi:.3f}+{-_Va:.2f} = {_Vb:.3f}\ \mathrm{{V}}")
        elif _Va > 0:
            _vb_sym = r"V_{bi}-V_a"
            _lines.append(rf"V_{{bi}}-V_a &= {_Vbi:.3f}-{_Va:.2f} = {_Vb:.3f}\ \mathrm{{V}}")
        else:
            _vb_sym = r"V_{bi}"
        _vb_term = _vb_sym if _vb_sym == r"V_{bi}" else f"({_vb_sym})"
        _lines.append(
            rf"W = {_xsym} &= \left[\frac{{2\varepsilon_s {_vb_term}}}{{e{_Nsym}}}\right]^{{1/2}} = {_W_um:.3f}\ \mu\mathrm{{m}}"
        )
        _lines.append(
            rf"|\mathcal{{E}}_{{max}}| &= \frac{{e{_Nsym}W}}{{\varepsilon_s}} = {sci_tex(_Emax)}\ \mathrm{{V/cm}}, \qquad C' = \frac{{\varepsilon_s}}{{W}} = {sci_tex(EPS_S / _W)}\ \mathrm{{F/cm^2}}"
        )
    elif not _rect and not (_deg_surface and _barrier < 0.0):
        _lines.append(rf"|\mathcal{{E}}(0)| &= {sci_tex(_Emax)}\ \mathrm{{V/cm}}")
    _eqs = mo.md(
        "$$\n\\begin{aligned}\n" + " \\\\\n".join(_lines) + "\n\\end{aligned}\n$$"
    )
    if _rect:
        _expl = mo.md(
            rf"\(V_{{bi}} > 0\): the semiconductor bands bend {'up' if _sc == 'n' else 'down'} toward the metal and a space charge region of ionized {'donors' if _sc == 'n' else 'acceptors'} forms. The barrier \(e{_bsym}\) seen from the metal does not change with bias; the barrier from the semiconductor, \(e(V_{{bi}} - V_a)\), does."
        )
    else:
        _expl = mo.md(
            rf"\(V_{{bi}} < 0\): {'electrons' if _sc == 'n' else 'holes'} accumulate at the surface and the bands bend {'down' if _sc == 'n' else 'up'} by \(e|V_{{bi}}| = {abs(_Vbi):.3f}\,\mathrm{{eV}}\). The accumulation layer is computed from Poisson's equation with Boltzmann statistics, not the depletion approximation."
        )
    _pin = mo.callout(
        mo.md(
            r"These are ideal barriers. Measured barrier heights \(\phi_{Bn}\), \(\phi_{Bp}\) differ: interface states pin the Fermi level and image-force lowering reduces the barrier. For W–Si, \(\phi_{Bn} \approx 0.52\,\mathrm{V}\) measured, against the ideal \(0.54\,\mathrm{V}\)."
        ),
        kind="info",
    )
    mo.vstack(_notes + [_head, _fig, _fig2, _eqs, _expl, _pin])
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "What the plots are doing": mo.md(
                r"""
    **(a)** Before contact the vacuum levels line up. The metal Fermi level is the energy zero, so the vacuum level is at \(e\phi_m\), \(E_c = e(\phi_m - \chi)\), and the semiconductor \(E_F\) is \(e(\phi_m - \phi_s)\) above the metal \(E_F\).

    **(b)** After contact \(E_F\) is constant at equilibrium. \(E_c\) at the interface stays at \(e(\phi_m - \chi)\) above the metal \(E_F\) for any bias, so \(e\phi_{B0}\) (or \(e\phi_{Bp}\) on p type) is fixed. \(V\) is applied to the anode. On n type the metal is positive in forward bias, which raises the semiconductor \(E_F\) by \(eV_a\); on p type the semiconductor is positive, which lowers its \(E_F\) by \(eV_a\). The band-edge step from the interface to the neutral region is \(e(V_{bi} - V_a)\).

    **(c)–(e)** For a rectifying contact the depletion approximation gives a uniform \(\rho = \pm eN\) out to \(W\), a linear field peaking at the interface, and a parabolic potential, as in a one-sided pn junction with the metal as the heavily doped side. An equal and opposite surface charge sits on the metal. \(\phi\) is zero in the neutral semiconductor, as on the slides.

    For an ohmic contact the majority carriers accumulate at the surface. The charge is mobile, so the depletion approximation does not apply; the profile comes from Poisson's equation with \(n = N_d\exp(e\phi/kT)\) (or the hole equivalent).
    """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding

    Use the explorer, then open the answers.

    1. At the defaults (W on n-type Si, \(N_d = 10^{16}\,\mathrm{cm}^{-3}\), \(V_a = 0\)), read \(\phi_{B0}\), \(V_{bi}\), \(x_n\), and \(|\mathcal{E}_{max}|\). Then apply \(V_R = 0.5\,\mathrm{V}\) (\(V = -0.5\,\mathrm{V}\)). What happens to \(\phi_{B0}\) and \(x_n\)?
    2. Switch to p-type Si, \(N_a = 10^{16}\,\mathrm{cm}^{-3}\). Which metals in Table 9.1 give an ohmic contact?
    3. Back on n-type Si at \(N_d = 10^{16}\,\mathrm{cm}^{-3}\), does any metal in Table 9.1 give an ideal ohmic contact? What does that suggest for real n-type contacts?
    """)
    return


@app.cell
def _(EPS_S, E_CHARGE, METALS, contact_params, mo, np, sci_tex):
    _p = contact_params(METALS["W"], "n", 1e16)
    _W0 = np.sqrt(2 * EPS_S * _p["V_bi"] / (E_CHARGE * 1e16))
    _W5 = np.sqrt(2 * EPS_S * (_p["V_bi"] + 0.5) / (E_CHARGE * 1e16))
    _Emax = E_CHARGE * 1e16 * _W0 / EPS_S
    _pp = contact_params(METALS["W"], "p", 1e16)
    _ohm_p = sorted(
        [k for k, v in METALS.items() if contact_params(v, "p", 1e16)["V_bi"] < 0], key=lambda k: METALS[k]
    )
    _ohm_n = [k for k, v in METALS.items() if contact_params(v, "n", 1e16)["V_bi"] < 0]
    _lowest = min(METALS, key=METALS.get)
    _pl = contact_params(METALS[_lowest], "n", 1e16)
    mo.accordion(
        {
            "Answers": mo.md(
                rf"""
    1. \(\phi_{{B0}} = {_p['barrier']:.2f}\,\mathrm{{V}}\), \(V_{{bi}} = {_p['V_bi']:.3f}\,\mathrm{{V}}\), \(x_n = {_W0 * 1e4:.3f}\,\mu\mathrm{{m}}\), \(|\mathcal{{E}}_{{max}}| = {sci_tex(_Emax)}\,\mathrm{{V/cm}}\), as in Example 9.1. At \(V_R = 0.5\,\mathrm{{V}}\), \(\phi_{{B0}}\) is unchanged and \(x_n\) grows to \({_W5 * 1e4:.2f}\,\mu\mathrm{{m}}\).
    2. On p-type Si, \(\phi_s = {_pp['phi_s']:.2f}\,\mathrm{{V}}\). An ohmic contact needs \(\phi_m > \phi_s\): {", ".join(_ohm_p)}.
    3. {"None. " if not _ohm_n else ""}On n-type Si, \(\phi_s = {_p['phi_s']:.3f}\,\mathrm{{V}}\), and the lowest work function in the table, {_lowest} at \({METALS[_lowest]:.2f}\,\mathrm{{V}}\), still gives \(V_{{bi}} = {_pl['V_bi']:.3f}\,\mathrm{{V}} > 0\). Real ohmic contacts on n-type Si use a heavily doped surface layer, so the barrier is thin enough to tunnel through.
    """
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
