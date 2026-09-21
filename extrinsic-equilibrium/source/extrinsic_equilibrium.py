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

    BLUE = "#0072B2"
    ORANGE = "#D55E00"
    GREEN = "#009E73"
    NAVY = "#0f4c81"

    # Lecture Table 4.1 and slide 8 (Neamen Ch. 4). kT = 0.0259 eV at 300 K.
    KT300 = 0.0259
    T0 = 300.0
    MATERIALS = {
        "Silicon": {
            "Eg": 1.12,
            "Nc300": 2.8e19,
            "Nv300": 1.04e19,
            "mn": 1.08,
            "mp": 0.56,
            "ni300_table": 1.5e10,
        },
        "Germanium": {
            "Eg": 0.66,
            "Nc300": 1.04e19,
            "Nv300": 6.0e18,
            "mn": 0.55,
            "mp": 0.37,
            "ni300_table": 2.4e13,
        },
        "Gallium arsenide": {
            "Eg": 1.42,
            "Nc300": 4.7e17,
            "Nv300": 7.0e18,
            "mn": 0.067,
            "mp": 0.48,
            "ni300_table": 1.8e6,
        },
    }

    def sci_tex(x, digits=2):
        x = float(x)
        if not np.isfinite(x) or x == 0.0:
            return "0"
        sign = "-" if x < 0 else ""
        ax = abs(x)
        exp = int(np.floor(np.log10(ax)))
        man = ax / (10.0**exp)
        return rf"{sign}{man:.{digits}f}\times 10^{{{exp}}}"

    def kT_eV(T):
        return KT300 * (float(T) / T0)

    def scale_Ncv(N300, T):
        return float(N300) * (float(T) / T0) ** 1.5

    def ni_scaled(ni300, Eg, T):
        """Accepted table ni(300 K), with the T dependence of (4.23).

        Table 4.1 Nc, Nv, Eg in (4.23) do not recover the accepted ni(300 K)
        (Si: 1.5e10, not ~7e9). Anchor to the table, then scale as T^{3/2}
        and exp[-Eg/2k (1/T - 1/300)]. This is not textbook Example 4.3,
        which applies (4.23) directly.
        """
        T = float(T)
        return (
            float(ni300)
            * (T / T0) ** 1.5
            * np.exp(-0.5 * float(Eg) * (1.0 / kT_eV(T) - 1.0 / KT300))
        )

    def ni_from_423(Nc300, Nv300, Eg, T=300.0):
        """Eq. (4.23) at temperature T, with Nc, Nv scaled as T^{3/2}."""
        Nc = scale_Ncv(Nc300, T)
        Nv = scale_Ncv(Nv300, T)
        return np.sqrt(Nc * Nv) * np.exp(-float(Eg) / (2.0 * kT_eV(T)))

    def style_ax(ax):
        ax.tick_params(direction="out", length=4, width=1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return (
        BLUE,
        GREEN,
        KT300,
        MATERIALS,
        NAVY,
        ORANGE,
        kT_eV,
        ni_from_423,
        ni_scaled,
        plt,
        scale_Ncv,
        sci_tex,
        style_ax,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Extrinsic Semiconductor in Thermal Equilibrium
    **ECE335 - companion to lectures**

    This is a calculator for semiconductors in the extrinsic regime, where the dopants are fully ionized and
    $n_i$ is still small compared with the net doping. In this case,

    - The majority carrier concentration is set by compensated doping
    - $n_0 p_0 = n_i^2$
    - $E_F$ lies between $E_c$ and $E_v$ when the sample is nondegenerate

    Temperature is limited to about $200$–$400\,\mathrm{K}$, the usual
    design range on the lecture $n_0(T)$ sketch.
    """)
    return


@app.cell
def _(mo):
    material_select = mo.ui.dropdown(
        options=["Silicon", "Germanium", "Gallium arsenide"],
        value="Silicon",
        label="Material",
    )
    Na_log_slider = mo.ui.slider(
        start=10.0,
        stop=18.0,
        value=14.0,
        step=0.1,
        show_value=True,
        label=r"log₁₀(N_a / cm⁻³)",
    )
    Nd_log_slider = mo.ui.slider(
        start=10.0,
        stop=18.0,
        value=16.0,
        step=0.1,
        show_value=True,
        label=r"log₁₀(N_d / cm⁻³)",
    )
    T_slider = mo.ui.slider(
        start=200,
        stop=400,
        value=300,
        step=5,
        show_value=True,
        label="T (K)",
    )
    _header = mo.md("### Parameters")
    _controls = mo.hstack(
        [material_select, Na_log_slider, Nd_log_slider, T_slider],
        justify="start",
        gap=1.2,
        wrap=True,
    )
    carrier_controls = mo.vstack([_header, _controls])
    carrier_controls
    return Na_log_slider, Nd_log_slider, T_slider, material_select


@app.cell
def _(
    BLUE,
    GREEN,
    KT300,
    MATERIALS,
    NAVY,
    Na_log_slider,
    Nd_log_slider,
    ORANGE,
    T_slider,
    kT_eV,
    material_select,
    mo,
    ni_from_423,
    ni_scaled,
    np,
    plt,
    scale_Ncv,
    sci_tex,
    style_ax,
):
    _mat_name = material_select.value
    _par = MATERIALS[_mat_name]
    _T = float(T_slider.value)
    _Na = 10.0 ** float(Na_log_slider.value)
    _Nd = 10.0 ** float(Nd_log_slider.value)
    _Eg = float(_par["Eg"])
    _mn = float(_par["mn"])
    _mp = float(_par["mp"])
    _Nc = scale_Ncv(_par["Nc300"], _T)
    _Nv = scale_Ncv(_par["Nv300"], _T)
    _kT = kT_eV(_T)
    _ni300 = float(_par["ni300_table"])
    _ni = ni_scaled(_ni300, _Eg, _T)
    _ni2 = _ni**2
    _Nnet = _Nd - _Na
    _Ntot = _Nd + _Na

    # (4.60) / (4.62), numerically stable when |Nnet| >> ni (p-type Si/GaAs at 200 K).
    _disc = np.sqrt(_Nnet * _Nnet + 4.0 * _ni2)
    if _Nnet >= 0.0:
        _n0 = 0.5 * (_Nnet + _disc)
        _p0 = _ni2 / _n0
    else:
        _p0 = 0.5 * (-_Nnet + _disc)
        _n0 = _ni2 / _p0
    _n0_approx = max(_Nnet, 0.0)
    _p0_approx = max(-_Nnet, 0.0)
    if _n0_approx > 0.0:
        _p0_from_approx = _ni2 / _n0_approx
    else:
        _p0_from_approx = 0.0
    if _p0_approx > 0.0:
        _n0_from_approx = _ni2 / _p0_approx
    else:
        _n0_from_approx = 0.0

    if abs(_Nnet) < 1e-6 * _Ntot:
        _dtype = "fully compensated (behaves intrinsic)"
    elif _Nd > _Na:
        _dtype = "n type ($N_d>N_a$); electrons majority"
    else:
        _dtype = "p type ($N_a>N_d$); holes majority"

    _extrinsic = abs(_Nnet) >= 10.0 * _ni
    _n0_safe = max(_n0, 1e-300)
    _p0_safe = max(_p0, 1e-300)
    _Ev = 0.0
    _Ec = _Eg
    _Emid = 0.5 * _Eg
    _EFi = _Emid + 0.5 * _kT * np.log(_Nv / _Nc)
    # (4.65): uses the same ni as n0, so Na = Nd gives EF = EFi.
    _EF_minus_EFi = _kT * np.log(_n0_safe / _ni)
    _EF = _EFi + _EF_minus_EFi
    _Ec_minus_EF = _Ec - _EF
    _EF_minus_Ev = _EF - _Ev
    _Ec_minus_EFi = _Ec - _EFi
    _EFi_minus_Ev = _EFi - _Ev
    _np_prod = _n0 * _p0
    _n0_boltz = _Nc * np.exp(np.clip(-_Ec_minus_EF / _kT, -80.0, 80.0))
    _p0_boltz = _Nv * np.exp(np.clip(-_EF_minus_Ev / _kT, -80.0, 80.0))
    _in_gap = (_EF > _Ev) and (_EF < _Ec)
    _boltz_ok = min(_Ec_minus_EF, _EF_minus_Ev) >= 3.0 * _kT
    _si_deg = (_mat_name == "Silicon") and (max(_n0, _p0) >= 3.0e17)
    _si = MATERIALS["Silicon"]
    _ni300_si_formula = ni_from_423(_si["Nc300"], _si["Nv300"], _si["Eg"], 300.0)
    _ni_Tfac = (_T / 300.0) ** 1.5
    _ni_exp = np.exp(-0.5 * _Eg * (1.0 / _kT - 1.0 / KT300))

    if not _in_gap:
        _regime = (
            r"Degenerate: $E_F$ is outside the gap. Boltzmann (4.11)/(4.19) "
            r"do not apply; the lecture then uses the Fermi–Dirac integral "
            r"(Neamen §4.3.3). Easy to reach in n-GaAs, where $N_c$ is only "
            rf"${sci_tex(_par['Nc300'])}\,\mathrm{{cm}}^{{-3}}$ at $300\,\mathrm{{K}}$."
        )
    elif (not _boltz_ok) or _si_deg:
        _deg_bits = []
        if not _boltz_ok:
            _deg_bits.append(
                rf"$(E_c-E_F)$ or $(E_F-E_v)$ is less than $3kT={3.0 * _kT:.3f}\,\mathrm{{eV}}$"
            )
        if _si_deg:
            _deg_bits.append(
                r"Si majority concentration is above the lecture value "
                r"$\sim 3\times 10^{17}\,\mathrm{cm}^{-3}$ (Neamen Ex. 4.13)"
            )
        _regime = (
            r"The Boltzmann nondegenerate assumption is strained ("
            + "; ".join(_deg_bits)
            + r"). The lecture then wants the Fermi–Dirac integral (Neamen §4.3.3)."
        )
    elif not _extrinsic:
        _regime = (
            rf"Net doping $|N_d-N_a|={sci_tex(abs(_Nnet))}\,\mathrm{{cm}}^{{-3}}$ "
            rf"is not $\gg n_i={sci_tex(_ni)}\,\mathrm{{cm}}^{{-3}}$. "
            r"The quadratic (4.60)/(4.62) is still used, but this is no longer a "
            r"clean extrinsic sample (especially likely for Ge near $400\,\mathrm{K}$, "
            r"or when $N_a=N_d$)."
        )
    else:
        _regime = (
            r"Extrinsic and nondegenerate: complete ionization, "
            r"$|N_d-N_a|\gg n_i$, and $E_F$ several $kT$ from both edges."
        )

    _pad = 0.10 * _Eg
    _fig, _ax = plt.subplots(figsize=(8.8, 6.6))
    _xl, _xr = 0.26, 0.78
    _ax.fill_between([_xl, _xr], _Ec, _Ec + _pad, color=BLUE, alpha=0.18, lw=0)
    _ax.fill_between([_xl, _xr], _Ev - _pad, _Ev, color=ORANGE, alpha=0.18, lw=0)
    _ax.fill_between([_xl, _xr], _Ev, _Ec, color="#F4F4F4", alpha=0.9, lw=0)
    _ax.plot([_xl, _xr], [_Ec, _Ec], color=BLUE, lw=3, zorder=3)
    _ax.plot([_xl, _xr], [_Ev, _Ev], color=ORANGE, lw=3, zorder=3)
    _ax.plot(
        [_xl, _xr],
        [_EFi, _EFi],
        color=GREEN,
        lw=2.4,
        ls=":",
        zorder=4,
    )
    _ax.plot(
        [_xl, _xr],
        [_EF, _EF],
        color=NAVY,
        lw=2.8,
        ls="--",
        zorder=5,
    )

    _xlab = 0.82
    _ax.text(_xlab, _Ec, r"$E_c$", color=BLUE, va="center", ha="left", fontsize=16)
    _ax.text(_xlab, _Ev, r"$E_v$", color=ORANGE, va="center", ha="left", fontsize=16)
    _d_fi = abs(_EF - _EFi)
    _fi_off = 0.035 * _Eg if _d_fi < 0.04 * _Eg else 0.0
    _ax.text(
        _xlab,
        _EFi - _fi_off,
        r"$E_{Fi}$",
        color=GREEN,
        va="center",
        ha="left",
        fontsize=16,
    )
    _ax.text(
        _xlab,
        _EF + _fi_off,
        r"$E_F$",
        color=NAVY,
        va="center",
        ha="left",
        fontsize=16,
    )

    _x_eg = 0.12
    _ax.annotate(
        "",
        xy=(_x_eg, _Ec),
        xytext=(_x_eg, _Ev),
        arrowprops=dict(arrowstyle="<->", color="#6B6B6B", lw=1.6),
    )
    _ax.text(
        _x_eg + 0.02,
        _Emid,
        rf"$E_g={_Eg:.2f}\,\mathrm{{eV}}$",
        color="#6B6B6B",
        va="center",
        ha="left",
        fontsize=14,
        rotation=90,
    )

    _xarr = 0.32
    _ax.annotate(
        "",
        xy=(_xarr, _EF),
        xytext=(_xarr, _Ec),
        arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.6),
    )
    _ax.text(
        _xarr + 0.025,
        0.5 * (_Ec + _EF),
        rf"$E_c-E_F={_Ec_minus_EF:.3f}\,\mathrm{{eV}}$",
        color=NAVY,
        va="center",
        ha="left",
        fontsize=13,
    )
    _xarr2 = 0.50
    _ax.annotate(
        "",
        xy=(_xarr2, _Ev),
        xytext=(_xarr2, _EF),
        arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.6),
    )
    _ax.text(
        _xarr2 + 0.025,
        0.5 * (_EF + _Ev),
        rf"$E_F-E_v={_EF_minus_Ev:.3f}\,\mathrm{{eV}}$",
        color=ORANGE,
        va="center",
        ha="left",
        fontsize=13,
    )

    _xfi = 0.70
    _ax.annotate(
        "",
        xy=(_xfi, _EFi),
        xytext=(_xfi, _Ec),
        arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.5),
    )
    _ax.annotate(
        "",
        xy=(_xfi, _Ev),
        xytext=(_xfi, _EFi),
        arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.5),
    )
    _ax.text(
        _xfi - 0.02,
        0.5 * (_Ec + _EFi),
        rf"$E_c-E_{{Fi}}={_Ec_minus_EFi:.3f}\,\mathrm{{eV}}$",
        color=GREEN,
        va="center",
        ha="right",
        fontsize=13,
    )
    _ax.text(
        _xfi - 0.02,
        0.5 * (_EFi + _Ev),
        rf"$E_{{Fi}}-E_v={_EFi_minus_Ev:.3f}\,\mathrm{{eV}}$",
        color=GREEN,
        va="center",
        ha="right",
        fontsize=13,
    )

    _ax.set_xlim(0.0, 1.10)
    _ax.set_ylim(min(_Ev - _pad, _EF - _pad), max(_Ec + _pad, _EF + _pad))
    _ax.set_ylabel("Energy (eV), $E_v=0$")
    _ax.set_xticks([])
    _ax.set_title(
        rf"{_mat_name}: $T={_T:.0f}\,\mathrm{{K}}$, "
        rf"$N_a={sci_tex(_Na)}$, $N_d={sci_tex(_Nd)}\,\mathrm{{cm}}^{{-3}}$"
    )
    style_ax(_ax)
    _ax.grid(False)
    _ax.spines["bottom"].set_visible(False)
    _fig.tight_layout()

    if _extrinsic and (_Nd >= _Na):
        _lim_header = mo.md(r"**Extrinsic limit** $|N_d-N_a|\gg n_i$")
        _eq_lim = mo.md(
            rf"""
    $$
    \begin{{aligned}}
    n_0 &\approx N_d-N_a = {sci_tex(_n0_approx)}\ \mathrm{{cm}}^{{-3}} \\
    p_0 &\approx n_i^2/n_0 = {sci_tex(_p0_from_approx)}\ \mathrm{{cm}}^{{-3}}
    \end{{aligned}}
    $$
    """
        )
    elif _extrinsic:
        _lim_header = mo.md(r"**Extrinsic limit** $|N_d-N_a|\gg n_i$")
        _eq_lim = mo.md(
            rf"""
    $$
    \begin{{aligned}}
    p_0 &\approx N_a-N_d = {sci_tex(_p0_approx)}\ \mathrm{{cm}}^{{-3}} \\
    n_0 &\approx n_i^2/p_0 = {sci_tex(_n0_from_approx)}\ \mathrm{{cm}}^{{-3}}
    \end{{aligned}}
    $$
    """
        )
    else:
        _lim_header = mo.md(r"**Extrinsic limit** (not applied)")
        _eq_lim = mo.md(
            r"$|N_d-N_a|$ is not $\gg n_i$ (near-intrinsic or fully compensated). "
            r"The quadratic above is the working result; do not use $n_0\approx N_d-N_a$."
        )

    _eq_boltz = mo.md(
        rf"""
    $$
    \begin{{aligned}}
    N_c\exp\left[-(E_c-E_F)/kT\right] &= {sci_tex(_n0_boltz)}\ \mathrm{{cm}}^{{-3}} \\
    N_v\exp\left[-(E_F-E_v)/kT\right] &= {sci_tex(_p0_boltz)}\ \mathrm{{cm}}^{{-3}}
    \end{{aligned}}
    $$
    """
    )
    _boltz_note = mo.md(
        rf"""
    These equal the quadratic $n_0={sci_tex(_n0)}$ and $p_0={sci_tex(_p0)}$ only if
    $n_i$ came from (4.23). Here $E_F$ is placed from the **table** $n_i$ via (4.65),
    so the Boltzmann check is close but not identical. The working concentrations
    are always the quadratic and $n_0 p_0=n_i^2$.
    """
    )
    _eq_ec_ef = mo.md(
        rf"""
    $$
    E_c-E_F=E_g-(E_F-E_v)={_Ec_minus_EF:.3f}\ \mathrm{{eV}}
    $$
    """
    )
    _eq_ef_ev = mo.md(
        rf"""
    $$
    E_F-E_v=E_g-(E_c-E_F)={_EF_minus_Ev:.3f}\ \mathrm{{eV}}
    $$
    """
    )
    _eq_ef_efi = mo.md(
        rf"""
    $$
    E_F-E_{{Fi}}=kT\ln(n_0/n_i)
    ={_kT:.5f}\ln\left({sci_tex(_n0)}/{sci_tex(_ni)}\right)
    ={_EF_minus_EFi:.3f}\ \mathrm{{eV}}
    $$
    """
    )
    _gap_note = mo.md(
        r"$E_F$ is placed from (4.65), so $N_a=N_d$ gives $E_F=E_{Fi}$. "
        r"$E_c-E_F$ and $E_F-E_v$ are the rest of $E_g$."
    )

    _info = mo.vstack(
        [
            mo.md(
                rf"""
    **Regime.** {_regime}

    **P or N Type.**
    $N_a={sci_tex(_Na)}\ \mathrm{{cm}}^{{-3}}$,
    $N_d={sci_tex(_Nd)}\ \mathrm{{cm}}^{{-3}}$,
    $N_d-N_a={sci_tex(_Nnet)}\ \mathrm{{cm}}^{{-3}}$,
    $N_d+N_a={sci_tex(_Ntot)}\ \mathrm{{cm}}^{{-3}}$ (scattering still sees the sum).
    {_dtype}.

    **Material at this $T$** (Table 4.1 $N_c$, $N_v$ scaled as $T^{{3/2}}$; $E_g$ held fixed;
    $kT=0.0259\times T/300={_kT:.4f}\ \mathrm{{eV}}$).
    $m_n^*={_mn:g}\,m_0$, $m_p^*={_mp:g}\,m_0$, $E_g={_Eg:.2f}\ \mathrm{{eV}}$.
    $N_c={sci_tex(_Nc)}\ \mathrm{{cm}}^{{-3}}$,
    $N_v={sci_tex(_Nv)}\ \mathrm{{cm}}^{{-3}}$.
    $n_i$ is the lecture/table value at $300\,\mathrm{{K}}$
    (${sci_tex(_ni300)}\ \mathrm{{cm}}^{{-3}}$), times the $T^{{3/2}}$ and
    exponential factors from (4.23). That is **not** textbook Example 4.3,
    which applies (4.23) directly.

    **Intrinsic concentration**
    """
            ),
            mo.md(
                rf"""
    $$
    n_i(T)=n_i(300)\left(\frac{{T}}{{300}}\right)^{{3/2}}
    \exp\left[\frac{{E_g}}{{2k}}\left(\frac{{1}}{{300}}-\frac{{1}}{{T}}\right)\right]
    $$
    """
            ),
            mo.md(
                rf"""
    $$
    n_i({_T:.0f}\,\mathrm{{K}})
    =({sci_tex(_ni300)})\times{_ni_Tfac:.4f}\times{_ni_exp:.4e}
    ={sci_tex(_ni)}\ \mathrm{{cm}}^{{-3}}
    $$
    """
            ),
            mo.md(
                rf"""
    **Note on $n_i$.** Equation (4.23),
    $n_i=\sqrt{{N_c N_v}}\exp(-E_g/2kT)$, is the derivation in the text.
    If you insert Table 4.1 for **silicon** at $300\,\mathrm{{K}}$
    ($N_c=2.8\times 10^{{19}}$, $N_v=1.04\times 10^{{19}}$, $E_g=1.12\,\mathrm{{eV}}$),
    you get $n_i\approx {sci_tex(_ni300_si_formula)}\ \mathrm{{cm}}^{{-3}}$, not the
    accepted $1.5\times 10^{{10}}\ \mathrm{{cm}}^{{-3}}$ used in the examples.
    (Those table entries are rounded separately, so they are not a consistent set.)

    This calculator uses the **accepted** $n_i(300\,\mathrm{{K}})$ from the lecture
    table — ${sci_tex(_ni300)}\ \mathrm{{cm}}^{{-3}}$ for {_mat_name} — and multiplies
    by the $T$ factors of (4.23). Textbook Example 4.3 instead applies (4.23) from
    $N_c$ and $N_v$, so silicon $n_i(400\,\mathrm{{K}})$ is $2.38\times 10^{{12}}$
    there and ${sci_tex(ni_scaled(_si["ni300_table"], _si["Eg"], 400.0))}$ here.

    $E_F$ is placed from (4.65), $E_F-E_{{Fi}}=kT\ln(n_0/n_i)$, using this same
    $n_i$. Then $N_a=N_d$ puts $E_F$ on $E_{{Fi}}$. The minority concentration is
    $n_i^2$ divided by the majority. Do not also evaluate $N_c\exp[-(E_c-E_F)/kT]$
    as if $n_i$ had come from (4.23).
    """
            ),
            mo.md("**Charge neutrality with complete ionization**, (4.60) and (4.62)"),
            mo.md(
                rf"""
    $$
    n_0=\frac{{N_d-N_a}}{{2}}+\sqrt{{\left(\frac{{N_d-N_a}}{{2}}\right)^2+n_i^2}}
    =\frac{{{sci_tex(_Nnet)}}}{{2}}
    +\sqrt{{\left(\frac{{{sci_tex(_Nnet)}}}{{2}}\right)^2+({sci_tex(_ni)})^2}}
    ={sci_tex(_n0)}\ \mathrm{{cm}}^{{-3}}
    $$
    """
            ),
            _lim_header,
            _eq_lim,
            mo.md("**Boltzmann check** (4.11), (4.19), using the plotted $E_F$"),
            _eq_boltz,
            _boltz_note,
            mo.md(
                r"**The $n_0 p_0$ product** is independent of doping and of $E_F$"
            ),
            mo.md(
                rf"""
    $$
    n_0 p_0=({sci_tex(_n0)})({sci_tex(_p0)})={sci_tex(_np_prod)}\ \mathrm{{cm}}^{{-6}}
    $$
    """
            ),
            mo.md(
                rf"""
    $$
    n_i^2=({sci_tex(_ni)})^2={sci_tex(_ni2)}\ \mathrm{{cm}}^{{-6}}
    \Rightarrow n_0 p_0/n_i^2={_np_prod / _ni2:.4f}
    $$
    """
            ),
            mo.md(r"**Fermi level**, (4.65) and the gap, with $E_v=0$"),
            mo.md(
                rf"""
    $$
    E_g=E_c-E_v={_Eg:.3f}\ \mathrm{{eV}}
    $$
    """
            ),
            mo.md(
                rf"""
    $$
    E_c-E_{{Fi}}={_Ec_minus_EFi:.3f}\ \mathrm{{eV}}
    $$
    """
            ),
            mo.md(
                rf"""
    $$
    E_{{Fi}}-E_v={_EFi_minus_Ev:.3f}\ \mathrm{{eV}}
    $$
    """
            ),
            _gap_note,
            _eq_ec_ef,
            _eq_ef_ev,
            mo.md(
                rf"""
    $$
    E_{{Fi}}=E_{{\mathrm{{midgap}}}}+\frac{{1}}{{2}} kT\ln(N_v/N_c)
    ={_Emid:.3f}+\frac{{1}}{{2}}({_kT:.4f})\ln\left({sci_tex(_Nv)}/{sci_tex(_Nc)}\right)
    ={_EFi:.3f}\ \mathrm{{eV}}
    $$
    """
            ),
            _eq_ef_efi,
        ]
    )

    _explain = mo.accordion(
        {
            "What the plot is doing": mo.md(
                r"""
    $E_v$ is drawn at $0$ and $E_c$ at $E_g$. $E_{Fi}$ is **not** exactly
    midgap unless $N_c=N_v$. Doping moves $E_F$ toward the majority band:
    donors pull $E_F$ up, acceptors pull it down. Compensation uses only the
    **difference** $N_d-N_a$ for the carrier type; $n_0 p_0=n_i^2$ is unchanged.
    Raising $T$ at fixed doping increases $n_i$ and $N_{c,v}$ and moves $E_F$
    back toward $E_{Fi}$.
    """
            )
        }
    )

    mo.vstack([_fig, _info, _explain])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding

    1. Starting from $N_d\gg N_a$, raise $N_a$ until it equals $N_d$. Which
       way does $E_F$ move, and what happens to $n_0 p_0$?
    2. Why can you **not** get the minority concentration by subtracting
       $N_a$ from $N_d$?
    3. At fixed doping, raise $T$ from $200\,\mathrm{K}$ to $400\,\mathrm{K}$.
       Why does $E_F$ move toward $E_{Fi}$ even though the sample is still
       labelled extrinsic on the lecture sketch?
    4. Switch from Si to Ge at the same $N_d$, $N_a$, and $T=400\,\mathrm{K}$.
       Why does the “extrinsic” warning appear more readily for Ge?
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Answers and suggested checks": mo.md(
                r"""
    1. $E_F$ falls toward $E_{Fi}$. At $N_a=N_d$ the sample is fully
       compensated: $n_0=p_0=n_i$ and $E_F=E_{Fi}$. The product $n_0 p_0$
       never leaves $n_i^2$.
    2. $N_d-N_a$ is the **net** ionized charge that the majority carrier
       must cancel. The minority is $n_i^2$ divided by the majority, a much
       smaller number. Subtracting two large dopings does not produce it.
    3. $n_i$ grows exponentially with $T$. Then
       $E_F-E_{Fi}=kT\ln(n_0/n_i)$ shrinks even if $n_0$ is still $\approx N_d-N_a$.
       The Fermi level returning toward midgap is the same trend that later
       becomes the intrinsic region of Fig. 4.16.
    4. Ge has a much smaller $E_g$, so $n_i$ at $400\,\mathrm{K}$ is huge
       compared with Si or GaAs. $|N_d-N_a|\gg n_i$ fails first in Ge.
    """
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
