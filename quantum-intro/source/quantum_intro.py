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

    def tex_sci(value):
        mantissa, exponent = f"{value:.3e}".split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"

    def photon_energy_eV(lambda_nm):
        return HC_EV_NM / lambda_nm

    def debroglie_nm(E_eV):
        return H / np.sqrt(2.0 * M_E * E_eV * Q) * 1e9

    def infinite_well_E_eV(n, a_nm):
        a = a_nm * 1e-9
        return (n**2 * np.pi**2 * HBAR**2) / (2.0 * M_E * a**2) / Q

    def step_coefficients_eV(E_eV, V0_eV):
        """Potential step (V=0 for x<0, V=V0 for x>0), unit-amplitude
        incidence from the left. Same algebra used inline by the 'Potential
        step' activity, extracted here as a function so it is directly
        testable (previously it was only computed inline inside the plotting
        cell)."""
        k = np.sqrt(2 * M_E * E_eV * Q) / HBAR * 1e-9
        q = np.lib.scimath.sqrt(2 * M_E * (E_eV - V0_eV) * Q) / HBAR * 1e-9
        r = (k - q) / (k + q)
        t = 2 * k / (k + q)
        R = abs(r) ** 2
        T = float(q.real / k * abs(t) ** 2)
        return r, t, k, q, R, T

    def barrier_coefficients(E_eV, V0_eV, L_nm):
        """Equal-mass rectangular barrier, including the exact E=V0 limit."""
        k = np.sqrt(2 * M_E * E_eV * Q) / HBAR
        k2 = np.lib.scimath.sqrt(2 * M_E * (E_eV - V0_eV) * Q) / HBAR
        L = L_nm * 1e-9
        c = np.cos(k2 * L)
        sinc = np.sinc(k2 * L / np.pi)
        # Match to the transmitted wave at x=L, using sin(qL)/q=L*sinc(qL/pi).
        # This basis remains well defined at q=0, where the interior is linear.
        t = np.exp(-1j * k * L) / (c - 0.5j * (k*k + k2*k2) * L * sinc / k)
        r = -0.5j * (k*k - k2*k2) * L * sinc * t * np.exp(1j*k*L) / k
        A = 1 + r
        B = 1j * (1 - r)
        return r, A, B, t, k, k2, float(abs(t)**2), float(abs(r)**2)

    def psi_barrier(x, r, A, B, t, k, k2, L_m):
        x = np.asarray(x)
        out = np.zeros_like(x, dtype=complex)
        left = x < 0
        mid = (x >= 0) & (x <= L_m)
        right = x > L_m
        out[left] = np.exp(1j*k*x[left]) + r*np.exp(-1j*k*x[left])
        dx = x[mid] - L_m
        out[mid] = t*np.exp(1j*k*L_m) * (np.cos(k2*dx) + 1j*k*dx*np.sinc(k2*dx/np.pi))
        out[right] = t*np.exp(1j*k*x[right])
        return out

    def well_interval_probability(n, left_fraction, right_fraction):
        def primitive(u):
            return u - np.sin(2*n*np.pi*u)/(2*n*np.pi)
        return primitive(right_fraction) - primitive(left_fraction)

    def hydrogen_radial_R(state, rho):
        """Dimensionless hydrogen radial function a0^(3/2)*R(r), rho=r/a0.
        Same algebra used inline by the 'Hydrogen transitions' activity,
        extracted here as a function so it is directly testable (previously
        it was only computed inline inside the plotting cell)."""
        return 2*np.exp(-rho) if state=="1s" else (2-rho)*np.exp(-rho/2)/(2*np.sqrt(2))

    def hydrogen_transition_eV(ni, nf):
        """Level energies and photon energy for a ni->nf hydrogen
        transition. Same algebra used inline by the 'Hydrogen transitions'
        activity, extracted here as a function so it is directly testable."""
        Ei = -RY_EV/ni**2
        Ef = -RY_EV/nf**2
        return Ei, Ef, Ei-Ef

    def electron_configuration(Z):
        """Ground-state electron configuration for Z=1..14, filling
        1s,2s,2p,3s,3p in that order (no aufbau exceptions occur in this
        range). Same algebra used inline by the 'Quantum numbers and shell
        capacity' activity, extracted here as a function so it is directly
        testable."""
        remaining = Z
        config = []
        for label, capacity in [("1s",2),("2s",2),("2p",6),("3s",2),("3p",6)]:
            occupied = min(remaining, capacity)
            if occupied: config.append((label, occupied))
            remaining -= occupied
        return config

    return (
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
        barrier_coefficients,
        debroglie_nm,
        electron_configuration,
        hydrogen_radial_R,
        hydrogen_transition_eV,
        infinite_well_E_eV,
        photon_energy_eV,
        plt,
        psi_barrier,
        step_coefficients_eV,
        style_ax,
        tex_sci,
        well_interval_probability,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Quantum Physics
    **ECE335 - companion to lectures**

    Use these activities alongside lectures and Chapter 2. Before moving a slider,
    predict what will change. Compare the result with your prediction, then open
    the explanation.

    **Neamen Chapter 2:** photons and matter waves (§2.1), probability and stationary
    states (§2.2), wells and scattering (§2.3), and atomic states (§2.4).
    The electron-diffraction and wave-packet activities illustrate §2.1.

    Calculations use the free-electron mass and nonrelativistic mechanics unless
    stated otherwise. Here $KE$ denotes kinetic energy, $V(x)$ potential energy, and $E$ total
    energy. For photons, $E=h\nu$. We use $j^2=-1$, as in Neamen.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Photoelectric effect

    **Predict:** At fixed wavelength above threshold, what changes when the
    intensity doubles? Can increasing intensity cause emission below threshold?

    Use the wavelength and work function to calculate $E=h\nu=hc/\lambda$ and,
    when emission is possible, $KE_{\max}=h\nu-\varphi$. The selected work
    functions are illustrative surface values. This model assumes single-photon
    emission and a constant collection efficiency.
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
        label="optical intensity I (W/m²)",
    )
    pe_controls = mo.vstack(
        [
    
            mo.hstack([pe_lambda, pe_metal, pe_intensity], justify="start", gap=1.5),
        ]
    )
    pe_controls
    return pe_intensity, pe_lambda, pe_metal


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
    pe_intensity,
    pe_lambda,
    pe_metal,
    photon_energy_eV,
    plt,
    style_ax,
    tex_sci,
):
    _lam = float(pe_lambda.value)
    _phi = float(pe_metal.value)
    _I = float(pe_intensity.value)
    _Eph = photon_energy_eV(_lam)
    _f = C / (_lam * 1e-9)
    _K = _Eph - _phi
    _ejected = _K >= 0.0
    _Kshow = _K
    _f0 = _phi * Q / H

    _f_plot = np.linspace(0.0, 1.6e15, 500)
    _K_line = H * _f_plot / Q - _phi
    _K_line = np.where(_K_line > 0.0, _K_line, np.nan)

    _fig, _ax = plt.subplots(figsize=(8.2, 4.4), layout="constrained")
    _ax.plot(_f_plot / 1e14, _K_line, color=BLUE, lw=2.6, label=rf"$\varphi = {_phi:.2f}$ eV")
    _nu_ext = np.linspace(0, _f0, 200)
    _ax.plot(_nu_ext/1e14, H*_nu_ext/Q-_phi, "--", color=BLUE, label="extrapolation")
    _ax.axvline(_f0 / 1e14, color=ORANGE, ls="--", lw=1.4, label=r"threshold $\nu_0 = \varphi/h$")
    _ax.plot([_f0 / 1e14], [0.0], "o", color=ORANGE, ms=8)
    _ax.plot(
        [_f / 1e14],
        [_Kshow],
        "o",
        color=GOLD,
        ms=11,
        zorder=5,
        label="selected frequency" if _ejected else "below threshold: no emission",
    )
    if _ejected:
        _ax.vlines(_f / 1e14, 0.0, _Kshow, color=GOLD, lw=1.6, alpha=0.7)
    _ax.axhline(0.0, color=NAVY, lw=0.8)
    _ax.set_xlim(0.0, 16.0)
    _ax.set_ylim(-_phi-.5, 5.4)
    _ax.set_xlabel(r"optical frequency $\nu$  ($10^{14}$ Hz)")
    _ax.set_ylabel(r"$KE_{\mathrm{max}}$ (eV)")
    _ax.legend(frameon=False, loc="upper left")
    style_ax(_ax)

    if _ejected:
        _status = (
            f"Electrons are ejected.  $KE_{{\\max}} = {_K:.3f}$ eV.  "
            r"(the kinetic energy does **not** change with intensity)."
        )
    else:
        _status = (
            f"Below threshold: $h\\nu = {_Eph:.3f}$ eV $< \\varphi = {_phi:.2f}$ eV.  "
            "No photoemission in this single-photon model."
        )

    _info = mo.md(
        rf"""
    Photon: $\lambda = {_lam:.0f}$ nm,  $\nu = {_f/1e14:.2f}\times 10^{{14}}$ Hz,  $E = {_Eph:.3f}$ eV.

    Photon flux (number arriving per unit area per unit time):


    $$
    \Phi=\frac{{I}}{{h\nu}}=\frac{{I\lambda}}{{hc}}
    ={tex_sci(_I/(_Eph*Q))}\ \mathrm{{photons\,m^{{-2}}s^{{-1}}}}.
    $$



    Divide optical intensity $I={_I:.2f}$ W/m² by the energy per photon
    $h\nu={tex_sci(_Eph*Q)}$ J. For a beam of area $A$, optical power is
    $P=IA$ and the total photon arrival rate is $P/(h\nu)=A\Phi$.
    This incident flux exists even below the emission threshold.

    {_status}

    Energy available after the work function: $h\nu-\varphi = {_K:.3f}$ eV.
    Below threshold, the dashed line is an extrapolation: a negative value means
    no emission, not negative emitted kinetic energy.
    """
    )
    mo.vstack([_fig, _info, mo.accordion({"Explanation": mo.md(r"""At fixed frequency, photon flux is $I/(h\nu)$. Doubling intensity doubles the ideal emission rate above threshold, while $KE_{\max}=h\nu-\varphi$ stays fixed. Below threshold, this single-photon model gives no emission.""")})])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## de Broglie wavelength

    **Predict:** If the electron’s kinetic energy increases by a factor of four,
    how does its wavelength change?

    For a nonrelativistic electron, $\lambda=h/\sqrt{2m_0 KE}$.
    Compare the wavelength with the marked atomic spacings. This comparison
    helps assess diffraction and confinement, but does not decide whether every
    quantum effect can be neglected.
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
        label="log10(KE / eV) for an electron",
    )
    db_controls = mo.vstack(
        [
    
            db_logE,
        ]
    )
    db_controls
    return (db_logE,)


@app.cell
def _(BLUE, GOLD, GREEN, ORANGE, db_logE, debroglie_nm, mo, np, plt, style_ax):
    _E = 10.0 ** float(db_logE.value)
    _lam_e = debroglie_nm(_E)
    _E_plot = np.logspace(-2.0, 3.0, 400)
    _lam_plot = debroglie_nm(_E_plot)
    _a_Si = 0.543
    _bond = 0.235

    _fig, _ax = plt.subplots(figsize=(8.2, 4.4), layout="constrained")
    _ax.loglog(_E_plot, _lam_plot, color=BLUE, lw=2.6, label=r"$\lambda = h/\sqrt{2m KE}$")
    _ax.axhline(_a_Si, color=GREEN, ls="--", lw=1.5, label=rf"Si lattice $a = {_a_Si}$ nm")
    _ax.axhline(_bond, color=ORANGE, ls=":", lw=1.8, label=rf"Si–Si bond $\approx {_bond}$ nm")


    _ax.plot([_E], [_lam_e], "o", color=GOLD, ms=11, zorder=5)
    _ax.set_xlabel("electron kinetic energy $KE$ (eV)")
    _ax.set_ylabel(r"de Broglie $\lambda$ (nm)")
    _ax.set_xlim(1e-2, 1e3)
    _ax.set_ylim(3e-2, 20)
    _ax.legend(frameon=False, loc="upper right")
    style_ax(_ax)

    _thermal = debroglie_nm(0.02585)
    _info = mo.md(
        rf"""
    At $KE = {_E:.3g}$ eV,  $\lambda = {_lam_e:.3g}$ nm.

    As a reference, $KE=k_BT=0.0259$ eV gives $\lambda\approx {_thermal:.2f}$ nm
    at 300 K. Electrons at a given temperature have a distribution of energies.
    For a classical gas in three dimensions, the mean kinetic energy is $3k_BT/2$.

    Check $KE=25$ eV and $KE=100$ eV: the wavelength should halve.
    """
    )
    mo.vstack([_fig, _info, mo.accordion({"Explanation": mo.md(r"""Since $\lambda\propto KE^{-1/2}$, quadrupling $KE$ halves $\lambda$. At 25 eV the wavelength is about 0.245 nm; at 100 eV it is about 0.123 nm.""")})])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Davisson–Germer experiment

    **Predict:** At a fixed detector angle, why does changing accelerating voltage
    produce intensity peaks? At fixed voltage, where should the angular peak occur?

    Electrons accelerated from rest through voltage $V_a$ have $KE=qV_a$ and


    $$
    \lambda=\frac{h}{\sqrt{2mqV_a}}.
    $$


    Davisson and Germer measured electrons scattered by a nickel crystal.
    Neamen Fig. 2.3 shows an angular distribution. Here you can vary both $V_a$
    and the detector angle $\theta$, measured from the outward surface normal.

    **Model:** a normally incident wave scatters coherently from $N=8$ equally
    spaced surface rows, with projected row spacing $d=0.215$ nm. Each row has
    the same scattering amplitude. The path difference is $d\sin\theta$:


    $$
    \frac{I_e(\theta,V_a)}{I_{e,\max}}=
    \left|\frac{1}{N}\sum_{r=0}^{N-1}e^{jr\delta}\right|^2,
    \qquad\delta=\frac{2\pi d\sin\theta}{\lambda}.
    $$


    Peaks occur when $d\sin\theta=n\lambda$. Thus at fixed $\theta$, peak
    positions are equally spaced in $\sqrt{V_a}$.
    These are calculated diffraction curves, not digitized measurements.
    The model omits multiple scattering and the energy dependence of atomic
    scattering strengths in real nickel.

    [Davisson and Germer, Physical Review 30, 705 (1927)](https://doi.org/10.1103/PhysRev.30.705).
    """)
    return


@app.cell
def _(mo):
    dg_voltage = mo.ui.slider(start=20, stop=400, step=1, value=54,
        show_value=True, label="accelerating voltage Vₐ (V)")
    dg_angle = mo.ui.slider(start=25, stop=75, step=1, value=50,
        show_value=True, label="detector angle θ (degrees)")
    mo.vstack([dg_voltage, dg_angle])
    return dg_angle, dg_voltage


@app.cell
def _(debroglie_nm, np):
    def dg_intensity(voltage, theta_deg):
        delta = 2*np.pi*0.215*np.sin(np.deg2rad(theta_deg))/debroglie_nm(voltage)
        return abs(np.mean(np.exp(1j*np.asarray(delta)[..., None]*np.arange(8)), axis=-1))**2

    return (dg_intensity,)


@app.cell
def _(
    BLUE,
    ORANGE,
    debroglie_nm,
    dg_angle,
    dg_intensity,
    dg_voltage,
    mo,
    np,
    plt,
    style_ax,
):
    _voltage = float(dg_voltage.value)
    _theta = float(dg_angle.value)
    _rootv = np.linspace(np.sqrt(20), 20, 2400)
    _angles = np.linspace(0, 90, 1800)
    _lam = debroglie_nm(_voltage)
    _fig, (_a, _b) = plt.subplots(2, 1, figsize=(8.5, 8), layout="constrained")
    _a.plot(_rootv, dg_intensity(_rootv**2, _theta), color=BLUE)
    _a.plot(np.sqrt(_voltage), dg_intensity(_voltage, _theta), 'o', color=ORANGE)
    _a.set_xlabel(r"$\sqrt{V_a}$ (V$^{1/2}$)")
    _a.set_title(rf"Fixed detector angle: $\theta={_theta:.0f}^\circ$")
    _b.plot(_angles, dg_intensity(_voltage, _angles), color=BLUE)
    _b.plot(_theta, dg_intensity(_voltage, _theta), 'o', color=ORANGE)
    _b.set_xlabel(r"detector angle $\theta$ (degrees)")
    _b.set_title(rf"Angular pattern at $V_a={_voltage:.0f}$ V")
    for _axis in (_a, _b):
        _axis.set_ylabel("relative electron intensity")
        _axis.set_ylim(0, 1.08)
        style_ax(_axis)
    _peakv = (debroglie_nm(1)/(0.215*np.sin(np.deg2rad(_theta))))**2
    _peak_angle = (rf"The first nonzero angular maximum is at ${np.rad2deg(np.arcsin(_lam/.215)):.1f}^\circ$."
        if _lam <= .215 else "The wavelength exceeds the row spacing, so no first-order angular maximum is possible.")
    mo.vstack([_fig, mo.md(rf"""
    $KE={_voltage:.0f}$ eV and $\lambda={_lam:.4f}$ nm. {_peak_angle}
    At the selected detector angle, the first-order voltage is ${_peakv:.1f}$ V.
    The orange markers show the same $(V_a,\theta)$ in both scans.
    """), mo.accordion({"Explanation":mo.md(r"""
    Changing voltage changes the electron wavelength. At a diffraction maximum,
    waves from successive rows arrive in phase. Between maxima they partly cancel.
    At 54 V the calculated wavelength is about 0.167 nm and the first-order peak
    is near 51°. The peak at 0° has zero path difference and is the specular order.
    The spacing of peaks on the square-root-voltage axis follows directly from
    $1/\lambda\propto\sqrt{V_a}$.
    """)})])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Wavepacket and position–momentum uncertainty

    **Predict:** Narrow the Gaussian envelope. What happens to the range of
    wavenumbers needed to construct the packet? Does changing the carrier change
    either width?

    At a fixed time, use a normalized Gaussian envelope multiplying a carrier:


    $$
    \psi(x)=\frac{1}{(\pi w^2)^{1/4}}
    e^{-x^2/(2w^2)}e^{jk_0x}.
    $$


    The real part contains a cosine carrier and the imaginary part a sine carrier.
    Both fit within the same envelope. Here $w$ controls its width and $k_0$ is the
    carrier wavenumber; these are packet parameters, not new energy symbols.

    With the symmetric Fourier-transform convention,


    $$
    \widetilde\psi(k)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}
    \psi(x)e^{-jkx}dx
    =\frac{\sqrt{w}}{\pi^{1/4}}e^{-w^2(k-k_0)^2/2}.
    $$


    The inverse transform uses $e^{jkx}$ with the same $1/\sqrt{2\pi}$ factor.
    The corresponding probability densities are


    $$
    |\psi(x)|^2=\frac{1}{\sqrt\pi w}e^{-x^2/w^2},\qquad
    |\widetilde\psi(k)|^2=\frac{w}{\sqrt\pi}e^{-w^2(k-k_0)^2}.
    $$


    Each density integrates to 1. The Fourier plot below is calculated from the
    sampled complex waveform by a discrete Fourier transform and compared with
    the analytical expression.

    Measure the **full width at half maximum (FWHM) of each probability density**:


    $$
    \Delta x_{\mathrm{FWHM}}=2\sqrt{\ln2}\,w,\qquad
    \Delta k_{\mathrm{FWHM}}=\frac{2\sqrt{\ln2}}{w}.
    $$


    Since $p=\hbar k$,


    $$
    \Delta x_{\mathrm{FWHM}}\Delta k_{\mathrm{FWHM}}=4\ln2,\qquad
    \Delta x_{\mathrm{FWHM}}\Delta p_{\mathrm{FWHM}}=4\ln2\,\hbar.
    $$


    This is the Gaussian result using FWHM. It illustrates the reciprocal widths
    behind the uncertainty principle; FWHM does not have a universal uncertainty
    bound for arbitrary waveforms. Some textbooks use a bound of $\hbar/2$ assuming
    standard deviation as the widths, and Neaman uses a bound of $\hbar$.
    """)
    return


@app.cell
def _(mo):
    pkt_width = mo.ui.slider(start=.4, stop=3.5, step=.1, value=1,
        show_value=True, label="Gaussian width parameter w (nm)")
    pkt_carrier = mo.ui.slider(start=0, stop=10, step=.5, value=6,
        show_value=True, label="carrier wavenumber k₀ (nm⁻¹)")
    mo.vstack([pkt_width, pkt_carrier])
    return pkt_carrier, pkt_width


@app.cell
def _(np):
    def gaussian_packet(w, k0):
        x = np.linspace(-256, 256, 65536, endpoint=False)
        dx = x[1]-x[0]
        psi = (np.pi*w*w)**(-.25)*np.exp(-x*x/(2*w*w))*np.exp(1j*k0*x)
        k = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx))*2*np.pi
        transform = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi)))*dx/np.sqrt(2*np.pi)
        exact = np.sqrt(w)/np.pi**.25*np.exp(-w*w*(k-k0)**2/2)
        return x, psi, k, transform, exact

    def density_fwhm(x, density):
        half = density.max()/2
        above = np.flatnonzero(density >= half)
        lo, hi = above[0], above[-1]
        left = np.interp(half, density[lo-1:lo+1], x[lo-1:lo+1])
        right = np.interp(half, density[hi:hi+2][::-1], x[hi:hi+2][::-1])
        return right-left, left, right

    return density_fwhm, gaussian_packet


@app.cell
def _(
    BLUE,
    GREEN,
    HBAR,
    ORANGE,
    density_fwhm,
    gaussian_packet,
    mo,
    np,
    pkt_carrier,
    pkt_width,
    plt,
    style_ax,
    tex_sci,
):
    _w, _k0 = float(pkt_width.value), float(pkt_carrier.value)
    _x, _psi, _k, _ft, _exact = gaussian_packet(_w, _k0)
    _px, _pk = abs(_psi)**2, abs(_ft)**2
    _dx, _xl, _xr = density_fwhm(_x, _px)
    _dk, _kl, _kr = density_fwhm(_k, _pk)
    _fig, _axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    _a, _b, _c, _d = _axes.flat
    _env = (np.pi*_w*_w)**(-.25)*np.exp(-_x*_x/(2*_w*_w))
    _a.plot(_x, _psi.real, color=BLUE, label=r"Re $\psi$")
    _a.plot(_x, _psi.imag, color=ORANGE, label=r"Im $\psi$")
    _a.plot(_x, _env, '--', color=GREEN, label="envelope")
    _a.plot(_x, -_env, '--', color=GREEN)
    _a.set_ylabel(r"$\psi$ (nm$^{-1/2}$)")
    _a.legend(frameon=False, loc="upper right")
    _a.set_title("Gaussian × carrier")
    _a.set_ylim(-1.1*_env.max(), 1.8*_env.max())
    _b.plot(_k, _ft.real, color=BLUE, label="Re (FFT)")
    _b.plot(_k, _ft.imag, color=ORANGE, label="Im (FFT)")
    _b.plot(_k, _exact, '--', color=GREEN, label="analytical")
    _b.set_ylabel(r"$\widetilde\psi$ (nm$^{1/2}$)")
    _b.legend(frameon=False, loc="upper right")
    _b.set_title("Fourier transform")
    _b.set_ylim(-.05*_exact.max(), 1.8*_exact.max())
    for _axis, _domain, _density, _lo, _hi, _label in [
        (_c,_x,_px,_xl,_xr,r"$|\psi|^2$ (nm$^{-1}$)"),
        (_d,_k,_pk,_kl,_kr,r"$|\widetilde\psi|^2$ (nm)")]:
        _axis.plot(_domain,_density,color=BLUE)
        _axis.fill_between(_domain,0,_density,color=BLUE,alpha=.15)
        _axis.hlines(_density.max()/2,_lo,_hi,color=ORANGE,lw=3,label="FWHM")
        _axis.set_ylabel(_label)
        _axis.legend(frameon=False)
    _c.set_title(rf"Density FWHM: {_dx:.3f} nm")
    _d.set_title(rf"Density FWHM: {_dk:.3f} nm$^{{-1}}$")
    for _axis in (_a,_c):
        _axis.set_xlim(-10,10)
        _axis.set_xlabel("x (nm)")
    for _axis in (_b,_d):
        _axis.set_xlim(-5,15)
        _axis.set_xlabel(r"$k$ (nm$^{-1}$)")
    for _axis in _axes.flat:
        style_ax(_axis)
    mo.vstack([_fig,mo.md(rf"""
    From half-maximum crossings of the computed densities:
    $\Delta x_{{\mathrm{{FWHM}}}}\Delta k_{{\mathrm{{FWHM}}}}={_dx*_dk:.4f}$
    (analytical: $4\ln2={4*np.log(2):.4f}$).
    Thus $\Delta x_{{\mathrm{{FWHM}}}}\Delta p_{{\mathrm{{FWHM}}}}
    ={_dx*_dk:.4f}\hbar={tex_sci(_dx*_dk*HBAR)}$ J·s.

    Numerical integrals: $\int|\psi|^2dx={np.trapezoid(_px,_x):.3f}$ and
    $\int|\widetilde\psi|^2dk={np.trapezoid(_pk,_k):.3f}$.
    """),mo.accordion({"Explanation":mo.md(r"""
    Halving $w$ halves the position FWHM and doubles the wavenumber FWHM.
    The product stays fixed. Changing $k_0$ shifts the Fourier peak without
    changing either width: it changes the mean momentum, not its spread.
    The carrier phase is visible in $\psi$ and cancels from $|\psi|^2$.
    """)})])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Stationary-state phase

    **Predict:** Can a wavefunction change with time while its probability density
    stays fixed? Compare its real and imaginary parts with $|\Psi|^2$ below.

    For a state of definite energy in a static potential,
    $\Psi(x,t)=\psi(x)e^{-jEt/\hbar}$. The complex phase changes but its modulus is 1.
    Here $j$ is the imaginary unit.
    """)
    return


@app.cell
def _(mo):
    phase_angle = mo.ui.slider(start=0, stop=2, step=0.05, value=0,
        show_value=True, label="phase Et/ℏ (multiples of π)")
    phase_angle
    return (phase_angle,)


@app.cell
def _(BLUE, GREEN, ORANGE, mo, np, phase_angle, plt, style_ax):
    _x = np.linspace(0, 1, 500)
    _spatial = np.sqrt(2)*np.sin(np.pi*_x)
    _wave = _spatial*np.exp(-1j*np.pi*phase_angle.value)
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    _a.plot(_x, _wave.real, color=BLUE, label=r"Re $\Psi$")
    _a.plot(_x, _wave.imag, color=ORANGE, label=r"Im $\Psi$")
    _a.set_ylim(-1.6, 1.6)
    _a.set_ylabel("amplitude (scaled)")
    _a.legend(frameon=False)
    _b.plot(_x, abs(_wave)**2, color=GREEN)
    _b.set_ylim(0, 2.2)
    _b.set_ylabel(r"$a|\Psi|^2$")
    for _axis in (_a, _b):
        _axis.set_xlabel("x/a")
        style_ax(_axis)
    mo.vstack([_fig, mo.accordion({"Explanation": mo.md(r"""
    $|e^{-jEt/\hbar}|^2=1$, so the phase changes neither the density nor its
    integral. This is a **stationary state**. A superposition of different
    energies can have a density that varies with time.
    """)})])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Infinite potential well

    **Predict:** Halve the well width at fixed $n$. What happens to the energy?
    Then choose $n=2$ and locate the interior node.

    For $V(x)=0$ on $0<x<a$ and infinite walls,


    $$
    \psi_n(x)=\sqrt{\frac{2}{a}}\sin\!\left(\frac{n\pi x}{a}\right),
    \qquad \psi_n=0\text{ outside the well}.
    $$


    The factor $\sqrt{2/a}$ makes


    $$
    \int_0^a|\psi_n(x)|^2dx=\frac{2}{a}\int_0^a\sin^2(n\pi x/a)dx=1.
    $$


    With $a$ and $x$ in nm, $\psi_n$ has units nm$^{-1/2}$.

    Select a spatial interval and calculate its probability by integrating
    $|\psi_n|^2$. The density has units of inverse length. Its value at a point
    is not itself a probability.
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
    well_interval = mo.ui.range_slider(start=0.0, stop=1.0, step=0.05,
        value=[0.25, 0.75], show_value=True, label="interval endpoints x/a")
    well_controls = mo.vstack(
        [
    
            mo.hstack([well_a, well_n], justify="start", gap=2),
            well_interval,
        ]
    )
    well_controls
    return well_a, well_interval, well_n


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
    well_interval,
    well_interval_probability,
    well_n,
):
    _a = float(well_a.value)
    _n = int(well_n.value)
    _En = infinite_well_E_eV(_n, _a)
    _E1 = infinite_well_E_eV(1, _a)
    _x = np.linspace(0.0, _a, 600)
    _psi = np.sqrt(2.0 / _a) * np.sin(_n * np.pi * _x / _a)
    _prob = _psi**2

    _u, _v = well_interval.value
    _Pinterval = well_interval_probability(_n, _u, _v)
    _fig, (_ax, _axP, _axE) = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    _ax.plot(_x, _psi, color=BLUE, lw=2.4)
    _ax.axhline(0, color=NAVY, lw=0.8)
    _ax.set_xlabel("x (nm)")
    _ax.set_ylabel(r"$\psi_n$ (nm$^{-1/2}$)")
    _ax.set_title(f"Wavefunction, n={_n}")
    _axP.plot(_x, _prob, color=ORANGE, lw=2.4)
    _inside = (_x >= _u*_a) & (_x <= _v*_a)
    _axP.fill_between(_x, 0, _prob, where=_inside, color=ORANGE, alpha=.35)
    _axP.set_xlabel("x (nm)")
    _axP.set_ylabel(r"$|\psi_n|^2$ (nm$^{-1}$)")
    _axP.set_title(f"Shaded probability: {_Pinterval:.3f}")
    for _axis in (_ax, _axP):
        _axis.set_xlim(0, _a)
        style_ax(_axis)

    _ns = np.arange(1, 7)
    _Es = infinite_well_E_eV(_ns, _a)
    for _nn, _EE in zip(_ns, _Es):
        _col = GOLD if _nn == _n else BLUE
        _axE.hlines(_EE, 0.2, 0.8, color=_col, lw=2.6 if _nn == _n else 1.6)
        _axE.text(0.85, _EE, rf"$n={_nn}$", va="center", fontsize=16, color=_col)
    _axE.axhline(KT_300K, color=GREEN, ls="--", lw=1.5, label=rf"$kT(300\,\mathrm{{K}}) = {KT_300K:.3f}$ eV")
    _axE.set_xlim(0.0, 1.35)
    _axE.set_ylim(-0.05 * _Es[-1], 1.15 * _Es[-1])
    _axE.set_xticks([])
    _axE.set_ylabel("$E_n$ (eV)")
    _axE.set_title(r"$E_n \propto n^2/a^2$")
    style_ax(_axE)
    _axE.spines["bottom"].set_visible(False)

    _info = mo.md(
        rf"""


    $$
    E_n = \frac{{n^2\pi^2\hbar^2}}{{2ma^2}} = {_En:.3f}\ \mathrm{{eV}}
    \quad (E_1 = {_E1:.3f}\ \mathrm{{eV}}).
    $$



    $P({_u:.2f}a < x < {_v:.2f}a)={_Pinterval:.4f}$.
    Numerical check: $\int_0^a|\psi_n|^2dx={np.trapezoid(_prob, _x):.4f}$ for the selected $n$ and $a$.

    The first level spacing is $E_2-E_1=3E_1={3*_E1:.4f}$ eV.
    The dashed line marks $k_BT={KT_300K:.3f}$ eV at 300 K, a reference energy scale, not a level-broadening model.
    Temperature does not change the allowed levels of this fixed potential.
    """
    )
    mo.vstack([_fig, _info, mo.accordion({"Explanation": mo.md(r"""Halving $a$ multiplies $E_n$ by four. For $n=1$, the central half of the well has probability $1/2+1/\pi\approx0.8183$. For $n=2$ it has probability 0.5. A zero at one point is a node in the density; probabilities refer to finite intervals.""")})])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Finite potential well

    **Predict:** Replace the infinite walls by a finite potential $V_0$.
    Does the wavefunction still vanish at the edges? How does the ground-state
    energy compare with the infinite well of the same width?

    Place the well on $-a/2<x<a/2$: $V(x)=0$ inside and $V(x)=V_0$ outside.
    A bound state has $0<E<V_0$. Because $V(-x)=V(x)$, the eigenfunctions
    can be chosen even or odd about $x=0$:


    $$
    k=\frac{\sqrt{2mE}}{\hbar},\qquad
    \kappa=\frac{\sqrt{2m(V_0-E)}}{\hbar}.
    $$


    Inside, even states are $A\cos(kx)$ and odd states $A\sin(kx)$.
    Outside, the tails decay as $e^{-\kappa(|x|-a/2)}$ and match the value
    at each edge. Continuity of the derivative gives


    $$
    k\tan(ka/2)=\kappa\quad\text{(even)},\qquad
    -k\cot(ka/2)=\kappa\quad\text{(odd)}.
    $$


    The calculation solves these conditions for the allowed energies and
    normalizes over **all space**, including both tails:


    $$
    1=A^2\left[\frac a2\mathbin{\pm}\frac{\sin(ka)}{2k}
      +\frac{b^2}{\kappa}\right],
    $$


    where the plus sign and $b=\cos(ka/2)$ apply to even states; the minus
    sign and $b=\sin(ka/2)$ apply to odd states.
    """)
    return


@app.cell
def _(mo):
    finite_a = mo.ui.slider(start=.5,stop=4,step=.1,value=2,
        show_value=True,label="well width a (nm)")
    finite_V0 = mo.ui.slider(start=.1,stop=5,step=.1,value=1,
        show_value=True,label="outside potential V₀ (eV)")
    mo.vstack([finite_a,finite_V0])
    return finite_V0, finite_a


@app.cell
def _(HBAR, M_E, Q, np):
    def finite_well_states(a, V0):
        # nm units: hbar²/(2m) in eV nm².
        kinetic_factor = HBAR**2/(2*M_E*Q)*1e18
        z0 = a/2*np.sqrt(V0/kinetic_factor)
        states = []
        # Phase form avoids tangent poles: z + asin(z/z0) = n*pi/2.
        for n in range(1, int(np.ceil(2*z0/np.pi))+2):
            target = n*np.pi/2
            if target >= z0+np.pi/2:
                break
            lo, hi = 0., z0
            for iteration in range(80):
                z = (lo+hi)/2
                if z+np.arcsin(z/z0) < target:
                    lo = z
                else:
                    hi = z
            z = (lo+hi)/2
            k = 2*z/a
            energy = kinetic_factor*k*k
            kap = np.sqrt((V0-energy)/kinetic_factor)
            even = n%2 == 1
            edge = np.cos(z) if even else np.sin(z)
            interior = a/2 + (1 if even else -1)*np.sin(2*z)/(2*k)
            norm = 1/np.sqrt(interior+edge*edge/kap)
            outside = norm*norm*edge*edge/kap
            states.append((energy,k,kap,even,norm,outside))
        return states

    def finite_well_wave(x, a, state):
        energy,k,kap,even,norm,outside = state
        x = np.asarray(x)
        inner = np.cos(k*x) if even else np.sin(k*x)
        edge = np.cos(k*a/2) if even else np.sin(k*a/2)
        tail = edge*np.exp(-kap*np.maximum(np.abs(x)-a/2,0))
        if not even:
            tail = tail*np.sign(x)
        return norm*np.where(np.abs(x)<=a/2,inner,tail)

    return finite_well_states, finite_well_wave


@app.cell
def _(finite_V0, finite_a, finite_well_states, mo):
    finite_states = finite_well_states(float(finite_a.value),float(finite_V0.value))
    finite_state = mo.ui.dropdown(
        options={f"n={i+1}: E={state[0]:.4f} eV":i for i,state in enumerate(finite_states)},
        value=f"n=1: E={finite_states[0][0]:.4f} eV",label="bound state")
    finite_state
    return finite_state, finite_states


@app.cell
def _(
    BLUE,
    ORANGE,
    finite_V0,
    finite_a,
    finite_state,
    finite_states,
    finite_well_wave,
    infinite_well_E_eV,
    mo,
    np,
    plt,
    style_ax,
):
    _a, _V0 = float(finite_a.value), float(finite_V0.value)
    _index = int(finite_state.value)
    _state = finite_states[_index]
    _E,_k,_kap,_even,_norm,_outside = _state
    _half = _a/2
    _pad = max(_half, 7/_kap)
    _xmin, _xmax = -_half-_pad, _half+_pad
    _x = np.unique(np.concatenate([
        np.linspace(_xmin,-_half,3000),np.linspace(-_half,_half,3000),np.linspace(_half,_xmax,3000)]))
    _wave = finite_well_wave(_x,_a,_state)
    _fig,(_aE,_aW,_aP)=plt.subplots(3,1,figsize=(8.5,10),layout="constrained")
    _aE.plot([_xmin,-_half,-_half,_half,_half,_xmax],[_V0,_V0,0,0,_V0,_V0],color=ORANGE,label=r"$V(x)$")
    for _i,_st in enumerate(finite_states):
        _aE.hlines(_st[0],-_half,_half,color=BLUE,lw=3 if _i==_index else 1,alpha=1 if _i==_index else .35)
    _aE.set_ylabel("energy (eV)")
    _aE.set_title(f"{len(finite_states)} bound states; selected n={_index+1}")
    _aE.legend(frameon=False)
    _aW.plot(_x,_wave,color=BLUE)
    _aW.axhline(0,color="gray",lw=.8)
    _aW.set_ylabel(r"$\psi$ (nm$^{-1/2}$)")
    _aP.plot(_x,abs(_wave)**2,color=BLUE)
    _aP.fill_between(_x,0,abs(_wave)**2,where=np.abs(_x)>_half,color=ORANGE,alpha=.35)
    _aP.set_ylabel(r"$|\psi|^2$ (nm$^{-1}$)")
    for _axis in (_aE,_aW,_aP):
        _axis.axvline(-_half,color="gray",ls=":")
        _axis.axvline(_half,color="gray",ls=":")
        _axis.set_xlim(_xmin,_xmax)
        _axis.set_xlabel("x (nm)")
        style_ax(_axis)
    mo.vstack([_fig,mo.md(rf"""
    Selected energy: $E={_E:.4f}$ eV. The infinite-well value for the same
    $n$ and $a$ is ${infinite_well_E_eV(_index+1,_a):.5f}$ eV.
    The state is **{'even' if _even else 'odd'}** about $x=0$.

    The total probability outside the well is ${_outside:.4f}$
    ({100*_outside:.2f}%). The numerical integral over the plotted domain is
    $\int|\psi|^2dx={np.trapezoid(abs(_wave)**2,_x):.4f}$; normalization includes
    the small tails beyond the plot analytically. The amplitude decay length is
    $1/\kappa={1/_kap:.4f}$ nm.
    """),mo.accordion({"Explanation":mo.md(r"""
    A finite wall allows a decaying tail. Both the wavefunction and its derivative
    are continuous at each edge. Spreading beyond the well lowers the energy
    relative to infinite walls. Increasing $V_0$ shrinks the tails and brings the
    energies toward the infinite-well values. Only finitely many bound states fit
    below $V_0$; above it are scattering states.
    """)})])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Potential step

    **Predict:** For $E<V_0$, can the density extend into the step even when
    no probability current flows through it? What changes when $E>V_0$?

    The potential is zero for $x<0$ and $V_0$ for $x>0$.
    Matching $\psi$ and $\psi'$ at $x=0$ gives the reflection and transmission.
    Compare this semi-infinite step with the finite barrier in the next activity.
    """)
    return


@app.cell
def _(mo):
    step_ratio = mo.ui.slider(start=0.1, stop=3, step=0.05, value=0.5,
        show_value=True, label="E/V₀ (V₀ = 1 eV)")
    step_ratio
    return (step_ratio,)


@app.cell
def _(BLUE, ORANGE, mo, np, plt, step_coefficients_eV, step_ratio, style_ax):
    _E = float(step_ratio.value)
    _r, _t, _k, _q, _R, _T = step_coefficients_eV(_E, 1.0)
    _xl = np.linspace(-2,0,700)
    _xr = np.linspace(0,2,700)
    _left = np.exp(1j*_k*_xl)+_r*np.exp(-1j*_k*_xl)
    _right = _t*np.exp(1j*_q*_xr)
    _fig, (_a,_b) = plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    _a.plot(_xl,abs(_left)**2,color=BLUE)
    _a.plot(_xr,abs(_right)**2,color=BLUE)
    _a.axvspan(0,2,color=ORANGE,alpha=.15)
    _a.set_xlabel("x (nm)")
    _a.set_ylabel("density / incident density")
    _a.set_title("Density at the step")
    _ratios=np.linspace(.01,3,600)
    _wave1=np.sqrt(_ratios)
    _wave2=np.lib.scimath.sqrt(_ratios-1)
    _reflection=abs((_wave1-_wave2)/(_wave1+_wave2))**2
    _b.plot(_ratios,_reflection,color=ORANGE,label="R")
    _b.plot(_ratios,1-_reflection,color=BLUE,label="T")
    _b.axvline(_E,ls="--",color="gray")
    _b.set_xlabel(r"$E/V_0$")
    _b.set_ylabel("fraction of incident current")
    _b.set_ylim(-.03,1.05)
    _b.legend(frameon=False)
    for _axis in (_a,_b): style_ax(_axis)
    if _E < 1:
        _description=rf"$E<V_0$: $\psi$ decays over $1/\kappa={1/_q.imag:.3f}$ nm. The density decays over $1/(2\kappa)$."
    elif np.isclose(_E,1):
        _description=r"$E=V_0$: the limiting bounded solution on the right is constant, and its probability current is zero."
    else:
        _description=r"$E>V_0$: a transmitted travelling wave exists, with lower kinetic energy $E-V_0$. Reflection is still nonzero."
    mo.vstack([_fig,mo.md(rf"$R={_R:.4f}$, $T={_T:.4f}$, $R+T={_R+_T:.4f}$. {_description}"),
        mo.accordion({"Explanation":mo.md(r"""
        For $E<V_0$, the right-hand solution is a decaying exponential.
        Its phase does not vary with $x$, so $J=(\hbar/m)\operatorname{Im}(\psi^*\psi')=0$.
        Penetration means nonzero density in the forbidden region. Transmission
        requires outgoing current. A finite barrier can support that current on
        its far side.

        For $E>V_0$, $T=(k_2/k_1)|t|^2$. The wave-number factor accounts for
        the different velocities on the two sides.
        """)})])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Rectangular barrier and tunneling

    **Predict:** For $E<V_0$, what happens when the barrier width doubles?
    Does the transmitted electron lose energy?

    The calculation matches $\psi$ and $\psi'$ at both interfaces, with the same
    mass everywhere and $V=0$ on both sides. Inside the finite barrier, both
    exponential terms are needed. For $E=V_0$, the interior solution is linear.
    $T$ is a ratio of transmitted to incident probability current.
    """)
    return


@app.cell
def _(mo):
    tun_E = mo.ui.slider(
        start=0.20,
        stop=5.00,
        value=1.00,
        step=0.05,
        show_value=True,
        label="electron energy E (eV)",
    )
    tun_V = mo.ui.slider(
        start=0.50,
        stop=20.00,
        value=3.00,
        step=0.10,
        show_value=True,
        label="barrier height V0 (eV)",
    )
    tun_L = mo.ui.slider(
        start=0.10,
        stop=2.50,
        value=0.80,
        step=0.05,
        show_value=True,
        label="barrier width a (nm)",
    )
    tun_controls = mo.vstack(
        [
    
            mo.hstack([tun_E, tun_V, tun_L], justify="start", gap=1.2),
        ]
    )
    tun_controls
    return tun_E, tun_L, tun_V


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
    tun_V,
):
    _E = float(tun_E.value)
    _V0 = float(tun_V.value)
    _L = float(tun_L.value)
    _r, _A, _B, _t, _k, _k2, _T, _R = barrier_coefficients(_E, _V0, _L)
    _Lm = _L * 1e-9
    _x = np.linspace(-2.0 * _L, 3.0 * _L, 1400)
    _psi = psi_barrier(_x * 1e-9, _r, _A, _B, _t, _k, _k2, _Lm)
    _prob = np.abs(_psi) ** 2
    _V = np.where((_x >= 0.0) & (_x <= _L), _V0, 0.0)

    _L_scan = np.linspace(0.05, 2.50, 250)
    _T_scan = np.array([barrier_coefficients(_E, _V0, _Ls)[6] for _Ls in _L_scan])

    _fig, (_axW, _axT) = plt.subplots(1, 2, figsize=(9.6, 4.4), layout="constrained")
    _axW.fill_between(_x, 0.0, _V, color=ORANGE, alpha=0.22, step=None)
    _axW.plot(_x, _V, color=ORANGE, lw=2.2, label="$V(x)$")
    _axW.axhline(_E, color=GREEN, ls="--", lw=1.6, label=rf"$E = {_E:.2f}$ eV")
    _axP = _axW.twinx()
    _axP.plot(_x, _prob, color=BLUE, lw=2.0, label=r"$|\psi|^2$")
    _axW.set_xlim(_x.min(), _x.max())
    _axW.set_ylim(-0.05 * max(_V0, _E), 1.25 * max(_V0, _E))
    _axP.set_ylim(0.0, 1.15 * max(_prob.max(), 1e-12))
    _axW.set_xlabel(r"$x$ (nm)")
    _axW.set_ylabel("$V$ (eV)")
    _axP.set_ylabel(r"$|\psi|^2$")
    _axW.set_title("Potential and relative density")
    style_ax(_axW)
    _axW.spines["right"].set_visible(True)
    _h1, _l1 = _axW.get_legend_handles_labels()
    _h2, _l2 = _axP.get_legend_handles_labels()
    _axW.legend(_h1 + _h2, _l1 + _l2, frameon=False, loc="upper right", fontsize=16)

    _axT.semilogy(_L_scan, _T_scan, color=NAVY, lw=2.4)
    _axT.plot([_L], [_T], "o", color=GOLD, ms=10, zorder=5)
    _axT.set_xlim(0.0, 2.55)
    _axT.set_ylim(max(float(_T_scan.min()) * 0.5, 1e-100), 1.5)
    _axT.set_xlabel(r"barrier width $a$ (nm)")
    _axT.set_ylabel("transmission $T$")
    _axT.set_title(r"$T(a)$ at this $E$, $V_0$")
    style_ax(_axT)

    if _E < _V0:
        _kappa = np.sqrt(2.0 * M_E * (_V0 - _E) * Q) / HBAR
        _exp_approx = float(np.exp(-2.0 * _kappa * _Lm))
        _regime = (
            rf"$E < V_0$ (classically forbidden). "
            rf"$\kappa = {_kappa*1e-9:.2f}$ nm$^{{-1}}$,  "
            rf"$\kappa a={_kappa*_Lm:.2f}$. Thick-barrier estimate "
            rf"$T\approx16(E/V_0)(1-E/V_0)e^{{-2\kappa a}}={16*(_E/_V0)*(1-_E/_V0)*_exp_approx:.3e}$ "
            r"requires $\kappa a\gg1$."
        )
    elif np.isclose(_E, _V0):
        _regime = r"$E=V_0$: the interior solution is linear. The calculation uses this exact limit."
    else:
        _regime = (
            r"$E > V_0$: part of the wave still reflects at the step. "
            r"$T$ oscillates with $a$ (resonances when an integer number of "
            r"half-waves fit in the barrier)."
        )

    _info = mo.md(
        rf"""
    $E = {_E:.2f}$ eV,  $V_0 = {_V0:.2f}$ eV,  $a = {_L:.2f}$ nm.

    Exact matching:  $T = {_T:.3e}$,  $R = {_R:.3e}$,  $R+T = {_R+_T:.4f}$.

    {_regime}

    The density is relative to an incident wave of amplitude 1. It is not normalized
    to one over the infinite line. $R+T=1$ checks current conservation.

    Try a thinner-barrier variation of Neamen Example 2.5: $E=2$ eV, $V_0=20$ eV, $a=0.1$ nm.
    The exact transmission is about 0.0188. Then double $a$.
    """
    )
    mo.vstack([_fig, _info, mo.accordion({"Explanation": mo.md(r"""For a sufficiently thick barrier, doubling $a$ reduces $T$ by a further factor approximately $e^{-2\kappa a}$. Total energy remains conserved. The exponential approximation needs $\kappa a\gg1$; the plotted matching solution also works for thin barriers and at $E=V_0$.""")})])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Hydrogen transitions and radial probability

    **Predict:** Does the maximum of the 1s density per unit volume occur at the
    same radius as the maximum of the radial density?

    Compare 1s and 2s. The radial factor $r^2$ accounts for the increasing volume
    of a spherical shell. The energy controls below describe ideal hydrogen
    level differences; they do not specify transition selection rules or rates.
    """)
    return


@app.cell
def _(mo):
    atom_state=mo.ui.dropdown(options=["1s","2s"],value="1s",label="radial state")
    atom_radius=mo.ui.slider(start=0,stop=15,step=.1,value=1,show_value=True,label="upper radius r/a₀")
    atom_ni=mo.ui.slider(start=1,stop=6,step=1,value=3,show_value=True,label="initial n")
    atom_nf=mo.ui.slider(start=1,stop=6,step=1,value=2,show_value=True,label="final n")
    mo.vstack([mo.hstack([atom_state,atom_radius]),mo.hstack([atom_ni,atom_nf])])
    return atom_nf, atom_ni, atom_radius, atom_state


@app.cell
def _(
    BLUE,
    HC_EV_NM,
    ORANGE,
    atom_nf,
    atom_ni,
    atom_radius,
    atom_state,
    hydrogen_radial_R,
    hydrogen_transition_eV,
    mo,
    np,
    plt,
    style_ax,
):
    _rho=np.linspace(0,15,1500)
    # Dimensionless R(r)*a0^(3/2), normalized by integral R^2*rho^2 drho = 1.
    _radial=hydrogen_radial_R(atom_state.value,_rho)
    _volume=_radial**2/(4*np.pi)
    _density=_rho**2*_radial**2
    _limit=float(atom_radius.value)
    _grid=np.linspace(0,_limit,1001)
    _rgrid=hydrogen_radial_R(atom_state.value,_grid)
    _prob=float(np.trapezoid(_grid**2*_rgrid**2,_grid))
    _fig,(_a,_b)=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    _a.plot(_rho,_volume,color=BLUE)
    _a.set_ylabel(r"$a_0^3|\psi|^2$")
    _a.set_title("Density per unit volume")
    _b.plot(_rho,_density,color=ORANGE)
    _b.fill_between(_rho,0,_density,where=_rho<=_limit,color=ORANGE,alpha=.3)
    _b.set_ylabel(r"$a_0 P(r)$")
    _b.set_title("Radial probability density")
    for _axis in (_a,_b):
        _axis.set_xlabel(r"$r/a_0$")
        style_ax(_axis)
    _ni=int(atom_ni.value)
    _nf=int(atom_nf.value)
    _Ei,_Ef,_delta=hydrogen_transition_eV(_ni,_nf)
    if _ni==_nf:
        _transition="The levels are the same. The energy difference is zero, so there is no transition photon."
    else:
        _process="emission" if _delta>0 else "absorption"
        _transition=rf"For {_process}: photon energy $|E_i-E_f|={abs(_delta):.3f}$ eV, wavelength ${HC_EV_NM/abs(_delta):.1f}$ nm."
    mo.vstack([_fig,mo.md(rf"""
    In the selected {atom_state.value} state, $P(0<r<{_limit:.1f}a_0)={_prob:.4f}$.
    $a_0=0.0529$ nm.

    $E_i={_Ei:.3f}$ eV and $E_f={_Ef:.3f}$ eV. {_transition}
    """),mo.accordion({"Explanation":mo.md(r"""
    The 1s volume density peaks at the nucleus. The radial density
    $P(r)=4\pi r^2|\psi_{100}|^2$ peaks at $r=a_0$ because a spherical shell
    has volume $4\pi r^2dr$. The 2s radial wavefunction has a node at $r=2a_0$.
    A radial probability distribution describes distances, not a circular orbit.

    Try $n_i=3$, $n_f=2$: the energy difference gives a wavelength near 656 nm.
    Swapping the levels gives absorption at the same photon energy.
    """)})])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Quantum numbers and shell capacity

    **Predict:** How many spatial orbitals and electron states belong to $n=3$?
    Use the shell selector to list all allowed combinations. Then compare the
    capacity of a shell with the occupied states in a neutral atom.
    """)
    return


@app.cell
def _(mo):
    shell_n=mo.ui.slider(start=1,stop=4,step=1,value=2,show_value=True,label="shell n")
    atom_Z=mo.ui.slider(start=1,stop=14,step=1,value=10,show_value=True,label="neutral atom: electron count Z")
    mo.hstack([shell_n,atom_Z])
    return atom_Z, shell_n


@app.cell
def _(atom_Z, electron_configuration, mo, shell_n):
    _n=int(shell_n.value)
    _rows=[]
    for _ell in range(_n):
        _mvalues=", ".join(str(_m) for _m in range(-_ell,_ell+1))
        _rows.append(f"| {_n}{'spdf'[_ell]} | {_ell} | {_mvalues} | {2*_ell+1} | {2*(2*_ell+1)} |")
    _config=[rf"{_label}^{{{_occupied}}}" for _label,_occupied in electron_configuration(int(atom_Z.value))]
    _names=["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si"]
    _configuration=r"\,".join(_config)
    mo.vstack([mo.md("\n".join([
        "| subshell | ℓ | allowed m | spatial orbitals | electron capacity |",
        "|:--|:--|:--|--:|--:|",*_rows])),mo.md(rf"""
    Shell $n={_n}$ contains ${_n**2}$ spatial orbitals and has capacity
    $2n^2={2*_n**2}$ electrons.

    Neutral **{_names[int(atom_Z.value)-1]}** has configuration ${_configuration}$.
    This filling order applies to the first fourteen elements shown here.
    """),mo.accordion({"Explanation":mo.md(r"""
    Each orbital has two spin projections, $s=+1/2$ and $-1/2$.
    Pauli exclusion allows one electron per complete set $(n,\ell,m,s)$.
    Summing $2(2\ell+1)$ over $\ell=0,\ldots,n-1$ gives $2n^2$.

    For $n=3$, the capacities are 2 in 3s, 6 in 3p, and 10 in 3d, totaling 18.
    Silicon occupies only four of these states in its ground-state configuration.
    Pauli specifies capacities; electron interactions also determine the order
    in which subshells fill. These are Neamen’s labels $(n,\ell,m,s)$.
    """)})])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Check your understanding

    Explain each result before opening the answers.

    1. A metal emits electrons under 400 nm light. What changes when intensity doubles?
    2. What happens to $\lambda$ if kinetic energy quadruples?
    3. For the ground state of a well, what is the probability in its central half?
    4. Why does a forbidden-region tail at a step not imply transmitted current?
    5. What happens to the total energy during elastic tunneling?
    6. How many electrons can occupy a 2p subshell?
    """)
    return


@app.cell
def _(mo):
    mo.accordion({"Answers and suggested checks":mo.md(r"""
    1. At fixed frequency, photon flux and the ideal emission rate double.
       The maximum kinetic energy stays fixed.
    2. The wavelength halves because $\lambda\propto KE^{-1/2}$.
    3. $P=1/2+1/\pi\approx0.8183$. Select $n=1$ and $x/a$ from 0.25 to 0.75.
       For $n=2$, the same interval gives 0.5.
    4. A single decaying exponential has zero net probability current.
       A finite barrier also has a transmitted travelling wave beyond its far interface.
    5. Total energy is conserved. The lower transmitted amplitude means a smaller
       transmission probability, not a smaller electron energy.
    6. Six: three spatial orbitals, each with two spin projections.

    """)})
    return


if __name__ == "__main__":
    app.run()
