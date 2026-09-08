# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.5.3",
#     "plotly==7.0.0",
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

    return (go,)


@app.cell
def _(go, np):
    BLUE = "#0072B2"
    ORANGE = "#D55E00"
    GREEN = "#009E73"
    GOLD = "#E69F00"
    SKY = "#56B4E9"
    NAVY = "#0f4c81"

    CUBE_EDGES = [
        ([0, 1], [0, 0], [0, 0]),
        ([0, 0], [0, 1], [0, 0]),
        ([0, 0], [0, 0], [0, 1]),
        ([1, 1], [0, 1], [0, 0]),
        ([1, 1], [0, 0], [0, 1]),
        ([0, 1], [1, 1], [0, 0]),
        ([0, 0], [1, 1], [0, 1]),
        ([0, 1], [0, 0], [1, 1]),
        ([0, 0], [0, 1], [1, 1]),
        ([1, 1], [1, 1], [0, 1]),
        ([1, 1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [1, 1]),
    ]

    APF_CLOSED = {
        "SC": r"\pi/6",
        "BCC": r"\pi\sqrt{3}/8",
        "FCC": r"\pi\sqrt{2}/6",
        "Diamond": r"\pi\sqrt{3}/16",
    }
    TOUCH_ALONG = {
        "SC": "the cube edge",
        "BCC": "the body diagonal",
        "FCC": "the face diagonal",
        "Diamond": r"1/4 of the body diagonal",
    }

    def unique_points(pts, tol=1e-6):
        unique = [pts[0]]
        for p in pts[1:]:
            if all(np.linalg.norm(p - u) > tol for u in unique):
                unique.append(p)
        return np.array(unique)

    def lattice_atom_positions(lattice_type, a=1.0):
        """Conventional-cell atom sites used for drawing (including shared corners)."""
        corners = np.array(
            [
                [0, 0, 0],
                [a, 0, 0],
                [0, a, 0],
                [0, 0, a],
                [a, a, 0],
                [a, 0, a],
                [0, a, a],
                [a, a, a],
            ],
            dtype=float,
        )
        faces = np.array(
            [
                [a / 2, a / 2, 0],
                [a / 2, a / 2, a],
                [a / 2, 0, a / 2],
                [a / 2, a, a / 2],
                [0, a / 2, a / 2],
                [a, a / 2, a / 2],
            ],
            dtype=float,
        )
        if lattice_type == "SC":
            return corners
        if lattice_type == "BCC":
            return np.vstack([corners, [[a / 2, a / 2, a / 2]]])
        if lattice_type == "FCC":
            return np.vstack([corners, faces])
        if lattice_type == "Diamond":
            fcc = np.vstack([corners, faces])
            basis = fcc + np.array([a / 4, a / 4, a / 4])
            inside = np.all((basis >= -1e-6) & (basis <= a + 1e-6), axis=1)
            return np.vstack([fcc, basis[inside]])
        return corners

    def get_lattice_apf_info(lattice_type, a):
        """Hard-sphere radius, atoms per conventional cell, and volumetric APF.

        Atoms touch along the closest-packed direction of that lattice.
        APF is computed from n × (4/3)πr³ / a³, not looked up.
        """
        if lattice_type == "SC":
            r = a / 2.0
            n_atoms = 1
        elif lattice_type == "BCC":
            r = np.sqrt(3.0) * a / 4.0
            n_atoms = 2
        elif lattice_type == "FCC":
            r = np.sqrt(2.0) * a / 4.0
            n_atoms = 4
        elif lattice_type == "Diamond":
            r = np.sqrt(3.0) * a / 8.0
            n_atoms = 8
        else:
            r = a / 2.0
            n_atoms = 1
        apf = n_atoms * (4.0 / 3.0) * np.pi * r**3 / a**3
        return r, n_atoms, apf

    def add_cube_edges(fig, a, color="black", width=2):
        for edge in CUBE_EDGES:
            fig.add_trace(
                go.Scatter3d(
                    x=[a * edge[0][0], a * edge[0][1]],
                    y=[a * edge[1][0], a * edge[1][1]],
                    z=[a * edge[2][0], a * edge[2][1]],
                    mode="lines",
                    line=dict(color=color, width=width),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    def add_hard_sphere(fig, pos, r, color=BLUE, opacity=0.65):
        u_vals = np.linspace(0.0, 2.0 * np.pi, 20)
        v_vals = np.linspace(0.0, np.pi, 15)
        x_sphere = pos[0] + r * np.outer(np.cos(u_vals), np.sin(v_vals)).flatten()
        y_sphere = pos[1] + r * np.outer(np.sin(u_vals), np.sin(v_vals)).flatten()
        z_sphere = pos[2] + r * np.outer(np.ones(len(u_vals)), np.cos(v_vals)).flatten()
        fig.add_trace(
            go.Mesh3d(
                x=x_sphere,
                y=y_sphere,
                z=z_sphere,
                alphahull=0,
                color=color,
                opacity=opacity,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    def create_apf_figure(lattice_type="SC", a=1.0, show_spheres=True):
        r, n_atoms, apf = get_lattice_apf_info(lattice_type, a)
        positions = lattice_atom_positions(lattice_type, a)
        fig = go.Figure()
        add_cube_edges(fig, a)
        if show_spheres:
            for pos in positions:
                add_hard_sphere(fig, pos, r)
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode="markers",
                    marker=dict(
                        size=8, color=BLUE, line=dict(color=NAVY, width=1)
                    ),
                    showlegend=False,
                )
            )
        fig.update_layout(
            title=dict(
                text=(
                    f"{lattice_type}: APF = {apf:.3f} ({100.0 * apf:.1f}%)"
                    f"<br><sub>r = {r:.3f}a, n = {n_atoms} atoms/cell"
                    " | Drag to rotate</sub>"
                ),
                x=0.5,
            ),
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
                aspectmode="cube",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                xaxis=dict(range=[-0.3 * a, 1.3 * a]),
                yaxis=dict(range=[-0.3 * a, 1.3 * a]),
                zaxis=dict(range=[-0.3 * a, 1.3 * a]),
            ),
            height=500,
            width=600,
            margin=dict(l=0, r=0, t=80, b=0),
        )
        return fig

    def make_silicon_density_figure():
        """One diamond unit cell: interior, corner, and face atoms in three colors."""
        corners = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        faces = np.array(
            [
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 1.0],
                [0.5, 0.0, 0.5],
                [0.5, 1.0, 0.5],
                [0.0, 0.5, 0.5],
                [1.0, 0.5, 0.5],
            ]
        )
        interior = np.array(
            [
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.25],
                [0.25, 0.75, 0.75],
                [0.75, 0.25, 0.75],
            ]
        )
        fig = go.Figure()
        for edge in CUBE_EDGES:
            fig.add_trace(
                go.Scatter3d(
                    x=edge[0],
                    y=edge[1],
                    z=edge[2],
                    mode="lines",
                    line=dict(color="gray", width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        subcube = [
            ([0, 0.5], [0, 0], [0, 0]),
            ([0, 0], [0, 0.5], [0, 0]),
            ([0, 0], [0, 0], [0, 0.5]),
            ([0.5, 0.5], [0, 0.5], [0, 0]),
            ([0.5, 0.5], [0, 0], [0, 0.5]),
            ([0, 0.5], [0.5, 0.5], [0, 0]),
            ([0, 0], [0.5, 0.5], [0, 0.5]),
            ([0, 0.5], [0, 0], [0.5, 0.5]),
            ([0, 0], [0, 0.5], [0.5, 0.5]),
            ([0.5, 0.5], [0.5, 0.5], [0, 0.5]),
            ([0.5, 0.5], [0, 0.5], [0.5, 0.5]),
            ([0, 0.5], [0.5, 0.5], [0.5, 0.5]),
        ]
        for edge in subcube:
            fig.add_trace(
                go.Scatter3d(
                    x=edge[0],
                    y=edge[1],
                    z=edge[2],
                    mode="lines",
                    line=dict(color=ORANGE, width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        all_sites = np.vstack([corners, faces, interior])
        bond_max = np.sqrt(3.0) / 4.0 + 0.01
        first_bond = True
        for p in interior:
            for q in all_sites:
                dist = np.linalg.norm(p - q)
                if 0.05 < dist < bond_max:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[p[0], q[0]],
                            y=[p[1], q[1]],
                            z=[p[2], q[2]],
                            mode="lines",
                            line=dict(color=GOLD, width=6),
                            name="Tetrahedral bonds",
                            showlegend=first_bond,
                            hoverinfo="skip",
                        )
                    )
                    first_bond = False
        fig.add_trace(
            go.Scatter3d(
                x=interior[:, 0],
                y=interior[:, 1],
                z=interior[:, 2],
                mode="markers",
                marker=dict(size=12, color=ORANGE),
                name="Completely inside (4)",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=corners[:, 0],
                y=corners[:, 1],
                z=corners[:, 2],
                mode="markers",
                marker=dict(size=12, color=BLUE),
                name="Corners (8)",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=faces[:, 0],
                y=faces[:, 1],
                z=faces[:, 2],
                mode="markers",
                marker=dict(size=12, color=GREEN),
                name="Faces (6)",
            )
        )
        fig.update_layout(
            title=dict(
                text="Si diamond unit cell<br><sub>Drag to rotate | Scroll to zoom</sub>",
                x=0.5,
            ),
            scene=dict(
                xaxis_title="x [a]",
                yaxis_title="y [a]",
                zaxis_title="z [a]",
                aspectmode="cube",
                camera=dict(eye=dict(x=1.7, y=1.5, z=1.2)),
                xaxis=dict(range=[-0.15, 1.15]),
                yaxis=dict(range=[-0.15, 1.15]),
                zaxis=dict(range=[-0.15, 1.15]),
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=520,
            width=560,
            margin=dict(l=0, r=0, t=60, b=0),
        )
        return fig

    return (
        APF_CLOSED,
        BLUE,
        NAVY,
        ORANGE,
        SKY,
        TOUCH_ALONG,
        create_apf_figure,
        get_lattice_apf_info,
        make_silicon_density_figure,
        unique_points,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Packing Density in Crystals
    **ECE335 - companion to lectures**

    Atoms in a crystal are modeled as hard spheres that touch along the
    closest-packed direction. Packing density is the fraction of space those
    spheres occupy — in the 3D unit cell, or in a chosen Miller plane.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Volumetric Packing Fraction

    The **atomic packing factor (APF)** is the fraction of the conventional
    cubic cell occupied by atoms:

    $$\mathrm{APF} = \frac{n \times \tfrac{4}{3}\pi r^3}{a^3}$$

    $n$ is the number of atoms per conventional cell and $r$ is fixed by the
    touching condition. APF does **not** depend on $a$: $r$ scales with $a$.

    | Lattice | Atoms touch along | Atomic radius $r$ | APF |
    |:--------:|:--------------:|:-----------------:|:----:|
    | SC      | Edge              | $a/2$             | $\pi/6 \approx 0.524$ |
    | BCC     | Body diagonal     | $\sqrt{3}\,a/4$   | $\pi\sqrt{3}/8 \approx 0.680$ |
    | FCC     | Face diagonal     | $\sqrt{2}\,a/4$   | $\pi\sqrt{2}/6 \approx 0.740$ |
    | Diamond | ¼ body diagonal   | $\sqrt{3}\,a/8$   | $\pi\sqrt{3}/16 \approx 0.340$ |
    """)
    return


@app.cell
def _(mo):
    lattice_dropdown = mo.ui.dropdown(
        options=["SC", "BCC", "FCC", "Diamond"],
        value="SC",
        label="Lattice type",
    )
    a_slider = mo.ui.slider(
        start=0.5,
        stop=2.0,
        step=0.1,
        value=1.0,
        label="Lattice constant a",
        show_value=True,
    )
    display_dropdown = mo.ui.dropdown(
        options=["Spheres", "Points"],
        value="Spheres",
        label="Display mode",
    )
    apf_controls = mo.vstack(
        [
            mo.md("### Atomic packing factor visualization"),
            mo.hstack(
                [lattice_dropdown, a_slider, display_dropdown],
                justify="start",
                gap=2,
            ),
        ]
    )
    return a_slider, apf_controls, display_dropdown, lattice_dropdown


@app.cell
def _(
    APF_CLOSED,
    TOUCH_ALONG,
    a_slider,
    apf_controls,
    create_apf_figure,
    display_dropdown,
    get_lattice_apf_info,
    lattice_dropdown,
    mo,
):
    _lat = lattice_dropdown.value
    _a = float(a_slider.value)
    _r, _n, _apf = get_lattice_apf_info(_lat, _a)
    _fig = create_apf_figure(
        _lat, _a, show_spheres=(display_dropdown.value == "Spheres")
    )
    _info = mo.md(
        rf"""
    **{_lat}.** Atoms touch along {TOUCH_ALONG[_lat]}.

    $$r = {_r:.4f},\quad n = {_n},\quad
    \mathrm{{APF}} = \frac{{{_n}\times\tfrac{{4}}{{3}}\pi r^3}}{{a^3}}
    = {APF_CLOSED[_lat]} = {_apf:.4f}\;({100.0 * _apf:.2f}\%).$$
    """
    )
    mo.vstack([apf_controls, _fig, _info], align="center")
    return


@app.cell
def _(make_silicon_density_figure, mo):
    _a_nm = 0.543
    _a_cm = _a_nm * 1e-7
    _V = _a_cm**3
    _n_si = 8.0 / _V
    _header = mo.md("## Density of silicon")
    _math = mo.md(
        rf"""
    Silicon is **diamond cubic**. The conventional cube has lattice constant
    $a = 0.543\,\mathrm{{nm}}$.

    **Atoms in one unit cell**

    - 4 atoms completely inside the cell $\rightarrow$ count as $4$
    - 8 atoms on the corners, each shared by 8 cells:
      $8 \times \tfrac{{1}}{{8}} = 1$
    - 6 atoms on the faces, each shared by 2 cells:
      $6 \times \tfrac{{1}}{{2}} = 3$
    - Total: $N = 4 + 1 + 3 = 8$

    **Cell volume**

    $$V = a^3 = (0.543\,\mathrm{{nm}})^3
    = (5.43\times 10^{{-8}}\,\mathrm{{cm}})^3
    = {_V:.2e}\,\mathrm{{cm}}^3$$

    **Density of silicon atoms**

    $$n_{{\mathrm{{Si}}}} = \frac{{N}}{{V}}
    = \frac{{8}}{{{_V:.2e}\,\mathrm{{cm}}^3}}
    = {_n_si:.2e}\,\mathrm{{cm}}^{{-3}}
    \approx 5\times 10^{{22}}\,\mathrm{{cm}}^{{-3}}$$
    """
    )
    _fig = make_silicon_density_figure()
    mo.vstack(
        [
            _header,
            mo.hstack(
                [_math, _fig],
                justify="start",
                align="center",
                widths=[0.48, 0.52],
                gap=2,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Planar Packing Fraction in Miller Planes

    The **planar (areal) packing fraction** is the fraction of a Miller plane
    covered by atoms:

    $$\text{Planar PF} = \frac{n_{\text{atoms}} \times \pi r^2}{A_{\text{2D cell}}}$$

    $n_{\text{atoms}}$ is the number of atoms in the 2D repeat cell of that
    plane and $A_{\text{2D cell}}$ is that cell's area. $r$ is the same
    hard-sphere radius as in the volumetric APF.

    | Lattice | Plane | $n_{\text{atoms}}$ | $r$ | $A_{\text{2D}}$ | Areal PF |
    |:--------:|:-----:|:-----------------:|:---------------:|:---------------:|:--------:|
    | FCC     | (111) | 1                 | $\sqrt{2}\,a/4$ | $\sqrt{3}\,a^2/4$ | 90.7% (close-packed) |
    | FCC     | (100) | 2                 | $\sqrt{2}\,a/4$ | $a^2$           | 78.5% |
    | FCC     | (110) | 2                 | $\sqrt{2}\,a/4$ | $\sqrt{2}\,a^2$ | 55.5% |
    | BCC     | (110) | 2                 | $\sqrt{3}\,a/4$ | $\sqrt{2}\,a^2$ | 83.3% (close-packed) |
    | BCC     | (100) | 1                 | $\sqrt{3}\,a/4$ | $a^2$           | 58.9% |
    | BCC     | (111) | 1                 | $\sqrt{3}\,a/4$ | $\sqrt{3}\,a^2$ | 34.0% |
    """)
    return


@app.cell
def _(
    BLUE,
    NAVY,
    ORANGE,
    SKY,
    get_lattice_apf_info,
    go,
    mo,
    np,
    unique_points,
):
    def generate_sc(n, a=1.0):
        xs, ys, zs = np.mgrid[0:n, 0:n, 0:n]
        return np.vstack((xs.ravel(), ys.ravel(), zs.ravel())).T * a

    def generate_bcc(n, a=1.0):
        sc = generate_sc(n, a)
        centers = np.array(
            [
                [i + 0.5, j + 0.5, k + 0.5]
                for i in range(n)
                for j in range(n)
                for k in range(n)
            ]
        ) * a
        return np.vstack((sc, centers))

    def generate_fcc(n, a=1.0):
        sc = generate_sc(n, a)
        shifts = np.array(
            [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
        )
        faces = np.array(
            [
                (np.array([i, j, k], dtype=float) + s) * a
                for i in range(n)
                for j in range(n)
                for k in range(n)
                for s in shifts
            ]
        )
        return np.vstack((sc, faces))

    def generate_diamond(n, a=1.0):
        fcc_pts = generate_fcc(n, a)
        return np.vstack((fcc_pts, fcc_pts + 0.25 * a))

    def get_plane_cell_intersection(h, k, l, a):
        """Polygon where plane hx + ky + lz = a cuts the cube [0, a]³."""
        if h == 0 and k == 0 and l == 0:
            return np.array([])
        edges = []
        for fixed_dim in range(3):
            for v1 in [0.0, a]:
                for v2 in [0.0, a]:
                    start = np.zeros(3)
                    start[(fixed_dim + 1) % 3] = v1
                    start[(fixed_dim + 2) % 3] = v2
                    direction = np.zeros(3)
                    direction[fixed_dim] = a
                    edges.append((start, direction))

        normal = np.array([h, k, l], dtype=float)
        pts = []
        for start, direction in edges:
            denom = np.dot(normal, direction)
            if abs(denom) < 1e-10:
                continue
            t = (a - np.dot(normal, start)) / denom
            if -1e-10 <= t <= 1.0 + 1e-10:
                pt = start + t * direction
                if all(-1e-10 <= pt[i] <= a + 1e-10 for i in range(3)):
                    pts.append(pt)
        if len(pts) < 3:
            return np.array([])
        pts = unique_points(np.array(pts))
        if len(pts) < 3:
            return np.array([])

        centroid = pts.mean(axis=0)
        v0 = pts[0] - centroid
        v0 = v0 / (np.linalg.norm(v0) + 1e-10)
        v1 = np.cross(normal, v0)
        v1 = v1 / (np.linalg.norm(v1) + 1e-10)
        angles = [
            np.arctan2(np.dot(p - centroid, v1), np.dot(p - centroid, v0))
            for p in pts
        ]
        return pts[np.argsort(angles)]

    def get_atoms_on_plane(lattice_type, h, k, l, a, tol=1e-4):
        if lattice_type == "SC":
            pts = generate_sc(3, a)
        elif lattice_type == "BCC":
            pts = generate_bcc(3, a)
        elif lattice_type == "FCC":
            pts = generate_fcc(3, a)
        elif lattice_type == "Diamond":
            pts = generate_diamond(3, a)
        else:
            pts = generate_sc(3, a)
        distances = np.abs(h * pts[:, 0] + k * pts[:, 1] + l * pts[:, 2] - a)
        on_plane = pts[distances < tol]
        return on_plane

    def get_areal_pf_calculation(lattice_type, h, k, l, a):
        """Planar packing fraction from the 2D cell of plane (hkl).

        Returns (pf, n_atoms, area_2d, r, area_formula, description), or
        Nones if that (lattice, plane) is not tabulated.
        """
        indices = tuple(sorted([abs(h), abs(k), abs(l)], reverse=True))
        r, _, _ = get_lattice_apf_info(lattice_type, a)

        n_atoms = None
        area_2d = None
        area_formula = None
        desc = None
        if lattice_type == "SC":
            if indices == (1, 0, 0):
                n_atoms, area_2d = 1, a**2
                area_formula, desc = "a²", "Square cell: 4 corners × ¼"
            elif indices == (1, 1, 0):
                n_atoms, area_2d = 1, np.sqrt(2.0) * a**2
                area_formula, desc = "√2·a²", "Rectangular cell a × √2 a"
            elif indices == (1, 1, 1):
                n_atoms, area_2d = 1, np.sqrt(3.0) * a**2
                area_formula, desc = "√3·a²", "Hexagonal cell, NN = √2·a"
        elif lattice_type == "FCC":
            if indices == (1, 1, 1):
                n_atoms, area_2d = 1, np.sqrt(3.0) * a**2 / 4.0
                area_formula = "√3·a²/4"
                desc = "Close-packed hexagonal, NN = a/√2"
            elif indices == (1, 0, 0):
                n_atoms, area_2d = 2, a**2
                area_formula = "a²"
                desc = "Square cell: 4 corners × ¼ + 1 center"
            elif indices == (1, 1, 0):
                n_atoms, area_2d = 2, np.sqrt(2.0) * a**2
                area_formula = "√2·a²"
                desc = "Rectangular cell: 4 corners × ¼ + 2 edges × ½"
        elif lattice_type == "BCC":
            if indices == (1, 1, 0):
                n_atoms, area_2d = 2, np.sqrt(2.0) * a**2
                area_formula = "√2·a²"
                desc = "Close-packed for BCC: 4 corners × ¼ + 1 center"
            elif indices == (1, 0, 0):
                n_atoms, area_2d = 1, a**2
                area_formula = "a²"
                desc = "Square cell: 4 corners × ¼ (body center on another plane)"
            elif indices == (1, 1, 1):
                n_atoms, area_2d = 1, np.sqrt(3.0) * a**2
                area_formula = "√3·a²"
                desc = "Hexagonal (SC sublattice only), NN = √2·a"
        elif lattice_type == "Diamond":
            if indices == (1, 1, 1):
                n_atoms, area_2d = 2, np.sqrt(3.0) * a**2 / 4.0
                area_formula = "√3·a²/4"
                desc = "2 atoms per hexagonal cell (FCC + basis)"
            elif indices == (1, 0, 0):
                n_atoms, area_2d = 4, a**2
                area_formula = "a²"
                desc = "Square cell: 4 atoms (FCC + basis layers)"
            elif indices == (1, 1, 0):
                n_atoms, area_2d = 4, np.sqrt(2.0) * a**2
                area_formula = "√2·a²"
                desc = "Rectangular cell with zigzag chains"

        if n_atoms is None:
            return None, None, None, r, None, None
        pf = n_atoms * np.pi * r**2 / area_2d
        return pf, n_atoms, area_2d, r, area_formula, desc

    def plane_basis_vectors(h, k, l):
        normal = np.array([h, k, l], dtype=float)
        normal = normal / np.linalg.norm(normal)
        temp = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array(
            [0.0, 1.0, 0.0]
        )
        u = np.cross(normal, temp)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        return u, v

    def plot_areal_packing(lattice_type="FCC", h=1, k=1, l=1, a=1.0):
        if h == 0 and k == 0 and l == 0:
            return mo.md("**(000) is not a valid Miller index.**")

        r, _, _ = get_lattice_apf_info(lattice_type, a)
        plane_pts = get_plane_cell_intersection(h, k, l, a)
        if len(plane_pts) < 3:
            return mo.md(
                f"**Plane ({h}{k}{l}) does not cut the unit cell properly.**"
            )

        pf, n_atoms, area_2d, r_calc, area_formula, desc = (
            get_areal_pf_calculation(lattice_type, h, k, l, a)
        )
        if pf is None:
            return mo.md(
                f"**Planar packing fraction is not tabulated for "
                f"{lattice_type} ({h}{k}{l}). Try (100), (110), or (111).**"
            )

        fig = go.Figure()
        for edge in [
            ([0, a], [0, 0], [0, 0]),
            ([0, 0], [0, a], [0, 0]),
            ([0, 0], [0, 0], [0, a]),
            ([a, a], [0, a], [0, 0]),
            ([a, a], [0, 0], [0, a]),
            ([0, a], [a, a], [0, 0]),
            ([0, 0], [a, a], [0, a]),
            ([0, a], [0, 0], [a, a]),
            ([0, 0], [0, a], [a, a]),
            ([a, a], [a, a], [0, a]),
            ([a, a], [0, a], [a, a]),
            ([0, a], [a, a], [a, a]),
        ]:
            fig.add_trace(
                go.Scatter3d(
                    x=edge[0],
                    y=edge[1],
                    z=edge[2],
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        n_pts = len(plane_pts)
        fig.add_trace(
            go.Mesh3d(
                x=plane_pts[:, 0],
                y=plane_pts[:, 1],
                z=plane_pts[:, 2],
                i=[0] * (n_pts - 2),
                j=list(range(1, n_pts - 1)),
                k=list(range(2, n_pts)),
                color=SKY,
                opacity=0.4,
                name=f"({h}{k}{l}) plane",
                showlegend=True,
            )
        )
        closed_pts = np.vstack([plane_pts, plane_pts[0]])
        fig.add_trace(
            go.Scatter3d(
                x=closed_pts[:, 0],
                y=closed_pts[:, 1],
                z=closed_pts[:, 2],
                mode="lines",
                line=dict(color=NAVY, width=4),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        atoms_on_plane = get_atoms_on_plane(lattice_type, h, k, l, a)
        if len(atoms_on_plane) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=atoms_on_plane[:, 0],
                    y=atoms_on_plane[:, 1],
                    z=atoms_on_plane[:, 2],
                    mode="markers",
                    marker=dict(
                        size=12, color=BLUE, line=dict(color=NAVY, width=2)
                    ),
                    name="Atoms on plane",
                    showlegend=True,
                )
            )
            u, v = plane_basis_vectors(h, k, l)
            theta = np.linspace(0.0, 2.0 * np.pi, 50)
            for atom in atoms_on_plane:
                circle = np.array(
                    [atom + r * (np.cos(t) * u + np.sin(t) * v) for t in theta]
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=circle[:, 0],
                        y=circle[:, 1],
                        z=circle[:, 2],
                        mode="lines",
                        line=dict(color=ORANGE, width=3),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        fig.update_layout(
            title=dict(
                text=(
                    f"{lattice_type} — plane ({h}{k}{l})"
                    f"<br><sub>Planar packing fraction = {100.0 * pf:.1f}%"
                    " | Drag to rotate</sub>"
                ),
                x=0.5,
            ),
            scene=dict(
                xaxis_title="x [a]",
                yaxis_title="y [a]",
                zaxis_title="z [a]",
                aspectmode="cube",
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
                xaxis=dict(range=[-0.2, a + 0.5]),
                yaxis=dict(range=[-0.2, a + 0.5]),
                zaxis=dict(range=[-0.2, a + 0.5]),
            ),
            height=600,
            width=800,
            margin=dict(l=0, r=0, t=80, b=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        calc_text = mo.md(
            rf"""
    **{lattice_type} ({h}{k}{l}).** {desc}

    - Atomic radius: $r = {r_calc:.4f}\,a$
    - Atoms per 2D cell: $n = {n_atoms}$
    - 2D cell area: $A = {area_formula} = {area_2d:.4f}\,a^2$
    - Planar PF: $n \times \pi r^2 / A = {n_atoms} \times \pi \times ({r_calc:.4f})^2 / {area_2d:.4f} = \mathbf{{{100.0 * pf:.1f}\%}}$
    """
        )
        return mo.vstack([fig, calc_text], align="center")

    lattice_type_dropdown2 = mo.ui.dropdown(
        options=["SC", "BCC", "FCC", "Diamond"],
        value="FCC",
        label="Lattice type",
    )
    h_slider2 = mo.ui.slider(start=0, stop=1, step=1, value=1, label="h")
    k_slider2 = mo.ui.slider(start=0, stop=1, step=1, value=1, label="k")
    l_slider2 = mo.ui.slider(start=0, stop=1, step=1, value=1, label="l")
    planar_controls = mo.vstack(
        [
            mo.md("### Planar packing fraction visualization"),
            mo.md(
                "Choose a lattice and Miller indices $(hkl)$. "
                "Calculated for (100), (110), and (111) (and their permutations)."
            ),
            mo.hstack(
                [lattice_type_dropdown2, h_slider2, k_slider2, l_slider2],
                justify="start",
                gap=2,
            ),
        ]
    )
    return (
        h_slider2,
        k_slider2,
        l_slider2,
        lattice_type_dropdown2,
        planar_controls,
        plot_areal_packing,
    )


@app.cell
def _(
    h_slider2,
    k_slider2,
    l_slider2,
    lattice_type_dropdown2,
    mo,
    planar_controls,
    plot_areal_packing,
):
    mo.vstack(
        [
            planar_controls,
            plot_areal_packing(
                lattice_type=lattice_type_dropdown2.value,
                h=h_slider2.value,
                k=k_slider2.value,
                l=l_slider2.value,
                a=1.0,
            ),
        ],
        align="center",
    )
    return


if __name__ == "__main__":
    app.run()
