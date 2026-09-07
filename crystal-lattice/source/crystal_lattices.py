# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np

    try:
        if "pyodide" in sys.modules:
            raise FileNotFoundError
        _test = Path(__file__).parent / "images"
        if _test.exists():
            ASSET_DIR = Path(__file__).parent
        else:
            raise FileNotFoundError
    except Exception:
        ASSET_DIR = None

    IMAGE_BASE = (
        "https://ece335.github.io/crystal-lattice/images"
        if ASSET_DIR is None
        else str(ASSET_DIR / "images")
    )
    return IMAGE_BASE, mo, np


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
def _(go, np):
    CUBE_CORNERS = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )
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
    FCC_FACE_CENTERS = np.array(
        [
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [0.5, 0.5, 1],
            [0.5, 1, 0.5],
            [1, 0.5, 0.5],
        ]
    )
    ARROW_COLORS = ["#D55E00", "#009E73", "#E69F00"]

    def unique_atoms(atoms, tol=0.01):
        unique = [atoms[0]]
        for atom in atoms[1:]:
            if not any(np.linalg.norm(atom - u) < tol for u in unique):
                unique.append(atom)
        return np.array(unique)

    def add_arrow(fig, start, end, color, name, row, col):
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=dict(color=color, width=6),
                name=name,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        direction = np.array(end) - np.array(start)
        fig.add_trace(
            go.Cone(
                x=[end[0]],
                y=[end[1]],
                z=[end[2]],
                u=[direction[0] * 0.3],
                v=[direction[1] * 0.3],
                w=[direction[2] * 0.3],
                colorscale=[[0, color], [1, color]],
                showscale=False,
                sizemode="absolute",
                sizeref=0.15,
                name=name,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    def add_cube_edges(fig, row, col, color="gray", width=2):
        for edge in CUBE_EDGES:
            fig.add_trace(
                go.Scatter3d(
                    x=edge[0],
                    y=edge[1],
                    z=edge[2],
                    mode="lines",
                    line=dict(color=color, width=width),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    def generate_interpenetrating_fcc(n_cells=2, offset=(0.25, 0.25, 0.25)):
        fcc1_list = []
        fcc2_list = []
        offset = np.array(offset)
        for ix in range(n_cells):
            for iy in range(n_cells):
                for iz in range(n_cells):
                    shift = np.array([ix, iy, iz])
                    fcc1_list.append(CUBE_CORNERS + shift)
                    fcc1_list.append(FCC_FACE_CENTERS + shift)
                    fcc2_list.append(CUBE_CORNERS + offset + shift)
                    fcc2_list.append(FCC_FACE_CENTERS + offset + shift)
        fcc1 = unique_atoms(np.vstack(fcc1_list))
        fcc2 = np.vstack(fcc2_list)
        margin = 0.01
        xmax = n_cells
        mask = np.all(
            (fcc2 >= -margin) & (fcc2 <= np.array([xmax, xmax, xmax]) + margin),
            axis=1,
        )
        fcc2 = unique_atoms(fcc2[mask])
        return fcc1, fcc2

    def make_interpenetrating_fcc_figure(
        name1,
        name2,
        color1,
        color2,
        title,
        n_cells=2,
    ):
        fcc1, fcc2 = generate_interpenetrating_fcc(n_cells=n_cells)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=fcc1[:, 0],
                y=fcc1[:, 1],
                z=fcc1[:, 2],
                mode="markers",
                marker=dict(size=10, color=color1, opacity=0.9),
                name=name1,
                visible=True,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=fcc2[:, 0],
                y=fcc2[:, 1],
                z=fcc2[:, 2],
                mode="markers",
                marker=dict(size=10, color=color2, opacity=0.9),
                name=name2,
                visible=True,
            )
        )

        bond_length = np.sqrt(3) / 4 + 0.01
        bond_start = 2
        for p1 in fcc2:
            for p2 in fcc1:
                if np.linalg.norm(p1 - p2) < bond_length:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[p1[0], p2[0]],
                            y=[p1[1], p2[1]],
                            z=[p1[2], p2[2]],
                            mode="lines",
                            line=dict(color="gray", width=3),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
        bond_end = len(fig.data)
        num_bonds = bond_end - bond_start

        for ix in range(n_cells):
            for iy in range(n_cells):
                for iz in range(n_cells):
                    cube_edges = [
                        ([0 + ix, 1 + ix], [0 + iy, 0 + iy], [0 + iz, 0 + iz]),
                        ([0 + ix, 0 + ix], [0 + iy, 1 + iy], [0 + iz, 0 + iz]),
                        ([0 + ix, 0 + ix], [0 + iy, 0 + iy], [0 + iz, 1 + iz]),
                        ([1 + ix, 1 + ix], [0 + iy, 1 + iy], [0 + iz, 0 + iz]),
                        ([1 + ix, 1 + ix], [0 + iy, 0 + iy], [0 + iz, 1 + iz]),
                        ([0 + ix, 1 + ix], [1 + iy, 1 + iy], [0 + iz, 0 + iz]),
                        ([0 + ix, 0 + ix], [1 + iy, 1 + iy], [0 + iz, 1 + iz]),
                        ([0 + ix, 1 + ix], [0 + iy, 0 + iy], [1 + iz, 1 + iz]),
                        ([0 + ix, 0 + ix], [0 + iy, 1 + iy], [1 + iz, 1 + iz]),
                        ([1 + ix, 1 + ix], [1 + iy, 1 + iy], [0 + iz, 1 + iz]),
                        ([1 + ix, 1 + ix], [0 + iy, 1 + iy], [1 + iz, 1 + iz]),
                        ([0 + ix, 1 + ix], [1 + iy, 1 + iy], [1 + iz, 1 + iz]),
                    ]
                    for edge in cube_edges:
                        fig.add_trace(
                            go.Scatter3d(
                                x=edge[0],
                                y=edge[1],
                                z=edge[2],
                                mode="lines",
                                line=dict(color="black", width=1.5, dash="dash"),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )

        total = len(fig.data)
        num_edges = total - bond_end
        vis_all = [True] * total
        vis_fcc1 = [True, False] + [False] * num_bonds + [True] * num_edges
        vis_fcc2 = [False, True] + [False] * num_bonds + [True] * num_edges
        vis_no_bonds = [True, True] + [False] * num_bonds + [True] * num_edges

        fig.update_layout(
            title=dict(
                text=(
                    f"{title}<br><sub>Two interpenetrating FCC lattices"
                    " | Click and drag to rotate | Scroll to zoom</sub>"
                ),
                x=0.5,
            ),
            scene=dict(
                xaxis_title="x [a]",
                yaxis_title="y [a]",
                zaxis_title="z [a]",
                aspectmode="data",
                camera=dict(eye=dict(x=2.0, y=2.0, z=1.5)),
                xaxis=dict(range=[-0.2, n_cells + 0.3]),
                yaxis=dict(range=[-0.2, n_cells + 0.3]),
                zaxis=dict(range=[-0.2, n_cells + 0.3]),
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                itemclick="toggle",
                itemdoubleclick="toggleothers",
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    x=0.5,
                    y=-0.05,
                    xanchor="center",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Show Both",
                            method="update",
                            args=[{"visible": vis_all}],
                        ),
                        dict(
                            label=f"{name1} only",
                            method="update",
                            args=[{"visible": vis_fcc1}],
                        ),
                        dict(
                            label=f"{name2} only",
                            method="update",
                            args=[{"visible": vis_fcc2}],
                        ),
                        dict(
                            label="Both (no bonds)",
                            method="update",
                            args=[{"visible": vis_no_bonds}],
                        ),
                    ],
                )
            ],
            height=750,
            width=900,
            margin=dict(l=0, r=0, t=80, b=80),
        )
        return fig

    return (
        ARROW_COLORS,
        CUBE_CORNERS,
        FCC_FACE_CENTERS,
        add_arrow,
        add_cube_edges,
        make_interpenetrating_fcc_figure,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Crystal Lattices
    **ECE335 — Introduction to Electronic Devices · Lecture 1**

    **References:** Neamen, *Semiconductor Physics and Devices*, Ch. 1, Sec. 1.1–1.6.
    """)
    return


@app.cell
def _(IMAGE_BASE, mo):
    _header = mo.md(
        r"""
    ## 1. Semiconductors

    Electronic devices are built in **semiconductors**. A semiconductor has an
    electrical resistivity that we can **engineer** — by doping, temperature, or light —
    unlike a metal (always conducting) or an insulator (always insulating).
    """
    )
    _table = mo.md(
        r"""
    | Material type | Resistivity |
    |:---:|:---:|
    | **Insulators** | $\rho > 10^{8}\ \Omega\cdot\mathrm{cm}$ |
    | **Conductors** | $\rho < 10^{-3}\ \Omega\cdot\mathrm{cm}$ |
    | **Semiconductors** | **controllable** |
    """
    )
    _img = mo.hstack(
        [mo.image(src=f"{IMAGE_BASE}/resistivity_chart.jpg", width="70%")],
        justify="center",
    )
    mo.vstack(
        [
            _header,
            mo.hstack([_table, _img], justify="start", align="center", gap=2),
        ]
    )
    return


@app.cell
def _(IMAGE_BASE, mo):
    _header = mo.md(
        r"""
    ## 2. Types of solids

    A **crystal** is an infinite periodic arrangement of atoms (or groups of atoms)
    in space. It has **discrete translational invariance**. Silicon used in
    integrated circuits is a single crystal.
    """
    )

    def _col(title, filename, caption):
        return mo.vstack(
            [
                mo.md(f"**{title}**"),
                mo.hstack(
                    [mo.image(src=f"{IMAGE_BASE}/{filename}", width=200)],
                    justify="center",
                ),
                mo.hstack(
                    [
                        mo.md(
                            f'<span style="font-size:16px; color:#555">{caption}</span>'
                        )
                    ],
                    justify="center",
                ),
            ],
            align="center",
            gap=0.3,
        )

    _table = mo.md(
        r"""
    | Type | Description | Example | Applications |
    |:---:|:---:|:---:|:---:|
    | **Crystalline** | Periodic, long-range order | Single-crystal Si | Most semiconductor devices |
    | **Polycrystalline** | Crystal domains, short-range order | Poly-Si | Transistor gates, resistors |
    | **Amorphous** | No crystal structure | a-Si | Low-cost solar cells |
    """
    )
    _photos = mo.hstack(
        [
            _col("Crystalline", "crystalline-Si.jpg", "Single-crystal Si"),
            _col("Polycrystalline", "poly-Si.jpg", "Poly-Si gates"),
            _col("Amorphous", "a-Si.jpg", "a-Si solar cells"),
        ],
        justify="space-around",
    )
    mo.vstack([_header, _table, _photos])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Crystal definitions

    | Term | Definition |
    |:---:|:---|
    | **Lattice** | A set of points in space, $\vec{R} = u_1\vec{a}_1 + u_2\vec{a}_2 + u_3\vec{a}_3$, with integers $\{u_i\}$ and primitive vectors $\{\vec{a}_i\}$ |
    | **Basis** | The group of atoms attached to each lattice point, $\vec{b}_j = v_{j,1}\vec{a}_1 + v_{j,2}\vec{a}_2 + v_{j,3}\vec{a}_3$ |
    | **Unit cell** | The volume that is repeated to fill all space |
    | **Primitive cell** | The smallest unit cell (one lattice point per cell) |
    | **Lattice constant** | The periodicity of the lattice, usually written $a$ |

    $$\boxed{\textbf{Lattice} + \textbf{Basis} = \textbf{Crystal structure}}$$

    There is only **one lattice point per primitive cell**, but lattice points on
    faces and edges are shared among adjacent conventional cells. The primitive
    cell is not unique; the number of atoms in a primitive cell is.
    """)
    return


@app.cell
def _(IMAGE_BASE, mo):
    def _panel(title, filename):
        return mo.vstack(
            [
                mo.md(f"**{title}**"),
                mo.image(src=f"{IMAGE_BASE}/{filename}", width="100%"),
            ],
            align="start",
        )

    mo.hstack(
        [
            _panel("2D square lattice", "lattice_2Dsquare.jpg"),
            _panel("1-atom basis", "basis_1atom.jpg"),
            _panel("2-atom basis", "basis_2atom.jpg"),
            _panel("Unit cell", "unitcell.jpg"),
        ],
        align="start",
        widths="equal",
        justify="space-between",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 3D crystal lattices

    There are **14 Bravais lattices**. The cubic ones that matter for
    semiconductors are below. Click and drag the plots to rotate.

    | Lattice | Atoms in the conventional cube | Atoms per conventional cell |
    |:---|:---|:---:|
    | **Simple cubic (SC)** | Corners | $8\times\tfrac{1}{8} = 1$ |
    | **Body-centered cubic (BCC)** | Corners + body center | $1 + 1 = 2$ |
    | **Face-centered cubic (FCC)** | Corners + face centers | $1 + 6\times\tfrac{1}{2} = 4$ |

    Colored arrows are a set of **primitive vectors** $\vec{a}_1,\vec{a}_2,\vec{a}_3$.
    """)
    return


@app.cell
def _(
    ARROW_COLORS,
    CUBE_CORNERS,
    FCC_FACE_CENTERS,
    add_arrow,
    add_cube_edges,
    go,
    make_subplots,
):
    _fig3d = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=(
            "Simple Cubic (SC)",
            "Body-Centered Cubic (BCC)",
            "Face-Centered Cubic (FCC)",
        ),
        horizontal_spacing=0.05,
    )

    _fig3d.add_trace(
        go.Scatter3d(
            x=CUBE_CORNERS[:, 0],
            y=CUBE_CORNERS[:, 1],
            z=CUBE_CORNERS[:, 2],
            mode="markers",
            marker=dict(size=10, color="#0072B2"),
            name="Corner atoms",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    add_cube_edges(_fig3d, 1, 1)
    for _i, (_start, _end) in enumerate(
        [([0, 0, 0], [1, 0, 0]), ([0, 0, 0], [0, 1, 0]), ([0, 0, 0], [0, 0, 1])]
    ):
        add_arrow(_fig3d, _start, _end, ARROW_COLORS[_i], rf"a_{_i+1}", 1, 1)

    _fig3d.add_trace(
        go.Scatter3d(
            x=CUBE_CORNERS[:, 0],
            y=CUBE_CORNERS[:, 1],
            z=CUBE_CORNERS[:, 2],
            mode="markers",
            marker=dict(size=10, color="#0072B2"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    _fig3d.add_trace(
        go.Scatter3d(
            x=[0.5],
            y=[0.5],
            z=[0.5],
            mode="markers",
            marker=dict(size=12, color="#D55E00"),
            name="Center atom",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    add_cube_edges(_fig3d, 1, 2)
    _bcc_primitive = [
        ([0.5, 0.5, 0.5], [1, 1, 0]),
        ([0.5, 0.5, 0.5], [1, 0, 1]),
        ([0.5, 0.5, 0.5], [0, 1, 1]),
    ]
    for _i, (_start, _end) in enumerate(_bcc_primitive):
        add_arrow(_fig3d, _start, _end, ARROW_COLORS[_i], rf"a_{_i+1}", 1, 2)

    _fig3d.add_trace(
        go.Scatter3d(
            x=CUBE_CORNERS[:, 0],
            y=CUBE_CORNERS[:, 1],
            z=CUBE_CORNERS[:, 2],
            mode="markers",
            marker=dict(size=10, color="#0072B2"),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    _fig3d.add_trace(
        go.Scatter3d(
            x=FCC_FACE_CENTERS[:, 0],
            y=FCC_FACE_CENTERS[:, 1],
            z=FCC_FACE_CENTERS[:, 2],
            mode="markers",
            marker=dict(size=10, color="#009E73"),
            name="Face-center atoms",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    add_cube_edges(_fig3d, 1, 3)
    _fcc_primitive = [
        ([0, 0, 0], [0.5, 0.5, 0]),
        ([0, 0, 0], [0.5, 0, 0.5]),
        ([0, 0, 0], [0, 0.5, 0.5]),
    ]
    for _i, (_start, _end) in enumerate(_fcc_primitive):
        add_arrow(_fig3d, _start, _end, ARROW_COLORS[_i], rf"a_{_i+1}", 1, 3)

    for _color, _label in zip(ARROW_COLORS, [r"$\vec{a}_1$", r"$\vec{a}_2$", r"$\vec{a}_3$"]):
        _fig3d.add_trace(
            go.Scatter3d(
                x=[None, None],
                y=[None, None],
                z=[None, None],
                mode="lines",
                line=dict(width=3, color=_color),
                name=_label,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    _camera = dict(eye=dict(x=1.5, y=1.5, z=1.5))
    _scene = dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z",
        aspectmode="cube",
        camera=_camera,
    )
    _fig3d.update_layout(
        height=500,
        width=1200,
        title_text="Interactive 3D cubic lattices with primitive vectors",
        legend=dict(yanchor="top", y=-0.2, xanchor="left", x=0.0, orientation="h"),
        margin=dict(b=80),
        scene=_scene,
        scene2=_scene,
        scene3=_scene,
    )
    _fig3d
    return


@app.cell
def _(IMAGE_BASE, mo):
    _header = mo.md("### 4.1 Diamond")
    _text = mo.md(
        r"""
    Silicon and germanium (Group IV) crystallize in the **diamond** structure:

    - FCC Bravais lattice with a **2-atom basis**
      - $\vec{b}_1 = (0,0,0)$
      - $\vec{b}_2 = \bigl(\tfrac{1}{4},\tfrac{1}{4},\tfrac{1}{4}\bigr)a$
    - Equivalently: **two interpenetrating FCC lattices** offset by $a/4$ along the body diagonal
    - **8 atoms** per conventional cubic cell
    - Each atom has **4 nearest neighbours** (tetrahedral bonding)
    """
    )
    _img = mo.hstack(
        [mo.image(src=f"{IMAGE_BASE}/diamond_structure.jpg", width="50%")],
        justify="center",
    )
    mo.vstack(
        [
            _header,
            mo.hstack([_text, _img], justify="start", align="center", gap=2),
        ]
    )
    return


@app.cell
def _(make_interpenetrating_fcc_figure):
    make_interpenetrating_fcc_figure(
        name1="FCC lattice 1 (blue)",
        name2="FCC lattice 2 (red, offset by a/4)",
        color1="#0072B2",
        color2="#D55E00",
        title="Diamond lattice: 8 unit cells (2×2×2)",
    )
    return


@app.cell
def _(IMAGE_BASE, mo):
    _header = mo.md("### 4.2 Zinc blende")
    _text = mo.md(
        r"""
    **Zinc blende** is the same geometry as diamond, but the two basis atoms
    are different species — typically a **Group III** atom and a **Group V** atom.

    - Same FCC + 2-atom basis as diamond
    - One FCC sublattice is the cation (e.g. Ga, In); the other is the anion (e.g. As, P)
    - **Examples:** GaAs, InP, ZnS
    - III–V compounds are important for **optoelectronics** and **high-speed electronics**
    """
    )
    _img = mo.hstack(
        [mo.image(src=f"{IMAGE_BASE}/zinc_blende_structure.jpg", width="40%")],
        justify="center",
    )
    mo.vstack(
        [
            _header,
            mo.hstack([_text, _img], justify="start", align="center", gap=2),
        ]
    )
    return


@app.cell
def _(make_interpenetrating_fcc_figure):
    make_interpenetrating_fcc_figure(
        name1="Group III (Ga)",
        name2="Group V (As)",
        color1="#0072B2",
        color2="#E69F00",
        title="Zinc blende (GaAs): 8 unit cells (2×2×2)",
    )
    return


@app.cell
def _(IMAGE_BASE, mo):
    _header = mo.md(
        r"""
    ## 5. Elemental and compound semiconductors

    - **Elemental** semiconductors are a single Group IV element: Si, Ge, C (diamond)
    - **Compound** semiconductors combine two (or more) elements
      - III–V: GaAs, InP, GaN, Al$_x$Ga$_{1-x}$As
      - II–VI: CdTe, ZnSe
    """
    )
    _img = mo.hstack(
        [mo.image(src=f"{IMAGE_BASE}/periodictable.jpg", width="70%")],
        justify="center",
    )
    _materials = mo.hstack(
        [mo.image(src=f"{IMAGE_BASE}/semiconductormaterials.jpg", width="80%")],
        justify="center",
    )
    mo.vstack([_header, _img, _materials], gap=1)
    return


@app.cell
def _(IMAGE_BASE, mo):
    _header = mo.md(
        r"""
    ## 6. Atomic bonding in semiconductors

    Silicon (atomic number 14) has **4 valence electrons**. In the diamond lattice
    each Si atom forms **covalent bonds** with 4 neighbours: the atoms share
    electron pairs so that each has a filled outer shell.

    - **Coordination number** = number of nearest neighbours = **4** for diamond and zinc blende
    - In GaAs, Ga (3 valence electrons) and As (5 valence electrons) share electrons
      in the same tetrahedral geometry — that is zinc blende
    """
    )
    _img = mo.hstack(
        [mo.image(src=f"{IMAGE_BASE}/diamond_coordinates.jpg", width="50%")],
        justify="center",
    )
    _bonds = mo.md(
        r"""
    Other bonding types (for context):

    | Bond | What holds the crystal together | Example |
    |:---|:---|:---|
    | **Covalent** | Shared electron pairs | Si, Ge, diamond, GaAs |
    | **Ionic** | Coulomb attraction of opposite ions | NaCl |
    | **Metallic** | Delocalized valence electrons | Na, Al, Cu |
    | **van der Waals** | Dipole–dipole attraction | Molecular crystals, layered 2D materials |
    """
    )
    mo.vstack([_header, _img, _bonds])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Imperfections and impurities

    Real crystals are not perfect. The defects we care about in devices are:

    - **Vacancies** — a missing atom on a lattice site
    - **Interstitials** — an extra atom squeezed between lattice sites
    - **Substitutional impurities** — a foreign atom on a lattice site

    **Doping** is the controlled introduction of substitutional impurities:

    - Group V on a Si site (P, As, Sb) → extra electron → **n-type**
    - Group III on a Si site (B, Al, Ga) → missing electron (hole) → **p-type**

    This is how we make the resistivity of a semiconductor **engineerable**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaways

    1. Devices are built in **crystals**. Structure and orientation affect performance.
    2. **Lattice + basis = crystal structure.**
    3. Si and Ge: **diamond** (FCC + 2-atom basis). III–V compounds: **zinc blende**.
    4. In both, each atom has **4 nearest neighbours** (tetrahedral covalent bonding).
    5. **Doping** replaces lattice atoms with Group III or V impurities and sets the carrier type.
    """)
    return


if __name__ == "__main__":
    app.run()
