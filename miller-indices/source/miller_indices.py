# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
#     "matplotlib",
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
        "https://ece335.github.io/miller-indices/images"
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
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    return go, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Miller Indices
    **ECE335 - companion to lectures**

    Miller indices label **planes** and **directions** in a crystal. Device properties
    (cleavage, oxidation rate, carrier mobility, wafer flats) depend on which plane
    and direction you use.
    """)
    return


@app.cell
def _(IMAGE_BASE, mo):
    _header = mo.md(
        r"""
    ## Planes: $(hkl)$

    Recipe, for a cubic unit cell of side $a$:

    1. Put the origin on a lattice point, with axes along the cube edges.
    2. Find where the plane cuts the axes, in units of $a$. Example: $(a, 3a, 2a)$.
    3. Take **reciprocals**: $\bigl(1, \tfrac{1}{3}, \tfrac{1}{2}\bigr)$.
    4. Clear fractions (lowest common multiple): $(6,2,3)$.
    5. Write in **round brackets**: $(623)$.
    """
    )
    _img = mo.hstack(
        [mo.image(src=f"{IMAGE_BASE}/miller_plane_623.jpg", width=300)],
        justify="center",
    )
    mo.vstack(
        [
            _header,
            mo.hstack([_img], justify="center"),
        ]
    )
    return


@app.cell
def _(IMAGE_BASE, mo):
    _header = mo.md(
        r"""
    ## Special cases

    - **No intercept** (plane parallel to an axis): treat the intercept as $\infty$, so the reciprocal is $0$. A plane parallel to $z$ is $(hk0)$.
    - **Negative intercept**: overbar, e.g. $(\bar{1}00)$.
    - **Family of equivalent planes** (cubic): curly brackets. $\{100\}$ means $(100)$, $(010)$, $(001)$, $(\bar{1}00)$, $(0\bar{1}0)$, $(00\bar{1})$.
    """
    )
    _img = mo.hstack(
        [mo.image(src=f"{IMAGE_BASE}/miller_planes_100_110_111.jpg", width="60%")],
        justify="center",
    )
    mo.vstack([_header, _img])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Directions: $[hkl]$

    A lattice direction is the vector

    $$\vec{r} = h\,\vec{a}_1 + k\,\vec{a}_2 + l\,\vec{a}_3$$

    | Object | Brackets | Example |
    |:---|:---:|:---|
    | One plane | $(hkl)$ | $(111)$ |
    | Family of equivalent planes | $\{hkl\}$ | $\{111\}$ |
    | One direction | $[hkl]$ | $[110]$ |
    | Family of equivalent directions | $\langle hkl\rangle$ | $\langle 110\rangle$ |

    **In a cubic crystal, $[hkl]$ is perpendicular to the plane $(hkl)$.**
    """)
    return


@app.cell
def _(np, plt):
    _fig, _ax = plt.subplots(figsize=(4.2, 4.2))
    _origin = np.array([0.15, 0.15])
    _x_end = np.array([0.85, 0.15])
    _y_end = np.array([0.75, 0.45])
    _z_end = np.array([0.15, 0.85])
    _x_int = _origin + 0.5 * (_x_end - _origin)
    _y_int = _origin + 0.7 * (_y_end - _origin)
    _z_int = _origin + 0.6 * (_z_end - _origin)
    _arrow = dict(
        head_width=0.025, head_length=0.02, fc="black", ec="black", linewidth=1.5
    )
    _ax.arrow(
        _origin[0],
        _origin[1],
        (_x_end - _origin)[0] * 0.95,
        (_x_end - _origin)[1],
        **_arrow,
    )
    _ax.arrow(
        _origin[0],
        _origin[1],
        (_y_end - _origin)[0] * 0.95,
        (_y_end - _origin)[1] * 0.95,
        **_arrow,
    )
    _ax.arrow(
        _origin[0],
        _origin[1],
        (_z_end - _origin)[0],
        (_z_end - _origin)[1] * 0.95,
        **_arrow,
    )
    _ax.text(_x_end[0] + 0.03, _x_end[1], r"$x$", fontsize=16, ha="left", va="center")
    _ax.text(_y_end[0] + 0.03, _y_end[1] + 0.02, r"$y$", fontsize=16, ha="left", va="center")
    _ax.text(_z_end[0], _z_end[1] + 0.05, r"$z$", fontsize=16, ha="center", va="bottom")
    _ax.add_patch(
        plt.Polygon(
            [_x_int, _y_int, _z_int],
            fill=True,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=2,
            alpha=0.7,
        )
    )
    _ax.annotate(
        "",
        xy=_x_int,
        xytext=_z_int,
        arrowprops=dict(arrowstyle="->", color="#C44536", lw=2.5),
    )
    _ax.annotate(
        "",
        xy=_x_int,
        xytext=_y_int,
        arrowprops=dict(arrowstyle="->", color="#C44536", lw=2.5),
    )
    _ax.text(_x_int[0], _x_int[1] - 0.06, r"$\frac{m}{h}$", fontsize=16, ha="center", va="top")
    _ax.text(_y_int[0] + 0.05, _y_int[1], r"$\frac{m}{k}$", fontsize=16, ha="left", va="center")
    _ax.text(_z_int[0] - 0.05, _z_int[1], r"$\frac{m}{l}$", fontsize=16, ha="right", va="center")
    _v1_mid = 0.5 * (_z_int + _x_int)
    _v2_mid = 0.5 * (_y_int + _x_int)
    _ax.text(
        _v1_mid[0] - 0.02,
        _v1_mid[1] + 0.06,
        r"$\vec{v}_1 = \left(\frac{m}{h}\vec{a}_1 - \frac{m}{l}\vec{a}_3\right)$",
        fontsize=11,
        ha="center",
        va="bottom",
    )
    _ax.text(
        _v2_mid[0] + 0.12,
        _v2_mid[1] - 0.03,
        r"$\vec{v}_2 = \left(\frac{m}{h}\vec{a}_1 - \frac{m}{k}\vec{a}_2\right)$",
        fontsize=11,
        ha="left",
        va="top",
    )
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1)
    _ax.set_aspect("equal")
    _ax.axis("off")
    _fig.tight_layout()
    miller_proof_fig = _fig
    return (miller_proof_fig,)


@app.cell
def _(miller_proof_fig, mo):
    _header = mo.md(r"## Why $(hkl) \perp [hkl]$ in a cubic crystal")
    _text = mo.md(
        r"""
    1. Intercepts of $(hkl)$ are $\tfrac{m}{h}$, $\tfrac{m}{k}$, $\tfrac{m}{l}$ along $x,y,z$ ($m$ an integer).
    2. Two vectors in the plane:
       $\vec{v}_1 = \tfrac{m}{h}\vec{a}_1 - \tfrac{m}{l}\vec{a}_3$,
       $\vec{v}_2 = \tfrac{m}{h}\vec{a}_1 - \tfrac{m}{k}\vec{a}_2$.
    3. Any vector in the plane is $\vec{s} = n_1\vec{v}_1 + n_2\vec{v}_2$.
    4. $\vec{s}\cdot(h\vec{a}_1 + k\vec{a}_2 + l\vec{a}_3) = 0$.
    5. So every in-plane vector is perpendicular to $[hkl]$: the plane $(hkl)$ is normal to $[hkl]$.
    """
    )
    mo.vstack(
        [
            _header,
            mo.hstack(
                [_text, miller_proof_fig],
                justify="start",
                align="center",
                widths=[0.58, 0.42],
                gap=2,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Useful formulas (cubic, lattice constant $a$)

    Angle between directions $[x_1 y_1 z_1]$ and $[x_2 y_2 z_2]$:

    $$\cos\alpha = \frac{x_1 x_2 + y_1 y_2 + z_1 z_2}{\sqrt{x_1^2+y_1^2+z_1^2}\,\sqrt{x_2^2+y_2^2+z_2^2}}$$

    Example: $[100]$ and $[110]$ give $\alpha = 45^\circ$.

    Spacing of adjacent $(hkl)$ planes:

    $$\boxed{d_{(hkl)} = \dfrac{a}{\sqrt{h^2 + k^2 + l^2}}}$$

    | Planes | $d$ |
    |:---:|:---:|
    | $(100)$ | $a$ |
    | $(110)$ | $a/\sqrt{2}$ |
    | $(111)$ | $a/\sqrt{3}$ |
    """)
    return


@app.cell
def _(go, mo, np):
    def miller_plain_index(n):
        if n < 0:
            return f"{abs(int(n))}\u0305"
        return str(int(n))

    def miller_plain(h, k, l, kind="plane"):
        s = miller_plain_index(h) + miller_plain_index(k) + miller_plain_index(l)
        if kind == "plane":
            return f"({s})"
        if kind == "family":
            return f"\u27e8{s}\u27e9"
        return f"[{s}]"

    def miller_tex_index(n):
        if n < 0:
            return rf"\bar{{{abs(int(n))}}}"
        return str(int(n))

    def miller_tex(h, k, l, kind="plane"):
        s = miller_tex_index(h) + miller_tex_index(k) + miller_tex_index(l)
        if kind == "plane":
            return rf"({s})"
        if kind == "family":
            return rf"\langle{s}\rangle"
        return rf"[{s}]"

    def miller_html_index(n):
        if n < 0:
            return (
                f"<span style='text-decoration:overline'>{abs(int(n))}</span>"
            )
        return str(int(n))

    def miller_html(h, k, l, kind="plane"):
        s = miller_html_index(h) + miller_html_index(k) + miller_html_index(l)
        if kind == "plane":
            return f"({s})"
        if kind == "family":
            return f"&langle;{s}&rangle;"
        return f"[{s}]"

    def plot_miller_plane(h, k, l):
        """Plot the Miller plane in a unit cell by showing intercepts."""
        fig_miller = go.Figure()
        plane_label = miller_plain(h, k, l, "plane")
        dir_label = miller_plain(h, k, l, "direction")

        _box = 1.0
        _grid = [-_box, 0.0, _box]
        cube_edges = []
        for _a in _grid:
            for _b in _grid:
                cube_edges.append(([-_box, _box], [_a, _a], [_b, _b]))
                cube_edges.append(([_a, _a], [-_box, _box], [_b, _b]))
                cube_edges.append(([_a, _a], [_b, _b], [-_box, _box]))

        for edge in cube_edges:
            fig_miller.add_trace(
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

        fig_miller.add_trace(
            go.Scatter3d(
                x=[-_box, _box],
                y=[0, 0],
                z=[0, 0],
                mode="lines",
                line=dict(color="red", width=4),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_miller.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[-_box, _box],
                z=[0, 0],
                mode="lines",
                line=dict(color="green", width=4),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_miller.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[-_box, _box],
                mode="lines",
                line=dict(color="blue", width=4),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        _lab = _box + 0.2
        fig_miller.add_trace(
            go.Scatter3d(
                x=[_lab],
                y=[0],
                z=[0],
                mode="text",
                text=["x"],
                textfont=dict(size=14, color="red"),
                showlegend=False,
            )
        )
        fig_miller.add_trace(
            go.Scatter3d(
                x=[0],
                y=[_lab],
                z=[0],
                mode="text",
                text=["y"],
                textfont=dict(size=14, color="green"),
                showlegend=False,
            )
        )
        fig_miller.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[_lab],
                mode="text",
                text=["z"],
                textfont=dict(size=14, color="blue"),
                showlegend=False,
            )
        )

        if h == 0 and k == 0 and l == 0:
            fig_miller.add_annotation(
                text="Invalid: (0,0,0) is not a valid Miller index",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="red"),
            )
        else:
            x_int = 1.0 / h if h != 0 else None
            y_int = 1.0 / k if k != 0 else None
            z_int = 1.0 / l if l != 0 else None
            intercept_points = []

            if x_int is not None and abs(x_int) <= _box + 1e-9:
                fig_miller.add_trace(
                    go.Scatter3d(
                        x=[x_int],
                        y=[0],
                        z=[0],
                        mode="markers",
                        marker=dict(size=10, color="red"),
                        name=f"x-intercept: {x_int:.2f}a",
                    )
                )
                intercept_points.append([x_int, 0, 0])
            if y_int is not None and abs(y_int) <= _box + 1e-9:
                fig_miller.add_trace(
                    go.Scatter3d(
                        x=[0],
                        y=[y_int],
                        z=[0],
                        mode="markers",
                        marker=dict(size=10, color="green"),
                        name=f"y-intercept: {y_int:.2f}a",
                    )
                )
                intercept_points.append([0, y_int, 0])
            if z_int is not None and abs(z_int) <= _box + 1e-9:
                fig_miller.add_trace(
                    go.Scatter3d(
                        x=[0],
                        y=[0],
                        z=[z_int],
                        mode="markers",
                        marker=dict(size=10, color="blue"),
                        name=f"z-intercept: {z_int:.2f}a",
                    )
                )
                intercept_points.append([0, 0, z_int])

            if len(intercept_points) == 3:
                pts = np.array(intercept_points)
                fig_miller.add_trace(
                    go.Mesh3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        i=[0],
                        j=[1],
                        k=[2],
                        opacity=0.6,
                        color="cyan",
                        name=f"{plane_label} plane",
                        showlegend=True,
                    )
                )
                for i in range(3):
                    p1 = intercept_points[i]
                    p2 = intercept_points[(i + 1) % 3]
                    fig_miller.add_trace(
                        go.Scatter3d(
                            x=[p1[0], p2[0]],
                            y=[p1[1], p2[1]],
                            z=[p1[2], p2[2]],
                            mode="lines",
                            line=dict(color="darkblue", width=4),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
            elif len(intercept_points) == 2:
                p1, p2 = intercept_points
                if h == 0:
                    vertices = [
                        [p1[0] - _box, p1[1], p1[2]],
                        [p1[0] + _box, p1[1], p1[2]],
                        [p2[0] + _box, p2[1], p2[2]],
                        [p2[0] - _box, p2[1], p2[2]],
                    ]
                elif k == 0:
                    vertices = [
                        [p1[0], p1[1] - _box, p1[2]],
                        [p1[0], p1[1] + _box, p1[2]],
                        [p2[0], p2[1] + _box, p2[2]],
                        [p2[0], p2[1] - _box, p2[2]],
                    ]
                else:
                    vertices = [
                        [p1[0], p1[1], p1[2] - _box],
                        [p1[0], p1[1], p1[2] + _box],
                        [p2[0], p2[1], p2[2] + _box],
                        [p2[0], p2[1], p2[2] - _box],
                    ]
                verts = np.array(vertices)
                fig_miller.add_trace(
                    go.Mesh3d(
                        x=verts[:, 0],
                        y=verts[:, 1],
                        z=verts[:, 2],
                        i=[0, 0],
                        j=[1, 2],
                        k=[2, 3],
                        opacity=0.6,
                        color="cyan",
                        name=f"{plane_label} plane",
                        showlegend=True,
                    )
                )
                fig_miller.add_trace(
                    go.Scatter3d(
                        x=[p1[0], p2[0]],
                        y=[p1[1], p2[1]],
                        z=[p1[2], p2[2]],
                        mode="lines",
                        line=dict(color="darkblue", width=4),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            elif len(intercept_points) == 1:
                p = intercept_points[0]
                if h != 0:
                    vertices = [
                        [p[0], -_box, -_box],
                        [p[0], _box, -_box],
                        [p[0], _box, _box],
                        [p[0], -_box, _box],
                    ]
                elif k != 0:
                    vertices = [
                        [-_box, p[1], -_box],
                        [_box, p[1], -_box],
                        [_box, p[1], _box],
                        [-_box, p[1], _box],
                    ]
                else:
                    vertices = [
                        [-_box, -_box, p[2]],
                        [_box, -_box, p[2]],
                        [_box, _box, p[2]],
                        [-_box, _box, p[2]],
                    ]
                verts = np.array(vertices)
                fig_miller.add_trace(
                    go.Mesh3d(
                        x=verts[:, 0],
                        y=verts[:, 1],
                        z=verts[:, 2],
                        i=[0, 0],
                        j=[1, 2],
                        k=[2, 3],
                        opacity=0.6,
                        color="cyan",
                        name=f"{plane_label} plane",
                        showlegend=True,
                    )
                )

            normal = np.array([h, k, l], dtype=float)
            normal_length = np.linalg.norm(normal)
            if normal_length > 0:
                normal_unit = normal / normal_length * 0.5
                center = np.array([0.0, 0.0, 0.0])
                fig_miller.add_trace(
                    go.Scatter3d(
                        x=[center[0], center[0] + normal_unit[0]],
                        y=[center[1], center[1] + normal_unit[1]],
                        z=[center[2], center[2] + normal_unit[2]],
                        mode="lines",
                        line=dict(color="magenta", width=5),
                        name=f"Normal {dir_label}",
                    )
                )
                fig_miller.add_trace(
                    go.Cone(
                        x=[center[0] + normal_unit[0]],
                        y=[center[1] + normal_unit[1]],
                        z=[center[2] + normal_unit[2]],
                        u=[normal_unit[0] * 0.3],
                        v=[normal_unit[1] * 0.3],
                        w=[normal_unit[2] * 0.3],
                        colorscale=[[0, "magenta"], [1, "magenta"]],
                        showscale=False,
                        sizemode="absolute",
                        sizeref=0.1,
                        showlegend=False,
                    )
                )

        if h != 0 or k != 0 or l != 0:
            d_spacing = 1.0 / np.sqrt(h**2 + k**2 + l**2)
            title_text = (
                f"Plane {miller_html(h, k, l, 'plane')} | "
                f"Interplane spacing = {d_spacing:.3f}a"
            )
        else:
            title_text = "Miller Plane Calculator"

        fig_miller.update_layout(
            title=dict(text=title_text, x=0.5),
            scene=dict(
                xaxis_title="x [a]",
                yaxis_title="y [a]",
                zaxis_title="z [a]",
                aspectmode="cube",
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
                xaxis=dict(range=[-_box - 0.35, _box + 0.35]),
                yaxis=dict(range=[-_box - 0.35, _box + 0.35]),
                zaxis=dict(range=[-_box - 0.35, _box + 0.35]),
            ),
            height=600,
            width=700,
            margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig_miller

    h_input = mo.ui.slider(value=1, start=-3, stop=3, step=1, label="h")
    k_input = mo.ui.slider(value=1, start=-3, stop=3, step=1, label="k")
    l_input = mo.ui.slider(value=1, start=-3, stop=3, step=1, label="l")
    return (
        h_input,
        k_input,
        l_input,
        miller_html,
        miller_plain,
        miller_tex,
        plot_miller_plane,
    )


@app.cell
def _(h_input, k_input, l_input, miller_tex, mo, plot_miller_plane):
    _h = h_input.value
    _k = k_input.value
    _l = l_input.value

    def _int_str(n):
        if n == 0:
            return r"\infty"
        return f"{1.0 / n:.2f}a" if abs(n) != 1 else ("a" if n == 1 else "-a")

    if _h == 0 and _k == 0 and _l == 0:
        _info = mo.md("**(000) is not a valid Miller index.**")
    else:
        _d = 1.0 / (_h**2 + _k**2 + _l**2) ** 0.5
        _plane = miller_tex(_h, _k, _l, "plane")
        _dir = miller_tex(_h, _k, _l, "direction")
        _info = mo.md(
            rf"""
    Intercepts: $x={_int_str(_h)}$, $y={_int_str(_k)}$, $z={_int_str(_l)}$.  
    Reciprocals $\to$ plane ${_plane}$. Magenta arrow: cubic normal ${_dir}$.  
    $d/a = 1/\sqrt{{{_h}^2+{_k}^2+{_l}^2}} = {_d:.3f}$.
    """
        )

    mo.vstack(
        [
            mo.md("## Interactive Miller indices"),
            mo.md(
                "Set $(h,k,l)$ to see the plane, axis intercepts, and the normal $[hkl]$. Drag to rotate."
            ),
            mo.hstack([h_input, k_input, l_input], justify="start", gap=2),
            _info,
            plot_miller_plane(_h, _k, _l),
        ],
        align="center",
    )
    return


@app.cell
def _(go, miller_html, miller_plain, miller_tex, mo, np, plt):
    DIAMOND_BASIS = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ]
    )
    SI_NN = np.sqrt(3.0) / 4.0
    SI_N_CELLS = 2
    SI_BOX_LO = -0.5 * SI_N_CELLS
    SI_BOX_HI = 0.5 * SI_N_CELLS
    SI_FCC1_COLOR = "#0072B2"
    SI_FCC2_COLOR = "#D55E00"
    SI_OFF_EDGE = "#888888"

    def _unique_points(pts, decimals=8):
        _rounded = np.round(np.asarray(pts, dtype=float), decimals=decimals)
        _, _idx = np.unique(_rounded, axis=0, return_index=True)
        return np.asarray(pts, dtype=float)[np.sort(_idx)]

    def _canon_vec(v):
        _v = np.array(v, dtype=float)
        for _c in _v:
            if abs(_c) > 1e-12:
                return _v if _c > 0 else -_v
        return _v

    def in_plane_lattice_vectors(h, k, l):
        _n = np.array([int(h), int(k), int(l)], dtype=int)
        _seen = set()
        _vecs = []
        for _x in range(-3, 4):
            for _y in range(-3, 4):
                for _z in range(-3, 4):
                    _v = np.array([_x, _y, _z], dtype=int)
                    if np.all(_v == 0) or int(_n @ _v) != 0:
                        continue
                    _g = int(np.gcd.reduce(np.abs(_v)))
                    _v = (_v // _g).astype(int)
                    _key = tuple(_v.tolist())
                    if _key not in _seen:
                        _seen.add(_key)
                        _vecs.append(_v)
        _vecs.sort(key=lambda t: (int(t @ t), tuple(t.tolist())))
        _v1 = _canon_vec(_vecs[0])
        _v2 = None
        for _v in _vecs[1:]:
            _vc = _canon_vec(_v)
            if np.linalg.norm(np.cross(_v1, _vc)) > 0.5:
                _v2 = _vc
                break
        if np.dot(np.cross(_v1, _v2), _n.astype(float)) < 0:
            _v2 = -_v2
        return _v1, _v2

    def miller_plane_eq_tex(h, k, l, C):
        _parts = []
        for _coef, _name in ((h, "x"), (k, "y"), (l, "z")):
            if _coef == 0:
                continue
            _mag = abs(int(_coef))
            _term = _name if _mag == 1 else rf"{_mag}{_name}"
            if not _parts:
                _parts.append(_term if _coef > 0 else f"-{_term}")
            elif _coef > 0:
                _parts.append(f" + {_term}")
            else:
                _parts.append(f" - {_term}")
        _left = "".join(_parts) if _parts else "0"
        _C = float(C)
        if abs(_C - round(_C)) < 1e-8:
            _right = str(int(round(_C)))
        else:
            _right = f"{_C:.4g}"
        return rf"{_left} = {_right}"

    def miller_intercepts_tex(h, k, l):
        _bits = []
        for _idx, _name in ((h, "x"), (k, "y"), (l, "z")):
            if _idx == 0:
                _bits.append(rf"{_name}=\infty")
            elif abs(int(_idx)) == 1:
                _sign = "-" if _idx < 0 else ""
                _bits.append(rf"{_name}={_sign}a")
            else:
                _bits.append(rf"{_name}=a/{int(_idx)}")
        return ",\\ ".join(_bits)

    def cube_grid_segments(n_cells=SI_N_CELLS):
        _segs = []
        _n = int(n_cells)
        _origin = -0.5 * _n
        for _a in range(_n + 1):
            for _b in range(_n + 1):
                for _c in range(_n):
                    _segs.append(
                        np.array(
                            [
                                [_c, _a, _b],
                                [_c + 1, _a, _b],
                            ],
                            dtype=float,
                        )
                        + _origin
                    )
                    _segs.append(
                        np.array(
                            [
                                [_a, _c, _b],
                                [_a, _c + 1, _b],
                            ],
                            dtype=float,
                        )
                        + _origin
                    )
                    _segs.append(
                        np.array(
                            [
                                [_a, _b, _c],
                                [_a, _b, _c + 1],
                            ],
                            dtype=float,
                        )
                        + _origin
                    )
        return _segs

    def silicon_view_geometry(h, k, l):
        _v1, _v2 = in_plane_lattice_vectors(h, k, l)
        _n = np.array([h, k, l], dtype=float)
        _n_hat = _n / np.linalg.norm(_n)
        _fcc1, _fcc2 = diamond_fcc_cells()
        _pts = np.vstack([_fcc1, _fcc2])
        _is_fcc1 = np.concatenate(
            [
                np.ones(len(_fcc1), dtype=bool),
                np.zeros(len(_fcc2), dtype=bool),
            ]
        )
        _N = h * _pts[:, 0] + k * _pts[:, 1] + l * _pts[:, 2]
        # Miller intercepts a/h, a/k, a/l  ⇒  hx + ky + lz = 1  (coords in units of a)
        _C = 1.0
        _on = np.abs(_N - _C) < 1e-8
        _u = _v1 / np.linalg.norm(_v1)
        _w = _v2 - np.dot(_v2, _u) * _u
        _vax = _w / np.linalg.norm(_w)
        if np.dot(np.cross(_u, _vax), _n_hat) < 0:
            _vax = -_vax
        _xy = np.column_stack([_pts @ _u, _pts @ _vax])
        return {
            "h": int(h),
            "k": int(k),
            "l": int(l),
            "pts": _pts,
            "xy": _xy,
            "on": _on,
            "N": _N,
            "C": _C,
            "eq_tex": miller_plane_eq_tex(h, k, l, _C),
            "intercepts_tex": miller_intercepts_tex(h, k, l),
            "v1": _v1,
            "v2": _v2,
            "u": _u,
            "vax": _vax,
            "n_hat": _n_hat,
            "fcc1": _fcc1,
            "fcc2": _fcc2,
            "is_fcc1": _is_fcc1,
            "n_on": int(np.count_nonzero(_on)),
            "n_off": int(np.count_nonzero(~_on)),
            "d_over_a": 1.0 / np.sqrt(h**2 + k**2 + l**2),
            "plane_verts": plane_box_vertices(h, k, l, _C),
            "cube_segs": cube_grid_segments(),
        }

    def plot_silicon_view(geom, show_fcc1, show_fcc2):
        _pts = geom["pts"]
        _xy = geom["xy"]
        _on = geom["on"]
        _is_fcc1 = geom["is_fcc1"]
        _v1 = geom["v1"]
        _v2 = geom["v2"]
        _u = geom["u"]
        _vax = geom["vax"]
        _h, _k, _l = geom["h"], geom["k"], geom["l"]
        _vis = np.zeros(len(_pts), dtype=bool)
        if show_fcc1:
            _vis |= _is_fcc1
        if show_fcc2:
            _vis |= ~_is_fcc1
        _on1 = _vis & _on & _is_fcc1
        _on2 = _vis & _on & ~_is_fcc1
        _off = _vis & ~_on

        _fig, _ax = plt.subplots(figsize=(5.2, 5.2))
        _idx = np.flatnonzero(_vis)
        for _a, _i in enumerate(_idx):
            for _j in _idx[_a + 1 :]:
                if abs(np.linalg.norm(_pts[_i] - _pts[_j]) - SI_NN) < 1e-6:
                    _ax.plot(
                        [_xy[_i, 0], _xy[_j, 0]],
                        [_xy[_i, 1], _xy[_j, 1]],
                        color="#c8c8c8",
                        lw=1.2,
                        zorder=1,
                        solid_capstyle="round",
                    )
        for _seg in geom["cube_segs"]:
            _ax.plot(
                _seg @ _u,
                _seg @ _vax,
                linestyle=(0, (5, 4)),
                color="#444444",
                lw=1.4,
                zorder=2,
            )
        if np.any(_off):
            _ax.scatter(
                _xy[_off, 0],
                _xy[_off, 1],
                s=150,
                facecolors="white",
                edgecolors=SI_OFF_EDGE,
                linewidths=1.6,
                zorder=3,
                label="Not on the plane",
            )
        if np.any(_on1):
            _ax.scatter(
                _xy[_on1, 0],
                _xy[_on1, 1],
                s=190,
                c=SI_FCC1_COLOR,
                edgecolors="black",
                linewidths=0.6,
                zorder=4,
                label="FCC 1",
            )
        if np.any(_on2):
            _ax.scatter(
                _xy[_on2, 0],
                _xy[_on2, 1],
                s=190,
                c=SI_FCC2_COLOR,
                edgecolors="black",
                linewidths=0.6,
                zorder=4,
                label="FCC 2",
            )
        _dir = miller_tex(_h, _k, _l, "family")
        _in1 = miller_tex(
            int(round(_v1[0])), int(round(_v1[1])), int(round(_v1[2])), "direction"
        )
        _in2 = miller_tex(
            int(round(_v2[0])), int(round(_v2[1])), int(round(_v2[2])), "direction"
        )
        _ax.set_aspect("equal")
        if abs(np.dot(_v1, _v2)) < 1e-8:
            _ax.set_xlabel(rf"${_in1}$  (units of $a$)", fontsize=16)
            _ax.set_ylabel(rf"${_in2}$  (units of $a$)", fontsize=16)
        else:
            _ax.set_xlabel(r"in-plane (units of $a$)", fontsize=16)
            _ax.set_ylabel(r"in-plane (units of $a$)", fontsize=16)
            _cell_xy = np.array(
                [0.5 * (_v1 + _v2) @ _u, 0.5 * (_v1 + _v2) @ _vax]
            )
            for _vec, _text in ((_v1, rf"${_in1}$"), (_v2, rf"${_in2}$")):
                _p1 = np.array([_vec @ _u, _vec @ _vax])
                _mid = 0.5 * _p1
                _edge = _p1.copy()
                _nrm = np.array([-_edge[1], _edge[0]], dtype=float)
                _nrm = _nrm / (np.linalg.norm(_nrm) + 1e-12)
                if np.dot(_nrm, _cell_xy - _mid) > 0:
                    _nrm = -_nrm
                _pos = _mid + 0.16 * _nrm
                _ax.text(
                    _pos[0],
                    _pos[1],
                    _text,
                    fontsize=16,
                    ha="center",
                    va="center",
                )
        _ax.tick_params(labelsize=16)
        _ax.set_title(
            rf"Si (diamond), viewed along ${_dir}$",
            fontsize=16,
            pad=10,
        )
        _ax.legend(
            frameon=False,
            fontsize=16,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=3,
        )
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _fig.tight_layout()
        return _fig

    def diamond_fcc_cells(n_cells=SI_N_CELLS):
        _corners = np.array(
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
        _faces = np.array(
            [
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 1.0],
                [0.5, 0.0, 0.5],
                [0.5, 1.0, 0.5],
                [0.0, 0.5, 0.5],
                [1.0, 0.5, 0.5],
            ]
        )
        _fcc1_list = []
        _origin = -0.5 * float(n_cells)
        for _ix in range(n_cells):
            for _iy in range(n_cells):
                for _iz in range(n_cells):
                    _shift = np.array([_ix, _iy, _iz], dtype=float) + _origin
                    _fcc1_list.append(_corners + _shift)
                    _fcc1_list.append(_faces + _shift)
        _fcc1 = _unique_points(np.vstack(_fcc1_list))
        _fcc2 = _fcc1 + np.array([0.25, 0.25, 0.25])
        _lo = _origin
        _hi = _origin + float(n_cells)
        _fcc2 = _fcc2[np.all((_fcc2 >= _lo - 1e-8) & (_fcc2 <= _hi + 1e-8), axis=1)]
        return _fcc1, _unique_points(_fcc2)

    def plane_box_vertices(h, k, l, C=1.0, lo=SI_BOX_LO, hi=SI_BOX_HI):
        _n = np.array([h, k, l], dtype=float)
        _corners = np.array(
            [
                [_x, _y, _z]
                for _x in (lo, hi)
                for _y in (lo, hi)
                for _z in (lo, hi)
            ]
        )
        _pts = []
        for _i in range(8):
            for _j in range(_i + 1, 8):
                _d = _corners[_j] - _corners[_i]
                if np.count_nonzero(np.abs(_d) > 1e-12) != 1:
                    continue
                _n0 = float(_n @ _corners[_i] - C)
                _n1 = float(_n @ _corners[_j] - C)
                if _n0 * _n1 > 1e-14:
                    continue
                if abs(_n0) < 1e-12 and abs(_n1) < 1e-12:
                    _pts.append(_corners[_i])
                    _pts.append(_corners[_j])
                    continue
                _den = _n0 - _n1
                if abs(_den) < 1e-14:
                    continue
                _t = np.clip(_n0 / _den, 0.0, 1.0)
                _pts.append(_corners[_i] + _t * _d)
        if len(_pts) == 0:
            return np.zeros((0, 3))
        _pts = _unique_points(_pts)
        if len(_pts) < 3:
            return _pts
        _centroid = _pts.mean(axis=0)
        _n_hat = _n / (np.linalg.norm(_n) + 1e-15)
        _tmp = (
            np.array([1.0, 0.0, 0.0])
            if abs(_n_hat[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        _u = np.cross(_n_hat, _tmp)
        _u = _u / np.linalg.norm(_u)
        _v = np.cross(_n_hat, _u)
        _rel = _pts - _centroid
        _ang = np.arctan2(_rel @ _v, _rel @ _u)
        return _pts[np.argsort(_ang)]

    def plot_silicon_3d(geom, show_fcc1, show_fcc2):
        _fig = go.Figure()
        _h, _k, _l = geom["h"], geom["k"], geom["l"]
        _plane_label = miller_plain(_h, _k, _l, "plane")
        _lo, _hi = SI_BOX_LO, SI_BOX_HI
        _xs, _ys, _zs = [], [], []
        for _seg in geom["cube_segs"]:
            _xs.extend([_seg[0, 0], _seg[1, 0], None])
            _ys.extend([_seg[0, 1], _seg[1, 1], None])
            _zs.extend([_seg[0, 2], _seg[1, 2], None])
        _fig.add_trace(
            go.Scatter3d(
                x=_xs,
                y=_ys,
                z=_zs,
                mode="lines",
                line=dict(color="#444444", width=2, dash="dash"),
                name="Unit cells",
                hoverinfo="skip",
            )
        )
        _end = _hi + 0.25
        _start = _lo - 0.25
        for _axis, _color, _label, _txt in (
            (([_start, _end], [0, 0], [0, 0]), "red", "x", [_end + 0.1, 0, 0]),
            (([0, 0], [_start, _end], [0, 0]), "green", "y", [0, _end + 0.1, 0]),
            (([0, 0], [0, 0], [_start, _end]), "blue", "z", [0, 0, _end + 0.1]),
        ):
            _fig.add_trace(
                go.Scatter3d(
                    x=_axis[0],
                    y=_axis[1],
                    z=_axis[2],
                    mode="lines",
                    line=dict(color=_color, width=4),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            _fig.add_trace(
                go.Scatter3d(
                    x=[_txt[0]],
                    y=[_txt[1]],
                    z=[_txt[2]],
                    mode="text",
                    text=[_label],
                    textfont=dict(size=14, color=_color),
                    showlegend=False,
                )
            )

        _verts = geom["plane_verts"]
        if len(_verts) >= 3:
            _nvert = len(_verts)
            _fig.add_trace(
                go.Mesh3d(
                    x=_verts[:, 0],
                    y=_verts[:, 1],
                    z=_verts[:, 2],
                    i=[0] * (_nvert - 2),
                    j=list(range(1, _nvert - 1)),
                    k=list(range(2, _nvert)),
                    opacity=0.35,
                    color="cyan",
                    name=f"{_plane_label} plane",
                    showlegend=True,
                )
            )
            _closed = np.vstack([_verts, _verts[0]])
            _fig.add_trace(
                go.Scatter3d(
                    x=_closed[:, 0],
                    y=_closed[:, 1],
                    z=_closed[:, 2],
                    mode="lines",
                    line=dict(color="darkblue", width=4),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        _fcc1 = geom["fcc1"]
        _fcc2 = geom["fcc2"]
        _on = geom["on"]
        _is_fcc1 = geom["is_fcc1"]
        _on1 = _on[_is_fcc1]
        _on2 = _on[~_is_fcc1]

        def _add_on_plane(_pts, _color, _name):
            if len(_pts) == 0:
                return
            _fig.add_trace(
                go.Scatter3d(
                    x=_pts[:, 0],
                    y=_pts[:, 1],
                    z=_pts[:, 2],
                    mode="markers",
                    marker=dict(
                        size=11,
                        color=_color,
                        opacity=1.0,
                        line=dict(width=1.5, color="black"),
                    ),
                    name=_name,
                )
            )

        _off_list = []
        if show_fcc1:
            _add_on_plane(_fcc1[_on1], SI_FCC1_COLOR, "FCC 1")
            if np.any(~_on1):
                _off_list.append(_fcc1[~_on1])
        if show_fcc2:
            _add_on_plane(_fcc2[_on2], SI_FCC2_COLOR, "FCC 2")
            if np.any(~_on2):
                _off_list.append(_fcc2[~_on2])
        if _off_list:
            _p = np.vstack(_off_list)
            _fig.add_trace(
                go.Scatter3d(
                    x=_p[:, 0],
                    y=_p[:, 1],
                    z=_p[:, 2],
                    mode="markers",
                    marker=dict(
                        size=7,
                        color="white",
                        opacity=1.0,
                        line=dict(width=2, color=SI_OFF_EDGE),
                    ),
                    name="Not on the plane",
                )
            )
        if show_fcc1 and show_fcc2:
            for _p1 in _fcc2:
                for _p2 in _fcc1:
                    if abs(np.linalg.norm(_p1 - _p2) - SI_NN) < 1e-6:
                        _fig.add_trace(
                            go.Scatter3d(
                                x=[_p1[0], _p2[0]],
                                y=[_p1[1], _p2[1]],
                                z=[_p1[2], _p2[2]],
                                mode="lines",
                                line=dict(color="#888888", width=3),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )

        _n_hat = geom["n_hat"]
        _vax = geom["vax"]
        _eye = 2.4 * _n_hat
        _fig.update_layout(
            title=dict(
                text=(
                    f"Si diamond, 2×2×2 cells | plane "
                    f"{miller_html(_h, _k, _l, 'plane')} | Drag to rotate"
                ),
                x=0.5,
            ),
            scene=dict(
                xaxis_title="x [a]",
                yaxis_title="y [a]",
                zaxis_title="z [a]",
                aspectmode="cube",
                camera=dict(
                    eye=dict(x=float(_eye[0]), y=float(_eye[1]), z=float(_eye[2])),
                    up=dict(
                        x=float(_vax[0]),
                        y=float(_vax[1]),
                        z=float(_vax[2]),
                    ),
                ),
                xaxis=dict(range=[_lo - 0.35, _hi + 0.45]),
                yaxis=dict(range=[_lo - 0.35, _hi + 0.45]),
                zaxis=dict(range=[_lo - 0.35, _hi + 0.45]),
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(size=11),
                itemclick="toggle",
                itemdoubleclick="toggleothers",
            ),
            height=520,
            width=520,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return _fig

    si_h = mo.ui.slider(value=1, start=-2, stop=2, step=1, label="h")
    si_k = mo.ui.slider(value=1, start=-2, stop=2, step=1, label="k")
    si_l = mo.ui.slider(value=0, start=-2, stop=2, step=1, label="l")
    si_fcc1 = mo.ui.checkbox(value=True, label="FCC 1 (corners and face centers)")
    si_fcc2 = mo.ui.checkbox(value=True, label="FCC 2 (offset by a/4)")
    return (
        plot_silicon_3d,
        plot_silicon_view,
        si_fcc1,
        si_fcc2,
        si_h,
        si_k,
        si_l,
        silicon_view_geometry,
    )


@app.cell
def _(
    miller_tex,
    mo,
    plot_silicon_3d,
    plot_silicon_view,
    si_fcc1,
    si_fcc2,
    si_h,
    si_k,
    si_l,
    silicon_view_geometry,
):
    _h = si_h.value
    _k = si_k.value
    _l = si_l.value
    if _h == 0 and _k == 0 and _l == 0:
        _body = mo.md(r"**$\langle 000\rangle$ is not a valid direction.**")
    else:
        _geom = silicon_view_geometry(_h, _k, _l)
        _dir = miller_tex(_h, _k, _l, "family")
        _plane = miller_tex(_h, _k, _l, "plane")
        _eq = _geom["eq_tex"]
        _intercepts = _geom["intercepts_tex"]
        _n_on = _geom["n_on"]
        _n_off = _geom["n_off"]
        _d = _geom["d_over_a"]
        _info = mo.md(
            rf"""
    Viewing along ${_dir}$. Cubic Si $\Rightarrow$ you face ${_plane}$.

    Miller intercepts of ${_plane}$: ${_intercepts}$ (coordinates in units of $a$).
    The plane through those intercepts is

    $${_eq}$$

    Both plots use the same diamond sites: FCC 1 (corners and face centers) and
    FCC 2 (FCC 1 $+\,(a/4,a/4,a/4)$) in the $2\times 2\times 2$ block
    $-a\le x,y,z\le a$.
    An atom is on the plane when $\lvert hx+ky+lz-1\rvert=0$. **{_n_on}** atoms
    satisfy that; **{_n_off}** do not. Filled blue = FCC 1, filled red = FCC 2,
    open grey = not on the plane. Spacing
    $d/a = 1/\sqrt{{{_h}^2+{_k}^2+{_l}^2}} = {_d:.3f}$.
    """
        )
        _body = mo.vstack(
            [
                _info,
                mo.hstack([si_fcc1, si_fcc2], justify="start", gap=2),
                mo.hstack(
                    [
                        plot_silicon_3d(_geom, si_fcc1.value, si_fcc2.value),
                        plot_silicon_view(
                            _geom, si_fcc1.value, si_fcc2.value
                        ),
                    ],
                    justify="start",
                    align="start",
                    gap=1,
                    widths=[1, 1],
                ),
            ],
            gap=0.4,
        )

    mo.vstack(
        [
            mo.md("## Silicon crystallographic directions"),
            mo.md(
                r"""
    Silicon is diamond cubic. Choose a viewing direction $\langle hkl\rangle$
    (angle brackets) to look into the crystal. Because Si is cubic, $\langle hkl\rangle$
    is perpendicular to $(hkl)$: you are looking at that plane.
    """
            ),
            mo.hstack([si_h, si_k, si_l], justify="start", gap=2),
            _body,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
