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
        return f"({s})" if kind == "plane" else f"[{s}]"

    def miller_tex_index(n):
        if n < 0:
            return rf"\bar{{{abs(int(n))}}}"
        return str(int(n))

    def miller_tex(h, k, l, kind="plane"):
        s = miller_tex_index(h) + miller_tex_index(k) + miller_tex_index(l)
        return rf"({s})" if kind == "plane" else rf"[{s}]"

    def miller_html_index(n):
        if n < 0:
            return (
                f"<span style='text-decoration:overline'>{abs(int(n))}</span>"
            )
        return str(int(n))

    def miller_html(h, k, l, kind="plane"):
        s = miller_html_index(h) + miller_html_index(k) + miller_html_index(l)
        return f"({s})" if kind == "plane" else f"[{s}]"

    def plot_miller_plane(h, k, l):
        """Plot the Miller plane in a unit cell by showing intercepts."""
        fig_miller = go.Figure()
        plane_label = miller_plain(h, k, l, "plane")
        dir_label = miller_plain(h, k, l, "direction")

        cube_edges = [
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
                x=[-2, 2],
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
                y=[-2, 2],
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
                z=[-2, 2],
                mode="lines",
                line=dict(color="blue", width=4),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_miller.add_trace(
            go.Scatter3d(
                x=[2.2],
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
                y=[2.2],
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
                z=[2.2],
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

            if x_int is not None and -2 <= x_int <= 2:
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
            if y_int is not None and -2 <= y_int <= 2:
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
            if z_int is not None and -2 <= z_int <= 2:
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
                        [p1[0] - 2, p1[1], p1[2]],
                        [p1[0] + 2, p1[1], p1[2]],
                        [p2[0] + 2, p2[1], p2[2]],
                        [p2[0] - 2, p2[1], p2[2]],
                    ]
                elif k == 0:
                    vertices = [
                        [p1[0], p1[1] - 2, p1[2]],
                        [p1[0], p1[1] + 2, p1[2]],
                        [p2[0], p2[1] + 2, p2[2]],
                        [p2[0], p2[1] - 2, p2[2]],
                    ]
                else:
                    vertices = [
                        [p1[0], p1[1], p1[2] - 2],
                        [p1[0], p1[1], p1[2] + 2],
                        [p2[0], p2[1], p2[2] + 2],
                        [p2[0], p2[1], p2[2] - 2],
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
                        [p[0], -2, -2],
                        [p[0], 2, -2],
                        [p[0], 2, 2],
                        [p[0], -2, 2],
                    ]
                elif k != 0:
                    vertices = [
                        [-2, p[1], -2],
                        [2, p[1], -2],
                        [2, p[1], 2],
                        [-2, p[1], 2],
                    ]
                else:
                    vertices = [
                        [-2, -2, p[2]],
                        [2, -2, p[2]],
                        [2, 2, p[2]],
                        [-2, 2, p[2]],
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
                center = np.array([0.5, 0.5, 0.5])
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
                xaxis=dict(range=[-0.5, 1.5]),
                yaxis=dict(range=[-0.5, 1.5]),
                zaxis=dict(range=[-0.5, 1.5]),
            ),
            height=600,
            width=700,
            margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig_miller

    h_input = mo.ui.slider(value=1, start=-3, stop=3, step=1, label="h")
    k_input = mo.ui.slider(value=1, start=-3, stop=3, step=1, label="k")
    l_input = mo.ui.slider(value=1, start=-3, stop=3, step=1, label="l")
    return h_input, k_input, l_input, miller_tex, plot_miller_plane


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


if __name__ == "__main__":
    app.run()
