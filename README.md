# ECE335 interactive notebooks (GitHub Pages)

Public course site: **https://ece335.github.io**

Marimo **source** notebooks live in [ECE335/interactive-notebooks](https://github.com/ECE335/interactive-notebooks) (`lectures/`). This repository publishes the WASM HTML.

| Notebook | Source | URL |
|:---|:---|:---|
| Crystal Lattices | [lectures/crystal-lattices](https://github.com/ECE335/interactive-notebooks/tree/main/lectures/crystal-lattices) | [ece335.github.io/crystal-lattice](https://ece335.github.io/crystal-lattice) |
| Miller Indices | [lectures/miller-indices](https://github.com/ECE335/interactive-notebooks/tree/main/lectures/miller-indices) | [ece335.github.io/miller-indices](https://ece335.github.io/miller-indices) |

## Adding a notebook later

1. Add the marimo `.py` file in `ECE335/interactive-notebooks` under `lectures/<topic>/`.
2. Export with `marimo export html-wasm … --mode run` into a root folder here named for the public URL.
3. Add a card to `index.html`.
