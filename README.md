# ECE335 interactive notebooks

Public course site: **https://ece335.github.io**

Source notebooks live in `lectures/`. Each notebook is exported as a WASM app in a hyphenated folder at the repo root (for example `crystal-lattice/` → https://ece335.github.io/crystal-lattice).

| Notebook | Source | URL |
|:---|:---|:---|
| Crystal Lattices | `lectures/crystal-lattices/` | [ece335.github.io/crystal-lattice](https://ece335.github.io/crystal-lattice) |

## Adding a notebook later

1. Put the marimo `.py` file and images in `lectures/<topic>/`.
2. Export with `marimo export html-wasm … --mode run` into a root folder named for the public URL.
3. Add a card to `index.html`.
