# Crystal Lattices

Interactive 3D lattice visualizations for ECE335 Lecture 1, used alongside the slides.

**Student URL:** https://ece335.github.io/crystal-lattice

## Run locally

```bash
cd lectures/crystal-lattices
uv run marimo edit crystal_lattices.py
```

Or, if marimo is already installed:

```bash
marimo edit crystal_lattices.py
```

## Export for GitHub Pages

From this directory:

```bash
marimo check crystal_lattices.py
marimo export html-wasm crystal_lattices.py -o ../../crystal-lattice --mode run -f
cp -R images ../../crystal-lattice/images
mkdir -p ../../crystal-lattice/source/images
cp crystal_lattices.py ../../crystal-lattice/source/
cp -R images/. ../../crystal-lattice/source/images/
```

Then add noindex tags to `crystal-lattice/index.html` if the export overwrote them.
