---
title: Carly — Procedural Map Generator
---

# Carly

A procedural map generator using diamond-square heightmaps and Voronoi terrain
regions. Pure Python stdlib, zero dependencies.

## Usage

```bash
uv run carly                                    # default 32×24 map
uv run carly --width 64 --height 48 --rivers    # larger map with rivers
uv run carly --seed 8675309 --voronoi-regions 300
uv run carly --enable-voronoi -o mymap.png      # Voronoi normalization
```

## Options

| Flag                   | Default      | Description                          |
| ---------------------- | ------------ | ------------------------------------ |
| `--seed`               | `42`         | Random seed for reproducibility      |
| `--width`              | `32`         | Map width in tiles                   |
| `--height`             | `32`         | Map height in tiles                  |
| `--samples-per-tile`   | `8`          | Diamond-square samples per tile edge |
| `--voronoi-regions`    | `180`        | Number of Voronoi terrain regions    |
| `--edge-water-falloff` | `0.12`       | Edge water blend fraction (0–0.5)    |
| `--enable-voronoi`     | —            | Disable Voronoi normalization        |
| `--rivers`             | —            | Enable river generation              |
| `-o`, `--output`       | `output.png` | Output PNG path                      |

## Algorithms

- **Diamond-square** — classic fractal heightmap algorithm for natural-looking
  terrain elevation
- **Voronoi normalization** — clusters cells into distinct biomes based on
  nearest-region assignment (default)
- **River generation** — traces flow paths from high to low points, carving
  channels through the terrain
