from __future__ import annotations

import argparse
import sys

from ._shared import ansi
from .mapgen import generate_map, render
from .png import save_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Procedural map generator (diamond-square + Voronoi)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--width", type=int, default=32, help="Map width in tiles")
    parser.add_argument("--height", type=int, default=32, help="Map height in tiles")
    parser.add_argument("--samples-per-tile", type=int, default=8, help="Diamond-square samples per tile edge")
    parser.add_argument("--voronoi-regions", type=int, default=180, help="Number of Voronoi terrain regions")
    parser.add_argument("--edge-water-falloff", type=float, default=0.12, help="Edge water blend fraction")
    parser.add_argument("--enable-voronoi", action="store_true", help="Disable Voronoi normalization")
    parser.add_argument("--rivers", action="store_true", help="Enable river generation")
    parser.add_argument("-o", "--output", default="output.png", help="Output PNG path")
    args = parser.parse_args()

    if args.width < 8 or args.height < 8:
        print("error: --width and --height must be >= 8", file=sys.stderr)
        sys.exit(1)
    if args.samples_per_tile < 1:
        print("error: --samples-per-tile must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.voronoi_regions < 1:
        print("error: --voronoi-regions must be >= 1", file=sys.stderr)
        sys.exit(1)
    if not 0.0 <= args.edge_water_falloff <= 0.5:
        print("error: --edge-water-falloff must be between 0 and 0.5", file=sys.stderr)
        sys.exit(1)

    map_w = args.width * args.samples_per_tile
    map_h = args.height * args.samples_per_tile
    print(f"{ansi(92)}generating {args.width}x{args.height} tile map ({map_w}x{map_h}px)...{ansi(0)}", file=sys.stderr)

    map_data = generate_map(
        width=args.width,
        height=args.height,
        seed=args.seed,
        samples_per_tile=args.samples_per_tile,
        voronoi_regions=args.voronoi_regions,
        edge_water_falloff=args.edge_water_falloff,
        use_voronoi=args.enable_voronoi,
        use_rivers=args.rivers,
    )
    print(
        f"{ansi(90)}  terrain: {sum(1 for r in map_data.terrain for t in r if t not in {'ocean', 'coast'})} land cells",
        file=sys.stderr,
    )
    if args.rivers:
        print(f"{ansi(90)}  rivers:  {len(map_data.rivers)} cells{ansi(0)}", file=sys.stderr)

    print(f"{ansi(92)}rendering...{ansi(0)}", file=sys.stderr)
    pixels = render(map_data, scale=args.samples_per_tile)
    print(f"{ansi(92)}writing {args.output}...{ansi(0)}", file=sys.stderr)
    save_png(pixels, args.output)
    print(f"done: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
