from __future__ import annotations

from . import diamond_square as ds

WATER_TERRAINS = {"ocean", "coast"}
LAND_TERRAINS = {"grassland", "forest", "hills", "mountains"}

TERRAIN_COLOR_STOPS: list[tuple[float, tuple[int, int, int]]] = [
    (0.00, (28, 60, 112)),
    (0.20, (47, 102, 144)),
    (0.28, (194, 178, 128)),
    (0.45, (85, 128, 60)),
    (0.62, (46, 105, 42)),
    (0.78, (118, 108, 80)),
    (0.90, (170, 170, 170)),
    (1.00, (245, 245, 250)),
]

RIVER_COLOR = (20, 78, 142)


class MapData:
    def __init__(
        self,
        width: int,
        height: int,
        heights: list[list[float]],
        base_heights: list[list[float]],
        terrain: list[list[str]],
        rivers: set[tuple[int, int]],
    ) -> None:
        self.width = width
        self.height = height
        self.heights = heights
        self.base_heights = base_heights
        self.terrain = terrain
        self.rivers = rivers


def _terrain_color(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, value))
    for (lv, lc), (rv, rc) in zip(TERRAIN_COLOR_STOPS, TERRAIN_COLOR_STOPS[1:]):
        if value <= rv:
            t = 0.0 if rv == lv else (value - lv) / (rv - lv)
            return (
                int(lc[0] + (rc[0] - lc[0]) * t),
                int(lc[1] + (rc[1] - lc[1]) * t),
                int(lc[2] + (rc[2] - lc[2]) * t),
            )
    return TERRAIN_COLOR_STOPS[-1][1]


def classify_terrain(heights: list[list[float]]) -> list[list[str]]:
    terrain: list[list[str]] = []
    for row in heights:
        tr: list[str] = []
        for v in row:
            if v < 0.20:
                tr.append("ocean")
            elif v < 0.28:
                tr.append("coast")
            elif v < 0.45:
                tr.append("grassland")
            elif v < 0.62:
                tr.append("forest")
            elif v < 0.78:
                tr.append("hills")
            else:
                tr.append("mountains")
        terrain.append(tr)

    _normalize_inland_shallows(terrain)
    return terrain


def _neighbors4(x: int, y: int, w: int, h: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            out.append((nx, ny))
    return out


def _label_landmasses(terrain: list[list[str]]) -> list[list[int]]:
    h, w = len(terrain), len(terrain[0])
    ids = [[-1] * w for _ in range(h)]
    next_id = 0
    for r in range(h):
        for c in range(w):
            if terrain[r][c] not in LAND_TERRAINS or ids[r][c] != -1:
                continue
            stack = [(c, r)]
            while stack:
                cx, cy = stack.pop()
                if ids[cy][cx] != -1 or terrain[cy][cx] not in LAND_TERRAINS:
                    continue
                ids[cy][cx] = next_id
                for nx, ny in _neighbors4(cx, cy, w, h):
                    if ids[ny][nx] == -1 and terrain[ny][nx] in LAND_TERRAINS:
                        stack.append((nx, ny))
            next_id += 1
    return ids


def _floodfill_coast(start: tuple[int, int], terrain, land_ids):
    h, w = len(terrain), len(terrain[0])
    stack = [start]
    comp: set[tuple[int, int]] = set()
    touches_edge = False
    adj_ocean = False
    adj_lands: set[int] = set()
    while stack:
        cx, cy = stack.pop()
        if (cx, cy) in comp:
            continue
        if terrain[cy][cx] != "coast":
            continue
        comp.add((cx, cy))
        touches_edge = touches_edge or cx == 0 or cy == 0 or cx == w - 1 or cy == h - 1
        for nx, ny in _neighbors4(cx, cy, w, h):
            k = terrain[ny][nx]
            if k == "coast" and (nx, ny) not in comp:
                stack.append((nx, ny))
            elif k == "ocean":
                adj_ocean = True
            elif land_ids[ny][nx] != -1:
                adj_lands.add(land_ids[ny][nx])
    return comp, touches_edge, adj_ocean, adj_lands


def _normalize_inland_shallows(terrain) -> None:
    land_ids = _label_landmasses(terrain)
    visited: set[tuple[int, int]] = set()
    h, w = len(terrain), len(terrain[0])
    for r in range(h):
        for c in range(w):
            if terrain[r][c] != "coast" or (c, r) in visited:
                continue
            comp, touches_edge, adj_ocean, adj_lands = _floodfill_coast((c, r), terrain, land_ids)
            visited.update(comp)
            replacement = "grassland" if (not touches_edge and adj_ocean and len(adj_lands) >= 2) else "ocean"
            for cc, cr in comp:
                terrain[cr][cc] = replacement


def _neighbors8(x: int, y: int, w: int, h: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                out.append((nx, ny))
    return out


def trace_downhill(
    source: tuple[int, int],
    heights: list[list[float]],
    terrain: list[list[str]],
) -> list[tuple[int, int]]:
    h, w = len(terrain), len(terrain[0])
    current = source
    path = [current]
    seen = {current}
    for _ in range(w + h):
        nx, ny = current
        best_v = float("inf")
        best = current
        for nnx, nny in _neighbors4(nx, ny, w, h):
            v = heights[nny][nnx]
            if v < best_v:
                best_v = v
                best = (nnx, nny)
        bx, by = best
        if terrain[by][bx] in WATER_TERRAINS:
            path.append(best)
            return path
        if best in seen:
            return path
        if heights[by][bx] > heights[current[1]][current[0]] + 0.03:
            return path
        path.append(best)
        seen.add(best)
        current = best
    return path


def generate_rivers(heights: list[list[float]], terrain: list[list[str]]) -> set[tuple[int, int]]:
    h, w = len(terrain), len(terrain[0])
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if terrain[r][c] in {"hills", "mountains"}:
                candidates.append((c, r))
    candidates.sort(key=lambda p: heights[p[1]][p[0]], reverse=True)

    river_cells: set[tuple[int, int]] = set()
    max_rivers = max(3, min(7, w * h // 500))
    accepted = 0

    for c, r in candidates:
        if accepted >= max_rivers:
            break
        if (c, r) in river_cells:
            continue
        path = trace_downhill((c, r), heights, terrain)
        if len(path) < 6:
            continue
        bx, by = path[-1]
        if terrain[by][bx] not in WATER_TERRAINS:
            continue
        for cell in path:
            cx, cy = cell
            if terrain[cy][cx] not in WATER_TERRAINS:
                river_cells.add(cell)
        accepted += 1

    return river_cells


def _borders(terrain: list[list[str]]) -> list[list[bool]]:
    h, w = len(terrain), len(terrain[0])
    mask = [[False] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if terrain[r][c] in WATER_TERRAINS:
                continue
            for nx, ny in _neighbors8(c, r, w, h):
                if terrain[ny][nx] in WATER_TERRAINS:
                    mask[r][c] = True
                    break
    return mask


def generate_map(
    width: int,
    height: int,
    seed: int | None = None,
    samples_per_tile: int = 4,
    voronoi_regions: int = 180,
    edge_water_falloff: float = 0.12,
    use_voronoi: bool = True,
    use_rivers: bool = False,
) -> MapData:
    heights, base_heights = ds.generate_sampled_heightmaps(
        width,
        height,
        samples_per_tile,
        seed=seed,
        voronoi_regions=voronoi_regions,
        edge_water_falloff=edge_water_falloff,
        use_voronoi=use_voronoi,
    )
    terrain = classify_terrain(heights)
    rivers = generate_rivers(heights, terrain) if use_rivers else set()
    return MapData(width, height, heights, base_heights, terrain, rivers)


def render(map_data: MapData, scale: int = 1) -> list[list[tuple[int, int, int]]]:
    h, w = map_data.height, map_data.width
    out_h, out_w = h * scale, w * scale
    border_mask = _borders(map_data.terrain)
    pixels: list[list[tuple[int, int, int]]] = [[(0, 0, 0)] * out_w for _ in range(out_h)]

    for r in range(out_h):
        tr = r // scale
        for c in range(out_w):
            tc = c // scale
            t = map_data.terrain[tr][tc]
            if t in WATER_TERRAINS:
                color = _terrain_color(map_data.heights[tr][tc])
            else:
                color = _terrain_color(map_data.base_heights[tr][tc])

            if (tc, tr) in map_data.rivers:
                color = RIVER_COLOR
            elif border_mask[tr][tc]:
                color = _terrain_color(map_data.heights[tr][tc])

            pixels[r][c] = color

    return pixels
