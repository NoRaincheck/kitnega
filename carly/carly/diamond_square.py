from __future__ import annotations

import math
import random

WATER_THRESHOLD = 50 / 255


def _normalize(grid: list[list[float]]) -> list[list[float]]:
    lo = float("inf")
    hi = float("-inf")
    for row in grid:
        for v in row:
            if not math.isnan(v):
                lo = min(lo, v)
                hi = max(hi, v)
    rng = hi - lo or 1
    return [[(v - lo) / rng for v in row] for row in grid]


def _seed_border(grid: list[list[float]], value: float) -> None:
    size = len(grid)
    for i in range(size):
        for y, x in [(0, i), (size - 1, i), (i, 0), (i, size - 1)]:
            if math.isnan(grid[y][x]):
                grid[y][x] = value


def _diamond_step(grid, size, step, half, scale, rng) -> None:
    for y in range(half, size, step):
        for x in range(half, size, step):
            if not math.isnan(grid[y][x]):
                continue
            avg = (
                grid[y - half][x - half]
                + grid[y - half][x + half]
                + grid[y + half][x - half]
                + grid[y + half][x + half]
            ) / 4.0
            grid[y][x] = avg + rng.gauss(0, scale)


def _square_step(grid, size, step, half, scale, rng) -> None:
    for y in range(0, size, half):
        for x in range((y + half) % step, size, step):
            if not math.isnan(grid[y][x]):
                continue
            vals: list[float] = []
            for ny, nx in [(y - half, x), (y + half, x), (y, x - half), (y, x + half)]:
                if 0 <= ny < size and 0 <= nx < size:
                    v = grid[ny][nx]
                    if not math.isnan(v):
                        vals.append(v)
            grid[y][x] = (sum(vals) / len(vals)) + rng.gauss(0, scale)


def diamond_square(
    power: int, roughness: float = 0.5, seed: int | None = None, seed_border: bool | float = False
) -> list[list[float]]:
    rng = random.Random(seed)
    size = (1 << power) + 1
    grid = [[float("nan")] * size for _ in range(size)]

    if seed_border is not False:
        val = -2 * roughness if seed_border is True else float(seed_border)
        _seed_border(grid, val)

    corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
    for y, x in corners:
        if math.isnan(grid[y][x]):
            grid[y][x] = rng.gauss(0, roughness)

    step = size - 1
    scale = 1.0
    while step > 1:
        half = step // 2
        _diamond_step(grid, size, step, half, scale, rng)
        _square_step(grid, size, step, half, scale, rng)
        step = half
        scale *= roughness

    return _normalize(grid)


def _rescale_water(terrain: list[list[float]], target: float) -> list[list[float]]:
    flat = [v for row in terrain for v in row]
    lo, hi = 0.1, 10.0
    gamma = 1.0
    for _ in range(50):
        gamma = (lo + hi) / 2
        below = sum(1 for v in flat if v**gamma < WATER_THRESHOLD)
        ratio = below / len(flat)
        if abs(ratio - target) < 0.001:
            break
        if ratio < target:
            lo = gamma
        else:
            hi = gamma
    return [[v**gamma for v in row] for row in terrain]


def _resize_bilinear(grid: list[list[float]], new_w: int, new_h: int) -> list[list[float]]:
    h = len(grid)
    w = len(grid[0])
    result = [[0.0] * new_w for _ in range(new_h)]
    for y in range(new_h):
        for x in range(new_w):
            gx = x / new_w * (w - 1)
            gy = y / new_h * (h - 1)
            ix, fx = int(gx), gx - int(gx)
            iy, fy = int(gy), gy - int(gy)
            ix1 = min(ix + 1, w - 1)
            iy1 = min(iy + 1, h - 1)
            v00 = grid[iy][ix]
            v01 = grid[iy][ix1]
            v10 = grid[iy1][ix]
            v11 = grid[iy1][ix1]
            v = (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy
            result[y][x] = v
    return result


def generate_heightmap(
    width: int,
    height: int,
    seed: int | None = None,
    roughness: float = 0.55,
    water_ratio: float = 0.28,
) -> list[list[float]]:
    size = max(width, height)
    power = max(5, (size - 1).bit_length())
    terrain = diamond_square(power, roughness=roughness, seed=seed, seed_border=True)
    terrain = _rescale_water(terrain, water_ratio)
    terrain = _resize_bilinear(terrain, width, height)
    return _normalize(terrain)


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def apply_edge_ocean_falloff(terrain: list[list[float]], border_fraction: float = 0.12) -> list[list[float]]:
    if border_fraction <= 0:
        return terrain
    h = len(terrain)
    w = len(terrain[0])
    bp = max(1.0, min(w, h) * border_fraction)
    low = WATER_THRESHOLD * 0.25
    result = [[0.0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            dx = min(x, w - 1 - x)
            dy = min(y, h - 1 - y)
            t = min(dx, dy) / bp
            t = max(0.0, min(1.0, t))
            s = _smoothstep(t)
            result[y][x] = low * (1 - s) + terrain[y][x] * s
    return result


def _border_seed_points(hw: int, hh: int, region_count: int) -> list[tuple[float, float]]:
    spacing = max(2, int(max(hw, hh) / max(8, math.sqrt(region_count))))
    pts: list[tuple[float, float]] = []
    for x in range(0, hw, spacing):
        pts.append((0.0, float(x)))
        pts.append((float(hh - 1), float(x)))
    for y in range(0, hh, spacing):
        pts.append((float(y), 0.0))
        pts.append((float(y), float(hw - 1)))
    for y, x in [(0, 0), (0, hw - 1), (hh - 1, 0), (hh - 1, hw - 1)]:
        pts.append((float(y), float(x)))
    return pts


def voronoi_normalize(
    terrain: list[list[float]], num_regions: int, seed: int | None, relaxation_iters: int = 0
) -> list[list[float]]:
    h = len(terrain)
    w = len(terrain[0])
    rng = random.Random(seed)

    inner_count = max(1, num_regions)
    inner_points: list[tuple[float, float]] = [
        (rng.random() * h, rng.random() * w) for _ in range(inner_count)
    ]
    border_points = _border_seed_points(w, h, num_regions)
    all_points = inner_points + border_points
    num_points = len(all_points)

    for iteration in range(relaxation_iters + 1):
        do_relax = iteration < relaxation_iters
        sums = [0.0] * num_points
        counts = [0] * num_points
        labels = [[0] * w for _ in range(h)]

        if do_relax:
            sum_y = [0.0] * num_points
            sum_x = [0.0] * num_points

        for y in range(h):
            for x in range(w):
                best_d = float("inf")
                best_i = 0
                for i, (py, px) in enumerate(all_points):
                    d = (y - py) * (y - py) + (x - px) * (x - px)
                    if d < best_d:
                        best_d = d
                        best_i = i
                labels[y][x] = best_i
                sums[best_i] += terrain[y][x]
                counts[best_i] += 1
                if do_relax:
                    sum_y[best_i] += y
                    sum_x[best_i] += x

        if do_relax:
            for i in range(inner_count):
                if counts[i] > 0:
                    inner_points[i] = (sum_y[i] / counts[i], sum_x[i] / counts[i])
            all_points = inner_points + border_points

    means = [sums[i] / (counts[i] or 1) for i in range(num_points)]

    result = [[0.0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            result[y][x] = 0.82 * means[labels[y][x]] + 0.18 * terrain[y][x]
    return _normalize(result)


def generate_sampled_heightmaps(
    width: int,
    height: int,
    samples_per_tile: int,
    seed: int | None = None,
    roughness: float = 0.55,
    water_ratio: float = 0.28,
    voronoi_regions: int = 180,
    edge_water_falloff: float = 0.12,
    use_voronoi: bool = True,
) -> tuple[list[list[float]], list[list[float]]]:
    high_res = generate_heightmap(
        width * samples_per_tile,
        height * samples_per_tile,
        seed=seed,
        roughness=roughness,
        water_ratio=water_ratio,
    )
    high_res = apply_edge_ocean_falloff(high_res, edge_water_falloff)
    if use_voronoi and voronoi_regions > 0:
        high_res = voronoi_normalize(high_res, voronoi_regions, seed, relaxation_iters=voronoi_regions)
        high_res = apply_edge_ocean_falloff(high_res, edge_water_falloff)

    area_means: list[list[float]] = [[0.0] * width for _ in range(height)]
    center_values: list[list[float]] = [[0.0] * width for _ in range(height)]
    center = samples_per_tile // 2
    for r in range(height):
        for c in range(width):
            s = 0.0
            for dy in range(samples_per_tile):
                for dx in range(samples_per_tile):
                    s += high_res[r * samples_per_tile + dy][c * samples_per_tile + dx]
            area_means[r][c] = s / (samples_per_tile * samples_per_tile)
            center_values[r][c] = high_res[r * samples_per_tile + center][c * samples_per_tile + center]

    return area_means, center_values
