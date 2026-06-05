"""8x8 UI element primitives - borders, shadows, fills, decorations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pixelui.render.canvas import Color


@dataclass(frozen=True)
class Element:
    """An 8x8 pixel sprite defined as 8 hex rows (MSB-first)."""

    name: str
    pixels: list[int]  # 8 bytes, one per row

    @property
    def width(self) -> int:
        return 8

    @property
    def height(self) -> int:
        return 8


# Corner sprites (8x8 each)
CORNER_TOP_LEFT = Element(
    "corner-tl",
    [0b11111000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000],
)
CORNER_TOP_RIGHT = Element(
    "corner-tr",
    [0b01111110, 0b00000010, 0b00000010, 0b00000010, 0b00000010, 0b00000010, 0b00000010, 0b00000000],
)
CORNER_BOTTOM_LEFT = Element(
    "corner-bl",
    [0b00000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b11111000],
)
CORNER_BOTTOM_RIGHT = Element(
    "corner-br",
    [0b00000000, 0b00000010, 0b00000010, 0b00000010, 0b00000010, 0b00000010, 0b00000010, 0b01111110],
)

# Edge sprites
EDGE_TOP = Element(
    "edge-top",
    [0b01111110, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000],
)
EDGE_BOTTOM = Element(
    "edge-bottom",
    [0b00000000] * 7 + [0b01111110],
)
EDGE_LEFT = Element(
    "edge-left",
    [0b10000000] * 8,
)
EDGE_RIGHT = Element(
    "edge-right",
    [0b00000010] * 8,
)

# Fill / solid
FILL_SOLID = Element("fill-solid", [0xFF] * 8)
FILL_NONE = Element("fill-none", [0x00] * 8)

# Shadow
SHADOW_RIGHT = Element(
    "shadow-right",
    [0b00010000, 0b00100000, 0b01000000, 0b10000000, 0b10000000, 0b01000000, 0b00100000, 0b00010000],
)
SHADOW_BOTTOM = Element(
    "shadow-bottom",
    [0x00] * 7 + [0b00011100],
)

# Decorative elements
ARROW_RIGHT = Element(
    "arrow-right",
    [0x00, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00001000, 0b00010000, 0b00100000],
)
ARROW_LEFT = Element(
    "arrow-left",
    [0x00, 0b00010000, 0b00100000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00010000],
)
CHECKBOX_UNCHECKED = Element(
    "checkbox-unchecked",
    [0b01111110, 0b10000001, 0b10000001, 0b10000001, 0b10000001, 0b10000001, 0b10000001, 0b01111110],
)
CHECKBOX_CHECKED = Element(
    "checkbox-checked",
    [0b01111110, 0b10011001, 0b10011001, 0b10011001, 0b10100101, 0b11000011, 0b10011001, 0b01111110],
)
DIVIDER = Element("divider", [0xFF] + [0x00] * 7)

# ---------------------------------------------------------------------------
# HTML table border elements (1px black line, 3px white gap to content)
# ---------------------------------------------------------------------------
#
# Each 8×8 tile: 4px outer white, 1px black line, 3px gap white
# (top/left edges) or mirrored for bottom/right edges.
#
#   4px outer    1px line   3px gap to content
#   ┌───────────┬──────────┬──────────────────┐
#   │ ░░░░      │ ████████ │      ░░░░        │
#   └───────────┴──────────┴──────────────────┘
#

# Top edge: line at row 4, 3px gap below (rows 5‑7)
_T = 0xFF  # horizontal black line (full row)
TABLE_TOP = Element("table-top", [0x00] * 4 + [_T] + [0x00] * 3)

# Bottom edge: line at row 3, 3px gap above (rows 0‑2)
TABLE_BOTTOM = Element("table-bottom", [0x00] * 3 + [_T] + [0x00] * 4)

# Left edge: line at col 4 (bit 3 in MSB‑first), 3px gap to the right
_L = 0b00001000  # single black pixel at col 4
TABLE_LEFT = Element("table-left", [_L] * 8)

# Right edge: line at col 3 (bit 4 in MSB‑first), 3px gap to the left
_R = 0b00010000  # single black pixel at col 3
TABLE_RIGHT = Element("table-right", [_R] * 8)

# Corners — the black line crosses at the corner and continues
# Horizontal line only spans from the vertical edge to the tile edge
# (not into outer padding or gap areas on the other side).
_TL_H = 0b00001111  # cols 4-7 black (TL & BL: line goes right from left edge)
_TR_H = 0b11110000  # cols 0-3 black (TR & BR: line goes left from right edge)

TABLE_TL = Element(
    "table-tl",
    [
        0x00,
        0x00,
        0x00,
        0x00,  # top outer padding
        _TL_H,  # top line from left edge to right
        _L,
        _L,
        _L,  # left line continues downward
    ],
)
TABLE_TR = Element(
    "table-tr",
    [
        0x00,
        0x00,
        0x00,
        0x00,
        _TR_H,
        _R,
        _R,
        _R,
    ],
)
TABLE_BL = Element(
    "table-bl",
    [
        _L,
        _L,
        _L,  # left line continues upward
        _TL_H,  # bottom line from left edge to right
        0x00,
        0x00,
        0x00,
        0x00,  # bottom outer padding
    ],
)
TABLE_BR = Element(
    "table-br",
    [
        _R,
        _R,
        _R,
        _TR_H,
        0x00,
        0x00,
        0x00,
        0x00,
    ],
)

# Element registry
ELEMENTS: dict[str, Element] = {
    e.name: e
    for e in [
        CORNER_TOP_LEFT,
        CORNER_TOP_RIGHT,
        CORNER_BOTTOM_LEFT,
        CORNER_BOTTOM_RIGHT,
        EDGE_TOP,
        EDGE_BOTTOM,
        EDGE_LEFT,
        EDGE_RIGHT,
        FILL_SOLID,
        FILL_NONE,
        SHADOW_RIGHT,
        SHADOW_BOTTOM,
        ARROW_RIGHT,
        ARROW_LEFT,
        CHECKBOX_UNCHECKED,
        CHECKBOX_CHECKED,
        DIVIDER,
        TABLE_TL,
        TABLE_TR,
        TABLE_BL,
        TABLE_BR,
        TABLE_TOP,
        TABLE_BOTTOM,
        TABLE_LEFT,
        TABLE_RIGHT,
    ]
}


def get_element(name: str) -> Element | None:
    """Look up an element by name."""
    return ELEMENTS.get(name)


def export_element_spritesheet(
    filepath: str,
    fg_color: Color | None = None,
    bg_color: Color | None = None,
) -> int:
    """Pack all registered 8×8 elements into a spritesheet PNG.

    Elements are laid out in a grid (16 columns) with each cell being 8×8.
    Returns the PNG file size in bytes.
    """
    from pixelui.render.canvas import BLACK, WHITE, Canvas

    names = sorted(ELEMENTS.keys())
    cols = 16
    rows = (len(names) + cols - 1) // cols
    sheet_w = cols * 8
    sheet_h = rows * 8
    stride = (sheet_w + 7) // 8

    data = bytearray(sheet_h * stride)

    for idx, name in enumerate(names):
        elem = ELEMENTS[name]
        col = idx % cols
        row = idx // cols

        for py in range(8):
            src_byte = elem.pixels[py]
            byte_col = col * 8 // 8  # each element occupies 1 byte per row
            dst_offset = (row * 8 + py) * stride + byte_col
            data[dst_offset] |= src_byte

    canvas = Canvas(sheet_w, sheet_h)
    canvas.blit_from_bytes(
        bytes(data),
        sheet_w,
        sheet_h,
        0,
        0,
        0,
        0,
        sheet_w,
        sheet_h,
        fg_color=fg_color or BLACK,
        bg_color=bg_color or WHITE,
    )
    png_data = canvas.to_png(filepath)
    return len(png_data)
