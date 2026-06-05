"""Glyph bitmap storage, spritesheet packing, and lookup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pixelui.render.canvas import Color


@dataclass(frozen=True)
class Glyph:
    """An 8x16 pixel glyph stored as a list of 16 hex bytes (MSB-first)."""

    char: str  # The character this glyph represents
    pixels: list[int]  # 16 bytes, each representing one row (bit 7 = leftmost)

    @classmethod
    def from_hex(cls, char: str, rows: list[str]) -> Glyph:
        """Create a glyph from a list of 16 hex string rows."""
        if len(rows) != 16:
            raise ValueError(f"Glyph must have exactly 16 rows, got {len(rows)}")
        pixels = [int(r.strip(), 16) & 0xFF for r in rows]
        return cls(char=char, pixels=pixels)

    @classmethod
    def from_binary(cls, char: str, rows: list[bytes]) -> Glyph:
        """Create a glyph from raw byte rows."""
        if len(rows) != 16:
            raise ValueError(f"Glyph must have exactly 16 rows, got {len(rows)}")
        pixels = [b[0] & 0xFF for b in rows]
        return cls(char=char, pixels=pixels)

    @property
    def width(self) -> int:
        return 8

    @property
    def height(self) -> int:
        return 16


@dataclass(frozen=True)
class GlyphRect:
    """Position and size of a glyph within a spritesheet."""

    x: int  # Column offset in the spritesheet
    y: int  # Row offset in the spritesheet
    w: int = 8  # Width (always 8 for this project)
    h: int = 16  # Height (always 16 for this project)


@dataclass
class SpriteFont:
    """A sprite font - maps characters to glyphs and packs them into a spritesheet.

    Glyphs are stored in a grid layout determined by cols (default 16).
    """

    name: str = "pixelui-8x16"
    glyph_width: int = 8
    glyph_height: int = 16
    cols: int = 16
    glyphs: dict[str, Glyph] = field(default_factory=dict)
    _grid: list[GlyphRect] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._build_grid()

    # glyph registration

    def add_glyph(self, glyph: Glyph) -> None:
        """Register a single glyph."""
        if len(glyph.pixels) != 16:
            raise ValueError(f"Glyph rows must be 16, got {len(glyph.pixels)}")
        self.glyphs[glyph.char] = glyph
        self._build_grid()

    def add_glyphs(self, glyphs: list[Glyph]) -> None:
        """Register multiple glyphs at once."""
        for g in glyphs:
            self.glyphs[g.char] = g
        self._build_grid()

    # grid layout

    def _build_grid(self) -> None:
        """Arrange all glyphs into a grid of GlyphRect entries."""
        char_list = sorted(self.glyphs.keys())
        rects: dict[str, GlyphRect] = {}
        for i, ch in enumerate(char_list):
            col = i % self.cols
            row = i // self.cols
            rects[ch] = GlyphRect(
                x=col * self.glyph_width,
                y=row * self.glyph_height,
            )
        object.__setattr__(self, "_grid", rects)

    def get_rect(self, char: str) -> GlyphRect | None:
        """Get the spritesheet position for a character."""
        if not hasattr(self, "_grid"):
            return None
        return self._grid.get(char)

    # lookup

    def get_glyph(self, char: str) -> Glyph | None:
        """Return the Glyph for char, or None."""
        if char not in self.glyphs:
            return None
        return self.glyphs[char]

    def measure_text(self, text: str) -> tuple[int, int]:
        """Measure the pixel dimensions of a string."""
        width = len(text) * self.glyph_width
        height = self.glyph_height
        if "\n" in text:
            lines = text.split("\n")
            height = len(lines) * self.glyph_height
        return width, height

    # spritesheet generation

    def build_spritesheet(self) -> tuple[bytes, int, int]:
        """Pack all glyphs into a single 1-bit image.

        Returns (data, width, height) suitable for Canvas.blit_from_bytes.
        """
        if not self.glyphs:
            raise ValueError("No glyphs registered")

        char_list = sorted(self.glyphs.keys())
        rows_needed = (len(char_list) + self.cols - 1) // self.cols
        sheet_width = self.cols * self.glyph_width
        sheet_height = rows_needed * self.glyph_height
        stride = (sheet_width + 7) // 8

        data = bytearray(sheet_height * stride)

        for idx, ch in enumerate(char_list):
            glyph = self.glyphs[ch]
            col = idx % self.cols
            row = idx // self.cols

            sheet_row_start = row * self.glyph_height
            sheet_byte_col = col * (self.glyph_width // 8)

            for py in range(self.glyph_height):
                src_byte = glyph.pixels[py]
                dst_offset = (sheet_row_start + py) * stride + sheet_byte_col
                data[dst_offset] |= src_byte

        return bytes(data), sheet_width, sheet_height

    def export_spritesheet(
        self,
        filepath: str,
        fg_color: Color | None = None,
        bg_color: Color | None = None,
    ) -> int:
        """Build and save the spritesheet as a PNG. Returns file size."""
        from pixelui.render.canvas import BLACK, WHITE, Canvas

        raw, w, h = self.build_spritesheet()
        canvas = Canvas(w, h)
        canvas.blit_from_bytes(
            raw,
            w,
            h,
            0,
            0,
            0,
            0,
            w,
            h,
            fg_color=fg_color or BLACK,
            bg_color=bg_color or WHITE,
        )
        png_data = canvas.to_png(filepath)
        return len(png_data)
