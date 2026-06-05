"""Compositing renderer - layout to canvas to PNG."""

from __future__ import annotations

from dataclasses import dataclass, field

from pixelui.fonts.sprite_font import Glyph, SpriteFont
from pixelui.render.canvas import BLACK, WHITE, Canvas, Color
from pixelui.ui.elements import Element, get_element
from pixelui.ui.layout import LayoutResult


@dataclass
class Renderer:
    """Composites layout results onto an RGBA canvas and exports PNG.

    Parameters
    ----------
    font : SpriteFont
        The sprite font used for all text rendering.
    padding : int
        Default padding (in pixels) around rendered content.
    fg_color : Color
        RGBA tuple to draw for glyph/element "on" mask bits.
        Default is (0, 0, 0, 255) (black).
    bg_color : Color
        RGBA tuple to draw for glyph/element "off" mask bits
        and the initial canvas fill. Default is (255, 255, 255, 255) (white).
    """

    font: SpriteFont
    padding: int = 4
    fg_color: Color = BLACK
    bg_color: Color = WHITE

    _canvas: Canvas | None = field(init=False, default=None)
    _layout_result: LayoutResult | None = field(init=False, default=None)

    def render_layout(self, layout: LayoutResult) -> Canvas:
        """Render a LayoutResult onto a new canvas."""
        w = layout.width + 2 * self.padding
        h = layout.height + 2 * self.padding
        canvas = Canvas(w, h)
        canvas.clear(self.bg_color)

        for item in layout.lines:
            dx = item.x + self.padding
            dy = item.y + self.padding
            if item.kind == "element":
                elem = get_element(item.content)
                if elem is not None:
                    self.blit_element(canvas, elem, dx, dy)
            else:
                glyph = self.font.get_glyph(item.content)
                if glyph is None:
                    continue
                self.blit_glyph(canvas, glyph, dx, dy)

        self._canvas = canvas
        self._layout_result = layout
        return canvas

    def render_text(
        self,
        text: str,
        *,
        x_start: int = 0,
        y_start: int = 0,
        max_width_chars: int | None = None,
    ) -> Canvas:
        """Render plain text directly."""
        from pixelui.ui import layout_text as _layout

        result = _layout(text, self.font, x_start=x_start, y_start=y_start, max_width_chars=max_width_chars)
        return self.render_layout(result)

    def render_panel(
        self,
        title: str | None = None,
        body: str = "",
        *,
        x: int = 0,
        y: int = 0,
    ) -> Canvas:
        """Render a bordered panel."""
        from pixelui.ui import layout_panel as _layout

        result = _layout(title or "", body, font=self.font, x=x, y=y)
        return self.render_layout(result)

    def blit_element(self, canvas: Canvas, elem: Element, dst_x: int, dst_y: int) -> None:
        """Blit an 8×8 element tile onto canvas at (dst_x, dst_y).

        The element's 1-bit pixel data acts as a mask: 1-bits draw *fg_color*,
        0-bits draw *bg_color*.
        """
        for py in range(elem.height):
            src_byte = elem.pixels[py]
            row = dst_y + py
            if not (0 <= row < canvas.height):
                continue
            for px in range(elem.width):
                src_bit = (src_byte >> (7 - px)) & 1
                color = self.fg_color if src_bit else self.bg_color
                canvas.set_pixel(dst_x + px, row, color)

    def blit_glyph(self, canvas: Canvas, glyph: Glyph, dst_x: int, dst_y: int) -> None:
        """Blit a single Glyph onto canvas at (dst_x, dst_y).

        The glyph's 1-bit pixel data acts as a mask: 1-bits draw *fg_color*,
        0-bits draw *bg_color*.
        """
        for py in range(glyph.height):
            src_byte = glyph.pixels[py]
            canvas_row = dst_y + py
            if not (0 <= canvas_row < canvas.height):
                continue
            for px in range(glyph.width):
                src_bit = (src_byte >> (7 - px)) & 1
                color = self.fg_color if src_bit else self.bg_color
                canvas.set_pixel(dst_x + px, canvas_row, color)

    def render_bordered_box(
        self,
        text: str,
        *,
        x: int = 0,
        y: int = 0,
        border: str = "table",
    ) -> Canvas:
        """Render a tight bordered box around a line of text."""
        from pixelui.ui.layout import layout_bordered_box as _layout

        result = _layout(text, self.font, x=x, y=y, border=border)
        return self.render_layout(result)

    def to_png(self, filepath: str | None = None) -> bytes:
        """Export the last rendered canvas as PNG. Returns raw bytes."""
        if self._canvas is None:
            raise RuntimeError("No layout has been rendered yet")
        return self._canvas.to_png(filepath)
