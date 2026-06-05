"""Tests for pixelui."""

import struct

from pixelui.fonts.builtin import default_font
from pixelui.fonts.sprite_font import Glyph, SpriteFont
from pixelui.render.canvas import BLACK, WHITE, Canvas
from pixelui.render.renderer import Renderer
from pixelui.ui.elements import (
    ARROW_RIGHT,
    CHECKBOX_CHECKED,
    CHECKBOX_UNCHECKED,
    ELEMENTS,
    FILL_SOLID,
)
from pixelui.ui.layout import layout_panel, layout_text, wrap_text


class TestCanvas:
    def test_clear(self):
        c = Canvas(8, 4)
        assert c.get_pixel(0, 0) == WHITE
        c.set_pixel(3, 1, BLACK)
        assert c.get_pixel(3, 1) == BLACK

    def test_set_get_pixel(self):
        c = Canvas(8, 4)
        for x in range(8):
            c.set_pixel(x, 0, BLACK if (x % 2) == 0 else WHITE)
        for x in range(8):
            expected = BLACK if (x % 2) == 0 else WHITE
            assert c.get_pixel(x, 0) == expected

    def test_blit(self):
        src = Canvas(8, 4)
        for y in range(4):
            for x in range(8):
                src.set_pixel(x, y, BLACK if (x + y) % 2 == 0 else WHITE)
        dst = Canvas(16, 8)
        dst.blit(src, 0, 0, 4, 2, 8, 4)
        for y in range(4):
            for x in range(8):
                assert dst.get_pixel(x + 4, y + 2) == src.get_pixel(x, y)

    def test_to_png(self):
        c = Canvas(16, 16)
        data = c.to_png()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert b"IHDR" in data
        assert b"IDAT" in data
        assert b"IEND" in data

    def test_to_png_dimensions(self):
        c = Canvas(32, 16)
        png_data = c.to_png()
        width = struct.unpack(">I", png_data[16:20])[0]
        height = struct.unpack(">I", png_data[20:24])[0]
        assert width == 32
        assert height == 16

    def test_copy(self):
        c1 = Canvas(8, 4)
        c1.set_pixel(3, 1, BLACK)
        c2 = c1.copy()
        assert c2.get_pixel(3, 1) == BLACK
        c2.set_pixel(3, 1, WHITE)
        assert c1.get_pixel(3, 1) == BLACK

    def test_dimensions_error(self):
        try:
            Canvas(0, 4)
            assert False
        except ValueError:
            pass


class TestGlyph:
    def test_from_hex(self):
        g = Glyph.from_hex("A", ["0xFF"] * 16)
        assert g.char == "A"
        assert len(g.pixels) == 16

    def test_invalid_rows(self):
        try:
            Glyph.from_hex("X", ["0x00"] * 8)
            assert False
        except ValueError:
            pass


class TestSpriteFont:
    def test_default_font(self):
        font = default_font()
        assert len(font.glyphs) > 50
        assert font.get_glyph("A") is not None
        assert font.get_glyph("z") is not None

    def test_measure_text(self):
        font = default_font()
        w, h = font.measure_text("Hi!")
        assert w == 3 * 8
        assert h == 16

    def test_build_spritesheet(self):
        font = default_font()
        data, width, height = font.build_spritesheet()
        assert width == font.cols * 8
        assert height > 0

    def test_glyph_lookup_missing(self):
        font = SpriteFont(name="test")
        assert font.get_glyph("Z") is None


class TestRenderer:
    def test_render_text(self):
        font = default_font()
        renderer = Renderer(font)
        canvas = renderer.render_text("Hello!")
        assert isinstance(canvas, Canvas)

    def test_render_panel(self):
        font = default_font()
        renderer = Renderer(font)
        canvas = renderer.render_panel(
            title="Test",
            body="Body text here.",
        )
        assert isinstance(canvas, Canvas)
        assert canvas.width > 0
        assert canvas.height > 0

    def test_render_to_png(self):
        font = default_font()
        renderer = Renderer(font)
        renderer.render_text("ABCD")
        png_data = renderer.to_png()
        assert len(png_data) > 0
        assert png_data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_render_before_export_raises(self):
        font = default_font()
        renderer = Renderer(font)
        try:
            renderer.to_png()
            assert False
        except RuntimeError:
            pass


class TestLayout:
    def test_wrap_text(self):
        lines = wrap_text("Hello World Foo Bar", 10)
        assert all(len(line) <= 10 for line in lines)
        assert len(lines) >= 2

    def test_layout_text(self):
        font = default_font()
        result = layout_text("ABC", font)
        assert len(result.lines) == 3
        assert result.width == 3 * 8

    def test_layout_panel(self):
        font = default_font()
        result = layout_panel(
            title="Title",
            body="Body content.",
            font=font,
        )
        assert len(result.lines) > 0
        assert result.width > 0


class TestElements:
    def test_element_lookup(self):
        elem = ELEMENTS.get("corner-tl")
        assert elem is not None

    def test_get_element_by_name(self):
        from pixelui.ui.elements import get_element

        elem = get_element("arrow-right")
        assert elem is ARROW_RIGHT

    def test_fill_solid(self):
        assert all(p == 0xFF for p in FILL_SOLID.pixels)

    def test_checkbox_elements_differ(self):
        unchecked = CHECKBOX_UNCHECKED
        checked = CHECKBOX_CHECKED
        assert unchecked.pixels != checked.pixels
