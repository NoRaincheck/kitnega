"""pixelui - generate 1-bit spritesheet UI images from text."""

from pixelui import (
    Canvas,
    Renderer,
    default_font,
    export_element_spritesheet,
    get_element,
)
from pixelui.render.canvas import BLACK, WHITE
from pixelui.ui.layout import layout_bordered_box

SKY_BLUE = (135, 206, 235, 255)
DARK_BLUE = (20, 60, 120, 255)


def main() -> None:
    font = default_font()

    # ── Text output (tight, no extra padding, like the bordered box) ──
    canvas = Renderer(font, padding=0, fg_color=BLACK, bg_color=WHITE).render_text("Hello, PixelUI!")
    canvas.to_png("text_output.png")
    print(f"text_output.png: {canvas.width}x{canvas.height}")

    # ── Bordered box (tightly wrapped, no padding) ──
    layout = layout_bordered_box("Hello World!", font)
    # Use padding=0 so the border is flush with the image edges
    canvas = Renderer(font, padding=0, fg_color=BLACK, bg_color=WHITE).render_layout(layout)
    canvas.to_png("bordered_box.png")
    print(f"bordered_box.png: {canvas.width}x{canvas.height}")

    # ── Bordered box with sky-blue text background ──
    layout = layout_bordered_box("Sky Blue!", font)
    canvas = Canvas(layout.width, layout.height)
    canvas.clear(SKY_BLUE)
    border_r = Renderer(font, padding=0, fg_color=WHITE, bg_color=BLACK)
    text_r = Renderer(font, padding=0, fg_color=WHITE, bg_color=SKY_BLUE)
    for item in layout.lines:
        if item.kind == "element":
            elem = get_element(item.content)
            if elem is not None:
                border_r.blit_element(canvas, elem, item.x, item.y)
        else:
            glyph = font.get_glyph(item.content)
            if glyph is not None:
                text_r.blit_glyph(canvas, glyph, item.x, item.y)
    canvas.to_png("bordered_box_skyblue.png")
    print(f"bordered_box_skyblue.png: {canvas.width}x{canvas.height}")

    # ── Panel (existing) ──
    canvas = Renderer(font, fg_color=BLACK, bg_color=WHITE).render_panel(
        title="Hello, PixelUI!",
        body="This is a 1-bit\nspritesheet UI.\n\nBuilt with pure Python.",
    )
    canvas.to_png("output.png")
    print(f"output.png: {canvas.width}x{canvas.height}")

    # ── Spritesheet: 8x16 font ──
    font.export_spritesheet("spritesheet.png")
    print("spritesheet.png: 8x16 font glyphs")

    # ── Spritesheet: 8x8 UI elements ──
    export_element_spritesheet("elements_spritesheet.png")
    print("elements_spritesheet.png: 8x8 UI elements")


if __name__ == "__main__":
    main()
