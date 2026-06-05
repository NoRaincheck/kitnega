"""Built-in 8x16 pixel font loader.

Loads glyph data from the vendored Spleen 8x16 spritesheet.
Style: clean, minimal pixel art.
"""

from __future__ import annotations

from pixelui.fonts.spleen import load_spleen_glyphs
from pixelui.fonts.sprite_font import SpriteFont


def default_font() -> SpriteFont:
    """Create a :class:`SpriteFont` with the built-in character set.

    Includes all standard ASCII printable characters at 8×16 px each,
    loaded from the vendored Spleen 8x16 spritesheet.
    """
    font = SpriteFont(name="pixelui-default-8x16", cols=20)
    for glyph in load_spleen_glyphs():
        font.add_glyph(glyph)
    return font
