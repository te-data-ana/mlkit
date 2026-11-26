"""Input contract for the UCI Secondary Mushroom dataset."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, NonNegativeFloat

Color = Literal["n", "b", "g", "r", "p", "u", "e", "w", "y", "l", "o", "k", "f"]
Surface = Literal["i", "g", "y", "s", "h", "l", "k", "t", "w", "e", "d", "f"]


class MushroomInputFeatures(BaseModel):
    """Metrical and nominal mushroom characteristics, with normalized names."""

    model_config = ConfigDict(allow_inf_nan=False)

    cap_diameter: NonNegativeFloat
    cap_shape: Literal["b", "c", "x", "f", "s", "p", "o"]
    cap_surface: Surface | None
    cap_color: Color
    does_bruise_or_bleed: Literal["t", "f"]
    gill_attachment: Literal["a", "x", "d", "e", "s", "p", "f", "?"] | None
    gill_spacing: Literal["c", "d", "f"] | None
    gill_color: Color
    stem_height: NonNegativeFloat
    stem_width: NonNegativeFloat
    stem_root: Literal["b", "s", "c", "u", "e", "z", "r", "f"] | None
    stem_surface: Surface | None
    stem_color: Color
    veil_type: Literal["p", "u"] | None
    veil_color: Color | None
    has_ring: Literal["t", "f"]
    ring_type: (
        Literal["c", "e", "r", "g", "l", "p", "s", "z", "y", "m", "f", "?"] | None
    )
    spore_print_color: Color | None
    habitat: Literal["g", "l", "m", "p", "h", "u", "w", "d"]
    season: Literal["s", "u", "a", "w"]
