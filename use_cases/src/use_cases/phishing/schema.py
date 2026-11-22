from typing import Literal

from pydantic import BaseModel, Field


class PhishingInputFeatures(BaseModel):
    """Raw features shared by training and prediction."""

    example_int_feature: int = Field(ge=0, le=100)
    example_float_feature: float = Field(ge=0)
    example_cat_feature: Literal["C1", "C2"]
