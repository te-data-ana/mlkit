"""Input features from the CDC Diabetes Health Indicators dataset."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

Binary = Annotated[int, Field(ge=0, le=1)]
HealthDays = Annotated[int, Field(ge=0, le=30)]
GeneralHealth = Annotated[int, Field(ge=1, le=5)]
AgeCategory = Annotated[int, Field(ge=1, le=13)]
EducationCategory = Annotated[int, Field(ge=1, le=6)]
IncomeCategory = Annotated[int, Field(ge=1, le=8)]


class DiabetesInputFeatures(BaseModel):
    """Validated, normalized UCI dataset 891 feature codes."""

    model_config = ConfigDict(allow_inf_nan=False)

    high_bp: Binary
    high_chol: Binary
    chol_check: Binary
    bmi: NonNegativeInt
    smoker: Binary
    stroke: Binary
    heart_diseaseor_attack: Binary
    phys_activity: Binary
    fruits: Binary
    veggies: Binary
    hvy_alcohol_consump: Binary
    any_healthcare: Binary
    no_docbc_cost: Binary
    gen_hlth: GeneralHealth
    ment_hlth: HealthDays
    phys_hlth: HealthDays
    diff_walk: Binary
    sex: Binary
    age: AgeCategory
    education: EducationCategory
    income: IncomeCategory
