"""Input contract for UCI Predict Students' Dropout and Academic Success."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

Binary = Literal[0, 1]
Grade200 = Annotated[float, Field(ge=0, le=200)]
Grade20 = Annotated[float, Field(ge=0, le=20)]
Qualification = Literal[
    1,
    2,
    3,
    4,
    5,
    6,
    9,
    10,
    11,
    12,
    13,
    14,
    18,
    19,
    20,
    22,
    25,
    26,
    27,
    29,
    30,
    31,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
]
Occupation = Literal[
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    90,
    99,
    101,
    102,
    103,
    112,
    114,
    121,
    122,
    123,
    124,
    125,
    131,
    132,
    134,
    135,
    141,
    143,
    144,
    151,
    152,
    153,
    154,
    161,
    163,
    171,
    172,
    173,
    174,
    175,
    181,
    182,
    183,
    191,
    192,
    193,
    194,
    195,
]


class StudentsInputFeatures(BaseModel):
    """Student characteristics and first- and second-semester results."""

    model_config = ConfigDict(allow_inf_nan=False)

    marital_status: Literal[1, 2, 3, 4, 5, 6]
    application_mode: Literal[
        1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57
    ]
    application_order: Annotated[int, Field(ge=0, le=9)]
    course: Literal[
        33,
        171,
        8014,
        9003,
        9070,
        9085,
        9119,
        9130,
        9147,
        9238,
        9254,
        9500,
        9556,
        9670,
        9773,
        9853,
        9991,
    ]
    daytime_evening_attendance: Binary
    previous_qualification: Literal[
        1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43
    ]
    previous_qualification_grade: Grade200
    nacionality: Literal[
        1,
        2,
        6,
        11,
        13,
        14,
        17,
        21,
        22,
        24,
        25,
        26,
        32,
        41,
        62,
        100,
        101,
        103,
        105,
        108,
        109,
    ]
    mothers_qualification: Qualification
    fathers_qualification: Qualification
    mothers_occupation: Occupation
    fathers_occupation: Occupation
    admission_grade: Grade200
    displaced: Binary
    educational_special_needs: Binary
    debtor: Binary
    tuition_fees_up_to_date: Binary
    gender: Binary
    scholarship_holder: Binary
    age_at_enrollment: NonNegativeInt
    international: Binary
    curricular_units_1st_sem_credited: NonNegativeInt
    curricular_units_1st_sem_enrolled: NonNegativeInt
    curricular_units_1st_sem_evaluations: NonNegativeInt
    curricular_units_1st_sem_approved: NonNegativeInt
    curricular_units_1st_sem_grade: Grade20
    curricular_units_1st_sem_without_evaluations: NonNegativeInt
    curricular_units_2nd_sem_credited: NonNegativeInt
    curricular_units_2nd_sem_enrolled: NonNegativeInt
    curricular_units_2nd_sem_evaluations: NonNegativeInt
    curricular_units_2nd_sem_approved: NonNegativeInt
    curricular_units_2nd_sem_grade: Grade20
    curricular_units_2nd_sem_without_evaluations: NonNegativeInt
    unemployment_rate: float
    inflation_rate: float
    gdp: float
