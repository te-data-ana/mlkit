"""PhiUSIIL's precomputed numeric URL and webpage features."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt

Binary = Annotated[int, Field(ge=0, le=1)]


class PhishingInputFeatures(BaseModel):
    """Numeric features from UCI dataset 967, with normalized column names."""

    model_config = ConfigDict(allow_inf_nan=False)

    url_length: NonNegativeInt
    domain_length: NonNegativeInt
    is_domain_ip: Binary
    # url_similarity_index: NonNegativeFloat
    # char_continuation_rate: NonNegativeFloat
    # tld_legitimate_prob: NonNegativeFloat
    # url_char_prob: NonNegativeFloat
    tld_length: NonNegativeInt
    no_of_sub_domain: NonNegativeInt
    has_obfuscation: Binary
    no_of_obfuscated_char: NonNegativeInt
    obfuscation_ratio: NonNegativeFloat
    no_of_letters_in_url: NonNegativeInt
    letter_ratio_in_url: NonNegativeFloat
    no_of_degits_in_url: NonNegativeInt
    degit_ratio_in_url: NonNegativeFloat
    no_of_equals_in_url: NonNegativeInt
    no_of_q_mark_in_url: NonNegativeInt
    no_of_ampersand_in_url: NonNegativeInt
    no_of_other_special_chars_in_url: NonNegativeInt
    spacial_char_ratio_in_url: NonNegativeFloat
    is_https: Binary
    line_of_code: NonNegativeInt
    largest_line_length: NonNegativeInt
    has_title: Binary
    domain_title_match_score: NonNegativeFloat
    # url_title_match_score: NonNegativeFloat
    has_favicon: Binary
    robots: Binary
    is_responsive: Binary
    no_of_url_redirect: NonNegativeInt
    no_of_self_redirect: NonNegativeInt
    has_description: Binary
    no_of_popup: NonNegativeInt
    no_ofi_frame: NonNegativeInt
    has_external_form_submit: Binary
    has_social_net: Binary
    has_submit_button: Binary
    has_hidden_fields: Binary
    has_password_field: Binary
    bank: Binary
    pay: Binary
    crypto: Binary
    has_copyright_info: Binary
    no_of_image: NonNegativeInt
    no_of_css: NonNegativeInt
    no_of_js: NonNegativeInt
    no_of_self_ref: NonNegativeInt
    no_of_empty_ref: NonNegativeInt
    no_of_external_ref: NonNegativeInt
