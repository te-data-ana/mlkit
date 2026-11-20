import logging

import pandas as pd

from mlkit.use_cases import UseCase

logger = logging.getLogger(__name__)


def validate_data(df: pd.DataFrame, use_case: UseCase) -> pd.DataFrame:
    """Validate a DataFrame against an explicitly selected use-case contract.

    Args:
        df: Input DataFrame.
        use_case: Data contract to apply.

    Returns:
        The validated and type-coerced DataFrame.

    Raises:
        pydantic.ValidationError: If a record violates the schema.
        ValueError: If required feature columns are missing or columns repeat.
    """
    try:
        logger.info("Validating data schema for %d rows.", len(df))
        validated = use_case.validate(df)
        logger.info("Data validation passed for %d rows.", len(validated))
        return validated
    except Exception:
        logger.exception("Data validation failed.")
        raise
