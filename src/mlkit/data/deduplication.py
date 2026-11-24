"""Optional source-row deduplication for prepared datasets."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def deduplicate_prepared_data(
    source: pd.DataFrame, prepared: pd.DataFrame
) -> pd.DataFrame:
    """Keep prepared rows whose corresponding source rows are unique.

    All source columns participate in comparison, but the index does not.
    The first occurrence and original order are preserved.

    Args:
        source: Source rows before mapping or validation.
        prepared: Corresponding rows after preparation and validation.

    Returns:
        Prepared rows with duplicate source rows removed.

    Raises:
        ValueError: If source and prepared row counts differ.
    """
    if len(source) != len(prepared):
        raise ValueError("Source and prepared data must have the same row count.")
    keep = ~source.duplicated(keep="first").to_numpy()
    result = prepared.iloc[keep]
    logger.info("Removed %d duplicate source rows.", len(source) - len(result))
    return result
