from pathlib import Path

import pandas as pd


def read_jsonl_log(jsonl_path: Path) -> pd.DataFrame:
    """Convert a prediction log from JSONL to a pandas DataFrame.

    Flattens the nested ``features`` object into separate columns when
    present. Returns an empty DataFrame if the file does not exist,
    is empty, contains invalid JSONL, or does not contain any valid
    records.

    Args:
        jsonl_path: Path to the source JSONL prediction log.

    Returns:
        Flattened prediction log as a pandas DataFrame.
    """
    try:
        df = pd.read_json(jsonl_path, lines=True)
    except (FileNotFoundError, ValueError):
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if "features" not in df.columns:
        return df

    features = pd.json_normalize(df["features"])

    return pd.concat(
        [df.drop(columns=["features"]), features],
        axis=1,
    )
