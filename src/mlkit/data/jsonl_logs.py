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

    if "predicted_probabilities" in df.columns:
        probabilities = pd.json_normalize(df.pop("predicted_probabilities"))
        probabilities.columns = [
            f"predicted_probability_{name}" for name in probabilities.columns
        ]
        if set(probabilities.columns).intersection(df.columns):
            raise ValueError(
                "Prediction probability columns overlap existing event fields."
            )
        df = pd.concat([df, probabilities], axis=1)

    if "features" in df.columns:
        features = pd.json_normalize(df.pop("features"))
        return pd.concat([df, features], axis=1)
    else:
        return df
