"""Validation and resolution of reproducible random seeds."""

import argparse


def validate_seed(seed: int) -> int:
    """Validate a seed accepted by NumPy and scikit-learn."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Random seed must be an integer between 0 and 4294967295.")
    return seed


def resolve_seed(root: int, override: int | None = None) -> int:
    """Use a component override when provided, otherwise use the root seed."""
    validate_seed(root)
    return validate_seed(override) if override is not None else root


def parse_seed(value: str) -> int:
    """Parse a command-line seed."""
    try:
        return validate_seed(int(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Random seed must be an integer between 0 and 4294967295."
        ) from exc


def parse_seed_override(value: str) -> int | None:
    """Parse a component seed or the literal ``inherit``."""
    return None if value == "inherit" else parse_seed(value)
