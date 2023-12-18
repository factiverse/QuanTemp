"""Util functions for loading data."""
import json
from typing import Any, Dict


def read_json(path) -> Dict[Any, Any]:
    """Reads the fact check dataset from the given path.

    Args:
        path: The path to the dataset.

    Returns:
        A list of facts.
    """
    with open(path) as f:
        data = json.load(f)
    return data
