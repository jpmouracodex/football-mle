"""Small HTTP helper for fetching remote CSVs with a browser-like User-Agent."""
from __future__ import annotations

import io

import pandas as pd
import requests

__all__ = ["read_csv_url"]

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; football_mle/1.0)"}


def read_csv_url(url: str, *, encoding: str = "utf-8", timeout: int = 30, **kwargs: object) -> pd.DataFrame:
    """Fetch ``url`` and parse it as CSV into a ``DataFrame``.

    Uses ``requests`` with a User-Agent (some hosts reject the default one) and a
    timeout, then hands the decoded text to :func:`pandas.read_csv`.
    """
    response = requests.get(url, headers=_HEADERS, timeout=timeout)
    response.raise_for_status()
    text = response.content.decode(encoding, errors="ignore")
    return pd.read_csv(io.StringIO(text), **kwargs)
