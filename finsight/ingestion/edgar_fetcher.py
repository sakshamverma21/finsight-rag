"""Pull filings from SEC EDGAR via edgartools and normalize the metadata we need."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from edgar import Company

from finsight.config import init_edgar


@dataclass
class FilingRef:
    """A fetched filing plus the metadata every chunk/fact will inherit."""

    ticker: str
    form: str
    fiscal_year: int
    fiscal_quarter: int | None
    filing_date: str
    accession: str
    obj: Any  # edgartools TenK / TenQ object
    company_name: str = ""

    def base_meta(self) -> dict:
        return {
            "ticker": self.ticker,
            "form": self.form,
            "fiscal_year": self.fiscal_year,
            "fiscal_quarter": self.fiscal_quarter,
            "filing_date": self.filing_date,
            "accession": self.accession,
        }


def _fiscal_year(period_of_report) -> int:
    # period_of_report renders as an ISO date, e.g. "2025-09-27".
    return int(str(period_of_report)[:4])


def fetch_latest_10k(ticker: str) -> FilingRef:
    """Fetch the most recent *original* 10-K for a ticker as a FilingRef.

    Excludes amendments: `get_filings(form="10-K")` also matches "10-K/A", and an
    amendment (often just Part III) carries no financial statements. We keep only
    the exact "10-K" form and take the most recent.
    """
    init_edgar()
    candidates = [f for f in Company(ticker).get_filings(form="10-K") if f.form == "10-K"]
    if not candidates:
        raise ValueError(f"no original 10-K found for {ticker}")
    filing = max(candidates, key=lambda f: str(f.filing_date))
    return FilingRef(
        ticker=ticker,
        form="10-K",
        fiscal_year=_fiscal_year(filing.period_of_report),
        fiscal_quarter=None,
        filing_date=str(filing.filing_date),
        accession=filing.accession_no,
        obj=filing.obj(),
        company_name=str(getattr(filing, "company", "") or ""),
    )
