"""Synthetic temperatures and a strict CSV round trip. No network or external observations."""

import csv
from datetime import date, timedelta
import math
from pathlib import Path

import numpy as np

from .indices import DataError, SITES, validate_years

FIELDS = ("date", "site", "tmax_f", "tmin_f")


def generate(years=tuple(range(2005, 2025)), seed: int = 7) -> list[dict]:
    years = validate_years(years)
    if any(b != a + 1 for a, b in zip(years, years[1:])):
        raise ValueError("synthetic years must be contiguous")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    # These coefficients are invented for a classroom example, not fitted to data.
    year_shocks = {year: rng.normal(0, [3.0, 1.5, 3.0]) for year in range(years[0] - 1, years[-1] + 2)}
    start, stop = date(years[0] - 1, 10, 1), date(years[-1], 10, 1)
    output = []
    for index in range((stop - start).days):
        day = start + timedelta(days=index)
        day_number = day.timetuple().tm_yday
        cycle = 60.0 + 22.0 * math.cos(2 * math.pi * (day_number - 196) / 365.25)
        label = day.year + 1 if day.month >= 10 else day.year
        shared_year, reference_year, local_year = year_shocks[label]
        shared_day, reference_day, local_day = rng.normal(0, [4.0, 1.2, 2.5])
        means = {"REFERENCE": cycle + shared_year + reference_year + shared_day + reference_day,
                 "LOCAL": cycle + 3.0 + .85 * shared_year + local_year + .9 * shared_day + local_day}
        for site in SITES:
            half_range = float(rng.uniform(7.0, 12.0))
            output.append({"date": day, "site": site,
                           "tmax_f": round(float(means[site] + half_range), 6),
                           "tmin_f": round(float(means[site] - half_range), 6)})
    return output


def write_daily(path: Path, rows: list[dict]) -> None:
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "date": row["date"].isoformat()})


def load_daily(path: Path) -> list[dict]:
    output, seen = [], set()
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise DataError("unexpected temperature CSV header")
        for raw in reader:
            if set(raw) != set(FIELDS) or any(raw[field] is None for field in FIELDS):
                raise DataError("invalid temperature CSV row")
            try:
                day = date.fromisoformat(raw["date"])
                high, low = float(raw["tmax_f"]), float(raw["tmin_f"])
            except (TypeError, ValueError) as error:
                raise DataError("invalid date or temperature") from error
            if day.isoformat() != raw["date"] or raw["site"] not in SITES:
                raise DataError("invalid date format or site")
            if not math.isfinite(high) or not math.isfinite(low) or low > high:
                raise DataError("non-finite or reversed temperatures")
            position = day, raw["site"]
            if position in seen:
                raise DataError("duplicate date/site")
            seen.add(position)
            output.append({"date": day, "site": raw["site"], "tmax_f": high, "tmin_f": low})
    return output
