"""Mean-temperature degree days, with explicit seasons and full date coverage."""

from datetime import date, timedelta
import math

BASE_F = 65.0
SITES = ("REFERENCE", "LOCAL")


class DataError(ValueError):
    pass


def validate_years(years):
    values = tuple(years)
    if not values or any(type(year) is not int or not 2 <= year <= 9998 for year in values):
        raise ValueError("years must be integers between 2 and 9998")
    if values != tuple(sorted(set(values))):
        raise ValueError("years must be unique and sorted")
    return values


def season_dates(year: int, trigger: str) -> tuple[date, ...]:
    validate_years((year,))
    if trigger == "CDD":
        start, stop = date(year, 6, 1), date(year, 10, 1)
    elif trigger == "HDD":
        # Winter is labelled by its January: Oct-Dec belong to the following year.
        start, stop = date(year - 1, 10, 1), date(year, 4, 1)
    else:
        raise ValueError("trigger must be CDD or HDD")
    return tuple(start + timedelta(days=index) for index in range((stop - start).days))


def daily_degree_days(tmax_f: float, tmin_f: float, trigger: str) -> float:
    if not math.isfinite(tmax_f) or not math.isfinite(tmin_f) or tmin_f > tmax_f:
        raise DataError("temperatures must be finite and tmin must not exceed tmax")
    mean = (tmax_f + tmin_f) / 2
    if trigger == "CDD":
        return max(mean - BASE_F, 0.0)
    if trigger == "HDD":
        return max(BASE_F - mean, 0.0)
    raise ValueError("trigger must be CDD or HDD")


def seasonal_indices(rows: list[dict], years) -> list[dict]:
    years = validate_years(years)
    lookup = {}
    for row in rows:
        day, site = row["date"], row["site"]
        if type(day) is not date or site not in SITES:
            raise DataError("invalid date or site")
        position = day, site
        if position in lookup:
            raise DataError("duplicate date/site")
        # Validate both temperatures even for days outside the included seasons.
        daily_degree_days(row["tmax_f"], row["tmin_f"], "CDD")
        lookup[position] = row
    output = []
    for trigger in ("CDD", "HDD"):
        for year in years:
            days = season_dates(year, trigger)
            totals = {}
            for site in SITES:
                if any((day, site) not in lookup for day in days):
                    raise DataError(f"incomplete {trigger} season {year}, {site}; no imputation")
                totals[site] = math.fsum(daily_degree_days(lookup[day, site]["tmax_f"],
                                        lookup[day, site]["tmin_f"], trigger) for day in days)
            output.append({"trigger": trigger, "season_year": year,
                           "start_date": days[0].isoformat(), "end_date": days[-1].isoformat(),
                           "days": len(days), "reference_dd": totals["REFERENCE"], "local_dd": totals["LOCAL"]})
    return output
