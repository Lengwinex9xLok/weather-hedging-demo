"""A training-period linear tracking fit, evaluated on later season-years."""

from datetime import date
import numpy as np

from .indices import DataError


def vectors(x, y, minimum: int):
    try:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise DataError("index values must be representable as finite floats") from error
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < minimum:
        raise DataError("paired one-dimensional samples are required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise DataError("non-finite index values")
    return x, y


def fit(x, y) -> dict:
    x, y = vectors(x, y, 3)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            centered = x - x.mean()
            denominator = float(centered @ centered)
            if denominator <= 0:
                raise DataError("reference training index has zero variance")
            beta = float(centered @ (y - y.mean()) / denominator)
            alpha = float(y.mean() - beta * x.mean())
            local_mean = float(y.mean())
    except (FloatingPointError, OverflowError) as error:
        raise DataError("non-finite calculation in training fit") from error
    if not np.isfinite([denominator, beta, alpha, local_mean]).all():
        raise DataError("non-finite calculation in training fit")
    return {"alpha": alpha, "beta": beta, "local_train_mean": local_mean, "n_train": len(x)}


def evaluate(model: dict, x, y) -> dict:
    x, y = vectors(x, y, 2)
    try:
        alpha, beta, local_mean = (float(model[key]) for key in ("alpha", "beta", "local_train_mean"))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise DataError("model must contain finite coefficients and training mean") from error
    if not np.isfinite([alpha, beta, local_mean]).all():
        raise DataError("model must contain finite coefficients and training mean")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            prediction = alpha + beta * x
            residual = y - prediction
            baseline = y - local_mean
            baseline_variance = float(np.var(baseline, ddof=1))
            if baseline_variance <= 0:
                raise DataError("local test index has zero variance")
            residual_variance = float(np.var(residual, ddof=1))
            ratio = residual_variance / baseline_variance
            mean_error = float(residual.mean())
    except (FloatingPointError, OverflowError) as error:
        raise DataError("non-finite calculation in test evaluation") from error
    if not np.isfinite([baseline_variance, residual_variance, ratio, 1 - ratio, mean_error]).all():
        raise DataError("non-finite calculation in test evaluation")
    # OOS can be worse than the baseline: ratios above one are retained.
    return {"n_test": len(y), "baseline_variance_dd2": baseline_variance,
            "residual_variance_dd2": residual_variance, "residual_variance_ratio": ratio,
            "variance_reduction_fraction": 1 - ratio, "mean_tracking_error_dd": mean_error}


def analyze(indices: list[dict], train_end: int = 2018) -> tuple[dict, list[dict]]:
    if type(train_end) is not int:
        raise ValueError("train_end must be an integer")
    training_dates = [date.fromisoformat(row["end_date"]) for row in indices if row["season_year"] <= train_end]
    testing_dates = [date.fromisoformat(row["start_date"]) for row in indices if row["season_year"] > train_end]
    if not training_dates or not testing_dates:
        raise DataError("both training and test seasons are required")
    if max(training_dates) >= min(testing_dates):
        raise DataError("training dates must end before any test season begins")
    results, predictions = {}, []
    for trigger in ("CDD", "HDD"):
        rows = sorted((row for row in indices if row["trigger"] == trigger), key=lambda row: row["season_year"])
        years = [row["season_year"] for row in rows]
        if len(years) != len(set(years)):
            raise DataError("duplicate trigger/season-year")
        train = [row for row in rows if row["season_year"] <= train_end]
        test = [row for row in rows if row["season_year"] > train_end]
        model = fit([row["reference_dd"] for row in train], [row["local_dd"] for row in train])
        metrics = evaluate(model, [row["reference_dd"] for row in test], [row["local_dd"] for row in test])
        results[trigger] = {"model": model, "test": metrics,
            "train_season_years": [row["season_year"] for row in train],
            "test_season_years": [row["season_year"] for row in test],
            "ratio_unit": "dimensionless", "observation_unit": "paired season-year",
            "denominator": "sample variance of local degree-day index on these same test years (ddof=1)",
            "era": "synthetic; the dates are invented labels, not observed historical data"}
        for row in rows:
            prediction = model["alpha"] + model["beta"] * row["reference_dd"]
            predictions.append({**row, "split": "train" if row["season_year"] <= train_end else "test",
                                "predicted_local_dd": prediction, "tracking_error_dd": row["local_dd"] - prediction,
                                "baseline_error_dd": row["local_dd"] - model["local_train_mean"]})
    return results, predictions
