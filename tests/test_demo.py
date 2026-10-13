import csv
import copy
from datetime import date
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from weather_hedging_demo.data import generate, load_daily, write_daily
from weather_hedging_demo.indices import DataError, daily_degree_days, season_dates, seasonal_indices
from weather_hedging_demo.model import analyze, evaluate, fit
import weather_hedging_demo.run as workflow


class IndexTests(unittest.TestCase):
    def test_mean_temperature_and_base(self):
        self.assertEqual(daily_degree_days(80, 60, "CDD"), 5)
        self.assertEqual(daily_degree_days(70, 50, "HDD"), 5)
        self.assertEqual(daily_degree_days(75, 55, "CDD"), 0)
        self.assertEqual(daily_degree_days(75, 55, "HDD"), 0)

    def test_invalid_daily_temperatures_and_trigger(self):
        for high, low in ((50, 60), (float("nan"), 50), (80, float("inf"))):
            with self.assertRaises(DataError):
                daily_degree_days(high, low, "CDD")
        with self.assertRaises(ValueError):
            daily_degree_days(80, 60, "other")

    def test_season_year_and_leap_day(self):
        winter = season_dates(2024, "HDD")
        self.assertEqual(winter[0], date(2023, 10, 1))
        self.assertEqual(winter[-1], date(2024, 3, 31))
        self.assertEqual(len(winter), 183)
        self.assertIn(date(2024, 2, 29), winter)
        self.assertEqual(len(season_dates(2023, "HDD")), 182)
        self.assertEqual(len(season_dates(2024, "CDD")), 122)

    def fixture(self):
        days = sorted(set(season_dates(2024, "CDD") + season_dates(2024, "HDD")))
        return [{"date": day, "site": site, "tmax_f": 80., "tmin_f": 60.}
                for day in days for site in ("REFERENCE", "LOCAL")]

    def test_full_season_totals(self):
        rows = seasonal_indices(self.fixture(), (2024,))
        self.assertEqual(rows[0]["reference_dd"], 122 * 5)
        self.assertEqual(rows[0]["local_dd"], 122 * 5)
        self.assertEqual(rows[1]["reference_dd"], 0)

    def test_missing_day_fails_instead_of_changing_support(self):
        with self.assertRaisesRegex(DataError, "incomplete"):
            seasonal_indices(self.fixture()[1:], (2024,))

    def test_duplicate_day_site_fails(self):
        rows = self.fixture()
        with self.assertRaisesRegex(DataError, "duplicate"):
            seasonal_indices(rows + [dict(rows[0])], (2024,))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_seed_reproducibility_and_csv_roundtrip(self):
        rows = generate((2023, 2024), 7)
        self.assertEqual(rows, generate((2023, 2024), 7))
        self.assertNotEqual(rows, generate((2023, 2024), 8))
        path = self.root / "daily.csv"
        write_daily(path, rows)
        self.assertEqual(load_daily(path), rows)

    def test_invalid_seed_and_years(self):
        for seed in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                generate((2023, 2024), seed)
        for years in ((2024, 2023), (2023, 2023), (2022, 2024), (True,)):
            with self.assertRaises(ValueError):
                generate(years)

    def test_csv_rejects_duplicate_nonfinite_reversed_or_unknown_site(self):
        header = "date,site,tmax_f,tmin_f\n"
        examples = ["2024-01-01,LOCAL,70,50\n2024-01-01,LOCAL,70,50\n",
                    "2024-01-01,LOCAL,nan,50\n", "2024-01-01,LOCAL,40,50\n",
                    "2024-01-01,OTHER,70,50\n"]
        for number, content in enumerate(examples):
            path = self.root / f"bad-{number}.csv"
            path.write_text(header + content, encoding="utf-8")
            with self.subTest(number=number), self.assertRaises(DataError):
                load_daily(path)

    def test_bad_header_fails(self):
        path = self.root / "bad.csv"
        path.write_text("date,site,tmax_f\n", encoding="utf-8")
        with self.assertRaises(DataError):
            load_daily(path)


def fake_indices():
    rows = []
    for trigger in ("CDD", "HDD"):
        for year in range(2005, 2025):
            days = season_dates(year, trigger)
            x = float((year - 2004) * 100)
            rows.append({"trigger": trigger, "season_year": year, "start_date": days[0].isoformat(),
                         "end_date": days[-1].isoformat(), "days": len(days), "reference_dd": x, "local_dd": 5 + 2 * x})
    return rows


class ModelTests(unittest.TestCase):
    def test_exact_linear_fit_and_test(self):
        model = fit([10, 20, 30, 40], [25, 45, 65, 85])
        self.assertAlmostEqual(model["beta"], 2)
        self.assertAlmostEqual(model["alpha"], 5)
        result = evaluate(model, [50, 60, 70], [105, 125, 145])
        self.assertAlmostEqual(result["residual_variance_ratio"], 0)

    def test_oos_ratio_above_one_and_negative_reduction_are_retained(self):
        model = fit([1, 2, 3], [2, 4, 6])
        result = evaluate(model, [1, 2, 3], [3, 2, 1])
        self.assertAlmostEqual(result["residual_variance_ratio"], 9)
        self.assertAlmostEqual(result["variance_reduction_fraction"], -8)

    def test_mean_bias_is_reported_separately_from_variance(self):
        model = fit([1, 2, 3], [2, 4, 6])
        result = evaluate(model, [4, 5, 6], [18, 20, 22])
        self.assertAlmostEqual(result["residual_variance_ratio"], 0)
        self.assertAlmostEqual(result["mean_tracking_error_dd"], 10)

    def test_degenerate_nonfinite_and_unpaired_samples_fail(self):
        for x, y in (([1, 1, 1], [1, 2, 3]), ([1, 2], [1, 2]),
                     ([1, 2, 3], [1, 2]), ([1, 2, 3], [1, np.nan, 3])):
            with self.assertRaises(DataError):
                fit(x, y)
        with self.assertRaises(DataError):
            evaluate(fit([1, 2, 3], [2, 4, 6]), [4, 5], [7, 7])

    def test_test_changes_cannot_change_fitted_coefficients(self):
        rows = fake_indices()
        first, _ = analyze(rows)
        changed = [{**row, "local_dd": row["local_dd"] + (1000 if row["season_year"] > 2018 else 0)} for row in rows]
        second, _ = analyze(changed)
        for trigger in ("CDD", "HDD"):
            self.assertEqual(first[trigger]["model"], second[trigger]["model"])
            self.assertEqual(first[trigger]["train_season_years"], list(range(2005, 2019)))
            self.assertEqual(first[trigger]["test_season_years"], list(range(2019, 2025)))

    def test_finite_training_inputs_that_overflow_are_rejected(self):
        with self.assertRaisesRegex(DataError, "non-finite calculation"):
            fit([1e200, 2e200, 3e200], [2e200, 4e200, 6e200])

    def test_test_predictions_or_variances_that_overflow_are_rejected(self):
        model = fit([1, 2, 3], [2, 4, 6])
        for x, y in (([1e308, 1.1e308], [1, 2]), ([4, 5, 6], [1e200, 2e200, 3e200])):
            with self.subTest(x=x), self.assertRaisesRegex(DataError, "non-finite calculation"):
                evaluate(model, x, y)

    def test_nonfinite_model_values_are_rejected(self):
        for key in ("alpha", "beta", "local_train_mean"):
            for value in (float("nan"), float("inf")):
                model = {**fit([1, 2, 3], [2, 4, 6]), key: value}
                with self.subTest(key=key, value=value), self.assertRaises(DataError):
                    evaluate(model, [4, 5, 6], [8, 10, 12])

    def test_ratio_overflow_is_rejected_without_clipping(self):
        model = {"alpha": 0., "beta": 1., "local_train_mean": 0.}
        with self.assertRaisesRegex(DataError, "non-finite calculation"):
            evaluate(model, [1, 2, 3], [1e-155, 2e-155, 3e-155])

    def test_input_order_is_irrelevant(self):
        self.assertEqual(analyze(fake_indices()), analyze(list(reversed(fake_indices()))))

    def test_overlapping_weather_dates_fail(self):
        rows = fake_indices()
        rows[-1] = {**rows[-1], "start_date": "2017-01-01"}
        with self.assertRaisesRegex(DataError, "training dates"):
            analyze(rows)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_complete_run_plot_files_counts_and_hashes(self):
        output = self.root / "result"
        result = workflow.run(output)
        self.assertEqual(result["daily_rows"], 14610)
        self.assertEqual(result["paired_season_rows"], 40)
        for trigger in ("CDD", "HDD"):
            self.assertEqual(result["results"][trigger]["model"]["n_train"], 14)
            self.assertEqual(result["results"][trigger]["test"]["n_test"], 6)
        manifest = workflow.verify(output)
        self.assertEqual(manifest["status"], "complete")
        for name in ("index_relationship.png", "test_tracking_errors.png"):
            self.assertEqual((output / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        with self.assertRaises(FileExistsError):
            workflow.run(output)

    def test_output_tampering_is_detected(self):
        output = self.root / "result"
        workflow.run(output)
        with (output / "results.json").open("ab") as handle:
            handle.write(b" ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            workflow.verify(output)

    def test_missing_manifest_metadata_is_rejected(self):
        output = self.root / "result"
        workflow.run(output)
        path = output / "run_manifest.json"
        original = json.loads(path.read_text())
        for key in ("version", "synthetic", "parameters", "runtime", "code_files"):
            changed = copy.deepcopy(original)
            del changed[key]
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.subTest(key=key), self.assertRaises(ValueError):
                workflow.verify(output)
        path.write_text(json.dumps({"status": "complete", "files": original["files"]}), encoding="utf-8")
        with self.assertRaises(ValueError):
            workflow.verify(output)
        path.write_text(json.dumps(original), encoding="utf-8")
        workflow.verify(output)

    def test_invalid_manifest_metadata_types_and_values_are_rejected(self):
        output = self.root / "result"
        workflow.run(output)
        path = output / "run_manifest.json"
        original = json.loads(path.read_text())
        cases = [(["version"], "unknown"), (["version"], []), (["synthetic"], False),
                 (["parameters"], []), (["parameters", "seed"], True),
                 (["parameters", "seed"], -1), (["parameters", "base_f"], float("nan")),
                 (["parameters", "base_f"], 10**400),
                 (["parameters", "season_years"], [2024, 2023]),
                 (["parameters", "season_years"], [2023, 2023]),
                 (["parameters", "season_years"], "2005-2024"),
                 (["parameters", "train_end"], 2024), (["parameters", "train_end"], True),
                 (["runtime", "python"], ""), (["runtime", "numpy"], 2.4),
                 (["code_files"], {}), (["code_files", "model.py"], "not-a-hash"),
                 (["files"], []), (["files", "results.json"], "not-a-hash")]
        for keys, value in cases:
            changed = copy.deepcopy(original)
            target = changed
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.subTest(keys=keys, value=value), self.assertRaises(ValueError):
                workflow.verify(output)
        path.write_text("[]", encoding="utf-8")
        with self.assertRaises(ValueError):
            workflow.verify(output)
        path.write_text(json.dumps(original), encoding="utf-8")
        workflow.verify(output)

    def test_plot_failure_produces_no_valid_output(self):
        output = self.root / "result"
        with patch.object(workflow, "draw", side_effect=RuntimeError("injected plot failure")):
            with self.assertRaises(RuntimeError):
                workflow.run(output)
        self.assertFalse(output.exists())
        stage = next(self.root.glob('.result.staging-*'))
        self.assertTrue((stage / "diagnostics.json").exists())
        self.assertFalse((stage / "run_manifest.json").exists())

    def test_module_cli_from_source(self):
        source = str(Path(__file__).resolve().parents[1] / "src")
        launcher = ("import sys,runpy;sys.path.insert(0," + repr(source) + ");"
                    "sys.argv=['weather_hedging_demo',*sys.argv[1:]];"
                    "runpy.run_module('weather_hedging_demo',run_name='__main__')")
        for args in (("run", str(self.root / "cli")), ("verify", str(self.root / "cli"))):
            result = subprocess.run([sys.executable, "-I", "-B", "-c", launcher, *args],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsInstance(json.loads(result.stdout), dict)


if __name__ == "__main__":
    unittest.main()
