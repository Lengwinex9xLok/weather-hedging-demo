# Local validation

Validated on 13 October 2026 with Python 3.11.9 on Windows, NumPy 2.4.4 and
Matplotlib 3.10.9. This is a local review draft, not a published research result.

## Checks completed

- Installed the standalone package into a new virtual environment.
- Ran `python -B -m unittest discover -s tests -v`: **27 tests passed** after the
  manifest and numerical checks were added (the initial version had 21 tests).
- Used the installed console entry to generate `demo_output/run-02` with seed 7.
  `run-01` is preserved as the earlier run.
- Verified the run's six artifact hashes against its complete manifest.
- Ran `pip check`: no broken requirements.
- Opened both generated images and checked titles, units, legends and margins.

The unit-test CLI check explicitly loads source code. The installed console entry
and module entry were checked separately from outside the project directory; they
do not depend on the research repository.

## Follow-up fixes

Manifest verification now requires a supported version, the synthetic flag, valid
run parameters, non-empty runtime versions, and the expected code/artifact hash
inventories. Missing metadata and invalid types fail even if artifact hashes match.
Code hashes are recorded identities; an old run is not required to match today's
source files. The complete earlier `run-01` manifest remains valid.

Fitting and evaluation now reject calculation overflow, non-finite model values
and non-finite metrics with `DataError`. Finite ratios above one are still retained.
The added tests reproduce both review findings and cover these numerical cases.
The generator, seed, train/test years, formulas and default results did not change.

## Default synthetic run

The input contains 14,610 date/site rows (two fictional locations), covering
1 October 2004 through 30 September 2024. It produces 40 paired seasonal rows:
20 CDD and 20 HDD. For each index, 14 season-years train the relationship and the
six later season-years test it. Training dates end before any test season begins.

| Index | Test residual variance ratio | Test mean tracking error (°F-days) |
|---|---:|---:|
| CDD | 1.103786 | +110.827600 |
| HDD | 0.960965 | −138.437700 |

The ratio is dimensionless. Its observation unit is a paired season-year; the
denominator is the local index's sample variance on the same six test season-years
(2019–2024 labels, entirely synthetic). Means use those same observations. These
six points do not establish statistical significance or future performance.

The CDD variance reduction fails in this example. No generator parameters or
seed were changed after inspecting the result. A low variance ratio would not
rule out mean bias, which is why the mean error is reported separately.

## Failure cases checked

Tests reject incomplete seasons, duplicate date/site records, invalid temperatures,
overlapping train/test dates and degenerate index variance. They also check
that changing test values cannot change fitted coefficients, that a ratio above one
is retained, and that output tampering or an interrupted plotting step cannot be
accepted as a complete run.

## Limits and review scope

No observed station data, contract quotes, station mappings, private research
outputs, credentials or Git history were copied into this project. The research
repository's tracked/unignored file bytes and Git status were unchanged when checked.
The virtual environment and generated run directory are excluded by `.gitignore`.
The two README previews are copies of the synthetic run's images.

Windows installation and rendering were executed locally. The macOS/Linux command
paths in the README are documented, but were not executed on those platforms.
There is no climate-model validation, contract pricing, loss model, transaction-cost
model or proof of profitable trading. Hashes provide file-integrity checks, not
independent signatures or evidence of research validity.

No Git repository was initialized and nothing was committed, pushed or published.
