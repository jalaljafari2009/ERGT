# Adaptive Notebook ERGT-03

## Purpose

Stage 19 creates the first Colab-facing notebook for the adaptive relational
control path:

```text
notebooks/ERGT_03_Adaptive_Relational_Control.ipynb
```

The notebook does not change ERGT-01 or ERGT-02. It is a safety and execution
wrapper for the adaptive controller built in stages 17 and 18.

When opened directly through Colab from GitHub, the first setup cell prepares the
full repository automatically:

```text
git clone --depth 1 https://github.com/jalaljafari2009/ERGT.git /content/ERGT
cd /content/ERGT
add /content/ERGT to sys.path
```

## What The Notebook Proves

ERGT-03 is not a scientific claim run. It proves that the adaptive path can be
started safely:

```text
preflight contracts pass
full repository is available under /content/ERGT in Colab
adaptive trainer emits live 100-step rows
meta-control attention fields are visible during execution
fail-fast events stop and export evidence
report bundle is lightweight
checkpoints are excluded
Colab GPU can be released automatically
```

Only after this notebook passes should the project move to a guarded 100- or
200-step smoke against actual training.

## Fixed Artifact Contract

The output bundle name is fixed:

```text
ergt_03_adaptive_control_report_bundle.zip
```

The default local review path after Colab download is:

```text
C:\Users\Administrator\Downloads\ergt_03_adaptive_control_report_bundle.zip
```

The bundle is intentionally lightweight. It includes configs, metrics, progress
logs, controller logs, live diagnostic rows, live markdown tables, plot payloads,
fail-fast reports, and summaries. It excludes checkpoints and large model
artifacts.

## Live Display Contract

The notebook streams adaptive telemetry during execution every:

```text
100 steps
```

Each live event is emitted through `display_live_diagnostic_event`, which receives
the same snapshot that is written to disk. The displayed rows come from:

```text
live_diagnostic_rows
live_diagnostic_tables
live_diagnostic_plot_payloads
```

The table includes loss, geometry scale, alpha, memory state, memory persistence,
memory rigidity, noise risk, attention regime, future leakage, controller
decisions, and meta-control attention weights. The latest table is repeated after
the smoke stage only for review; the full live stream is emitted during the run.

## Failure Safety

The notebook verifies fail-fast behavior before any long run:

```text
future_leak_score > 0 -> failed_fast
```

On failure or completion, the notebook exports the report bundle. In Colab it can
also release the runtime automatically so GPU time is not consumed after the run
has stopped.

## Machine Artifacts

This stage adds:

```text
notebooks/ERGT_03_Adaptive_Relational_Control.ipynb
evaluation/adaptive_notebook_ergt_03.py
experiments/create_adaptive_notebook_ergt_03_report.py
tests/test_adaptive_notebook_ergt_03.py
```

Default report:

```text
runs/contracts/adaptive_notebook_ergt_03.json
```

Usage:

```bash
python experiments/create_adaptive_notebook_ergt_03_report.py
```

## Exit Criteria

This stage is complete when:

```text
notebook JSON parses
short smoke profile is default
guarded 2000 profile is declared
preflight contract tests are present
adaptive trainer is invoked
live 100-step rows, tables, and plots are exported
live 100-step tables are streamed during execution through the notebook callback
fail-fast report is created
fixed bundle name and local review path are declared
GitHub-to-Colab repository bootstrap is declared
auto shutdown hook is present
checkpoint artifacts are excluded
```

Next stage:

```text
Short Smoke and Failure-Safety Validation, now specified in
docs/40_short_smoke_failure_safety_validation.md
```
