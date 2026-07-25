# Active code map

This repository contains one reviewer-facing implementation path for the
formal Protocol A results.

## Formal execution

| Responsibility | Entry point |
| --- | --- |
| Prepare fold-specific Protocol A inputs | `scripts/protocol_a_full/prepare_protocol_a_inputs.py` |
| Train GeneSPT-GC and full GeneSPT | `scripts/protocol_a_full/run_protocol_a_genespt.py` |
| Run Tangram, TransImp, SpaIM, SpaGE and stPlus | `scripts/protocol_a_full/run_protocol_a_baselines.py` |
| Run stAI with a truth-isolated three-stage wrapper | `baseline_adapters/stai/run_stai_protocol_a.py` |
| Select and apply the GeneSPT readout | `scripts/protocol_a_full/run_protocol_a_validation_readout.py` |
| Centralized raw evaluation | `scripts/protocol_a_full/evaluate_protocol_a_raw_predictions.py` |
| Figure 3 controls | `scripts/protocol_a_full/run_protocol_a_figure3_controls.py` |

The model implementation imported by these entry points is under `main/`.
The authoritative metric implementation is `src/genespt/metrics.py`.

## Figures and source data

The current Figure 2-6 generators are the five
`scripts/protocol_a_full/generate_protocol_a_figure*.py` files. Reviewer-facing
source tables are assembled by
`scripts/protocol_a_full/build_reviewer_source_data.py`.

## Archive verification

All public archive operations are under `scripts/reproducibility/`:

- `verify_protocol_a_release.py` checks the formal method set, matrix counts,
  information-boundary records and file hashes.
- `recompute_protocol_a_benchmark.py` recomputes the 210 benchmark fold rows.
- `recompute_protocol_a_mechanism.py` recomputes the 90 Figure 3 mechanism rows.
- the two export scripts build compact benchmark and mechanism matrices without
  modifying the formal run.
- `regenerate_release_metadata.py` creates package structure and SHA256 files.

Exploratory queues, superseded plotters and pre-Protocol-A result collectors
are not part of the reviewer repository.
