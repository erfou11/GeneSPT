# Protocol A baseline adapters

This directory contains the exact six adapter entry points used for the
formal benchmark:

| Method | Adapter |
| --- | --- |
| Tangram | `tangram/run_tangram_mhpr_fold_from_split.py` |
| TransImp | `transimp/run_transimp_mhpr_fold_from_split.py` |
| SpaIM | `spaim/run_spaim_mhpr_fold_from_split.py` |
| SpaGE | `spage/run_spage_mhpr_fold_from_split.py` |
| stPlus | `stplus/run_stplus_mhpr_fold_from_split.py` |
| stAI | `stai/run_stai_protocol_a.py` |

The adapters enforce the frozen train/validation/test gene indices, fit each
external method on training genes only, use train-gene library sizes for ST
normalization, save complete final-test predictions, and defer every reported
metric to the centralized evaluator.

The adapters target the exact upstream revisions recorded in
`configs/protocol_a_baseline_versions.yaml` and
`docs/BASELINE_ADAPTATION.md`. Runtime parameters, protocol adaptations, and
failure rules are documented alongside them. The patches under `patches/` are
relative to those upstream revisions.
stAI uses the official source without an upstream patch; its three-stage
wrapper is documented in `stai/README.md`.

These adapters are training entry points. Matrix-level reproduction uses
`scripts/reproducibility/recompute_protocol_a_benchmark.py` with the extracted
data archive.
