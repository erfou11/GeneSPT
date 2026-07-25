# Baseline methods

The benchmark compares GeneSPT with six external methods: Tangram, TransImp,
SpaIM, SpaGE, stPlus, and stAI. Figure 6 uses the five methods available in
the completed downstream-analysis matrices.

The exact adapters are under `baseline_adapters/`. Upstream revisions,
parameters, runtime patches, information-boundary adaptations and failure
rules are documented in:

- `configs/protocol_a_baseline_versions.yaml`
- `docs/BASELINE_ADAPTATION.md`
- `manifests/protocol_a/FORMAL_BASELINE_RUN_MANIFEST.csv`

All methods use the same frozen whole-gene splits and centralized evaluator.
Full training uses the recorded upstream revisions. The archived compact
matrices support direct metric recomputation.
