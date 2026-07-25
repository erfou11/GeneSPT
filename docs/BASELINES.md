# Baseline methods

The formal benchmark includes exactly six external methods: Tangram,
TransImp, SpaIM, SpaGE, stPlus and stAI. No other external method is part of
the reported Protocol A panel. Figure 6 retains the five methods used in its
frozen downstream-analysis run and is not a second benchmark method list.

The exact adapters are under `baseline_adapters/`. Upstream revisions,
parameters, runtime patches, information-boundary adaptations and failure
rules are documented in:

- `configs/protocol_a_baseline_versions.yaml`
- `docs/BASELINE_ADAPTATION.md`
- `manifests/protocol_a/FORMAL_BASELINE_RUN_MANIFEST.csv`

All methods use the same frozen whole-gene splits and centralized evaluator.
Full third-party repositories are not vendored; reviewers can either check out
the recorded revisions for retraining or use the archived compact matrices for
one-command metric reproduction.
