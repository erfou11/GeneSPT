# Source anchor

The public source anchor is the root of the cloned Git repository, written as
`.` in repository-relative commands. The repository-root `compose.yaml`
bind-mounts that checkout at `/workspace/GeneSPT` inside the container. No
parent workspace, host-specific checkout path, or Dev Container file is part
of the public execution contract.

The source tree was refreshed from the cleaned active workbench on 2026-07-10.
Only `main/` and `scripts/` files retained by the post-cleanup dependency and
publication audit are tracked. Historical diffusion routes, superseded dataset
scripts, mistaken PSP configurations, one-off manuscript update helpers, and
old figure versions are intentionally outside the public repository and must
not be treated as active provenance.

## Archive path convention

Paths inside the reviewer archive are always documented relative to the
extracted archive root, for example:

```text
ground_truth/Cell2location_mouse_brain_ST8059048_shared12819/gene_names.txt
results_source_data/cell2location_psp_ablation_strict/cell2location_strict_psp_toggle_fold_metrics.csv
```

The host location of that root is supplied separately with `--archive-root`.
Generated repository outputs belong under ignored, repository-relative
`results/` or `figures/` paths and should be archived through Zenodo when they
become formal evidence.
