# Repository paths

Repository-relative commands use the root of the cloned Git repository,
written as `.`. The repository `compose.yaml` bind-mounts this checkout at
`/workspace/GeneSPT` inside the container.

## Archive path convention

Paths inside the Zenodo archive are documented relative to the extracted
archive root, for example:

```text
ground_truth_protocol_a/<dataset_id>/fold<fold>/truth.npz
prediction_matrices/<dataset_id>/<method>/fold<fold>/prediction.npz
```

The host location of that root is supplied separately with `--archive-root`.
Generated repository outputs belong under ignored, repository-relative
`results/` or `figures/` paths.
