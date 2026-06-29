# Demo data

The Code Ocean smoke test uses synthetic data generated at runtime by `scripts/reproduce_demo_results.py`.

This avoids uploading restricted participant EEG recordings while still allowing editors and reviewers to verify that the computational workflow runs end-to-end.

Expected real-data NPZ fields for full experiments:

- `eeg`: float32 array with shape `[N, C, T]`
- `text_embeddings`: float32 array with shape `[N, E]`
- `subject_ids`: optional int64 array with shape `[N]`
- `texts`: optional object/string array with shape `[N]`

Expected geometry NPZ fields:

- `eigenmodes`: float32 array with shape `[V, K]`
- `eigenvalues`: float32 array with shape `[K]`
- `region_names`: optional object/string array with shape `[V]`
