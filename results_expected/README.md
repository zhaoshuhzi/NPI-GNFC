# Expected outputs

A successful Code Ocean smoke test should create the following files in `/results` or `./results`:

- `demo_metrics.json`
- `demo_loss_curve.csv`
- `demo_source_field.npy`
- `demo_decoded_embeddings.npy`

The exact loss values may vary slightly across hardware and PyTorch versions, but the script should complete without errors and print the decoded embedding and source-field tensor shapes.
