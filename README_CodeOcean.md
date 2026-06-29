# Code Ocean reproducibility notes for CAGNNF

This repository can be imported into Code Ocean from the public Git URL:

```text
https://github.com/zhaoshuhzi/NPI-GNFC
```

The Code Ocean capsule should use a Python/PyTorch starter environment. Install dependencies with:

```bash
pip install -r requirements_codeocean.txt
```

Set `run.sh` as the file to run. The run script executes a lightweight synthetic-data demonstration of the CAGNNF computation path:

```bash
bash run.sh
```

The demo does not require restricted EEG data. It creates synthetic EEG, cortical geometry and text-embedding tensors, then runs a compact model that follows the intended pathway:

```text
scalp EEG -> NPI-like perturbation propagation -> source-space neural field -> semantic embedding reconstruction
```

Outputs are written to `/results` in Code Ocean or `./results` when run locally:

- `demo_metrics.json`
- `demo_loss_curve.csv`
- `demo_source_field.npy`
- `demo_decoded_embeddings.npy`

Full manuscript-scale training requires the complete EEG datasets and preprocessing pipeline. This lightweight capsule demo is intended to verify the executable workflow and software dependencies for peer review.
