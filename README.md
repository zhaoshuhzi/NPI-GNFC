# NPI+GNFC

A PyTorch reference implementation of **NPI+GNFC** for non-invasive EEG-to-text modeling.

The implementation follows the model description used in the manuscript draft:

- **Geometric Neural Field Encoder**: estimates perturbation-driven effective connectivity from EEG and expands the dynamics into a continuous cortical neural field using geometric eigenmodes.
- **Meta-Diffusion Semantic Decoder**: aligns neural-field representations to text embeddings under limited calibration data, and reconstructs semantic embeddings through conditional diffusion.

This repository is intended as a clean, extensible research scaffold. It includes a runnable smoke test with synthetic data and data adapters for NPZ-formatted EEG, geometry, and text-embedding files.

## Repository layout

```text
npi_gnfc_github/
├── configs/default.yaml
├── examples/smoke_train.py
├── scripts/train_chineseeeg.py
├── src/npi_gnfc/
│   ├── config.py
│   ├── data.py
│   ├── diffusion.py
│   ├── encoder.py
│   ├── geometry.py
│   ├── losses.py
│   ├── maml.py
│   ├── metrics.py
│   ├── model.py
│   ├── npi.py
│   ├── train.py
│   └── utils.py
└── tests/test_smoke.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Smoke test

```bash
python examples/smoke_train.py --steps 5
```

Expected behavior: the script creates synthetic EEG, geometry, and text-embedding tensors, runs several optimization steps, and prints the training losses.

## Expected data format

### EEG + text embedding NPZ

The training script expects an `.npz` file containing:

```text
eeg:             float32 array, shape [N, C, T]
text_embeddings: float32 array, shape [N, E]
subject_ids:     int64 array, optional, shape [N]
texts:           object/string array, optional, shape [N]
```

where `N` is sample count, `C` is EEG channel count, `T` is time length, and `E` is text embedding dimension.

### Geometry NPZ

The geometry file should contain:

```text
eigenmodes:  float32 array, shape [V, K]
eigenvalues: float32 array, shape [K]
```

where `V` is the number of cortical vertices or parcels and `K` is the number of retained geometric eigenmodes.

Optional fields:

```text
region_names: object/string array, shape [V]
```

## Example training on real NPZ data

```bash
python scripts/train_chineseeeg.py \
  --data /path/to/chineseeeg_embeddings.npz \
  --geometry /path/to/hcp_geometry.npz \
  --config configs/default.yaml \
  --output runs/chineseeeg_npi_gnfc
```

## Notes

1. This implementation reconstructs and decodes **semantic embeddings**. Turning decoded embeddings into final text requires a tokenizer/LLM retrieval or generation layer. The repository keeps that step modular to avoid coupling the model to a specific language model.
2. HD95, CER, WER, BLEU, Dice, IoU, and MCD utilities are included in `metrics.py`.
3. The MAML component is implemented as a lightweight first-order adaptation utility that can be used to adapt the semantic projection layers with a small support set.

## Citation placeholder

If you use this code, please cite the associated NPI+GNFC manuscript after publication.
