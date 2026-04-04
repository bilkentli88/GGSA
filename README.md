# GGSA: Gradient-Guided Stress Adaptation for Streaming Intrusion Detection Under Degraded Telemetry

This repository contains the implementation of **GGSA (Gradient-Guided Stress Adaptation)**, a latency-aware online adaptation method for streaming intrusion detection under degraded telemetry.

The code accompanies the revised manuscript on GGSA and includes the main experimental scripts used for:

- **CIC-IDS2017**
- **ToN-IoT**
- **UNSW-NB15** (supplementary architectural extension)

## Overview

Real-time intrusion detection systems may remain accurate and stable in aggregate terms while still reacting too slowly under degraded telemetry. GGSA is designed to reduce this delay by strengthening local decision support near attack onset through **uncertainty-triggered, gradient-guided stress synthesis**.

The repository includes:

- controlled real-data streaming evaluation scripts
- comparisons against **Standard**, **PGD**, **DTR**, **RSA**, and **LSTM** baselines
- runtime-oriented evaluation outputs
- supplementary recurrent / multi-architecture experiments

## Main Idea

Under degraded telemetry, early attack signatures may move into sparsely supported regions near the decision boundary. In such cases, a detector may become conservative and delay positive predictions.

GGSA addresses this problem by generating local stress samples during uncertainty-critical periods and using them to strengthen local decision support near attack onset. The method is designed for **fully online** operation.

## Repository Structure

A suggested structure is:

```text
GGSA/
├── scripts/
│   ├── ggsa_CIC.py
│   ├── ggsa_TON_IOT.py
│   └── ggsa_UNSW.py
├── data/
├── results/
├── README.md
├── LICENSE
└── requirements.txt
```

You may adapt the folder organization as needed.

## Included Scripts

### 1. CIC-IDS2017
**Script:** `ggsa_CIC.py`

This script evaluates GGSA and baseline methods on the CIC-IDS2017 dataset under multiple telemetry degradation regimes. It includes:
- MLP-based GGSA
- Standard / PGD / DTR / RSA baselines
- LSTM baseline
- per-seed runtime reporting
- severity-based summaries

Expected dataset file:
- `CICIDS2017_day.csv`

### 2. ToN-IoT
**Script:** `ggsa_TON_IOT.py`

This script evaluates GGSA on ToN-IoT using a multi-attack protocol. It automatically selects attack categories subject to support and segment-validity constraints, then reports:
- per-attack, per-severity summaries
- aggregated summaries across attacks
- runtime-oriented metrics

Expected dataset file:
- `train_test_network.csv`

### 3. UNSW-NB15
**Script:** `ggsa_UNSW.py`

This script provides a supplementary architectural extension on UNSW-NB15. It compares:
- Standard
- PGD
- RSA
- LSTM-Base
- GGSA-MLP
- GGSA-LSTM

Expected dataset file:
- `UNSW_NB15.csv`

## Requirements

The scripts use standard Python scientific computing and deep learning libraries.

Typical dependencies include:

- Python 3.10+
- NumPy
- pandas
- scikit-learn
- PyTorch
- joblib

You can install them with:

```bash
pip install -r requirements.txt
```

## Example requirements.txt

A minimal `requirements.txt` can contain:

```text
numpy
pandas
scikit-learn
torch
joblib
```

## Datasets

Please obtain the datasets from their official sources and place the corresponding CSV files in the working directory or update the dataset filename variables inside the scripts.

Datasets used in this project:
- **CIC-IDS2017**
- **ToN-IoT**
- **UNSW-NB15** (supplementary experiment)

This repository does **not** redistribute the datasets.

## Running the Experiments

Example usage:

```bash
python ggsa_CIC.py
python ggsa_TON_IOT.py
python ggsa_UNSW.py
```

Before running a script, make sure the expected dataset file is present in the same directory or edit the filename variable at the top of the script.

## Methods Included

Depending on the script, the repository includes comparisons against some or all of the following methods:

- **Standard** online learning
- **PGD** (passive adversarial-style hardening baseline)
- **DTR** (dynamic threshold regulation)
- **RSA** (random stress augmentation)
- **LSTM** baseline
- **GGSA-MLP**
- **GGSA-LSTM** (supplementary UNSW experiment)

## Reproducibility Notes

The experiments use:
- fully online replay-style evaluation
- multiple fixed random seeds
- telemetry degradation severity regimes
- uncertainty-triggered repair logic
- Top-K feature selection for local stress synthesis

Important experimental settings are documented directly inside the scripts.

## Citation

If you use this repository, please cite the associated paper. Until final publication details are available, you may cite the repository as:

```bibtex
@misc{altay2026ggsa,
  author       = {A. T. Altay},
  title        = {GGSA: Gradient-Guided Stress Adaptation for Streaming Intrusion Detection Under Degraded Telemetry},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Repository name: GGSA}
}
```

## License

This project is released under the **MIT License**. See the `LICENSE` file for details.

## Contact

For questions about the code or the accompanying manuscript, please contact the repository author.
