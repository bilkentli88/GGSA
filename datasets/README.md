# Datasets Used in GGSA Experiments

This directory is intended to store the dataset files used in the GGSA experiments.

> **Note:** The datasets are **not redistributed** in this repository. Please download them from their official sources and place the required files in this directory before running the scripts.

## Expected Files

The scripts in this repository expect the following dataset files:

- `CICIDS2017_day.csv`
- `train_test_network.csv`
- `UNSW_NB15.csv`

## Dataset 1: CIC-IDS2017

- **Benchmark:** CIC-IDS2017
- **File expected by script:** `CICIDS2017_day.csv`
- **Script:** `scripts/ggsa_CIC.py`
- **Usage in this repository:** real-data evaluation under degraded telemetry
- **Target setting:** binary intrusion detection stream
- **Label handling:** `BENIGN -> 0`; all other labels -> `1`
- **Preprocessing in script:**
  - column names are stripped
  - numeric columns are selected
  - missing values are filled with `0`
  - values are clipped to the configured numeric range
- **Additional note:** this repository uses a replay-style sequential setting rather than a shuffled offline classification setting

## Dataset 2: ToN-IoT

- **Benchmark:** ToN-IoT Network dataset
- **File expected by script:** `train_test_network.csv`
- **Script:** `scripts/ggsa_TON_IOT.py`
- **Usage in this repository:** second real-data benchmark with a multi-attack protocol
- **Attack handling:** attack categories are selected automatically from the `type` column unless manually specified in the script
- **Typical evaluated attacks:** `ddos`, `dos`, `injection`, `password` (subject to the validity rules in the script)
- **Preprocessing in script:**
  - column names are stripped
  - the `type` column is normalized to lowercase
  - selected categorical columns (if present) are label-encoded
  - identifier / leaky columns are removed
  - missing values are filled with `0`

## Dataset 3: UNSW-NB15

- **Benchmark:** UNSW-NB15
- **File expected by script:** `UNSW_NB15.csv`
- **Script:** `scripts/ggsa_UNSW.py`
- **Usage in this repository:** supplementary architectural extension experiment
- **Target setting:** binary target constructed from `attack_cat == "Exploits"` by default
- **Preprocessing in script:**
  - selected non-feature columns such as `id`, `label`, and `attack_cat` are removed
  - numeric columns are selected
  - missing values are filled with `0`

## Reproducibility Note

Some public benchmark datasets exist in multiple processed versions or file organizations. To reduce ambiguity, this repository documents:

- the exact file names expected by the scripts
- the target labels or attack families used
- the preprocessing assumptions implemented in the code

Users are encouraged to verify that the downloaded dataset files match the expected structure before running the experiments.


