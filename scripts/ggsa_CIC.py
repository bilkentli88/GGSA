"""
GGSA: Gradient-Guided Stress Adaptation
=======================================

Cleaned reproduction script for the CIC-IDS2017 experiment.

What this script includes
-------------------------
1. GGSA on the dynamic MLP backbone
2. Baselines: Standard, PGD, DTR, RSA, and LSTM
3. Runtime / overhead measurements
4. Multi-seed evaluation across three telemetry degradation regimes

Important note
--------------
In this script, GGSA is implemented on the dynamic MLP backbone.
The LSTM model is included as a baseline comparator under the same
streaming protocol.

Expected input
--------------
Place the CIC dataset CSV file (for example: 'CICIDS2017_day.csv')
in the same folder as this script, or update DATA_FILENAME below.

Outputs
-------
The script writes a JSON results file containing:
- per-seed raw results
- summary statistics by method and severity regime
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


# ============================================================================
# USER SETTINGS
# ============================================================================

DATA_FILENAME = "CICIDS2017_day.csv"
OUTPUT_FILENAME = "results_ggsa_cic_with_lstm.json"

# Number of parallel jobs used across random seeds.
# Reduce this if your machine has limited CPU / RAM.
N_JOBS = 4


# ============================================================================
# GLOBAL EXPERIMENT CONFIGURATION
# ============================================================================

SEEDS: List[int] = [
    88, 109, 253, 371, 458, 555, 666, 793, 907, 1009,
    1103, 1201, 1301, 1409, 1511, 1601, 1789, 1877, 1971, 2025
]

# Shared experimental hyperparameters
TOPK_FEATURES = 20
HARDENING_STEPS = 5
PROBA_TRIGGER_LOW = 0.30
PROBA_TRIGGER_HIGH = 0.70
UNCERTAINTY_THRESHOLD = 0.60
FN_BIAS_WEIGHT = 5.0
PGD_EPSILON = 0.5
CLIP_VALUE = 1e12

# Streaming protocol
WARMUP_STEPS = 2000
PREDICT_START_INDEX = 2000

# LSTM baseline settings
SEQ_LEN = 8
LSTM_HIDDEN_DIM = 32
LSTM_NUM_LAYERS = 1

# Evaluation methods
METHODS: List[Tuple[str, str]] = [
    ("Standard", "mlp"),
    ("PGD", "mlp"),
    ("DTR", "mlp"),
    ("RSA", "mlp"),
    ("GGSA", "mlp"),
    ("LSTM", "lstm"),
]


# ============================================================================
# TELEMETRY DEGRADATION MODEL
# ============================================================================

@dataclass
class FaultParams:
    """Parameters controlling the telemetry degradation process."""
    noise_std: float
    dropout_p: float
    drift_scale: float
    burst_p: float
    burst_len: int
    burst_noise_mult: float


SEVERITIES: Dict[str, FaultParams] = {
    # S1 = Low
    "S1": FaultParams(
        noise_std=0.01,
        dropout_p=0.02,
        drift_scale=0.002,
        burst_p=0.002,
        burst_len=10,
        burst_noise_mult=2.0,
    ),
    # S2 = Medium
    "S2": FaultParams(
        noise_std=0.05,
        dropout_p=0.10,
        drift_scale=0.015,
        burst_p=0.01,
        burst_len=20,
        burst_noise_mult=3.0,
    ),
    # S3 = Severe
    "S3": FaultParams(
        noise_std=0.15,
        dropout_p=0.30,
        drift_scale=0.05,
        burst_p=0.02,
        burst_len=30,
        burst_noise_mult=6.0,
    ),
}


class TelemetryDegradation:
    """
    Simulates degraded telemetry by combining:
    - calibration drift
    - additive Gaussian noise
    - feature dropout
    - transient burst corruption
    """

    def __init__(self, params: FaultParams, rng: np.random.Generator) -> None:
        self.params = params
        self.rng = rng
        self.t = 0
        self.in_burst_until = -1

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply one step of degradation to a feature vector.

        The degradation sequence is:
        1. mild feature-wise calibration drift
        2. burst-state update
        3. additive noise
        4. feature dropout
        """
        x_faulted = x.copy()

        # 1) Calibration drift: only a subset of features drift at each step.
        drift_factor = 1.0 + (self.params.drift_scale * self.t)
        drift_mask = self.rng.random(x_faulted.shape[0]) < 0.20
        x_faulted[drift_mask] *= drift_factor

        # 2) Determine whether the current time step is inside a burst window.
        if self.t > self.in_burst_until and (self.rng.random() < self.params.burst_p):
            self.in_burst_until = self.t + self.params.burst_len
        in_burst = self.t <= self.in_burst_until

        # 3) Additive Gaussian noise
        current_std = self.params.noise_std
        if in_burst:
            current_std *= self.params.burst_noise_mult
        if current_std > 0:
            x_faulted += self.rng.normal(0.0, current_std, size=x_faulted.shape[0])

        # 4) Feature dropout
        current_dropout = self.params.dropout_p * (2.0 if in_burst else 1.0)
        current_dropout = min(0.99, current_dropout)
        if current_dropout > 0:
            x_faulted[self.rng.random(x_faulted.shape[0]) < current_dropout] = 0.0

        self.t += 1
        return x_faulted


# ============================================================================
# ONLINE MODELS
# ============================================================================

class DynamicMLP(nn.Module):
    """
    Lightweight MLP used as the main GGSA backbone and for the MLP baselines.

    Architecture:
        input -> 64 -> 32 -> 1
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x_tensor: torch.Tensor) -> float:
        with torch.no_grad():
            logits = self.forward(x_tensor)
            return torch.sigmoid(logits).item()

    def get_input_gradient(self, x_tensor: torch.Tensor, target_tensor: torch.Tensor) -> np.ndarray:
        """
        Compute gradient of the loss with respect to the input.
        This is used by GGSA and PGD.
        """
        x_grad = x_tensor.clone().detach().requires_grad_(True)
        logits = self.forward(x_grad)
        loss = self.loss_fn(logits, target_tensor).mean()
        loss.backward()
        return x_grad.grad.detach().cpu().numpy()

    def online_update(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor, weight: float = 1.0) -> float:
        """
        Perform one online update step.
        """
        self.optimizer.zero_grad()
        logits = self.forward(x_tensor)
        loss = (self.loss_fn(logits, y_tensor) * weight).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class DynamicLSTM(nn.Module):
    """
    Lightweight LSTM baseline for online sequence classification.

    Input shape:
        (batch, seq_len, input_dim)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)

    def predict_proba(self, x_tensor: torch.Tensor) -> float:
        with torch.no_grad():
            logits = self.forward(x_tensor)
            return torch.sigmoid(logits).item()

    def online_update(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor, weight: float = 1.0) -> float:
        self.optimizer.zero_grad()
        logits = self.forward(x_tensor)
        loss = (self.loss_fn(logits, y_tensor) * weight).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_detection_delay(y_true: np.ndarray, y_pred: np.ndarray, onset_idx: int) -> Optional[int]:
    """
    Return the number of time steps between the true attack onset and the first
    correct positive prediction. If no correct positive prediction is found,
    return None.
    """
    for t in range(onset_idx, len(y_true)):
        if y_true[t] == 1 and y_pred[t] == 1:
            return t - onset_idx
    return None


def get_topk_indices(grad: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of the K most sensitive features based on absolute gradient.
    """
    return np.argsort(np.abs(grad.ravel()))[-k:]


def build_model(model_type: str, input_dim: int) -> nn.Module:
    if model_type == "mlp":
        return DynamicMLP(input_dim)
    if model_type == "lstm":
        return DynamicLSTM(
            input_dim=input_dim,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def pad_sequence_buffer(
    buffer: Deque[np.ndarray],
    seq_len: int,
    input_dim: int,
) -> np.ndarray:
    """
    Left-pad a sequence buffer with zeros until it reaches seq_len.
    """
    items = list(buffer)
    if len(items) < seq_len:
        pad_count = seq_len - len(items)
        pads = [np.zeros(input_dim, dtype=np.float32) for _ in range(pad_count)]
        items = pads + items
    return np.stack(items[-seq_len:], axis=0)


def build_scaled_sequence_from_buffer(
    buffer_items: Deque[np.ndarray],
    scaler: StandardScaler,
    input_dim: int,
    seq_len: int,
) -> np.ndarray:
    """
    Build a padded sequence, then scale each time step using the same scaler.
    Output shape: (1, seq_len, input_dim)
    """
    seq_raw = pad_sequence_buffer(buffer_items, seq_len, input_dim)
    seq_scaled = np.vstack([
        scaler.transform(seq_raw[i].reshape(1, -1))
        for i in range(seq_len)
    ])
    return seq_scaled.reshape(1, seq_len, input_dim)


def make_lstm_tensor(
    buffer: Deque[np.ndarray],
    x_new: np.ndarray,
    scaler: StandardScaler,
    input_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Construct the LSTM input tensor from the recent raw stream buffer plus the
    current observation.
    """
    temp_buffer = deque(buffer, maxlen=SEQ_LEN)
    temp_buffer.append(x_new.copy())
    seq_scaled = build_scaled_sequence_from_buffer(
        temp_buffer,
        scaler,
        input_dim,
        SEQ_LEN,
    )
    return torch.tensor(seq_scaled, dtype=torch.float32, device=device)


def summarize_method(
    per_seed_results: Dict[str, Dict[str, Dict]],
    method_name: str,
) -> Dict[str, float]:
    """
    Aggregate summary statistics for one method across seeds.
    """
    delays = [
        per_seed_results[str(seed)][method_name]["delay"]
        for seed in SEEDS
        if per_seed_results[str(seed)][method_name]["delay"] is not None
    ]
    f1s = [per_seed_results[str(seed)][method_name]["f1"] for seed in SEEDS]
    flips = [per_seed_results[str(seed)][method_name]["flip"] for seed in SEEDS]
    infer_ms = [per_seed_results[str(seed)][method_name]["timing"]["avg_infer_ms"] for seed in SEEDS]
    update_ms = [per_seed_results[str(seed)][method_name]["timing"]["avg_update_ms"] for seed in SEEDS]
    repair_ms = [per_seed_results[str(seed)][method_name]["timing"]["avg_repair_ms_per_trigger"] for seed in SEEDS]
    repair_counts = [per_seed_results[str(seed)][method_name]["timing"]["repair_count"] for seed in SEEDS]

    if delays:
        delay_mean = float(np.mean(delays))
        delay_std = float(np.std(delays))
        delay_max = int(np.max(delays))
    else:
        delay_mean, delay_std, delay_max = -1.0, 0.0, -1

    return {
        "delay_mean": delay_mean,
        "delay_std": delay_std,
        "delay_max": delay_max,
        "f1_mean": float(np.mean(f1s)) if f1s else 0.0,
        "f1_std": float(np.std(f1s)) if f1s else 0.0,
        "flip_mean": float(np.mean(flips)) if flips else 0.0,
        "flip_std": float(np.std(flips)) if flips else 0.0,
        "avg_infer_ms_mean": float(np.mean(infer_ms)) if infer_ms else 0.0,
        "avg_update_ms_mean": float(np.mean(update_ms)) if update_ms else 0.0,
        "avg_repair_ms_per_trigger_mean": float(np.mean(repair_ms)) if repair_ms else 0.0,
        "repair_count_mean": float(np.mean(repair_counts)) if repair_counts else 0.0,
    }


# ============================================================================
# CORE EXPERIMENT LOOP
# ============================================================================

def run_single_seed(
    X: np.ndarray,
    y: np.ndarray,
    onset_idx: int,
    seed: int,
    fault_params: FaultParams,
) -> Tuple[int, Dict[str, Dict]]:
    """
    Run the full CIC experiment for one seed and one degradation regime.

    Returns
    -------
    seed : int
        The seed used.
    results : dict
        Method-level metrics and timing information.
    """
    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    results: Dict[str, Dict] = {}

    for method_name, model_type in METHODS:
        rng = np.random.default_rng(seed)
        injector = TelemetryDegradation(fault_params, rng)
        scaler = StandardScaler()

        input_dim = X.shape[1]
        model = build_model(model_type, input_dim).to(device)

        infer_time = 0.0
        update_time = 0.0
        repair_time = 0.0
        repair_count = 0

        # ------------------------------------------------------------------
        # Phase 1: warmup / initial online fit
        # ------------------------------------------------------------------
        warm_end = min(WARMUP_STEPS, len(X))
        warm_faulted = [injector.apply(X[i]) for i in range(warm_end)]
        scaler.fit(warm_faulted)

        if model_type == "mlp":
            warm_x_tensor = torch.tensor(
                scaler.transform(warm_faulted),
                dtype=torch.float32,
                device=device,
            )
        else:
            warm_sequences = []
            warm_buffer: Deque[np.ndarray] = deque(maxlen=SEQ_LEN)
            for i in range(warm_end):
                warm_buffer.append(warm_faulted[i].copy())
                seq_scaled = build_scaled_sequence_from_buffer(
                    warm_buffer,
                    scaler,
                    input_dim,
                    SEQ_LEN,
                )
                warm_sequences.append(seq_scaled[0])

            warm_x_tensor = torch.tensor(
                np.stack(warm_sequences, axis=0),
                dtype=torch.float32,
                device=device,
            )

        warm_y_tensor = torch.tensor(
            y[:warm_end].reshape(-1, 1),
            dtype=torch.float32,
            device=device,
        )

        # A few passes are used to initialize the model on the warmup prefix.
        for _ in range(5):
            model.online_update(warm_x_tensor, warm_y_tensor)

        # ------------------------------------------------------------------
        # Phase 2: online streaming evaluation
        # ------------------------------------------------------------------
        y_pred = np.zeros_like(y)
        stream_buffer: Deque[np.ndarray] = deque(maxlen=SEQ_LEN)

        for i in range(warm_end):
            stream_buffer.append(warm_faulted[i].copy())

        for t in range(warm_end, len(X)):
            x_raw = injector.apply(X[t])
            y_target = torch.tensor([[y[t]]], dtype=torch.float32, device=device)

            # Build model input for the current observation.
            if model_type == "mlp":
                x_scaled = scaler.transform(x_raw.reshape(1, -1))
                x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)
            else:
                x_tensor = make_lstm_tensor(
                    stream_buffer,
                    x_raw,
                    scaler,
                    input_dim,
                    device,
                )

            # 1) Inference timing
            t0 = time.perf_counter()
            proba = model.predict_proba(x_tensor)
            infer_time += time.perf_counter() - t0

            # 2) Prediction rule
            threshold = 0.5
            if method_name in ["GGSA", "RSA", "DTR"]:
                if PROBA_TRIGGER_LOW < proba < PROBA_TRIGGER_HIGH:
                    threshold = 0.35

            y_pred[t] = 1 if proba >= threshold else 0

            # 3) Standard online update
            t1 = time.perf_counter()
            model.online_update(x_tensor, y_target)
            update_time += time.perf_counter() - t1

            # 4) Additional robustness / repair logic
            is_missed_attack = (y[t] == 1) and (proba < UNCERTAINTY_THRESHOLD)
            dynamic_step = max(0.5, fault_params.noise_std * 5.0)

            # PGD baseline: passive adversarial hardening (MLP only)
            if method_name == "PGD" and model_type == "mlp":
                x_grad = model.get_input_gradient(x_tensor, y_target)
                perturbation = np.sign(x_grad) * PGD_EPSILON
                adv_tensor = torch.tensor(
                    x_tensor.detach().cpu().numpy() + perturbation,
                    dtype=torch.float32,
                    device=device,
                )

                t2 = time.perf_counter()
                model.online_update(adv_tensor, y_target)
                update_time += time.perf_counter() - t2

            # GGSA: gradient-guided repair (MLP only in this script)
            elif method_name == "GGSA" and model_type == "mlp" and is_missed_attack:
                repair_start = time.perf_counter()

                target_attack = torch.tensor([[1.0]], dtype=torch.float32, device=device)
                grad = model.get_input_gradient(x_tensor, target_attack)
                important_features = get_topk_indices(grad, TOPK_FEATURES)

                for _ in range(HARDENING_STEPS):
                    synthetic_raw = x_raw.copy()

                    # Add local stochastic perturbation on important features.
                    synthetic_raw[important_features] += rng.normal(
                        0,
                        fault_params.noise_std * 2.0,
                        size=len(important_features),
                    )

                    synthetic_scaled = scaler.transform(synthetic_raw.reshape(1, -1))

                    # Add structured gradient-guided shift in the same feature subspace.
                    synthetic_scaled[0, important_features] += (
                        np.sign(grad[0, important_features]) * dynamic_step
                    )

                    synthetic_tensor = torch.tensor(
                        synthetic_scaled,
                        dtype=torch.float32,
                        device=device,
                    )
                    model.online_update(
                        synthetic_tensor,
                        target_attack,
                        weight=FN_BIAS_WEIGHT,
                    )

                repair_time += time.perf_counter() - repair_start
                repair_count += 1

            # RSA: stochastic stress repair baseline (MLP only)
            elif method_name == "RSA" and model_type == "mlp" and is_missed_attack:
                repair_start = time.perf_counter()

                target_attack = torch.tensor([[1.0]], dtype=torch.float32, device=device)
                for _ in range(HARDENING_STEPS):
                    synthetic_raw = x_raw.copy()
                    random_idx = rng.choice(input_dim, TOPK_FEATURES, replace=False)
                    synthetic_raw[random_idx] += rng.normal(
                        0,
                        dynamic_step,
                        size=len(random_idx),
                    )

                    synthetic_scaled = scaler.transform(synthetic_raw.reshape(1, -1))
                    synthetic_tensor = torch.tensor(
                        synthetic_scaled,
                        dtype=torch.float32,
                        device=device,
                    )
                    model.online_update(
                        synthetic_tensor,
                        target_attack,
                        weight=FN_BIAS_WEIGHT,
                    )

                repair_time += time.perf_counter() - repair_start
                repair_count += 1

            stream_buffer.append(x_raw.copy())

        # ------------------------------------------------------------------
        # Metric calculation
        # ------------------------------------------------------------------
        delay = compute_detection_delay(y, y_pred, onset_idx)

        eval_y = y[PREDICT_START_INDEX:]
        eval_pred = y_pred[PREDICT_START_INDEX:]
        flip_rate = (
            np.mean(np.abs(eval_pred[1:] - eval_pred[:-1]))
            if len(eval_pred) > 1
            else 0.0
        )

        results[method_name] = {
            "delay": delay,
            "f1": float(f1_score(eval_y, eval_pred, zero_division=0)),
            "flip": float(flip_rate),
            "timing": {
                "infer_total_sec": float(infer_time),
                "update_total_sec": float(update_time),
                "repair_total_sec": float(repair_time),
                "repair_count": int(repair_count),
                "stream_steps": int(max(0, len(X) - warm_end)),
                "avg_infer_ms": float((infer_time / max(1, len(X) - warm_end)) * 1000.0),
                "avg_update_ms": float((update_time / max(1, len(X) - warm_end)) * 1000.0),
                "avg_repair_ms_per_trigger": (
                    float((repair_time / max(1, repair_count)) * 1000.0)
                    if repair_count > 0
                    else 0.0
                ),
            },
        }

    return seed, results


# ============================================================================
# DATA LOADING / PREPARATION
# ============================================================================

def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the CIC CSV file and return numeric features X and binary labels y.

    Labels are mapped as:
        benign -> 0
        any attack label -> 1
    """
    if not os.path.exists(csv_path):
        print(f"\n[!] Error: Could not find '{csv_path}' in the current directory.")
        print(f"    Current directory: {os.getcwd()}")
        print("    Please place the CSV file next to this script or update DATA_FILENAME.")
        if sys.stdout.isatty():
            input("\nPress Enter to exit...")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        print(f"[!] Error while reading CSV: {exc}")
        sys.exit(1)

    df.columns = [c.strip() for c in df.columns]

    possible_label_cols = [c for c in ["Label", "label", "Attack"] if c in df.columns]
    if not possible_label_cols:
        print("[!] Could not locate a label column. Expected one of: Label, label, Attack")
        sys.exit(1)

    label_col = possible_label_cols[0]
    y_raw = df[label_col].astype(str).values
    y = np.array([0 if v in ["BENIGN", "Benign", "0"] else 1 for v in y_raw], dtype=int)

    X = (
        df.select_dtypes(include=[np.number])
        .fillna(0)
        .clip(-CLIP_VALUE, CLIP_VALUE)
        .values
    )

    return X, y


def find_attack_onset(y: np.ndarray) -> int:
    """
    Find a representative attack onset index after warmup.

    The current strategy:
    - identify all contiguous attack segments
    - keep only those that start after the warmup phase
    - choose the longest such segment
    """
    segments: List[Tuple[int, int]] = []
    in_segment = False
    start = 0

    for i, value in enumerate(y):
        if value == 1 and not in_segment:
            in_segment = True
            start = i
        elif value == 0 and in_segment:
            segments.append((start, i - 1))
            in_segment = False

    if in_segment:
        segments.append((start, len(y) - 1))

    candidates = [seg for seg in segments if seg[0] > WARMUP_STEPS]
    if not candidates:
        return 0

    longest_segment = max(candidates, key=lambda seg: seg[1] - seg[0])
    return longest_segment[0]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    print("\n" + "=" * 78)
    print(" GGSA REPRODUCTION PROTOCOL FOR CIC-IDS2017 (WITH LSTM BASELINE)")
    print("=" * 78)
    print(f"Target data file: {DATA_FILENAME}")
    print(f"Parallel jobs: {N_JOBS}")
    print(f"LSTM sequence length: {SEQ_LEN}")
    print(f"LSTM hidden dimension: {LSTM_HIDDEN_DIM}")

    # 1) Load data
    X, y = load_dataset(DATA_FILENAME)

    # 2) Determine attack onset for delay calculation
    onset_idx = find_attack_onset(y)

    # 3) Trim the stream to keep the experiment lightweight
    end_idx = min(len(y), onset_idx + 2500)
    X = X[:end_idx]
    y = y[:end_idx]

    print(f"Loaded stream length: {len(y)}")
    print(f"Attack onset index used for delay calculation: {onset_idx}")

    final_results: Dict[str, Dict] = {}

    # 4) Run each severity regime
    for severity_name, severity_params in SEVERITIES.items():
        print(
            f"\nRunning severity regime {severity_name} "
            f"(noise={severity_params.noise_std}, dropout={severity_params.dropout_p})..."
        )

        seed_results = Parallel(n_jobs=N_JOBS)(
            delayed(run_single_seed)(X, y, onset_idx, seed, severity_params)
            for seed in SEEDS
        )

        per_seed_data = {str(seed): res for seed, res in seed_results}

        summary = {
            method_name: summarize_method(per_seed_data, method_name)
            for method_name, _ in METHODS
        }

        final_results[severity_name] = {
            "raw_data": per_seed_data,
            "summary": summary,
        }

        # Console summary table
        print("-" * 118)
        print(
            f"{'Method':<10} | {'Delay (Mean / Max)':<20} | {'F1':<8} | "
            f"{'Flip':<8} | {'Infer ms':<10} | {'Update ms':<10} | {'Repair ms':<10}"
        )
        print("-" * 118)

        for method_name, _ in METHODS:
            stats = summary[method_name]
            delay_str = f"{stats['delay_mean']:.2f} / {stats['delay_max']}"
            print(
                f"{method_name:<10} | "
                f"{delay_str:<20} | "
                f"{stats['f1_mean']:.3f}    | "
                f"{stats['flip_mean']:.4f}  | "
                f"{stats['avg_infer_ms_mean']:.4f}   | "
                f"{stats['avg_update_ms_mean']:.4f}   | "
                f"{stats['avg_repair_ms_per_trigger_mean']:.4f}"
            )

        print("-" * 118)

    # 5) Save output
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILENAME}")

    if sys.stdout.isatty():
        input("Press Enter to close...")


if __name__ == "__main__":
    main()
