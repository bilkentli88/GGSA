"""
GGSA Multi-Attack Protocol on ToN-IoT
=====================================

Cleaned reproduction script for the ToN-IoT experiments.

Purpose
-------
- Use ToN-IoT as a second real dataset with multiple attack categories.
- Apply the degraded-telemetry protocol across attack types.
- Report per-attack, per-severity, and aggregated summaries.

Notes
-----
- This script evaluates whether the delay-oriented behavior of GGSA extends
  beyond a single attack type.
- The MLP is used for Standard / PGD / DTR / RSA / GGSA-MLP.
- The LSTM is included as a baseline comparator under the same streaming protocol.

Expected input
--------------
Place `train_test_network.csv` in the same folder as this script,
or update DATA_FILENAME below.

Outputs
-------
A JSON results file containing:
- per-seed raw results
- per-attack summaries
- aggregate summaries across attacks
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
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================================
# USER CONFIGURATION
# ============================================================================

DATA_FILENAME = "train_test_network.csv"
OUTPUT_FILENAME = "results_ggsa_toniot_multi_attack_runtime.json"

# Primary save location. If this path is not writable, the script falls back
# to LOCAL_FALLBACK_DIR.
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
LOCAL_FALLBACK_DIR = "."

# Parallel jobs across seeds
N_JOBS = 2


# ============================================================================
# GLOBAL EXPERIMENT SETTINGS
# ============================================================================

SEEDS: List[int] = [
    88, 109, 253, 371, 458, 555, 666, 793, 907, 1009,
    1103, 1201, 1301, 1409, 1511, 1601, 1789, 1877, 1971, 2025
]

# Automatic attack-selection settings
EXCLUDED_LABELS = {"normal", "benign", "background", "none"}
ATTACKS_TO_RUN = None         # Example: ["injection", "password"]; None = auto-select
MAX_ATTACKS = 5
MIN_TOTAL_POSITIVES = 200
MIN_SEGMENT_LEN = 20
MIN_BENIGN_PRE = 1000
ONSET_OFFSET = 3000           # Similar spirit to warmup + pre-history guard

# GGSA / adaptation hyperparameters
TOPK_FEATURES = 20
HARDENING_STEPS = 5
PROBA_TRIGGER_LOW = 0.35
PROBA_TRIGGER_HIGH = 0.65
UNCERTAINTY_THRESHOLD = 0.60
FN_BIAS_WEIGHT = 5.0

# Streaming settings
WARMUP_STEPS = 1000
WINDOW_PRE = 2000
WINDOW_POST = 6000

# LSTM baseline settings
SEQ_LEN = 8
LSTM_HIDDEN_DIM = 32

# DTR settings
DTR_BUFFER_SIZE = 128
DTR_RETRAIN_EPOCHS = 2
DTR_TRIGGER_LOW = 0.40
DTR_TRIGGER_HIGH = 0.60
DTR_MIN_BUFFER_FOR_RETRAIN = 32


# ============================================================================
# MODELS
# ============================================================================

class DynamicMLP(nn.Module):
    """
    Lightweight MLP backbone used by:
    - Standard
    - PGD
    - DTR
    - RSA
    - GGSA-MLP
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

    def predict_proba(self, x_t: torch.Tensor) -> float:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x_t)).item()

    def get_input_gradient(self, x_t: torch.Tensor, target_t: torch.Tensor) -> np.ndarray:
        x_g = x_t.clone().detach().requires_grad_(True)
        loss = self.loss_fn(self.forward(x_g), target_t).mean()
        loss.backward()
        return x_g.grad.detach().cpu().numpy()

    def online_update(self, x_t: torch.Tensor, y_t: torch.Tensor, weight: float = 1.0) -> None:
        self.optimizer.zero_grad()
        loss = (self.loss_fn(self.forward(x_t), y_t) * weight).mean()
        loss.backward()
        self.optimizer.step()


class DynamicLSTM(nn.Module):
    """
    Lightweight LSTM baseline for online sequence classification.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, LSTM_HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(LSTM_HIDDEN_DIM, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def predict_proba(self, x_t: torch.Tensor) -> float:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x_t)).item()

    def get_input_gradient(self, x_t: torch.Tensor, target_t: torch.Tensor) -> np.ndarray:
        x_g = x_t.clone().detach().requires_grad_(True)
        loss = self.loss_fn(self.forward(x_g), target_t).mean()
        loss.backward()
        return x_g.grad.detach().cpu().numpy()

    def online_update(self, x_t: torch.Tensor, y_t: torch.Tensor, weight: float = 1.0) -> None:
        self.optimizer.zero_grad()
        loss = (self.loss_fn(self.forward(x_t), y_t) * weight).mean()
        loss.backward()
        self.optimizer.step()


# ============================================================================
# FAULT MODEL AND HELPERS
# ============================================================================

@dataclass
class FaultParams:
    noise_std: float
    dropout_p: float
    drift_scale: float
    burst_p: float
    burst_len: int
    burst_noise_mult: float


SEVERITIES: Dict[str, FaultParams] = {
    "S1": FaultParams(0.01, 0.02, 0.002, 0.002, 10, 2.0),
    "S2": FaultParams(0.05, 0.10, 0.015, 0.01, 20, 3.0),
    "S3": FaultParams(0.15, 0.30, 0.05, 0.02, 30, 6.0),
}


class TelemetryDegradation:
    """
    Simulates degraded telemetry via:
    - gradual drift
    - additive Gaussian noise
    - feature dropout
    - transient bursts
    """

    def __init__(self, params: FaultParams, rng: np.random.Generator) -> None:
        self.params = params
        self.rng = rng
        self.t = 0
        self.in_burst_until = -1
        self.drift: Optional[np.ndarray] = None

    def apply(self, x: np.ndarray) -> np.ndarray:
        x_faulted = x.copy().astype(np.float32)

        if self.drift is None:
            self.drift = np.zeros_like(x_faulted)

        # Decide whether a burst starts now.
        if self.t > self.in_burst_until and (self.rng.random() < self.params.burst_p):
            self.in_burst_until = self.t + self.params.burst_len
        in_burst = self.t <= self.in_burst_until

        # Gradual drift
        self.drift += self.rng.normal(
            0.0,
            self.params.drift_scale,
            size=x_faulted.shape[0],
        ).astype(np.float32)
        x_faulted += self.drift

        # Additive noise
        current_std = self.params.noise_std
        if in_burst:
            current_std *= self.params.burst_noise_mult
        x_faulted += self.rng.normal(
            0.0,
            current_std,
            size=x_faulted.shape[0],
        ).astype(np.float32)

        # Dropout corruption
        current_dropout = self.params.dropout_p * (2.0 if in_burst else 1.0)
        mask = self.rng.random(x_faulted.shape[0]) < current_dropout
        x_faulted[mask] = 0.0

        self.t += 1
        return x_faulted


def make_lstm_tensor(
    buffer: Deque[np.ndarray],
    x_new: np.ndarray,
    scaler: StandardScaler,
    input_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Construct the LSTM input tensor from the recent raw stream buffer plus the
    current observation. Left-padding with zeros is used when needed.
    """
    items = list(buffer)
    items.append(x_new.copy())
    if len(items) < SEQ_LEN:
        items = [np.zeros(input_dim, dtype=np.float32)] * (SEQ_LEN - len(items)) + items
    seq_scaled = scaler.transform(np.stack(items[-SEQ_LEN:], axis=0))
    return torch.tensor(
        seq_scaled.reshape(1, SEQ_LEN, input_dim),
        dtype=torch.float32,
        device=device,
    )


def extract_attack_segments(y: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract contiguous positive segments from a binary label vector.
    """
    segments: List[Tuple[int, int]] = []
    start = None

    for i, value in enumerate(y):
        if value == 1 and start is None:
            start = i
        elif value == 0 and start is not None:
            segments.append((start, i - 1))
            start = None

    if start is not None:
        segments.append((start, len(y) - 1))

    return segments


def choose_attack_onset(y: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    """
    Choose a usable onset segment subject to:
    - enough benign pre-history
    - enough total offset into the stream
    - minimum contiguous attack length

    The longest valid segment is chosen. If no segment is valid, return (None, None).
    """
    segments = extract_attack_segments(y)
    lengths = [end - start + 1 for start, end in segments]

    print(f"[diag] contiguous positive segments: {len(segments)}")
    if lengths:
        print(f"[diag] segment length stats -> min={min(lengths)}, median={int(np.median(lengths))}, max={max(lengths)}")
        print(f"[diag] top 10 segment lengths: {sorted(lengths, reverse=True)[:10]}")
    else:
        print("[diag] no positive segments found")

    candidates = []
    for start, end in segments:
        length = end - start + 1
        if start < ONSET_OFFSET:
            continue
        if start < MIN_BENIGN_PRE:
            continue
        if length < MIN_SEGMENT_LEN:
            continue
        candidates.append((length, start, end))

    print(f"[diag] valid candidates after rules: {len(candidates)}")
    if not candidates:
        return None, None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    length, start, end = candidates[0]
    print(f"[diag] chosen segment -> start={start}, end={end}, len={length}")
    return start, end


def safe_mean(xs: List[Optional[float]]) -> Optional[float]:
    values = [x for x in xs if x is not None]
    return None if not values else float(np.mean(values))


def safe_std(xs: List[Optional[float]]) -> Optional[float]:
    values = [x for x in xs if x is not None]
    if not values:
        return None
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


# ============================================================================
# CORE ENGINE
# ============================================================================

def run_single_seed(
    X: np.ndarray,
    y: np.ndarray,
    onset_idx: int,
    seed: int,
    fault_params: FaultParams,
) -> Dict[str, Dict]:
    """
    Run one seed for one attack type under one severity regime.
    """
    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    methods: List[Tuple[str, str]] = [
        ("Standard", "mlp"),
        ("PGD", "mlp"),
        ("DTR", "mlp"),
        ("RSA", "mlp"),
        ("GGSA-MLP", "mlp"),
        ("LSTM", "lstm"),
    ]

    results: Dict[str, Dict] = {}

    for method_name, model_type in methods:
        rng = np.random.default_rng(seed)
        injector = TelemetryDegradation(fault_params, rng)
        scaler = StandardScaler()

        input_dim = X.shape[1]
        actual_k = min(TOPK_FEATURES, input_dim)
        model: nn.Module = DynamicMLP(input_dim).to(device) if model_type == "mlp" else DynamicLSTM(input_dim).to(device)

        # ------------------------------------------------------------------
        # Phase 1: warmup
        # ------------------------------------------------------------------
        warm_end = min(WARMUP_STEPS, len(X))
        warm_samples = [injector.apply(X[i]) for i in range(warm_end)]
        scaler.fit(warm_samples)

        y_pred: List[int] = []
        y_true_sync: List[int] = []
        stream_buffer: Deque[np.ndarray] = deque(maxlen=SEQ_LEN)

        for i in range(warm_end):
            stream_buffer.append(warm_samples[i])

        # DTR / RSA replay buffer (MLP only)
        replay_x: Deque[np.ndarray] = deque(maxlen=DTR_BUFFER_SIZE)
        replay_y: Deque[np.ndarray] = deque(maxlen=DTR_BUFFER_SIZE)

        prev_hat = 0
        flip_count = 0

        runtime = {
            "infer_total": 0.0,
            "infer_steps": 0,
            "update_total": 0.0,
            "update_steps": 0,
            "repair_total": 0.0,
            "repair_triggers": 0,
        }

        # ------------------------------------------------------------------
        # Phase 2: online stream processing
        # ------------------------------------------------------------------
        for t in range(warm_end, len(X)):
            x_raw = injector.apply(X[t])
            y_tensor = torch.tensor([[float(y[t])]], dtype=torch.float32, device=device)

            infer_start = time.perf_counter()

            if model_type == "mlp":
                x_scaled = scaler.transform(x_raw.reshape(1, -1))
                x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)
            else:
                x_tensor = make_lstm_tensor(stream_buffer, x_raw, scaler, input_dim, device)

            y_true_sync.append(int(y[t]))
            proba = model.predict_proba(x_tensor)

            threshold = 0.35 if (PROBA_TRIGGER_LOW < proba < PROBA_TRIGGER_HIGH) and ("GGSA" in method_name) else 0.5
            y_hat = 1 if proba >= threshold else 0

            infer_end = time.perf_counter()
            runtime["infer_total"] += (infer_end - infer_start)
            runtime["infer_steps"] += 1

            y_pred.append(y_hat)
            if len(y_pred) > 1 and y_hat != prev_hat:
                flip_count += 1
            prev_hat = y_hat

            update_start = time.perf_counter()
            model.online_update(x_tensor, y_tensor)
            update_end = time.perf_counter()
            runtime["update_total"] += (update_end - update_start)
            runtime["update_steps"] += 1

            # Maintain replay memory for MLP-based methods that use it.
            if model_type == "mlp":
                replay_x.append(x_tensor.detach().cpu().numpy())
                replay_y.append(np.array([[float(y[t])]], dtype=np.float32))

            # --------------------------------------------------------------
            # Additional method-specific logic
            # --------------------------------------------------------------

            # PGD: passive adversarial hardening baseline
            if method_name == "PGD" and y[t] == 1 and proba < UNCERTAINTY_THRESHOLD and model_type == "mlp":
                repair_start = time.perf_counter()
                grad = model.get_input_gradient(
                    x_tensor,
                    torch.tensor([[1.0]], dtype=torch.float32, device=device),
                )[0]
                important_features = np.argsort(np.abs(grad))[-actual_k:]

                synthetic_scaled = x_tensor.detach().cpu().numpy().copy()
                synthetic_scaled[0, important_features] += (
                    np.sign(grad[important_features]) * max(0.5, fault_params.noise_std * 5.0)
                )

                synthetic_tensor = torch.tensor(
                    synthetic_scaled,
                    dtype=torch.float32,
                    device=device,
                )
                model.online_update(
                    synthetic_tensor,
                    torch.tensor([[1.0]], dtype=torch.float32, device=device),
                    weight=FN_BIAS_WEIGHT,
                )
                repair_end = time.perf_counter()
                runtime["repair_total"] += (repair_end - repair_start)
                runtime["repair_triggers"] += 1

            # DTR: threshold-regulation baseline with buffered retraining
            if method_name == "DTR" and model_type == "mlp":
                trigger = ((DTR_TRIGGER_LOW < proba < DTR_TRIGGER_HIGH) or (y[t] == 1 and y_hat == 0))
                if trigger and len(replay_x) >= DTR_MIN_BUFFER_FOR_RETRAIN:
                    repair_start = time.perf_counter()
                    bx = torch.tensor(np.concatenate(list(replay_x), axis=0), dtype=torch.float32, device=device)
                    by = torch.tensor(np.concatenate(list(replay_y), axis=0), dtype=torch.float32, device=device)
                    for _ in range(DTR_RETRAIN_EPOCHS):
                        model.online_update(bx, by, weight=1.5)
                    repair_end = time.perf_counter()
                    runtime["repair_total"] += (repair_end - repair_start)
                    runtime["repair_triggers"] += 1

            # RSA: stochastic repair baseline
            if method_name == "RSA" and model_type == "mlp":
                trigger = (y[t] == 1 and proba < UNCERTAINTY_THRESHOLD)
                if trigger and len(replay_x) >= DTR_MIN_BUFFER_FOR_RETRAIN:
                    positive_indices = [i for i, yy in enumerate(replay_y) if yy[0, 0] == 1.0]
                    if positive_indices:
                        repair_start = time.perf_counter()
                        selected = positive_indices[-min(16, len(positive_indices)):]
                        bx = torch.tensor(
                            np.concatenate([replay_x[i] for i in selected], axis=0),
                            dtype=torch.float32,
                            device=device,
                        )
                        by = torch.tensor(
                            np.concatenate([replay_y[i] for i in selected], axis=0),
                            dtype=torch.float32,
                            device=device,
                        )
                        for _ in range(2):
                            model.online_update(bx, by, weight=FN_BIAS_WEIGHT)
                        repair_end = time.perf_counter()
                        runtime["repair_total"] += (repair_end - repair_start)
                        runtime["repair_triggers"] += 1

            # GGSA-MLP: gradient-guided repair on the MLP backbone
            if "GGSA" in method_name and (y[t] == 1 and proba < UNCERTAINTY_THRESHOLD):
                repair_start = time.perf_counter()

                grad_full = model.get_input_gradient(
                    x_tensor,
                    torch.tensor([[1.0]], dtype=torch.float32, device=device),
                )

                # For LSTM-shaped gradients, use the last time step.
                grad = grad_full[0, -1, :] if model_type == "lstm" else grad_full[0]
                important_features = np.argsort(np.abs(grad))[-actual_k:]
                unit_grad = grad[important_features] / (np.linalg.norm(grad[important_features]) + 1e-8)
                step = max(0.5, fault_params.noise_std * 5.0)

                for _ in range(HARDENING_STEPS):
                    synthetic_raw = x_raw.copy()
                    synthetic_raw[important_features] += rng.normal(
                        0,
                        fault_params.noise_std,
                        size=len(important_features),
                    )

                    if model_type == "mlp":
                        synthetic_scaled = scaler.transform(synthetic_raw.reshape(1, -1))
                        synthetic_scaled[0, important_features] += unit_grad * step
                        synthetic_tensor = torch.tensor(
                            synthetic_scaled,
                            dtype=torch.float32,
                            device=device,
                        )
                    else:
                        synthetic_tensor = make_lstm_tensor(
                            stream_buffer,
                            synthetic_raw,
                            scaler,
                            input_dim,
                            device,
                        )
                        synthetic_tensor[0, -1, important_features] += torch.tensor(
                            unit_grad * step,
                            dtype=torch.float32,
                            device=device,
                        )

                    model.online_update(
                        synthetic_tensor,
                        torch.tensor([[1.0]], dtype=torch.float32, device=device),
                        weight=FN_BIAS_WEIGHT,
                    )

                repair_end = time.perf_counter()
                runtime["repair_total"] += (repair_end - repair_start)
                runtime["repair_triggers"] += 1

            stream_buffer.append(x_raw.copy())

        # ------------------------------------------------------------------
        # Metric calculation
        # ------------------------------------------------------------------
        delay = None
        start_idx = onset_idx - warm_end
        for idx in range(max(0, start_idx), len(y_pred)):
            if y_true_sync[idx] == 1 and y_pred[idx] == 1:
                delay = idx - start_idx
                break

        flip_rate = flip_count / max(1, len(y_pred) - 1)
        infer_ms = 1000.0 * runtime["infer_total"] / max(runtime["infer_steps"], 1)
        update_ms = 1000.0 * runtime["update_total"] / max(runtime["update_steps"], 1)
        repair_ms = (
            1000.0 * runtime["repair_total"] / max(runtime["repair_triggers"], 1)
            if runtime["repair_triggers"] > 0
            else 0.0
        )

        results[method_name] = {
            "delay": None if delay is None else int(delay),
            "f1": float(f1_score(y_true_sync, y_pred, zero_division=0)),
            "flip": float(flip_rate),
            "infer_ms": float(infer_ms),
            "update_ms": float(update_ms),
            "repair_ms": float(repair_ms),
            "triggers": int(runtime["repair_triggers"]),
        }

    return results


# ============================================================================
# DATA PREPARATION
# ============================================================================

def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare the ToN-IoT dataframe and return:
    - normalized dataframe
    - numeric feature matrix
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "type" not in df.columns:
        raise ValueError("Required column 'type' not found in train_test_network.csv")

    # Normalize attack labels
    df["type"] = df["type"].astype(str).str.strip().str.lower()

    # Encode selected categorical columns if they exist
    for col in ["proto", "service", "conn_state", "dns_query"]:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Remove potential leaky or identifier columns
    leaky_cols = [
        "ts", "src_ip", "dst_ip", "src_port", "dst_port",
        "label", "Label", "type"
    ]

    X = (
        df.select_dtypes(include=[np.number])
        .drop(columns=[c for c in leaky_cols if c in df.columns], errors="ignore")
        .fillna(0)
    )

    return df, X.values.astype(np.float32)


def choose_attack_types(df: pd.DataFrame) -> List[str]:
    """
    Select attack types to evaluate.

    If ATTACKS_TO_RUN is provided, use it directly (after filtering to existing labels).
    Otherwise:
    - exclude benign labels
    - require minimum support
    - keep at most MAX_ATTACKS by support
    """
    counts = df["type"].value_counts()
    print("[diag] type value counts:")
    print(counts.head(30))

    if ATTACKS_TO_RUN is not None:
        attacks = [a.strip().lower() for a in ATTACKS_TO_RUN]
        return [a for a in attacks if a in counts.index]

    candidates = []
    for attack, count in counts.items():
        if attack in EXCLUDED_LABELS:
            continue
        if int(count) < MIN_TOTAL_POSITIVES:
            continue
        candidates.append((attack, int(count)))

    candidates = sorted(candidates, key=lambda item: (-item[1], item[0]))[:MAX_ATTACKS]
    return [attack for attack, _ in candidates]


def summarize_results_list(results_list: List[Dict[str, Dict]]) -> Dict[str, Dict]:
    """
    Summarize per-seed results for one attack type under one severity regime.
    """
    methods = list(results_list[0].keys())
    summary: Dict[str, Dict] = {}

    for method_name in methods:
        delays = [r[method_name]["delay"] for r in results_list]
        f1s = [r[method_name]["f1"] for r in results_list]
        flips = [r[method_name]["flip"] for r in results_list]
        infer_vals = [r[method_name]["infer_ms"] for r in results_list]
        update_vals = [r[method_name]["update_ms"] for r in results_list]
        repair_vals = [r[method_name]["repair_ms"] for r in results_list]
        trigger_vals = [r[method_name]["triggers"] for r in results_list]

        summary[method_name] = {
            "delay_mean": safe_mean(delays),
            "delay_std": safe_std(delays),
            "f1_mean": safe_mean(f1s),
            "f1_std": safe_std(f1s),
            "flip_mean": safe_mean(flips),
            "flip_std": safe_std(flips),
            "infer_ms_mean": safe_mean(infer_vals),
            "infer_ms_std": safe_std(infer_vals),
            "update_ms_mean": safe_mean(update_vals),
            "update_ms_std": safe_std(update_vals),
            "repair_ms_mean": safe_mean(repair_vals),
            "repair_ms_std": safe_std(repair_vals),
            "triggers_mean": safe_mean(trigger_vals),
            "triggers_std": safe_std(trigger_vals),
        }

    return summary


def aggregate_across_attacks(all_results: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Aggregate attack-level summaries across all successfully evaluated attacks.
    """
    aggregate: Dict[str, Dict] = {}
    attacks = [attack for attack in all_results.keys() if attack not in {"_meta", "_aggregate"}]
    if not attacks:
        return aggregate

    severities = list(SEVERITIES.keys())
    methods = list(next(iter(all_results[attacks[0]].values()))["summary"].keys())

    for severity_name in severities:
        aggregate[severity_name] = {}
        for method_name in methods:
            delay_vals, f1_vals, flip_vals = [], [], []
            infer_vals, update_vals, repair_vals, trigger_vals = [], [], [], []

            for attack in attacks:
                stats = all_results[attack][severity_name]["summary"][method_name]

                if stats["delay_mean"] is not None:
                    delay_vals.append(stats["delay_mean"])
                if stats["f1_mean"] is not None:
                    f1_vals.append(stats["f1_mean"])
                if stats["flip_mean"] is not None:
                    flip_vals.append(stats["flip_mean"])
                if stats["infer_ms_mean"] is not None:
                    infer_vals.append(stats["infer_ms_mean"])
                if stats["update_ms_mean"] is not None:
                    update_vals.append(stats["update_ms_mean"])
                if stats["repair_ms_mean"] is not None:
                    repair_vals.append(stats["repair_ms_mean"])
                if stats["triggers_mean"] is not None:
                    trigger_vals.append(stats["triggers_mean"])

            aggregate[severity_name][method_name] = {
                "attack_avg_delay_mean": safe_mean(delay_vals),
                "attack_avg_delay_std": safe_std(delay_vals),
                "attack_avg_f1_mean": safe_mean(f1_vals),
                "attack_avg_f1_std": safe_std(f1_vals),
                "attack_avg_flip_mean": safe_mean(flip_vals),
                "attack_avg_flip_std": safe_std(flip_vals),
                "attack_avg_infer_ms_mean": safe_mean(infer_vals),
                "attack_avg_infer_ms_std": safe_std(infer_vals),
                "attack_avg_update_ms_mean": safe_mean(update_vals),
                "attack_avg_update_ms_std": safe_std(update_vals),
                "attack_avg_repair_ms_mean": safe_mean(repair_vals),
                "attack_avg_repair_ms_std": safe_std(repair_vals),
                "attack_avg_triggers_mean": safe_mean(trigger_vals),
                "attack_avg_triggers_std": safe_std(trigger_vals),
            }

    return aggregate


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    if not os.path.exists(DATA_FILENAME):
        print(f"Error: {DATA_FILENAME} not found.")
        sys.exit(1)

    print("=" * 90)
    print(" GGSA MULTI-ATTACK PROTOCOL ON ToN-IoT")
    print("=" * 90)
    print(f"Target data file: {DATA_FILENAME}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"Window: pre={WINDOW_PRE}, post={WINDOW_POST}")
    print(f"Min total positives: {MIN_TOTAL_POSITIVES}")
    print(f"Min contiguous segment length: {MIN_SEGMENT_LEN}")
    print(f"Min benign pre-history: {MIN_BENIGN_PRE}")
    print(f"Onset offset: {ONSET_OFFSET}")

    df = pd.read_csv(DATA_FILENAME, low_memory=False)
    df, X_all = preprocess_dataframe(df)

    selected_attacks = choose_attack_types(df)
    if not selected_attacks:
        print("[!] No attack types met the automatic inclusion criteria.")
        sys.exit(1)

    print(f"\nSelected attack types: {selected_attacks}")

    all_final_results: Dict[str, Dict] = {
        "_meta": {
            "data_file": DATA_FILENAME,
            "output_dir": OUTPUT_DIR,
            "seeds": SEEDS,
            "window_pre": WINDOW_PRE,
            "window_post": WINDOW_POST,
            "warmup_steps": WARMUP_STEPS,
            "min_total_positives": MIN_TOTAL_POSITIVES,
            "min_segment_len": MIN_SEGMENT_LEN,
            "selected_attacks": selected_attacks,
        }
    }

    for attack in selected_attacks:
        print("\n" + "-" * 90)
        print(f"Attack type: {attack}")
        print("-" * 90)

        y_all = (df["type"].values == attack).astype(int)
        positive_count = int(np.sum(y_all))
        negative_count = int(len(y_all) - positive_count)
        print(f"Binary class counts -> positive={positive_count}, negative={negative_count}")

        onset_idx, end_idx = choose_attack_onset(y_all)
        if onset_idx is None:
            print(f"[skip] No valid onset found for attack='{attack}' under current rules.")
            continue

        left = max(0, onset_idx - WINDOW_PRE)
        right = min(len(y_all), onset_idx + WINDOW_POST)
        if right - left < (WINDOW_PRE + min(WINDOW_POST, len(y_all) - onset_idx)):
            print(f"[diag] clipped window -> left={left}, right={right}, len={right-left}")
        else:
            print(f"[diag] window -> left={left}, right={right}, len={right-left}")

        X_stream = X_all[left:right]
        y_stream = y_all[left:right]
        onset_local = onset_idx - left

        attack_results: Dict[str, Dict] = {}
        for severity_name, severity_params in SEVERITIES.items():
            print(f"\n>>> Running {attack} under severity {severity_name} ...")
            results_list = Parallel(n_jobs=N_JOBS)(
                delayed(run_single_seed)(X_stream, y_stream, onset_local, seed, severity_params)
                for seed in SEEDS
            )

            summary = summarize_results_list(results_list)
            attack_results[severity_name] = {
                "per_seed": results_list,
                "summary": summary,
            }

            print(
                f"[summary:{severity_name}] delay means -> "
                + ", ".join(
                    [
                        f"{m}={summary[m]['delay_mean']:.3f}" if summary[m]["delay_mean"] is not None else f"{m}=None"
                        for m in summary.keys()
                    ]
                )
            )
            print(
                f"[runtime:{severity_name}] infer ms -> "
                + ", ".join(
                    [
                        f"{m}={summary[m]['infer_ms_mean']:.4f}" if summary[m]["infer_ms_mean"] is not None else f"{m}=None"
                        for m in summary.keys()
                    ]
                )
            )

        all_final_results[attack] = attack_results

    valid_attacks = [attack for attack in all_final_results.keys() if attack != "_meta"]
    if not valid_attacks:
        print("[!] No attack types produced valid experiments. Nothing to save.")
        sys.exit(1)

    all_final_results["_aggregate"] = aggregate_across_attacks(all_final_results)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_final_results, f, indent=2)
        saved_path = output_path
    except Exception as exc:
        fallback_path = os.path.join(LOCAL_FALLBACK_DIR, OUTPUT_FILENAME)
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(all_final_results, f, indent=2)
        saved_path = fallback_path
        print(f"[warn] Could not write to OUTPUT_DIR={OUTPUT_DIR}. Saved locally instead. Reason: {exc}")

    print("\n" + "=" * 90)
    print(f"SUCCESS! Results saved to {saved_path}")
    print(f"Included attacks: {valid_attacks}")
    print("=" * 90)


if __name__ == "__main__":
    main()
