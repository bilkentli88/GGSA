"""
GGSA Multi-Architecture Protocol on UNSW-NB15
=============================================

Cleaned reproduction script for the supplementary UNSW-NB15 experiments.

Purpose
-------
- Evaluate GGSA in a targeted UNSW-NB15 setting.
- Compare MLP and LSTM-based configurations under degraded telemetry.
- Measure delay on a sustained target-attack segment rather than on isolated positives.

What is included
----------------
Methods evaluated:
- Standard
- PGD
- RSA
- LSTM-Base
- GGSA-MLP
- GGSA-LSTM

Important notes
---------------
- This script is intended as a supplementary architectural extension experiment.
- The target attack family is controlled by TARGET_ATTACK below.
- Delay is computed on a contiguous target-attack segment selected from the stream.
"""

from __future__ import annotations

import json
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
# RESEARCH CONFIGURATION
# ============================================================================

DATA_FILENAME = "UNSW_NB15.csv"
OUTPUT_FILENAME = "results_ggsa_unsw_multi_arch_clean.json"

TARGET_ATTACK = "Exploits"
RUN_ONLY_SEVERITY = None        # None = run all severity regimes
TOPK_FEATURES = 5

SEEDS: List[int] = [
    88, 109, 253, 371, 458, 555, 666, 793, 907, 1009,
    1103, 1201, 1301, 1409, 1511, 1601, 1789, 1877, 1971, 2025
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_JOBS = 1 if DEVICE.type == "cuda" else 4

# Hyperparameters
HARDENING_STEPS = 5
PGD_STEPS = 5
PGD_STEP_SIZE = 0.02
PGD_EPSILON = 0.08
PROBA_TRIGGER_LOW = 0.30
PROBA_TRIGGER_HIGH = 0.70
UNCERTAINTY_THRESHOLD = 0.60
FN_BIAS_WEIGHT = 5.0

# Streaming / protocol settings
WARMUP_STEPS = 800
WARMUP_EPOCHS = 5
WINDOW_PRE = 2000
WINDOW_POST = 6000
SEQ_LEN = 8
MIN_ATTACK_SEGMENT_LEN = 50
LABEL_DELAY = 0


# ============================================================================
# MODELS
# ============================================================================

class DynamicMLP(nn.Module):
    """
    Lightweight MLP used for:
    - Standard
    - PGD
    - RSA
    - GGSA-MLP
    """

    def __init__(self, input_dim: int, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=torch.tensor([pos_weight], device=DEVICE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            return float(torch.sigmoid(self.forward(x)).item())

    def get_input_gradient(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.train()
        self.zero_grad(set_to_none=True)
        x_grad = x.clone().detach().requires_grad_(True)
        loss = self.loss_fn(self.forward(x_grad), target).mean()
        loss.backward()
        grad = x_grad.grad.detach()
        if not was_training:
            self.eval()
        return grad

    def online_update(self, x: torch.Tensor, y: torch.Tensor, weight: float = 1.0) -> None:
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = (self.loss_fn(self.forward(x), y) * weight).mean()
        loss.backward()
        self.optimizer.step()


class DynamicLSTM(nn.Module):
    """
    Lightweight LSTM used for:
    - LSTM-Base
    - GGSA-LSTM
    """

    def __init__(self, input_dim: int, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=torch.tensor([pos_weight], device=DEVICE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def predict_proba(self, x: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            return float(torch.sigmoid(self.forward(x)).item())

    def get_input_gradient(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.train()
        self.zero_grad(set_to_none=True)
        x_grad = x.clone().detach().requires_grad_(True)
        loss = self.loss_fn(self.forward(x_grad), target).mean()
        loss.backward()
        grad = x_grad.grad.detach()
        if not was_training:
            self.eval()
        return grad

    def online_update(self, x: torch.Tensor, y: torch.Tensor, weight: float = 1.0) -> None:
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = (self.loss_fn(self.forward(x), y) * weight).mean()
        loss.backward()
        self.optimizer.step()


# ============================================================================
# NOISE REGIMES
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
    "S1": FaultParams(0.02, 0.04, 0.004, 0.003, 12, 2.5),
    "S2": FaultParams(0.08, 0.15, 0.020, 0.012, 24, 3.5),
    "S3": FaultParams(0.20, 0.35, 0.060, 0.025, 36, 6.5),
}


class TelemetryDegradation:
    """
    Simulate degraded telemetry with:
    - slowly accumulating drift
    - additive Gaussian noise
    - burst-aware noise amplification
    - random feature dropout
    """

    def __init__(self, params: FaultParams, rng: np.random.Generator, feature_dim: int) -> None:
        self.params = params
        self.rng = rng
        self.t = 0
        self.in_burst_until = -1
        self.drift = np.zeros(feature_dim, dtype=np.float64)

    def apply(self, x: np.ndarray) -> np.ndarray:
        x_faulted = x.astype(np.float64, copy=True)

        # Slow drift accumulation
        self.drift += self.rng.normal(0.0, self.params.drift_scale, size=x_faulted.shape[0])
        x_faulted += self.drift

        # Burst-aware additive noise
        if self.t > self.in_burst_until and (self.rng.random() < self.params.burst_p):
            self.in_burst_until = self.t + self.params.burst_len

        burst_multiplier = self.params.burst_noise_mult if self.t <= self.in_burst_until else 1.0
        x_faulted += self.rng.normal(
            0.0,
            self.params.noise_std * burst_multiplier,
            size=x_faulted.shape[0],
        )

        # Random dropout corruption
        dropout_mask = self.rng.random(x_faulted.shape[0]) < self.params.dropout_p
        x_faulted[dropout_mask] = 0.0

        self.t += 1
        return x_faulted.astype(np.float32)


# ============================================================================
# HELPERS
# ============================================================================

def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_positive_weight(y: np.ndarray) -> float:
    """
    Compute a simple class-imbalance weight from the warmup prefix.
    """
    positive = max(int(y.sum()), 1)
    negative = max(len(y) - positive, 1)
    return float(negative / positive)


def find_positive_segments(y: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return contiguous positive segments from a binary label vector.
    """
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None

    for i, value in enumerate(y):
        if value == 1 and start is None:
            start = i
        elif value == 0 and start is not None:
            segments.append((start, i - 1))
            start = None

    if start is not None:
        segments.append((start, len(y) - 1))

    return segments


def select_target_segment(y: np.ndarray, min_len: int) -> Tuple[int, int]:
    """
    Select a sustained target-attack segment.

    Strategy:
    - Prefer the first segment with length >= min_len
    - Otherwise fall back to the longest available positive segment
    """
    segments = find_positive_segments(y)
    if not segments:
        raise ValueError(f"No positive segment found for attack '{TARGET_ATTACK}'.")

    long_enough = [(start, end) for start, end in segments if (end - start + 1) >= min_len]
    if long_enough:
        return long_enough[0]

    return max(segments, key=lambda seg: seg[1] - seg[0] + 1)


def build_model(model_type: str, input_dim: int, pos_weight: float) -> nn.Module:
    if model_type == "mlp":
        return DynamicMLP(input_dim, pos_weight=pos_weight).to(DEVICE)
    if model_type == "lstm":
        return DynamicLSTM(input_dim, pos_weight=pos_weight).to(DEVICE)
    raise ValueError(f"Unknown model_type: {model_type}")


def threshold_for_mode(mode: str, proba: float) -> float:
    """
    Apply the uncertainty-triggered lower threshold only for GGSA variants.
    """
    if "GGSA" in mode and PROBA_TRIGGER_LOW < proba < PROBA_TRIGGER_HIGH:
        return 0.35
    return 0.5


def warmup_train(
    model: nn.Module,
    model_type: str,
    scaler: StandardScaler,
    warm_x: np.ndarray,
    warm_y: np.ndarray,
) -> Deque[np.ndarray]:
    """
    Perform a supervised warmup fit before online streaming begins.

    Returns the raw warmup buffer, which is later reused by the LSTM path.
    """
    buffer: Deque[np.ndarray] = deque(maxlen=SEQ_LEN)
    for i in range(len(warm_x)):
        buffer.append(warm_x[i])

    if model_type == "mlp":
        x_tensor = torch.tensor(
            scaler.transform(warm_x),
            dtype=torch.float32,
            device=DEVICE,
        )
        y_tensor = torch.tensor(
            warm_y.reshape(-1, 1),
            dtype=torch.float32,
            device=DEVICE,
        )
        for _ in range(WARMUP_EPOCHS):
            model.online_update(x_tensor, y_tensor, weight=1.0)
    else:
        if len(warm_x) < SEQ_LEN:
            return buffer

        sequences = []
        labels = []
        scaled_warm = scaler.transform(warm_x)

        for i in range(SEQ_LEN - 1, len(warm_x)):
            sequences.append(scaled_warm[i - SEQ_LEN + 1:i + 1])
            labels.append(warm_y[i])

        x_tensor = torch.tensor(np.stack(sequences), dtype=torch.float32, device=DEVICE)
        y_tensor = torch.tensor(np.array(labels).reshape(-1, 1), dtype=torch.float32, device=DEVICE)

        for _ in range(WARMUP_EPOCHS):
            model.online_update(x_tensor, y_tensor, weight=1.0)

    return buffer


def pgd_positive_sample_mlp(model: DynamicMLP, xt_scaled: np.ndarray) -> np.ndarray:
    """
    Generate a PGD-style positive sample around the current scaled input.
    """
    base = torch.tensor(xt_scaled, dtype=torch.float32, device=DEVICE)
    adv = base.clone().detach()
    target = torch.tensor([[1.0]], dtype=torch.float32, device=DEVICE)

    for _ in range(PGD_STEPS):
        grad = model.get_input_gradient(adv, target)
        adv = adv + PGD_STEP_SIZE * torch.sign(grad)
        adv = torch.max(torch.min(adv, base + PGD_EPSILON), base - PGD_EPSILON).detach()

    return adv.cpu().numpy()


def compute_delay_on_segment(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_start: int,
    attack_end: int,
) -> Optional[int]:
    """
    Compute delay as the first correct positive prediction inside the chosen
    attack segment.
    """
    for i in range(attack_start, attack_end + 1):
        if y_true[i] == 1 and y_pred[i] == 1:
            return i - attack_start
    return None


def make_lstm_tensor_from_buffer(
    buffer: Deque[np.ndarray],
    scaler: StandardScaler,
) -> torch.Tensor:
    """
    Convert the current raw sequence buffer into a scaled LSTM input tensor.
    """
    seq_raw = np.stack(list(buffer))
    seq_scaled = scaler.transform(seq_raw).reshape(1, SEQ_LEN, -1)
    return torch.tensor(seq_scaled, dtype=torch.float32, device=DEVICE)


# ============================================================================
# CORE ENGINE
# ============================================================================

def run_single_seed(
    X: np.ndarray,
    y: np.ndarray,
    attack_start: int,
    attack_end: int,
    seed: int,
    fault_params: FaultParams,
) -> Dict[str, Dict[str, float]]:
    """
    Run one full seed for one severity regime.
    """
    set_all_seeds(seed)
    results: Dict[str, Dict[str, float]] = {}

    methods: List[Tuple[str, str]] = [
        ("Standard", "mlp"),
        ("PGD", "mlp"),
        ("RSA", "mlp"),
        ("LSTM-Base", "lstm"),
        ("GGSA-MLP", "mlp"),
        ("GGSA-LSTM", "lstm"),
    ]

    for method_name, model_type in methods:
        rng = np.random.default_rng(seed)
        injector = TelemetryDegradation(fault_params, rng, X.shape[1])

        # ------------------------------------------------------------------
        # Warmup phase
        # ------------------------------------------------------------------
        warm_x = np.array([injector.apply(X[i]) for i in range(WARMUP_STEPS)])
        warm_y = y[:WARMUP_STEPS].astype(np.float32)

        scaler = StandardScaler()
        scaler.fit(warm_x)

        pos_weight = get_positive_weight(warm_y)
        model = build_model(model_type, X.shape[1], pos_weight=pos_weight)
        buffer = warmup_train(model, model_type, scaler, warm_x, warm_y)

        y_pred = np.zeros_like(y)
        y_pred[:WARMUP_STEPS] = warm_y.astype(int)

        delayed_label_queue: Deque[Tuple[int, torch.Tensor, np.ndarray, np.ndarray, int]] = deque()

        # ------------------------------------------------------------------
        # Online phase
        # ------------------------------------------------------------------
        for t in range(WARMUP_STEPS, len(X)):
            x_raw = injector.apply(X[t])
            xt_scaled = scaler.transform(x_raw.reshape(1, -1))

            if model_type == "lstm":
                buffer.append(x_raw)
                if len(buffer) < SEQ_LEN:
                    # This should not happen after warmup, but keep it safe.
                    while len(buffer) < SEQ_LEN:
                        buffer.appendleft(np.zeros(X.shape[1], dtype=np.float32))
                xt = make_lstm_tensor_from_buffer(buffer, scaler)
            else:
                xt = torch.tensor(xt_scaled, dtype=torch.float32, device=DEVICE)

            proba = model.predict_proba(xt)
            threshold = threshold_for_mode(method_name, proba)
            y_pred[t] = 1 if proba >= threshold else 0

            delayed_label_queue.append((t, xt.detach().clone(), xt_scaled.copy(), x_raw.copy(), int(y[t])))
            if len(delayed_label_queue) <= LABEL_DELAY:
                continue

            update_t, update_xt, update_xt_scaled, update_x_raw, update_y = delayed_label_queue.popleft()
            y_tensor = torch.tensor([[update_y]], dtype=torch.float32, device=DEVICE)
            model.online_update(update_xt, y_tensor)

            # --------------------------------------------------------------
            # Additional repair / robustness logic
            # --------------------------------------------------------------
            if update_y == 1 and proba < UNCERTAINTY_THRESHOLD:

                # GGSA variants
                if "GGSA" in method_name:
                    grad_tensor = model.get_input_gradient(
                        update_xt,
                        torch.tensor([[1.0]], dtype=torch.float32, device=DEVICE),
                    )

                    if model_type == "mlp":
                        grad = grad_tensor.cpu().numpy().flatten()
                    else:
                        grad = grad_tensor[0, -1, :].cpu().numpy()

                    topk = min(TOPK_FEATURES, len(grad))
                    important_idx = np.argsort(np.abs(grad))[-topk:]
                    direction = np.sign(grad[important_idx])
                    direction[direction == 0] = 1.0

                    for _ in range(HARDENING_STEPS):
                        if model_type == "mlp":
                            synthetic_scaled = update_xt_scaled.copy()
                            synthetic_scaled[0, important_idx] += (
                                direction * (fault_params.noise_std * fault_params.burst_noise_mult)
                            )
                            synthetic_tensor = torch.tensor(
                                synthetic_scaled,
                                dtype=torch.float32,
                                device=DEVICE,
                            )
                        else:
                            repaired_last = scaler.transform(update_x_raw.reshape(1, -1))
                            repaired_last[0, important_idx] += (
                                direction * (fault_params.noise_std * fault_params.burst_noise_mult)
                            )

                            seq_raw = np.stack(list(buffer)[:-1] + [scaler.inverse_transform(repaired_last).flatten()])
                            seq_scaled = scaler.transform(seq_raw).reshape(1, SEQ_LEN, -1)
                            synthetic_tensor = torch.tensor(
                                seq_scaled,
                                dtype=torch.float32,
                                device=DEVICE,
                            )

                        model.online_update(
                            synthetic_tensor,
                            torch.tensor([[1.0]], dtype=torch.float32, device=DEVICE),
                            weight=FN_BIAS_WEIGHT,
                        )

                # RSA baseline
                elif method_name == "RSA":
                    feature_count = X.shape[1]
                    topk = min(TOPK_FEATURES, feature_count)
                    important_idx = rng.choice(feature_count, topk, replace=False)
                    direction = rng.choice([-1.0, 1.0], topk)

                    for _ in range(HARDENING_STEPS):
                        synthetic_scaled = update_xt_scaled.copy()
                        synthetic_scaled[0, important_idx] += (
                            direction * (fault_params.noise_std * fault_params.burst_noise_mult)
                        )
                        synthetic_tensor = torch.tensor(
                            synthetic_scaled,
                            dtype=torch.float32,
                            device=DEVICE,
                        )
                        model.online_update(
                            synthetic_tensor,
                            torch.tensor([[1.0]], dtype=torch.float32, device=DEVICE),
                            weight=FN_BIAS_WEIGHT,
                        )

                # PGD baseline
                elif method_name == "PGD":
                    adv = pgd_positive_sample_mlp(model, update_xt_scaled)
                    synthetic_tensor = torch.tensor(adv, dtype=torch.float32, device=DEVICE)
                    for _ in range(HARDENING_STEPS):
                        model.online_update(
                            synthetic_tensor,
                            torch.tensor([[1.0]], dtype=torch.float32, device=DEVICE),
                            weight=FN_BIAS_WEIGHT,
                        )

        # Flush delayed labels if LABEL_DELAY > 0
        while delayed_label_queue:
            _, update_xt, _, _, update_y = delayed_label_queue.popleft()
            y_tensor = torch.tensor([[update_y]], dtype=torch.float32, device=DEVICE)
            model.online_update(update_xt, y_tensor)

        # ------------------------------------------------------------------
        # Metric calculation
        # ------------------------------------------------------------------
        delay = compute_delay_on_segment(y, y_pred, attack_start, attack_end)
        attack_mask = np.arange(len(y)) >= WARMUP_STEPS
        flip_rate = (
            float(np.mean(np.abs(y_pred[WARMUP_STEPS + 1:] - y_pred[WARMUP_STEPS:-1])))
            if len(y) > WARMUP_STEPS + 1
            else 0.0
        )

        results[method_name] = {
            "delay": float(delay) if delay is not None else float(attack_end - attack_start + 1),
            "detected_within_segment": float(delay is not None),
            "f1": float(f1_score(y[attack_mask], y_pred[attack_mask], zero_division=0)),
            "flip": flip_rate,
            "attack_segment_start": float(attack_start),
            "attack_segment_end": float(attack_end),
        }

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    df = pd.read_csv(DATA_FILENAME, low_memory=False)

    # Binary target for the selected attack family
    y = (df["attack_cat"].astype(str).str.strip() == TARGET_ATTACK).astype(int).values

    X = (
        df.drop(columns=["id", "label", "attack_cat"], errors="ignore")
        .select_dtypes(include=[np.number])
        .fillna(0)
        .values
        .astype(np.float32)
    )

    if len(X) <= WARMUP_STEPS:
        raise ValueError("The selected stream is shorter than WARMUP_STEPS.")

    raw_attack_start, raw_attack_end = select_target_segment(y, MIN_ATTACK_SEGMENT_LEN)

    window_start = max(0, raw_attack_start - WINDOW_PRE)
    window_end = min(len(y), raw_attack_end + 1 + WINDOW_POST)

    X_stream = X[window_start:window_end]
    y_stream = y[window_start:window_end]
    attack_start = raw_attack_start - window_start
    attack_end = raw_attack_end - window_start

    severity_items = (
        [(RUN_ONLY_SEVERITY, SEVERITIES[RUN_ONLY_SEVERITY])]
        if RUN_ONLY_SEVERITY
        else list(SEVERITIES.items())
    )

    final_results: Dict[str, Dict] = {
        "metadata": {
            "data_filename": DATA_FILENAME,
            "target_attack": TARGET_ATTACK,
            "window_start": int(window_start),
            "window_end_exclusive": int(window_end),
            "attack_segment_start": int(raw_attack_start),
            "attack_segment_end": int(raw_attack_end),
            "min_attack_segment_len": int(MIN_ATTACK_SEGMENT_LEN),
            "warmup_steps": int(WARMUP_STEPS),
            "warmup_epochs": int(WARMUP_EPOCHS),
            "seq_len": int(SEQ_LEN),
            "label_delay": int(LABEL_DELAY),
            "topk_features": int(TOPK_FEATURES),
            "hardening_steps": int(HARDENING_STEPS),
            "pgd_steps": int(PGD_STEPS),
            "pgd_step_size": float(PGD_STEP_SIZE),
            "pgd_epsilon": float(PGD_EPSILON),
            "device": str(DEVICE),
            "methods": ["Standard", "PGD", "RSA", "LSTM-Base", "GGSA-MLP", "GGSA-LSTM"],
        },
        "results": {},
    }

    for severity_name, params in severity_items:
        print(f">>> Running {TARGET_ATTACK} regime {severity_name}...")
        final_results["results"][severity_name] = Parallel(n_jobs=N_JOBS)(
            delayed(run_single_seed)(X_stream, y_stream, attack_start, attack_end, seed, params)
            for seed in SEEDS
        )

    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)

    print(f"Multi-architecture UNSW run complete. Results saved to {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()
