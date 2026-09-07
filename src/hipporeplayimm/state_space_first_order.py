"""Exact first-order state-space replay recursions."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

from .encoding import LogEmissionTensor
from .models import _normalize_log_weights
from .state_space_utils import (
    _as_log_probs,
    _coerce_valid_bin_mask,
    _gaussian_transition_matrix,
    _mode_transition_matrix,
    _scaled_emissions,
    _uniform_log_prior,
    _uniform_probabilities,
)


def _score_stationary(
    emissions: LogEmissionTensor,
    valid_bin_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    log_likelihood = np.asarray(emissions.log_likelihood, dtype=float)
    if np.any(np.isnan(log_likelihood)) or np.any(log_likelihood == np.inf):
        raise ValueError("log_likelihood must not contain NaN or +inf")
    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, emissions.n_bins)
    active = np.ones(emissions.n_bins, dtype=bool) if valid_mask is None else valid_mask
    finite_on_active_support = np.isfinite(log_likelihood[:, active])
    if not np.all(np.any(finite_on_active_support, axis=1)):
        raise ValueError("at least one emission row has no finite likelihood mass")

    masked_log_likelihood = np.where(
        active[None, :] & np.isfinite(log_likelihood),
        log_likelihood,
        -np.inf,
    )
    log_weights = np.sum(masked_log_likelihood, axis=0) + _uniform_log_prior(
        emissions.n_bins,
        valid_bin_mask,
    )
    logp = float(logsumexp(log_weights))
    if not np.isfinite(logp):
        raise ValueError("stationary model has no finite path mass")
    posterior = _normalize_log_weights(log_weights)
    return logp, np.repeat(posterior[None, :], emissions.n_time, axis=0)


def _score_fragmented(
    emissions: LogEmissionTensor,
    valid_bin_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    scaled, offsets = _scaled_emissions(
        emissions.log_likelihood,
        valid_bin_mask=valid_bin_mask,
    )
    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, emissions.n_bins)
    if valid_mask is None:
        row_sums = scaled.sum(axis=1)
        n_valid = emissions.n_bins
        support = slice(None)
    else:
        row_sums = scaled[:, valid_mask].sum(axis=1)
        n_valid = int(np.sum(valid_mask))
        support = valid_mask
    if np.any(row_sums <= 0.0):
        raise ValueError("at least one emission row has no finite likelihood mass")
    posterior = np.zeros_like(scaled)
    posterior[:, support] = scaled[:, support] / row_sums[:, None]
    logp = float(np.sum(np.log(row_sums / n_valid) + offsets))
    return logp, _as_log_probs(posterior)


def _normalize_backward_message(message: np.ndarray) -> np.ndarray:
    """Bound backward messages without changing their relative state weights.

    Backward messages are defined only up to a common positive scale. Dividing
    by a forward likelihood scale can overflow even when the smoothed posterior
    is well-defined. Normalize by the largest message entry instead, jointly
    over all modes and positions for an IMM.
    """

    maximum = float(np.max(message))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("backward message has no finite positive mass")
    return message / maximum


def _forward_backward_first_order(
    log_likelihood: np.ndarray,
    transition: csr_matrix,
    valid_bin_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    n_time, n_bins = log_likelihood.shape
    scaled, offsets = _scaled_emissions(log_likelihood, valid_bin_mask=valid_bin_mask)
    filtered = np.zeros((n_time, n_bins), dtype=float)
    scales = np.zeros(n_time, dtype=float)

    alpha = scaled[0] * _uniform_probabilities(n_bins, valid_bin_mask)
    scales[0] = float(alpha.sum())
    if scales[0] <= 0.0:
        raise ValueError("first emission row has no finite likelihood mass")
    alpha /= scales[0]
    filtered[0] = alpha
    logp = float(np.log(scales[0]) + offsets[0])

    for time_index in range(1, n_time):
        alpha = np.asarray(transition @ alpha, dtype=float) * scaled[time_index]
        scales[time_index] = float(alpha.sum())
        if scales[time_index] <= 0.0:
            raise ValueError(f"emission row {time_index} has no finite predicted mass")
        alpha /= scales[time_index]
        filtered[time_index] = alpha
        logp += float(np.log(scales[time_index]) + offsets[time_index])

    smoothed = np.zeros_like(filtered)
    beta = np.ones(n_bins, dtype=float)
    smoothed[-1] = filtered[-1]
    for time_index in range(n_time - 1, 0, -1):
        beta = _normalize_backward_message(
            np.asarray(transition.T @ (scaled[time_index] * beta), dtype=float)
        )
        gamma = filtered[time_index - 1] * beta
        total = float(gamma.sum())
        smoothed[time_index - 1] = gamma / total if total > 0.0 else filtered[time_index - 1]
    return logp, _as_log_probs(smoothed)


def _forward_backward_first_order_time_varying(
    log_likelihood: np.ndarray,
    transitions: list[csr_matrix],
    valid_bin_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Forward/backward recursion with one transition matrix per step."""

    n_time, n_bins = log_likelihood.shape
    if len(transitions) != max(n_time - 1, 0):
        raise ValueError("transitions must contain one matrix per adjacent time-bin pair")
    scaled, offsets = _scaled_emissions(log_likelihood, valid_bin_mask=valid_bin_mask)
    filtered = np.zeros((n_time, n_bins), dtype=float)
    scales = np.zeros(n_time, dtype=float)

    alpha = scaled[0] * _uniform_probabilities(n_bins, valid_bin_mask)
    scales[0] = float(alpha.sum())
    if scales[0] <= 0.0:
        raise ValueError("first emission row has no finite likelihood mass")
    alpha /= scales[0]
    filtered[0] = alpha
    logp = float(np.log(scales[0]) + offsets[0])

    for time_index in range(1, n_time):
        transition = transitions[time_index - 1]
        alpha = np.asarray(transition @ alpha, dtype=float) * scaled[time_index]
        scales[time_index] = float(alpha.sum())
        if scales[time_index] <= 0.0:
            raise ValueError(f"emission row {time_index} has no finite predicted mass")
        alpha /= scales[time_index]
        filtered[time_index] = alpha
        logp += float(np.log(scales[time_index]) + offsets[time_index])

    smoothed = np.zeros_like(filtered)
    beta = np.ones(n_bins, dtype=float)
    smoothed[-1] = filtered[-1]
    for time_index in range(n_time - 1, 0, -1):
        transition = transitions[time_index - 1]
        beta = _normalize_backward_message(
            np.asarray(transition.T @ (scaled[time_index] * beta), dtype=float)
        )
        gamma = filtered[time_index - 1] * beta
        total = float(gamma.sum())
        smoothed[time_index - 1] = gamma / total if total > 0.0 else filtered[time_index - 1]
    return logp, _as_log_probs(smoothed)


def _score_first_order_imm(
    log_likelihood: np.ndarray,
    bin_centers: np.ndarray,
    *,
    stationary_sigma_cm: float,
    diffusion_sigma_cm: float,
    max_step_sigma: float,
    mode_stickiness: float,
    valid_bin_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    modes = ("stationary", "diffusion", "fragmented")
    n_modes = len(modes)
    n_time, n_bins = log_likelihood.shape
    transitions = {
        "stationary": _gaussian_transition_matrix(bin_centers, stationary_sigma_cm, max_step_sigma, valid_bin_mask=valid_bin_mask),
        "diffusion": _gaussian_transition_matrix(bin_centers, diffusion_sigma_cm, max_step_sigma, valid_bin_mask=valid_bin_mask),
        "fragmented": None,
    }
    mode_transition = _mode_transition_matrix(n_modes, mode_stickiness)
    scaled, offsets = _scaled_emissions(log_likelihood, valid_bin_mask=valid_bin_mask)
    filtered = np.zeros((n_time, n_modes, n_bins), dtype=float)
    scales = np.zeros(n_time, dtype=float)

    alpha = np.tile(scaled[0] * _uniform_probabilities(n_bins, valid_bin_mask) / n_modes, (n_modes, 1))
    scales[0] = float(alpha.sum())
    if scales[0] <= 0.0:
        raise ValueError("first emission row has no finite likelihood mass")
    alpha /= scales[0]
    filtered[0] = alpha
    logp = float(np.log(scales[0]) + offsets[0])

    for time_index in range(1, n_time):
        predicted = np.zeros_like(alpha)
        for dst_idx, dst_mode in enumerate(modes):
            dst = np.zeros(n_bins, dtype=float)
            for src_idx in range(n_modes):
                dst += mode_transition[src_idx, dst_idx] * _apply_transition(
                    transitions[dst_mode],
                    alpha[src_idx],
                    valid_bin_mask=valid_bin_mask,
                )
            predicted[dst_idx] = dst
        alpha = predicted * scaled[time_index][None, :]
        scales[time_index] = float(alpha.sum())
        if scales[time_index] <= 0.0:
            raise ValueError(f"emission row {time_index} has no finite predicted mass")
        alpha /= scales[time_index]
        filtered[time_index] = alpha
        logp += float(np.log(scales[time_index]) + offsets[time_index])

    smoothed = np.zeros_like(filtered)
    beta = np.ones((n_modes, n_bins), dtype=float)
    smoothed[-1] = filtered[-1]
    for time_index in range(n_time - 1, 0, -1):
        beta_prev = np.zeros_like(beta)
        for src_idx in range(n_modes):
            for dst_idx, dst_mode in enumerate(modes):
                beta_prev[src_idx] += mode_transition[src_idx, dst_idx] * _apply_transition_backward(
                    transitions[dst_mode],
                    scaled[time_index] * beta[dst_idx],
                    valid_bin_mask=valid_bin_mask,
                )
        beta = _normalize_backward_message(beta_prev)
        gamma = filtered[time_index - 1] * beta
        total = float(gamma.sum())
        smoothed[time_index - 1] = gamma / total if total > 0.0 else filtered[time_index - 1]

    return logp, _as_log_probs(smoothed.sum(axis=1)), smoothed.sum(axis=2)


def _apply_transition(
    transition: csr_matrix | None,
    weights: np.ndarray,
    valid_bin_mask: np.ndarray | None = None,
) -> np.ndarray:
    if transition is None:
        return _uniform_probabilities(weights.shape[0], valid_bin_mask) * float(weights.sum())
    return np.asarray(transition @ weights, dtype=float)


def _apply_transition_backward(
    transition: csr_matrix | None,
    values: np.ndarray,
    valid_bin_mask: np.ndarray | None = None,
) -> np.ndarray:
    if transition is None:
        valid_mask = _coerce_valid_bin_mask(valid_bin_mask, values.shape[0])
        if valid_mask is None:
            return np.full(values.shape, float(values.sum()) / values.shape[0], dtype=float)
        out = np.zeros(values.shape, dtype=float)
        out[valid_mask] = float(values[valid_mask].sum()) / int(np.sum(valid_mask))
        return out
    return np.asarray(transition.T @ values, dtype=float)
