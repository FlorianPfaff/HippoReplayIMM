"""Exact first-order state-space replay recursions."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

from .encoding import LogEmissionTensor
from .models import _normalize_log_weights
from .state_space_forward_backward import first_order_forward_backward, first_order_imm_forward_backward
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


def _forward_backward_first_order(
    log_likelihood: np.ndarray,
    transition: csr_matrix,
    valid_bin_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    return first_order_forward_backward(log_likelihood, transition, valid_bin_mask)


def _forward_backward_first_order_time_varying(
    log_likelihood: np.ndarray,
    transitions: list[csr_matrix],
    valid_bin_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Forward/backward recursion with one transition matrix per step."""

    return first_order_forward_backward(log_likelihood, transitions, valid_bin_mask)


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
    stationary = _gaussian_transition_matrix(bin_centers, stationary_sigma_cm, max_step_sigma, valid_bin_mask=valid_bin_mask)
    diffusion = _gaussian_transition_matrix(bin_centers, diffusion_sigma_cm, max_step_sigma, valid_bin_mask=valid_bin_mask)
    mode_transition = _mode_transition_matrix(3, mode_stickiness)
    return first_order_imm_forward_backward(
        log_likelihood, stationary, diffusion,
        [mode_transition] * max(len(log_likelihood) - 1, 0), valid_bin_mask,
    )


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
