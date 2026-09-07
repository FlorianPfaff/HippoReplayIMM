"""Stable sparse first-order inference shared by scalar and duration-aware models.

Keep the fast probability recursion while its positive products are representable.
If an intermediate could become subnormal or overflow, replay the entire event
in log space. Restarting is essential: a lost forward path cannot be recovered by
merely rescaling a backward message. Sparse spatial transitions stay sparse, and
IMM mode/position probabilities are always normalized jointly.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

from .state_space_utils import LOG_ZERO, _as_log_probs, _scaled_emissions, _uniform_probabilities

_LOG_TINY = float(np.log(np.finfo(float).tiny))
_LOG_MAX = float(np.log(np.finfo(float).max))


def _log_positive(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, -np.inf, dtype=float)
    np.log(values, out=out, where=values > 0.0)
    return out


def _minimum_log_positive(values: np.ndarray) -> float:
    positive = values[values > 0.0]
    return float(np.log(positive.min())) if positive.size else np.inf


class _SpatialTransition:
    def __init__(self, matrix: csr_matrix | None, uniform: np.ndarray):
        self.matrix = matrix
        self.uniform = uniform
        self.log_minimum = _minimum_log_positive(uniform if matrix is None else matrix.data)
        self._log_matrices: dict[bool, csr_matrix] = {}

    def apply(self, values: np.ndarray, *, backward: bool = False) -> np.ndarray:
        if self.matrix is None:
            if backward:
                return np.where(self.uniform > 0.0, self.uniform @ values, 0.0)
            return self.uniform * values.sum()
        matrix = self.matrix.T if backward else self.matrix
        return np.asarray(matrix @ values, dtype=float)

    def apply_log(self, values: np.ndarray, *, backward: bool = False) -> np.ndarray:
        if self.matrix is None:
            log_uniform = _log_positive(self.uniform)
            if backward:
                total = logsumexp(log_uniform + values)
                return np.where(self.uniform > 0.0, total, -np.inf)
            return log_uniform + logsumexp(values)
        if backward not in self._log_matrices:
            matrix = (self.matrix.T if backward else self.matrix).tocsr(copy=True)
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            matrix.data = np.log(matrix.data)
            self._log_matrices[backward] = matrix
        matrix = self._log_matrices[backward]
        out = np.full(matrix.shape[0], -np.inf, dtype=float)
        nonempty = np.diff(matrix.indptr) > 0
        if np.any(nonempty):
            terms = matrix.data + values[matrix.indices]
            out[nonempty] = np.logaddexp.reduceat(terms, matrix.indptr[:-1][nonempty])
        return out


class _Step:
    def __init__(self, spatial: tuple[_SpatialTransition, ...], modes: np.ndarray):
        self.spatial = spatial
        self.modes = modes
        self.log_modes = _log_positive(modes)
        self.log_minimum = min(kernel.log_minimum for kernel in spatial) + _minimum_log_positive(modes)

    def apply(self, values: np.ndarray, *, backward: bool = False, logarithmic: bool = False) -> np.ndarray:
        if len(self.spatial) == 1:
            kernel = self.spatial[0]
            result = kernel.apply_log(values[0], backward=backward) if logarithmic else kernel.apply(values[0], backward=backward)
            return result[None, :]
        if logarithmic:
            if backward:
                local = np.stack([kernel.apply_log(values[i], backward=True) for i, kernel in enumerate(self.spatial)])
                return logsumexp(self.log_modes[:, :, None] + local[None, :, :], axis=1)
            mixed = logsumexp(values[:, None, :] + self.log_modes[:, :, None], axis=0)
            return np.stack([kernel.apply_log(mixed[i]) for i, kernel in enumerate(self.spatial)])
        if backward:
            local = np.stack([kernel.apply(values[i], backward=True) for i, kernel in enumerate(self.spatial)])
            return self.modes @ local
        mixed = self.modes.T @ values
        return np.stack([kernel.apply(mixed[i]) for i, kernel in enumerate(self.spatial)])


def _log_recursion(shifted: np.ndarray, offsets: np.ndarray, prior: np.ndarray, steps: list[_Step]) -> tuple[float, np.ndarray]:
    """Recompute from the original log emissions, never clipped probabilities."""

    filtered = np.empty((len(shifted), *prior.shape), dtype=float)
    alpha = _log_positive(prior)
    logp = 0.0
    for time_index, emission in enumerate(shifted):
        if time_index:
            alpha = steps[time_index - 1].apply(alpha, logarithmic=True)
        alpha = alpha + emission
        scale = float(logsumexp(alpha))
        if not np.isfinite(scale):
            raise ValueError(f"emission row {time_index} has no finite predicted mass")
        alpha -= scale
        filtered[time_index] = alpha
        logp += scale + float(offsets[time_index])

    smoothed = np.empty_like(filtered)
    smoothed[-1] = filtered[-1]
    beta = np.zeros(prior.shape, dtype=float)
    for time_index in range(len(shifted) - 1, 0, -1):
        beta = steps[time_index - 1].apply(shifted[time_index] + beta, backward=True, logarithmic=True)
        # One common normalization preserves relative IMM mode odds.
        beta -= logsumexp(beta)
        gamma = filtered[time_index - 1] + beta
        smoothed[time_index - 1] = gamma - logsumexp(gamma)
    return float(logp), np.where(np.isneginf(smoothed), LOG_ZERO, smoothed)


def _forward_backward(log_likelihood: np.ndarray, uniform: np.ndarray, n_modes: int, steps: list[_Step]) -> tuple[float, np.ndarray]:
    n_time = len(log_likelihood)
    if not n_time:
        raise ValueError("emissions must contain at least one time bin")
    if len(steps) != n_time - 1:
        raise ValueError("transitions must contain one matrix per adjacent time-bin pair")
    active = uniform > 0.0
    scaled, offsets = _scaled_emissions(log_likelihood, valid_bin_mask=active)
    shifted = np.where(active[None, :], log_likelihood - offsets[:, None], -np.inf)
    minimum_emission = np.min(np.where(np.isfinite(shifted), shifted, np.inf), axis=1)
    prior = np.tile(uniform / n_modes, (n_modes, 1))

    def fallback() -> tuple[float, np.ndarray]:
        return _log_recursion(shifted, offsets, prior, steps)

    # Bound every positive product before evaluating it, including products in
    # sparse matvecs (whose underflow is not caught by NumPy's errstate).
    if _minimum_log_positive(prior) + minimum_emission[0] < _LOG_TINY:
        return fallback()
    filtered = np.empty((n_time, *prior.shape), dtype=float)
    scales = np.empty(n_time, dtype=float)
    alpha = prior * scaled[0]
    scales[0] = float(alpha.sum())
    alpha /= scales[0]
    filtered[0] = alpha
    logp = float(np.log(scales[0]) + offsets[0])
    for time_index in range(1, n_time):
        step = steps[time_index - 1]
        if _minimum_log_positive(alpha) + step.log_minimum + minimum_emission[time_index] < _LOG_TINY:
            return fallback()
        alpha = step.apply(alpha) * scaled[time_index]
        scales[time_index] = float(alpha.sum())
        if scales[time_index] <= 0.0 or not np.isfinite(scales[time_index]):
            return fallback()
        alpha /= scales[time_index]
        filtered[time_index] = alpha
        logp += float(np.log(scales[time_index]) + offsets[time_index])

    smoothed = np.empty_like(filtered)
    smoothed[-1] = filtered[-1]
    beta = np.ones(prior.shape, dtype=float)
    for time_index in range(n_time - 1, 0, -1):
        step = steps[time_index - 1]
        if (_minimum_log_positive(beta) + minimum_emission[time_index] + step.log_minimum < _LOG_TINY
                or np.log(beta.max()) - np.log(scales[time_index]) > _LOG_MAX - 1.0):
            return fallback()
        beta = step.apply(scaled[time_index] * beta, backward=True) / scales[time_index]
        if not np.all(np.isfinite(beta)) or _minimum_log_positive(filtered[time_index - 1]) + _minimum_log_positive(beta) < _LOG_TINY:
            return fallback()
        gamma = filtered[time_index - 1] * beta
        total = float(gamma.sum())
        if total <= 0.0 or not np.isfinite(total):
            return fallback()
        smoothed[time_index - 1] = gamma / total
    return logp, _as_log_probs(smoothed.reshape(n_time, -1)).reshape(smoothed.shape)


def first_order_forward_backward(log_likelihood: np.ndarray, transitions, valid_bin_mask=None) -> tuple[float, np.ndarray]:
    """Score fixed or per-step column-stochastic spatial transitions."""

    uniform = _uniform_probabilities(log_likelihood.shape[1], valid_bin_mask)
    matrices = transitions if isinstance(transitions, (list, tuple)) else [transitions] * max(len(log_likelihood) - 1, 0)
    cache: dict[int, _Step] = {}
    steps = []
    for matrix in matrices:
        key = id(matrix)
        if key not in cache:
            cache[key] = _Step((_SpatialTransition(matrix, uniform),), np.ones((1, 1)))
        steps.append(cache[key])
    logp, joint = _forward_backward(log_likelihood, uniform, 1, steps)
    return logp, joint[:, 0, :]


def first_order_imm_forward_backward(
    log_likelihood: np.ndarray,
    stationary_transition: csr_matrix,
    diffusion_transitions,
    mode_transitions: list[np.ndarray],
    valid_bin_mask=None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Score stationary/diffusion/fragmented IMM without a dense joint kernel."""

    uniform = _uniform_probabilities(log_likelihood.shape[1], valid_bin_mask)
    stationary = _SpatialTransition(stationary_transition, uniform)
    fragmented = _SpatialTransition(None, uniform)
    matrices = diffusion_transitions if isinstance(diffusion_transitions, (list, tuple)) else [diffusion_transitions] * max(len(log_likelihood) - 1, 0)
    if len(matrices) != len(mode_transitions):
        raise ValueError("transitions must contain one matrix per adjacent time-bin pair")
    spatial_cache: dict[int, _SpatialTransition] = {}
    steps = []
    for matrix, modes in zip(matrices, mode_transitions, strict=True):
        key = id(matrix)
        if key not in spatial_cache:
            spatial_cache[key] = _SpatialTransition(matrix, uniform)
        steps.append(_Step((stationary, spatial_cache[key], fragmented), modes))
    logp, joint = _forward_backward(log_likelihood, uniform, 3, steps)
    return logp, logsumexp(joint, axis=1), np.exp(logsumexp(joint, axis=2))
