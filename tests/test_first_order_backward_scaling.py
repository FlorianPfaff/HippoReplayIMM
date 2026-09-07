"""Backward-message scaling must not overflow or change smoothed marginals."""

from itertools import product

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye
from scipy.special import logsumexp

from hipporeplayimm.state_space_first_order import (
    _forward_backward_first_order,
    _forward_backward_first_order_time_varying,
    _score_first_order_imm,
)


def _enumerated_posterior(log_likelihood, transitions, prior):
    """Independent log-domain reference by enumeration of every latent path."""
    n_time, n_states = log_likelihood.shape
    paths = np.asarray(list(product(range(n_states), repeat=n_time)), dtype=int)
    with np.errstate(divide="ignore"):
        weights = np.log(prior[paths[:, 0]])
        for time in range(n_time):
            weights += log_likelihood[time, paths[:, time]]
            if time:
                weights += np.log(transitions[time - 1][paths[:, time], paths[:, time - 1]])
    evidence = logsumexp(weights)
    posterior = np.zeros_like(log_likelihood)
    for time in range(n_time):
        for state in range(n_states):
            posterior[time, state] = np.exp(logsumexp(weights[paths[:, time] == state]) - evidence)
    return evidence, posterior


@pytest.mark.parametrize("time_varying", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("offset", [0.0, 900.0])
def test_subnormal_forward_scale_has_finite_smoothed_posterior(time_varying, masked, offset):
    # Both stationary paths have likelihood exp(-720), hence a 50/50 posterior.
    # The previous beta / forward_scale update overflowed even though the
    # forward evidence, filtered marginals, and true smoothed result are finite.
    likelihood = np.array([[0.0, -720.0], [-720.0, 0.0]]) + offset
    mask = None
    if masked:
        likelihood = np.column_stack((likelihood, np.full(2, 10000.0)))
        mask = np.array([True, True, False])
    transition = eye(likelihood.shape[1], format="csr")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if time_varying:
            evidence, trajectory = _forward_backward_first_order_time_varying(likelihood, [transition], valid_bin_mask=mask)
        else:
            evidence, trajectory = _forward_backward_first_order(likelihood, transition, valid_bin_mask=mask)
    assert evidence == pytest.approx(-720.0 + 2.0 * offset, abs=1e-8)
    # Subnormal forward arithmetic has lower relative precision than normal floats.
    np.testing.assert_allclose(np.exp(trajectory[:, :2]), 0.5, atol=1e-9, rtol=0.0)
    if masked:
        np.testing.assert_array_equal(np.exp(trajectory[:, 2]), 0.0)


def test_imm_backward_scale_overflow_does_not_destroy_fragmented_posterior():
    # With identity spatial kernels and no mode switching, only the fragmented
    # mode can explain a final jump after this long, perfectly localized event.
    n_time, n_bins = 174, 64
    likelihood = np.full((n_time, n_bins), -np.inf)
    likelihood[:-1, 0] = 0.0
    likelihood[-1, 1] = 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        evidence, trajectory, modes = _score_first_order_imm(
            likelihood,
            np.arange(n_bins, dtype=float)[:, None],
            stationary_sigma_cm=0.01,
            diffusion_sigma_cm=0.01,
            max_step_sigma=5.0,
            mode_stickiness=1.0,
        )
    assert evidence == pytest.approx(-np.log(3.0) - n_time * np.log(n_bins), abs=1e-8)
    expected_positions = np.zeros_like(likelihood)
    expected_positions[:-1, 0] = 1.0
    expected_positions[-1, 1] = 1.0
    np.testing.assert_allclose(np.exp(trajectory), expected_positions, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(modes, np.tile([0.0, 0.0, 1.0], (n_time, 1)), atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("time_varying", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_first_order_smoothing_matches_enumerated_paths(time_varying, masked):
    rng = np.random.default_rng(2407)
    likelihood = rng.normal(-5.0, 3.0, (4, 3))
    likelihood[1, 0] = -np.inf
    mask = np.array([True, False, True]) if masked else None
    prior = np.ones(3) if mask is None else mask.astype(float)
    prior /= prior.sum()
    transitions = []
    for _ in range(3):
        matrix = rng.uniform(0.1, 1.0, (3, 3))
        if masked:
            matrix[~mask] = 0.0
        matrix /= matrix.sum(axis=0, keepdims=True)
        transitions.append(matrix)
    if not time_varying:
        transitions = [transitions[0]] * 3
    expected_evidence, expected_posterior = _enumerated_posterior(likelihood, transitions, prior)
    if time_varying:
        evidence, trajectory = _forward_backward_first_order_time_varying(
            likelihood, [csr_matrix(matrix) for matrix in transitions], valid_bin_mask=mask
        )
    else:
        evidence, trajectory = _forward_backward_first_order(likelihood, csr_matrix(transitions[0]), valid_bin_mask=mask)
    assert evidence == pytest.approx(expected_evidence, abs=1e-12)
    np.testing.assert_allclose(np.exp(trajectory), expected_posterior, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("stickiness", [0.0, 0.6, 1.0])
@pytest.mark.parametrize("masked", [False, True])
def test_imm_smoothing_preserves_relative_mode_weights(stickiness, masked):
    centers = np.array([[0.0], [1.0], [2.0]])
    likelihood = np.array([[0.0, -2.0, -4.0], [-3.0, -1.0, 0.0], [-2.0, 0.0, -1.0]])
    mask = np.array([True, False, True]) if masked else None
    prior_position = np.ones(3) if mask is None else mask.astype(float)
    prior_position /= prior_position.sum()
    spatial = []
    distance_squared = (centers - centers.T) ** 2
    for sigma in (0.4, 1.1):
        kernel = np.exp(-0.5 * distance_squared / sigma**2)
        kernel[distance_squared > (5.0 * sigma) ** 2] = 0.0
        if masked:
            kernel[~mask] = 0.0
        kernel /= kernel.sum(axis=0, keepdims=True)
        spatial.append(kernel)
    spatial.append(np.tile(prior_position[:, None], (1, 3)))
    mode_transition = np.full((3, 3), (1.0 - stickiness) / 2.0)
    np.fill_diagonal(mode_transition, stickiness)
    joint_transition = np.block([
        [mode_transition[src, dst] * spatial[dst] for src in range(3)]
        for dst in range(3)
    ])
    expected_evidence, joint_posterior = _enumerated_posterior(
        np.tile(likelihood, (1, 3)),
        [joint_transition] * 2,
        np.tile(prior_position / 3.0, 3),
    )
    expected = joint_posterior.reshape(3, 3, 3)
    evidence, trajectory, modes = _score_first_order_imm(
        likelihood,
        centers,
        stationary_sigma_cm=0.4,
        diffusion_sigma_cm=1.1,
        max_step_sigma=5.0,
        mode_stickiness=stickiness,
        valid_bin_mask=mask,
    )
    assert evidence == pytest.approx(expected_evidence, abs=1e-12)
    np.testing.assert_allclose(np.exp(trajectory), expected.sum(axis=1), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(modes, expected.sum(axis=2), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("scorer", ["fixed", "time-varying", "imm"])
def test_single_bin_needs_no_backward_message(scorer):
    likelihood = np.array([[-1.0, -3.0]])
    expected_evidence = logsumexp(likelihood[0]) - np.log(2.0)
    expected_posterior = np.exp(likelihood - logsumexp(likelihood, axis=1, keepdims=True))
    if scorer == "fixed":
        evidence, trajectory = _forward_backward_first_order(likelihood, eye(2, format="csr"))
    elif scorer == "time-varying":
        evidence, trajectory = _forward_backward_first_order_time_varying(likelihood, [])
    else:
        evidence, trajectory, modes = _score_first_order_imm(
            likelihood,
            np.array([[0.0], [1.0]]),
            stationary_sigma_cm=0.4,
            diffusion_sigma_cm=1.1,
            max_step_sigma=5.0,
            mode_stickiness=0.6,
        )
        np.testing.assert_allclose(modes, 1.0 / 3.0)
    assert evidence == pytest.approx(expected_evidence, abs=1e-12)
    np.testing.assert_allclose(np.exp(trajectory), expected_posterior, atol=1e-12, rtol=1e-12)
