"""Analytic and exhaustive-path regressions for first-order numerical stability."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye
from scipy.special import logsumexp

import hipporeplayimm.state_space as ss
import hipporeplayimm.state_space_forward_backward as numerics
from hipporeplayimm.duration_occupancy import _forward_backward_variable, _score_first_order_imm_variable
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space_first_order import (
    _forward_backward_first_order,
    _forward_backward_first_order_time_varying,
    _score_first_order_imm,
)


def _enumerate_paths(log_emissions, matrices, prior):
    """Independent dense enumeration; transitions are destination-by-source."""
    n_time, n_states = log_emissions.shape
    paths = np.array(list(itertools.product(range(n_states), repeat=n_time)))
    with np.errstate(divide="ignore"):
        log_weights = np.log(prior[paths[:, 0]])
        for t in range(n_time):
            log_weights += log_emissions[t, paths[:, t]]
            if t:
                log_weights += np.log(matrices[t - 1][paths[:, t], paths[:, t - 1]])
    logp = logsumexp(log_weights)
    posterior = np.full((n_time, n_states), -np.inf)
    for t in range(n_time):
        for state in range(n_states):
            posterior[t, state] = logsumexp(log_weights[paths[:, t] == state]) - logp
    return logp, posterior


def _score_first_order(variant, emissions, transitions, mask=None):
    if variant == "fixed":
        return _forward_backward_first_order(emissions, transitions[0], valid_bin_mask=mask)
    if variant == "varying":
        return _forward_backward_first_order_time_varying(emissions, transitions, valid_bin_mask=mask)
    return _forward_backward_variable(ss, emissions, transitions, valid_bin_mask=mask)


@pytest.mark.parametrize("variant", ["fixed", "varying", "duration"])
@pytest.mark.parametrize("penalty", [700.0, 720.0, 1000.0])
@pytest.mark.parametrize("masked", [False, True])
def test_opposing_observations_keep_both_identity_paths(variant, penalty, masked):
    emissions = np.array([[0.0, -penalty], [-penalty, 0.0]])
    mask = None
    if masked:
        # An inactive bin with overwhelming likelihood must not affect scaling.
        emissions = np.column_stack((emissions, [1e4, 1e4]))
        mask = np.array([True, True, False])
    transition = eye(emissions.shape[1], format="csr")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        logp, posterior = _score_first_order(variant, emissions, [transition], mask)
    assert logp == pytest.approx(-penalty, abs=1e-10)
    np.testing.assert_allclose(np.exp(posterior[:, :2]), 0.5, atol=1e-12)
    if masked:
        np.testing.assert_array_equal(np.exp(posterior[:, 2]), 0.0)


@pytest.mark.parametrize("variant", ["fixed", "varying", "duration"])
@pytest.mark.parametrize("penalty", [400.0, 700.0, 720.0])
def test_a_tiny_forward_path_can_become_the_only_feasible_path(variant, penalty):
    # Consecutive penalties lose this path in a probability-only filter;
    # backward-message normalization alone cannot reconstruct it.
    emissions = np.array([[0.0, -penalty, -np.inf], [0.0, -penalty, -np.inf], [-np.inf, 0.0, 0.0]])
    transition = eye(3, format="csr")
    logp, posterior = _score_first_order(variant, emissions, [transition, transition])
    assert logp == pytest.approx(-2.0 * penalty - np.log(3), abs=1e-10)
    np.testing.assert_allclose(np.exp(posterior), [[0, 1, 0]] * 3, atol=1e-12)


@pytest.mark.parametrize("variant", ["fixed", "varying", "duration"])
@pytest.mark.parametrize("force_log", [False, True])
def test_first_order_agrees_with_exhaustive_asymmetric_paths(variant, force_log, monkeypatch):
    if force_log:
        monkeypatch.setattr(numerics, "_LOG_TINY", 1.0)
    emissions = np.array([[-0.2, -1.0, -2.0], [-1.0, -0.3, -np.inf], [-0.5, -2.0, -0.4]])
    a = np.array([[0.7, 0.2, 0.0], [0.3, 0.6, 0.5], [0.0, 0.2, 0.5]])
    b = np.array([[0.4, 0.0, 0.5], [0.4, 0.3, 0.1], [0.2, 0.7, 0.4]])
    matrices = [a, a if variant == "fixed" else b]
    expected_logp, expected_posterior = _enumerate_paths(emissions, matrices, np.full(3, 1 / 3))
    logp, posterior = _score_first_order(variant, emissions, list(map(csr_matrix, matrices)))
    assert logp == pytest.approx(expected_logp, abs=1e-12)
    np.testing.assert_allclose(np.exp(posterior), np.exp(expected_posterior), atol=1e-12)


@pytest.mark.parametrize("variant", ["fixed", "varying", "duration"])
def test_row_offsets_preserve_tiny_path_posteriors(variant):
    emissions = np.array([[0, -400.0], [0, -400.0], [-1000.0, 0]])
    offsets = np.array([10000.0, -11000.0, 200.0])
    transitions = [eye(2, format="csr")] * 2
    logp, posterior = _score_first_order(variant, emissions, transitions)
    shifted_logp, shifted_posterior = _score_first_order(variant, emissions + offsets[:, None], transitions)
    assert shifted_logp == pytest.approx(logp + offsets.sum(), abs=1e-10)
    np.testing.assert_allclose(shifted_posterior, posterior, atol=1e-10)
    assert logp == pytest.approx(logsumexp(emissions.sum(axis=0)) - np.log(2), abs=1e-10)
    np.testing.assert_allclose(posterior, np.tile([-200.0, 0.0], (3, 1)), atol=1e-10)


@pytest.mark.parametrize("force_log", [False, True])
@pytest.mark.parametrize("stickiness", [0.0, 0.6, 1.0])
@pytest.mark.parametrize("variable", [False, True])
def test_imm_agrees_with_exhaustive_joint_paths(force_log, stickiness, variable, monkeypatch):
    if force_log:
        monkeypatch.setattr(numerics, "_LOG_TINY", 1.0)
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    emissions = np.array([[-0.2, -1.0], [-0.8, -0.1], [-0.3, -1.2]])
    stationary = ss._gaussian_transition_matrix(centers, 0.4, 4.0)
    diffusion = ss._gaussian_transition_matrix(centers, 0.8, 4.0)
    mode_a = ss._mode_transition_matrix(3, stickiness)
    mode_b = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [0.4, 0.5, 0.1]]) if variable else mode_a
    diffusions = [diffusion, csr_matrix([[0.9, 0.4], [0.1, 0.6]]) if variable else diffusion]
    matrices = []
    for modes, spatial_diff in zip([mode_a, mode_b], diffusions, strict=True):
        kernels = [stationary.toarray(), spatial_diff.toarray(), np.full((2, 2), 0.5)]
        joint = np.zeros((6, 6))
        for src in range(3):
            for dst in range(3):
                joint[dst * 2:(dst + 1) * 2, src * 2:(src + 1) * 2] = modes[src, dst] * kernels[dst]
        matrices.append(joint)
    expected_logp, expected = _enumerate_paths(np.tile(emissions, (1, 3)), matrices, np.full(6, 1 / 6))
    if variable:
        logp, trajectory, modes = _score_first_order_imm_variable(
            ss, emissions, centers, stationary_sigma_cm=0.4, diffusion_transitions=diffusions,
            max_step_sigma=4.0, mode_stickiness=stickiness, mode_transitions=[mode_a, mode_b],
        )
    else:
        logp, trajectory, modes = _score_first_order_imm(
            emissions, centers, stationary_sigma_cm=0.4, diffusion_sigma_cm=0.8,
            max_step_sigma=4.0, mode_stickiness=stickiness,
        )
    expected = np.exp(expected).reshape(3, 3, 2)
    assert logp == pytest.approx(expected_logp, abs=1e-12)
    np.testing.assert_allclose(np.exp(trajectory), expected.sum(axis=1), atol=1e-12)
    np.testing.assert_allclose(modes, expected.sum(axis=2), atol=1e-12)


@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_long_imm_retains_fragmented_mode_until_the_final_jump(variable, masked):
    n_time, n_bins = 174, 64
    centers = np.column_stack((10.0 * np.arange(n_bins), np.zeros(n_bins)))
    emissions = np.full((n_time, n_bins), -np.inf)
    emissions[:-1, 0] = 0.0
    emissions[-1, 1] = 0.0
    mask = None
    if masked:
        emissions = np.column_stack((emissions, np.full(n_time, 1000.0)))
        centers = np.vstack((centers, [10000.0, 0.0]))
        mask = np.r_[np.ones(n_bins, dtype=bool), False]
    kwargs = dict(stationary_sigma_cm=0.1, max_step_sigma=4.0, mode_stickiness=1.0, valid_bin_mask=mask)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if variable:
            result = _score_first_order_imm_variable(ss, emissions, centers, diffusion_transitions=eye(len(centers), format="csr"), **kwargs)
        else:
            result = _score_first_order_imm(emissions, centers, diffusion_sigma_cm=0.1, **kwargs)
    logp, trajectory, modes = result
    assert logp == pytest.approx(-np.log(3) - n_time * np.log(n_bins), abs=1e-9)
    np.testing.assert_allclose(modes, [[0.0, 0.0, 1.0]] * n_time, atol=1e-12)
    np.testing.assert_allclose(np.exp(trajectory).sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.exp(trajectory[:-1, 0]), 1.0, atol=1e-12)
    assert np.exp(trajectory[-1, 1]) == pytest.approx(1.0)


@pytest.mark.parametrize("return_trajectory", [False, True])
def test_public_duration_decoder_uses_stable_inference(return_trajectory):
    emissions = LogEmissionTensor(
        log_likelihood=np.array([[0.0, -1000.0], [-1000.0, 0.0]]),
        spike_counts=np.zeros((2, 1), dtype=int), times=np.array([0.01, 0.025]),
        dt=0.02, cell_ids=np.array([1]), n_spikes=0,
        bin_durations=np.array([0.02, 0.01]), transition_durations=np.array([0.015]),
    )
    centers = np.array([[0.0, 0.0], [100.0, 0.0]])
    model = ss.StateSpaceReplayModel(mode="diffusion", config=ss.StateSpaceDecoderConfig(mode="diffusion", diffusion_sigma_cm_sqrt_s=0.1))
    score = model.score(emissions, centers, occupancy_s=np.ones(2), return_trajectory=return_trajectory)
    assert score.log_likelihood == pytest.approx(-1000.0, abs=1e-10)
    np.testing.assert_allclose(np.exp(score.terminal_log_posterior), 0.5, atol=1e-12)
    if return_trajectory:
        np.testing.assert_allclose(np.exp(score.trajectory_log_posterior), 0.5, atol=1e-12)
    else:
        assert score.trajectory_log_posterior is None


@pytest.mark.parametrize("variant", ["fixed", "varying", "duration"])
def test_structurally_impossible_paths_still_raise(variant):
    emissions = np.array([[0.0, -np.inf], [-np.inf, 0.0]])
    with pytest.raises(ValueError, match="no finite predicted mass"):
        _score_first_order(variant, emissions, [eye(2, format="csr")])


def test_log_sparse_matvec_handles_empty_rows_duplicates_and_explicit_zeros():
    # Duplicate entries must be summed in probability space before taking logs.
    matrix = csr_matrix((np.array([0.2, 0.3, 0.0, 0.5]), np.array([0, 0, 1, 1]), np.array([0, 2, 2, 4])), shape=(3, 3))
    kernel = numerics._SpatialTransition(matrix, np.full(3, 1 / 3))
    values = np.array([-1000.0, -900.0, -800.0])
    for backward in (False, True):
        dense = matrix.toarray().T if backward else matrix.toarray()
        with np.errstate(divide="ignore"):
            expected = logsumexp(np.log(dense) + values[None, :], axis=1)
        np.testing.assert_allclose(kernel.apply_log(values, backward=backward), expected, atol=1e-12)


def test_ordinary_case_does_not_use_log_fallback(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("well-conditioned input should retain the fast probability recursion")
    monkeypatch.setattr(numerics, "_log_recursion", unexpected)
    _forward_backward_first_order(np.array([[-0.2, -0.8], [-0.4, -0.1]]), csr_matrix([[0.7, 0.2], [0.3, 0.8]]))
