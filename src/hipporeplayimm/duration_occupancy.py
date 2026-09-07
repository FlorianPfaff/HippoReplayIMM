"""Duration-aware state-space scoring with occupancy-mask support.

The historical duration patch replaced ``StateSpaceReplayModel.score`` but fell
back to the original scalar-dt implementation whenever callers supplied
``occupancy_s``.  Benchmark and decode paths routinely pass occupancy arrays,
even when the occupancy threshold is disabled, so duration metadata could be
silently ignored.  This module installs a final score implementation that keeps
per-transition durations and valid-occupancy masks active at the same time.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.special import logsumexp

from .duration_dynamics import attach_duration_metadata, transition_durations_s
from .evidence_reporting import DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
from .state_space_forward_backward import first_order_forward_backward, first_order_imm_forward_backward
from .state_space_utils import _first_order_imm_content_diagnostics


def apply_duration_occupancy_patch() -> None:
    """Install the duration-aware state-space scorer that honors occupancy masks."""

    import hipporeplayimm.state_space as ss

    if getattr(ss, "_duration_occupancy_patch_applied", False):
        return

    if getattr(ss.StateSpaceReplayModel.score, "_native_duration_occupancy_aware", False):
        ss._duration_occupancy_patch_applied = True
        return

    ss.StateSpaceReplayModel.__duration_occupancy_previous_score__ = ss.StateSpaceReplayModel.score
    ss.StateSpaceReplayModel.score = _score_state_space_duration_with_occupancy
    ss._duration_occupancy_patch_applied = True


def _score_state_space_duration_with_occupancy(
    self,
    emissions,
    bin_centers,
    candidate_indices=None,
    *,
    occupancy_s=None,
    return_trajectory: bool = True,
):
    import hipporeplayimm.state_space as ss

    if emissions.n_time == 0:
        raise ValueError("emissions must contain at least one time bin")
    if emissions.n_bins != bin_centers.shape[0]:
        raise ValueError("emissions.n_bins must match bin_centers rows")
    assert self.config is not None

    valid_bin_mask = ss._valid_bin_mask_from_occupancy(
        occupancy_s,
        self.config.valid_occupancy_threshold_s,
        emissions.n_bins,
    )
    durations = transition_durations_s(emissions)
    attach_duration_metadata(emissions)

    extra: dict[str, float | int | str] = {}
    terminal = None
    if self.mode == "stationary":
        logp, trajectory = ss._score_stationary(emissions, valid_bin_mask=valid_bin_mask)
        transition_sigma_cm = 0.0
    elif self.mode in {"fragmented", "jump"}:
        logp, trajectory = ss._score_fragmented(emissions, valid_bin_mask=valid_bin_mask)
        transition_sigma_cm = float("inf")
    elif self.mode == "diffusion":
        diffusion_sigmas = _per_transition_sigmas(
            self.config.diffusion_sigma_cm_sqrt_s,
            durations,
        )
        transition_sigma_cm = _representative_sigma(
            self.config.diffusion_sigma_cm_sqrt_s,
            durations,
            float(emissions.dt),
        )
        transitions = [
            ss._gaussian_transition_matrix(
                bin_centers,
                float(sigma),
                self.config.max_step_sigma,
                valid_bin_mask=valid_bin_mask,
            )
            for sigma in diffusion_sigmas
        ]
        logp, trajectory = _forward_backward_variable(
            ss,
            emissions.log_likelihood,
            transitions,
            valid_bin_mask=valid_bin_mask,
        )
    elif self.mode == "first-order-imm":
        diffusion_sigmas = _per_transition_sigmas(
            self.config.diffusion_sigma_cm_sqrt_s,
            durations,
        )
        transition_sigma_cm = _representative_sigma(
            self.config.diffusion_sigma_cm_sqrt_s,
            durations,
            float(emissions.dt),
        )
        diffusion_transitions = [
            ss._gaussian_transition_matrix(
                bin_centers,
                float(sigma),
                self.config.max_step_sigma,
                valid_bin_mask=valid_bin_mask,
            )
            for sigma in diffusion_sigmas
        ]
        logp, trajectory, mode_post = _score_first_order_imm_variable(
            ss,
            emissions.log_likelihood,
            bin_centers,
            stationary_sigma_cm=self.config.stationary_sigma_cm,
            diffusion_transitions=diffusion_transitions,
            max_step_sigma=self.config.max_step_sigma,
            mode_stickiness=self.config.imm_mode_stickiness,
            mode_transitions=_mode_transition_matrices(
                ss,
                3,
                self.config.imm_mode_stickiness,
                self.config.imm_switch_tau_s,
                durations,
            ),
            valid_bin_mask=valid_bin_mask,
        )
        names = ("stationary", "diffusion", "fragmented")
        event_mode_mass = mode_post.mean(axis=0)
        extra = {
            f"state_space_mode_{name}_terminal_probability": float(mode_post[-1, idx])
            for idx, name in enumerate(names)
        }
        extra.update(
            {
                f"state_space_mode_{name}_event_probability": float(event_mode_mass[idx])
                for idx, name in enumerate(names)
            }
        )
        extra.update(
            {
                "state_space_imm_modes": ",".join(names),
                "state_space_imm_evidence_support": "exact_full_grid",
                "state_space_imm_nonstationary_terminal_probability": float(mode_post[-1, 1:].sum()),
                "state_space_imm_nonstationary_event_probability": float(event_mode_mass[1:].sum()),
                "state_space_imm_mean_mode_entropy": ss._mean_entropy(ss._as_log_probs(mode_post)),
            }
        )
        extra.update(
            _first_order_imm_content_diagnostics(
                mode_post,
                trajectory,
                bin_centers,
                float(emissions.dt),
            )
        )
    elif self.mode == "trajectory-imm-exact-sparse":
        from .state_space_trajectory_imm import _score_trajectory_imm_exact_sparse

        logp, trajectory, terminal, mode_post, trajectory_imm_extra = _score_trajectory_imm_exact_sparse(
            emissions,
            bin_centers,
            self.config,
            durations,
            valid_bin_mask=valid_bin_mask,
            return_trajectory=return_trajectory,
        )
        transition_sigma_cm = float(
            trajectory_imm_extra["state_space_trajectory_imm_diffusion_transition_sigma_cm"]
        )
        names = ("stationary", "diffusion", "fragmented", "momentum_exact_sparse")
        extra = {
            f"state_space_mode_{name}_terminal_probability": float(mode_post[-1, idx])
            for idx, name in enumerate(names)
        } if mode_post is not None else {}
        extra.update(trajectory_imm_extra)
    elif self.mode == "displacement-momentum":
        from .state_space_displacement_momentum import _score_displacement_momentum_exact

        logp, trajectory, terminal, displacement_post, displacement_extra = _score_displacement_momentum_exact(
            emissions,
            bin_centers,
            self.config,
            durations,
            valid_bin_mask=valid_bin_mask,
            return_trajectory=return_trajectory,
        )
        transition_sigma_cm = float(displacement_extra["state_space_displacement_transition_sigma_cm"])
        extra = displacement_extra
    elif self.mode == "displacement-imm":
        from .state_space_displacement_imm import _DISPLACEMENT_IMM_MODES, _score_displacement_imm_exact

        logp, trajectory, terminal, mode_post, displacement_post, displacement_extra = _score_displacement_imm_exact(
            emissions,
            bin_centers,
            self.config,
            durations,
            valid_bin_mask=valid_bin_mask,
            return_trajectory=return_trajectory,
        )
        transition_sigma_cm = float(
            displacement_extra["state_space_displacement_imm_transition_sigma_cm"]
        )
        extra = {
            f"state_space_mode_{name.replace('-', '_')}_terminal_probability": float(mode_post[-1, idx])
            for idx, name in enumerate(_DISPLACEMENT_IMM_MODES)
        }
        extra.update(displacement_extra)
    elif self.mode == "momentum-exact-sparse":
        from .state_space_sparse_momentum import _score_sparse_momentum_exact

        logp, trajectory, terminal, sparse_extra = _score_sparse_momentum_exact(
            emissions,
            bin_centers,
            self.config,
            durations,
            valid_bin_mask=valid_bin_mask,
            return_trajectory=return_trajectory,
        )
        transition_sigma_cm = float(
            sparse_extra["state_space_momentum_transition_sigma_cm"]
        )
        extra = sparse_extra
    elif self.mode == "momentum":
        candidates = _duration_candidates(ss, self, emissions, bin_centers, candidate_indices, valid_bin_mask)
        momentum_sigmas = _per_transition_sigmas(
            self.config.momentum_sigma_cm_sqrt_s,
            durations,
        )
        transition_sigma_cm = _representative_sigma(
            self.config.momentum_sigma_cm_sqrt_s,
            durations,
            float(emissions.dt),
        )
        initial_sigma = _per_bin_sigma(
            self.config.momentum_initial_sigma_cm_sqrt_s,
            durations[0] if len(durations) else float(emissions.dt),
        )
        decays = _duration_adjusted_decays(self.config, durations, float(emissions.dt))
        time_scales = _time_scales(durations)
        logp, trajectory, masses = _score_momentum_duration(
            ss,
            emissions,
            bin_centers,
            candidates,
            sigmas_cm=momentum_sigmas,
            initial_sigma_cm=initial_sigma,
            velocity_decays=decays,
            time_scales=time_scales,
            valid_bin_mask=valid_bin_mask,
        )
        evidence_support = _path_model_evidence_support(
            ss,
            candidates,
            emissions.n_bins,
            valid_bin_mask,
            emissions.n_time,
        )
        candidate_support_label = "full_grid" if evidence_support == "exact_full_grid" else (
            "derived" if candidate_indices is None else "provided"
        )
        extra = {
            "mean_candidate_log_mass": float(np.mean(masses)),
            "min_candidate_log_mass": float(np.min(masses)),
            "mean_candidate_count": float(np.mean([len(curr) for curr in candidates])),
            "state_space_momentum_candidate_support": candidate_support_label,
            "state_space_momentum_trajectory_posterior": "smoothed_pair_marginal",
            "state_space_momentum_evidence_support": evidence_support,
            **ss._candidate_support_config_diagnostics("state_space_momentum", self.config),
            "state_space_momentum_candidate_selection": (
                "provided" if candidate_indices is not None else ss._candidate_selection_label(self.config)
            ),
        }
        if emissions.n_time == 1:
            extra.update(_single_bin_degenerate_diagnostics("state_space_momentum"))
    elif self.mode == "imm":
        candidates = _duration_candidates(ss, self, emissions, bin_centers, candidate_indices, valid_bin_mask)
        diffusion_sigmas = _per_transition_sigmas(
            self.config.diffusion_sigma_cm_sqrt_s,
            durations,
        )
        transition_sigma_cm = _representative_sigma(
            self.config.diffusion_sigma_cm_sqrt_s,
            durations,
            float(emissions.dt),
        )
        momentum_sigmas = _per_transition_sigmas(
            self.config.momentum_sigma_cm_sqrt_s,
            durations,
        )
        momentum_transition_sigma_cm = _representative_sigma(
            self.config.momentum_sigma_cm_sqrt_s,
            durations,
            float(emissions.dt),
        )
        initial_sigma = _per_bin_sigma(
            self.config.momentum_initial_sigma_cm_sqrt_s,
            durations[0] if len(durations) else float(emissions.dt),
        )
        decays = _duration_adjusted_decays(self.config, durations, float(emissions.dt))
        time_scales = _time_scales(durations)
        logp, trajectory, mode_post, masses = _score_imm_duration(
            ss,
            emissions,
            bin_centers,
            candidates,
            stationary_sigma_cm=self.config.stationary_sigma_cm,
            diffusion_sigmas_cm=diffusion_sigmas,
            momentum_sigmas_cm=momentum_sigmas,
            initial_momentum_sigma_cm=initial_sigma,
            velocity_decays=decays,
            time_scales=time_scales,
            mode_stickiness=self.config.imm_mode_stickiness,
            mode_transitions=_mode_transition_matrices(
                ss,
                4,
                self.config.imm_mode_stickiness,
                self.config.imm_switch_tau_s,
                durations,
            ),
            valid_bin_mask=valid_bin_mask,
        )
        evidence_support = _path_model_evidence_support(
            ss,
            candidates,
            emissions.n_bins,
            valid_bin_mask,
            emissions.n_time,
        )
        candidate_support_label = "full_grid" if evidence_support == "exact_full_grid" else (
            "derived" if candidate_indices is None else "provided"
        )
        names = ("stationary", "diffusion", "momentum", "jump")
        extra = {
            f"state_space_mode_{name}_terminal_probability": float(mode_post[-1, idx])
            for idx, name in enumerate(names)
        }
        extra.update(
            {
                "mean_candidate_log_mass": float(np.mean(masses)),
                "min_candidate_log_mass": float(np.min(masses)),
                "mean_candidate_count": float(np.mean([len(curr) for curr in candidates])),
                "state_space_imm_modes": ",".join(names),
                "state_space_imm_candidate_support": candidate_support_label,
                "state_space_imm_trajectory_posterior": "smoothed_pair_marginal",
                "state_space_imm_evidence_support": evidence_support,
                **ss._candidate_support_config_diagnostics("state_space_imm", self.config),
                "state_space_imm_candidate_selection": (
                    "provided" if candidate_indices is not None else ss._candidate_selection_label(self.config)
                ),
                "state_space_momentum_transition_sigma_cm": float(momentum_transition_sigma_cm),
                "state_space_momentum_initial_transition_sigma_cm": float(initial_sigma),
            }
        )
        if emissions.n_time == 1:
            extra.update(_single_bin_degenerate_diagnostics("state_space_imm"))
    else:  # pragma: no cover - StateSpaceReplayModel.__post_init__ validates this.
        raise ValueError(f"Unsupported state-space mode: {self.mode}")

    trajectory_to_return = None
    if trajectory is not None:
        terminal = trajectory[-1]
        mean_trajectory_entropy = ss._mean_entropy(trajectory)
        if return_trajectory:
            trajectory_to_return = trajectory
            trajectory_available = 1
        else:
            trajectory_available = 0
    elif terminal is not None:
        trajectory_available = 0
        mean_trajectory_entropy = float("nan")
    else:
        raise ValueError("state-space scorer did not return a trajectory or terminal posterior")

    diagnostics: dict[str, float | int | str] = {
        "state_space_mode": str(self.mode),
        "state_space_time_bin_s": float(emissions.dt),
        "state_space_transition_durations": ",".join(f"{duration:.12g}" for duration in durations),
        "state_space_trajectory_posterior": trajectory_available,
        "state_space_trajectory_time_bins": int(emissions.n_time),
        "state_space_stationary_sigma_cm": float(self.config.stationary_sigma_cm),
        "state_space_diffusion_sigma_cm_sqrt_s": float(self.config.diffusion_sigma_cm_sqrt_s),
        "state_space_max_step_sigma": float(self.config.max_step_sigma),
        "state_space_imm_mode_stickiness": float(self.config.imm_mode_stickiness),
        "state_space_imm_switch_tau_s": float(self.config.imm_switch_tau_s),
        "state_space_momentum_sigma_cm_sqrt_s": float(self.config.momentum_sigma_cm_sqrt_s),
        "state_space_momentum_initial_sigma_cm_sqrt_s": float(self.config.momentum_initial_sigma_cm_sqrt_s),
        "state_space_momentum_velocity_decay": float(self.config.momentum_velocity_decay),
        "state_space_displacement_radius_bins": int(getattr(self.config, "displacement_radius_bins", 0)),
        "state_space_displacement_position_sigma_cm": float(getattr(self.config, "displacement_position_sigma_cm", 0.0)),
        "state_space_displacement_transition_sigma_cm_sqrt_s": float(getattr(self.config, "displacement_transition_sigma_cm_sqrt_s", 0.0)),
        "state_space_displacement_prior_sigma_cm": float(getattr(self.config, "displacement_prior_sigma_cm", 0.0)),
        "state_space_momentum_velocity_decay_tau_s": float(self.config.momentum_velocity_decay_tau_s),
        "state_space_valid_occupancy_threshold_s": float(self.config.valid_occupancy_threshold_s),
        "state_space_transition_sigma_cm": float(transition_sigma_cm),
        "mean_trajectory_posterior_entropy": mean_trajectory_entropy,
        **extra,
    }
    if valid_bin_mask is not None:
        diagnostics.update(
            {
                "state_space_valid_bin_count": int(np.sum(valid_bin_mask)),
                "state_space_valid_bin_fraction": float(np.mean(valid_bin_mask)),
            }
        )
    diagnostics.update(ss._posterior_diagnostics(terminal, bin_centers))
    return ss.EventScore(
        str(self.name),
        float(logp),
        emissions.n_time,
        emissions.n_spikes,
        diagnostics=diagnostics,
        terminal_log_posterior=terminal,
        trajectory_log_posterior=trajectory_to_return,
    )


def _path_model_evidence_support(
    ss,
    candidates,
    n_bins: int,
    valid_bin_mask: np.ndarray | None,
    n_time: int,
) -> str:
    """Classify evidence support for candidate path models."""

    if int(n_time) <= 1:
        return DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
    return ss._candidate_evidence_support_label(candidates, n_bins, valid_bin_mask)


def _single_bin_degenerate_diagnostics(prefix: str) -> dict[str, int | str]:
    return {
        f"{prefix}_degenerate_reason": "single_time_bin_fragmented_marginal",
        f"{prefix}_required_min_time_bins": 2,
    }


def _duration_candidates(ss, model, emissions, bin_centers, candidate_indices, valid_bin_mask):
    if candidate_indices is None:
        candidate_emissions = _candidate_selection_emissions(emissions, valid_bin_mask)
        candidates = model.candidate_indices(
            candidate_emissions,
            bin_centers,
            valid_bin_mask=valid_bin_mask,
        )
    else:
        candidates = candidate_indices
    candidates = ss._validate_candidate_indices(candidates, emissions.n_time, emissions.n_bins)
    candidates = ss._restrict_candidates_to_valid_bins(
        candidates,
        emissions.log_likelihood,
        valid_bin_mask,
    )
    return ss._validate_candidate_indices(candidates, emissions.n_time, emissions.n_bins)


def _candidate_selection_emissions(emissions, valid_bin_mask):
    """Return emissions whose invalid occupancy bins cannot win internal beams."""

    if valid_bin_mask is None:
        return emissions

    mask = np.asarray(valid_bin_mask, dtype=bool)
    if mask.shape != (emissions.n_bins,):
        raise ValueError("valid_bin_mask must contain one boolean value per spatial bin")
    if not np.any(mask):
        raise ValueError("valid_bin_mask must contain at least one valid spatial bin")
    if bool(np.all(mask)):
        return emissions

    log_likelihood = np.asarray(emissions.log_likelihood, dtype=float).copy()
    log_likelihood[:, ~mask] = -np.inf
    return replace(emissions, log_likelihood=log_likelihood)


def _candidate_evidence_support(
    candidates,
    n_bins: int,
    valid_bin_mask: np.ndarray | None,
) -> str:
    """Classify candidate recursions as exact only when support is the full grid.

    Momentum and four-mode IMM recursions are lower-bound evidences when their
    second-order path support is pruned.  When every time bin contains every
    spatial state allowed by the occupancy mask, the same recursions are exact
    full-grid dynamic programs and should be allowed into comparable-evidence
    summaries.
    """

    if valid_bin_mask is None:
        expected = int(n_bins)
        for current in candidates:
            if np.unique(np.asarray(current, dtype=int)).size != expected:
                return "truncated_full_grid"
        return "exact_full_grid"

    valid = np.flatnonzero(np.asarray(valid_bin_mask, dtype=bool))
    for current in candidates:
        arr = np.unique(np.asarray(current, dtype=int))
        if arr.size != valid.size or not np.array_equal(np.sort(arr), valid):
            return "truncated_full_grid"
    return "exact_full_grid"


def _per_bin_sigma(sigma_cm_sqrt_s: float, dt_s: float) -> float:
    sigma = float(sigma_cm_sqrt_s)
    dt = float(dt_s)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_cm_sqrt_s must be finite and positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    return max(sigma * np.sqrt(dt), np.finfo(float).eps)


def _per_transition_sigmas(sigma_cm_sqrt_s: float, durations: np.ndarray) -> np.ndarray:
    return np.asarray([_per_bin_sigma(sigma_cm_sqrt_s, duration) for duration in durations], dtype=float)


def _representative_sigma(sigma_cm_sqrt_s: float, durations: np.ndarray, fallback_dt: float) -> float:
    dt = float(np.median(durations)) if len(durations) else float(fallback_dt)
    return _per_bin_sigma(sigma_cm_sqrt_s, dt)


def _duration_adjusted_decays(
    config_or_decay: object,
    durations: np.ndarray,
    reference_dt: float,
) -> np.ndarray:
    reference_dt = float(reference_dt)
    if not np.isfinite(reference_dt) or reference_dt <= 0.0:
        raise ValueError("reference dt must be finite and positive")

    durations = np.asarray(durations, dtype=float)
    if durations.size == 0:
        return np.empty(0, dtype=float)
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("transition durations must be finite and positive")
    if hasattr(config_or_decay, "momentum_velocity_decay"):
        tau_s = float(getattr(config_or_decay, "momentum_velocity_decay_tau_s", 0.0))
        if not np.isfinite(tau_s) or tau_s < 0.0:
            raise ValueError("momentum_velocity_decay_tau_s must be finite and nonnegative")
        if tau_s > 0.0:
            return np.asarray(np.exp(-durations / tau_s), dtype=float)
        decay = float(getattr(config_or_decay, "momentum_velocity_decay"))
    else:
        decay = float(config_or_decay)

    if not np.isfinite(decay) or decay < 0.0:
        raise ValueError("momentum_velocity_decay must be finite and nonnegative")
    return np.asarray([decay ** (float(duration) / reference_dt) for duration in durations], dtype=float)


def _mode_transition_matrices(
    ss,
    n_modes: int,
    mode_stickiness: float,
    imm_switch_tau_s: float,
    durations: np.ndarray,
) -> list[np.ndarray]:
    """Return one IMM mode-transition matrix per adjacent time-bin pair.

    ``mode_stickiness`` remains the legacy per-transition probability.  When
    ``imm_switch_tau_s`` is positive, the stickiness is derived from each actual
    transition duration as ``exp(-duration / tau)`` so a final partial replay bin
    or other variable-duration bin does not use the wrong switching rate.
    """

    durations = np.asarray(durations, dtype=float)
    tau_s = float(imm_switch_tau_s)
    if not np.isfinite(tau_s) or tau_s < 0.0:
        raise ValueError("imm_switch_tau_s must be finite and nonnegative")
    if tau_s == 0.0:
        return _resolve_mode_transitions(
            ss,
            int(n_modes),
            float(mode_stickiness),
            None,
            int(durations.size),
        )
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("transition durations must be finite and positive")
    return [
        ss._mode_transition_matrix(
            int(n_modes),
            float(np.exp(-float(duration) / tau_s)),
        )
        for duration in durations
    ]


def _resolve_mode_transitions(
    ss,
    n_modes: int,
    mode_stickiness: float,
    mode_transitions,
    n_transitions: int,
) -> list[np.ndarray]:
    """Validate provided IMM mode transitions or build a constant sequence."""

    if mode_transitions is None:
        transition = ss._mode_transition_matrix(int(n_modes), float(mode_stickiness))
        return [transition for _ in range(int(n_transitions))]
    if len(mode_transitions) != int(n_transitions):
        raise ValueError("mode_transitions must contain one matrix per transition")
    resolved = [np.asarray(matrix, dtype=float) for matrix in mode_transitions]
    expected_shape = (int(n_modes), int(n_modes))
    for matrix in resolved:
        if matrix.shape != expected_shape:
            raise ValueError("mode transition matrices must be square with one row and column per mode")
    return resolved


def _duration_adjusted_decays_from_config(config, durations: np.ndarray, reference_dt: float) -> np.ndarray:
    """Return transition-specific momentum velocity decays.

    Historically the duration-aware scorer scaled a per-bin decay by
    ``duration / reference_dt``.  That is still the backwards-compatible path.
    When ``momentum_velocity_decay_tau_s`` is positive, use the physical-time
    decay ``exp(-duration / tau)`` instead so the same setting is meaningful for
    1, 2, 3, or 5 ms replay bins.
    """

    tau_s = float(getattr(config, "momentum_velocity_decay_tau_s", 0.0))
    if not np.isfinite(tau_s) or tau_s < 0.0:
        raise ValueError("momentum_velocity_decay_tau_s must be finite and nonnegative")
    if tau_s == 0.0:
        return _duration_adjusted_decays(
            float(getattr(config, "momentum_velocity_decay", 0.95)),
            durations,
            reference_dt,
        )
    durations = np.asarray(durations, dtype=float)
    if np.any(durations <= 0.0) or not np.all(np.isfinite(durations)):
        raise ValueError("transition durations must be finite and positive")
    return np.exp(-durations / tau_s)


def _time_scales(durations: np.ndarray) -> np.ndarray:
    durations = np.asarray(durations, dtype=float)
    if durations.size == 0:
        return np.empty(0, dtype=float)
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("transition durations must be finite and positive")
    scales = np.ones_like(durations, dtype=float)
    if len(durations) > 1:
        scales[1:] = durations[1:] / durations[:-1]
    return scales


def _transition_at(transitions, transition_index: int):
    return transitions[transition_index] if isinstance(transitions, (list, tuple)) else transitions


def _forward_backward_variable(ss, log_likelihood, transitions, *, valid_bin_mask=None):
    return first_order_forward_backward(log_likelihood, transitions, valid_bin_mask)


def _score_first_order_imm_variable(
    ss,
    log_likelihood,
    bin_centers,
    *,
    stationary_sigma_cm,
    diffusion_transitions,
    max_step_sigma,
    mode_stickiness,
    mode_transitions=None,
    valid_bin_mask=None,
):
    mode_transitions = _resolve_mode_transitions(
        ss, 3, mode_stickiness, mode_transitions, max(len(log_likelihood) - 1, 0),
    )
    stationary = ss._gaussian_transition_matrix(
        bin_centers, stationary_sigma_cm, max_step_sigma, valid_bin_mask=valid_bin_mask,
    )
    return first_order_imm_forward_backward(
        log_likelihood, stationary, diffusion_transitions, mode_transitions, valid_bin_mask,
    )


def _score_momentum_duration(
    ss,
    emissions,
    bin_centers,
    candidates,
    *,
    sigmas_cm,
    initial_sigma_cm,
    velocity_decays,
    time_scales,
    valid_bin_mask=None,
):
    if emissions.n_time == 1:
        logp, trajectory = ss._score_fragmented(emissions, valid_bin_mask=valid_bin_mask)
        return logp, trajectory, [0.0]

    masses = ss._candidate_log_masses(emissions.log_likelihood, candidates)
    pair = ss._init_pair_log_alpha(
        emissions.log_likelihood,
        candidates[0],
        candidates[1],
        bin_centers,
        sigma_cm=float(initial_sigma_cm),
        valid_bin_mask=valid_bin_mask,
    )
    pair_alphas = [pair]
    for time_index in range(2, emissions.n_time):
        transition_index = time_index - 1
        pair = ss._advance_momentum_pair(
            pair,
            candidates[time_index - 2],
            candidates[time_index - 1],
            candidates[time_index],
            emissions.log_likelihood[time_index, candidates[time_index]],
            bin_centers,
            sigma_cm=float(sigmas_cm[transition_index]),
            velocity_decay=float(velocity_decays[transition_index]) * float(time_scales[transition_index]),
            valid_bin_mask=valid_bin_mask,
        )
        pair_alphas.append(pair)

    logp = float(logsumexp(pair_alphas[-1]))
    pair_betas = [np.zeros_like(pair_alphas[-1]) for _ in pair_alphas]
    for pair_index in range(len(pair_alphas) - 2, -1, -1):
        transition_index = pair_index + 1
        curr_time = pair_index + 2
        pair_betas[pair_index] = ss._backward_momentum_pair(
            pair_betas[pair_index + 1],
            candidates[pair_index],
            candidates[pair_index + 1],
            candidates[curr_time],
            emissions.log_likelihood[curr_time, candidates[curr_time]],
            bin_centers,
            sigma_cm=float(sigmas_cm[transition_index]),
            velocity_decay=float(velocity_decays[transition_index]) * float(time_scales[transition_index]),
            valid_bin_mask=valid_bin_mask,
        )

    trajectory = np.full((emissions.n_time, emissions.n_bins), ss.LOG_ZERO, dtype=float)
    for pair_index, (alpha, beta) in enumerate(zip(pair_alphas, pair_betas, strict=True)):
        pair_log_posterior = alpha + beta - logp
        if pair_index == 0:
            trajectory[0, candidates[0]] = logsumexp(pair_log_posterior, axis=1)
        trajectory[pair_index + 1, candidates[pair_index + 1]] = logsumexp(pair_log_posterior, axis=0)
    for time_index in range(emissions.n_time):
        trajectory[time_index] -= logsumexp(trajectory[time_index])
    return logp, trajectory, masses


def _score_imm_duration(
    ss,
    emissions,
    bin_centers,
    candidates,
    *,
    stationary_sigma_cm,
    diffusion_sigmas_cm,
    momentum_sigmas_cm,
    initial_momentum_sigma_cm,
    velocity_decays,
    time_scales,
    mode_stickiness,
    mode_transitions=None,
    valid_bin_mask=None,
):
    modes = ("stationary", "diffusion", "momentum", "jump")
    if emissions.n_time == 1:
        logp, trajectory = ss._score_fragmented(emissions, valid_bin_mask=valid_bin_mask)
        mode_posterior = np.full((1, len(modes)), 1.0 / len(modes), dtype=float)
        return logp, trajectory, mode_posterior, [0.0]

    mode_transitions = _resolve_mode_transitions(
        ss,
        len(modes),
        mode_stickiness,
        mode_transitions,
        max(emissions.n_time - 1, 0),
    )
    masses = ss._candidate_log_masses(emissions.log_likelihood, candidates)

    by_mode = [
        ss._init_imm_pair_log_alpha(
            emissions.log_likelihood,
            candidates[0],
            candidates[1],
            bin_centers,
            mode=mode,
            stationary_sigma_cm=stationary_sigma_cm,
            diffusion_sigma_cm=float(diffusion_sigmas_cm[0]),
            momentum_initial_sigma_cm=float(initial_momentum_sigma_cm),
            valid_bin_mask=valid_bin_mask,
        )
        for mode in modes
    ]
    log_pair = np.stack(by_mode, axis=0) - np.log(len(modes))
    pair_alphas = [log_pair]

    for time_index in range(2, emissions.n_time):
        transition_index = time_index - 1
        next_alpha = []
        mode_transition = mode_transitions[transition_index]
        with np.errstate(divide="ignore"):
            log_mode_transition = np.log(mode_transition)
        for dst_mode_index, dst_mode in enumerate(modes):
            mixed_prev = logsumexp(
                log_pair + log_mode_transition[:, dst_mode_index][:, None, None],
                axis=0,
            )
            next_alpha.append(
                ss._advance_imm_pair_log_alpha(
                    mixed_prev,
                    candidates[time_index - 2],
                    candidates[time_index - 1],
                    candidates[time_index],
                    emissions.log_likelihood[time_index, candidates[time_index]],
                    bin_centers,
                    mode=dst_mode,
                    stationary_sigma_cm=stationary_sigma_cm,
                    diffusion_sigma_cm=float(diffusion_sigmas_cm[transition_index]),
                    momentum_sigma_cm=float(momentum_sigmas_cm[transition_index]),
                    velocity_decay=float(velocity_decays[transition_index]) * float(time_scales[transition_index]),
                    valid_bin_mask=valid_bin_mask,
                )
            )
        log_pair = np.stack(next_alpha, axis=0)
        pair_alphas.append(log_pair)

    logp = float(logsumexp(pair_alphas[-1]))
    pair_betas = [np.zeros_like(pair_alphas[-1]) for _ in pair_alphas]
    for pair_index in range(len(pair_alphas) - 2, -1, -1):
        transition_index = pair_index + 1
        curr_time = pair_index + 2
        mode_transition = mode_transitions[transition_index]
        pair_betas[pair_index] = ss._backward_imm_pair(
            pair_betas[pair_index + 1],
            candidates[pair_index],
            candidates[pair_index + 1],
            candidates[curr_time],
            emissions.log_likelihood[curr_time, candidates[curr_time]],
            bin_centers,
            modes=modes,
            mode_transition=mode_transition,
            stationary_sigma_cm=stationary_sigma_cm,
            diffusion_sigma_cm=float(diffusion_sigmas_cm[transition_index]),
            momentum_sigma_cm=float(momentum_sigmas_cm[transition_index]),
            velocity_decay=float(velocity_decays[transition_index]) * float(time_scales[transition_index]),
            valid_bin_mask=valid_bin_mask,
        )

    trajectory = np.full((emissions.n_time, emissions.n_bins), ss.LOG_ZERO, dtype=float)
    mode_log_posterior = np.full((emissions.n_time, len(modes)), ss.LOG_ZERO, dtype=float)
    for pair_index, (alpha, beta) in enumerate(zip(pair_alphas, pair_betas, strict=True)):
        pair_log_posterior = alpha + beta - logp
        if pair_index == 0:
            trajectory[0, candidates[0]] = logsumexp(pair_log_posterior, axis=(0, 2))
            mode_log_posterior[0] = logsumexp(pair_log_posterior, axis=(1, 2))
        trajectory[pair_index + 1, candidates[pair_index + 1]] = logsumexp(pair_log_posterior, axis=(0, 1))
        mode_log_posterior[pair_index + 1] = logsumexp(pair_log_posterior, axis=(1, 2))
    for time_index in range(emissions.n_time):
        trajectory[time_index] -= logsumexp(trajectory[time_index])
    mode_log_posterior -= logsumexp(mode_log_posterior, axis=1)[:, None]
    return logp, trajectory, np.exp(mode_log_posterior), masses


def _uniform_probabilities(n_bins: int, valid_bin_mask: np.ndarray | None = None) -> np.ndarray:
    if valid_bin_mask is None:
        return np.full(n_bins, 1.0 / n_bins, dtype=float)
    mask = np.asarray(valid_bin_mask, dtype=bool)
    if mask.shape != (n_bins,):
        raise ValueError("valid_bin_mask must contain one boolean value per spatial bin")
    if not np.any(mask):
        raise ValueError("valid_bin_mask must contain at least one valid spatial bin")
    probabilities = np.zeros(n_bins, dtype=float)
    probabilities[mask] = 1.0 / int(np.sum(mask))
    return probabilities


def _uniform_backward(values: np.ndarray, valid_bin_mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if valid_bin_mask is None:
        return np.full(values.shape, float(values.sum()) / values.shape[0], dtype=float)
    mask = np.asarray(valid_bin_mask, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError("valid_bin_mask must contain one boolean value per spatial bin")
    if not np.any(mask):
        raise ValueError("valid_bin_mask must contain at least one valid spatial bin")
    out = np.zeros(values.shape, dtype=float)
    out[mask] = float(values[mask].sum()) / int(np.sum(mask))
    return out
