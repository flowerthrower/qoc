"""
Tests for PSU, SU and TRACEDIFF fidelity types.
"""

import pytest
import qutip as qt
import numpy as np
import collections

try:
    import jax.numpy as jnp
    _jax_available = True
except ImportError:
    _jax_available = False

from qutip_qoc.pulse_optim import optimize_pulses
from qutip_qoc.objective import Objective

Case = collections.namedtuple(
    "Case",
    [
        "objectives",
        "control_parameters",
        "tlist",
        "algorithm_kwargs",
        "optimizer_kwargs",
    ],
)

# --------------------------- System and Control ---------------------------


def sin(t, p):
    return p[0] * np.sin(p[1] * t + p[2])


def grad_sin(t, p, idx):
    if idx == 0:
        return np.sin(p[1] * t + p[2])
    if idx == 1:
        return p[0] * np.cos(p[1] * t + p[2]) * t
    if idx == 2:
        return p[0] * np.cos(p[1] * t + p[2])
    if idx == 3:
        return p[0] * np.cos(p[1] * t + p[2]) * p[1]  # w.r.t. time


p_guess = q_guess = [0, 0, 0]
p_bounds = q_bounds = [(-1, 1), (-1, 1), (-np.pi, np.pi)]

H_d = [0 * qt.sigmaz()]  # NO drift
H_c = [
    [qt.sigmax(), lambda t, p: sin(t, p), {"grad": grad_sin}],
    [qt.sigmay(), lambda t, q: sin(t, q), {"grad": grad_sin}],
]

H = H_d + H_c

# ------------------------------- Objective -------------------------------

# PSU (must not depend on global phase) state to state transfer
initial = qt.basis(2, 0)

PSU_state2state = Case(
    objectives=[Objective(initial, H, (-1j) * initial)],
    control_parameters={
        "p": {"guess": p_guess, "bounds": p_bounds},
        "q": {"guess": q_guess, "bounds": q_bounds},
    },
    tlist=np.linspace(0, 1, 100),
    algorithm_kwargs={"alg": "GOAT", "fid_type": "PSU"},
    optimizer_kwargs={
        "seed": 0,
    },
)

PSU_state2state_dual_annealing = PSU_state2state._replace(
    optimizer_kwargs={"method": "dual_annealing", "niter": 5},
)

# SU (must depend on global phase) state to state transfer
SU_state2state = PSU_state2state._replace(
    objectives=[Objective(initial, H, initial)],
    algorithm_kwargs={"alg": "GOAT", "fid_type": "SU"},
)


# unitary gate synthesis
initial_U = qt.qeye(2)

# PSU (must not depend on global phase)
PSU_unitary = PSU_state2state._replace(
    objectives=[Objective(initial_U, H, (-1j) * initial_U)]
)

# SU (must depend on global phase)
SU_unitary = SU_state2state._replace(objectives=[Objective(initial_U, H, initial_U)])


# TRACEDIFF (must depend on global phase) map synthesis
initial_map = qt.sprepost(qt.qeye(2), qt.qeye(2).dag())

L_d = [qt.liouvillian(0 * qt.sigmaz(), c_ops=[0 * qt.destroy(2)])]  # NO drift
L_c = [
    [qt.liouvillian(qt.sigmax()), lambda t, p: sin(t, p), {"grad": grad_sin}],
    [qt.liouvillian(qt.sigmay()), lambda t, q: sin(t, q), {"grad": grad_sin}],
]

L = L_d + L_c

TRCDIFF_map = PSU_unitary._replace(
    objectives=[Objective(initial_map, L, initial_map)],
    algorithm_kwargs={"alg": "GOAT", "fid_type": "TRACEDIFF"},
)

if _jax_available:
    # ----------------------- System and JAX Control ---------------------


    def sin_jax(t, p):
        return p[0] * jnp.sin(p[1] * t + p[2])


    Hc_jax = [
        [qt.sigmax(), lambda t, p: sin_jax(t, p)],
        [qt.sigmay(), lambda t, q: sin_jax(t, q)],
    ]

    H_jax = H_d + Hc_jax

    # ------------------------------- Objective -------------------------------

    # state to state transfer
    PSU_state2state_jax = PSU_state2state._replace(
        objectives=[Objective(initial, H_jax, (-1j) * initial)],
        algorithm_kwargs={"alg": "JOPT"},
    )

    SU_state2state_jax = SU_state2state._replace(
        objectives=[Objective(initial, H_jax, initial)], algorithm_kwargs={"alg": "JOPT"}
    )


    # unitary gate synthesis
    PSU_unitary_jax = PSU_unitary._replace(
        objectives=[Objective(initial_U, H_jax, (-1j) * initial_U)],
        algorithm_kwargs={"alg": "JOPT"},
    )

    SU_unitary_jax = SU_unitary._replace(
        objectives=[Objective(initial_U, H_jax, initial_U)],
        algorithm_kwargs={"alg": "JOPT"},
    )

    # map synthesis
    Lc_jax = [
        [qt.liouvillian(qt.sigmax()), lambda t, p: sin_jax(t, p)],
        [qt.liouvillian(qt.sigmay()), lambda t, q: sin_jax(t, q)],
    ]
    L_jax = L_d + Lc_jax

    TRCDIFF_map_jax = TRCDIFF_map._replace(
        objectives=[Objective(initial_map, L_jax, initial_map)],
        algorithm_kwargs={"alg": "JOPT", "fid_type": "TRACEDIFF"},
    )

else:
    # jax not available, set these to none so tests will be skipped
    PSU_state2state_jax = None
    SU_state2state_jax = None
    PSU_unitary_jax = None
    SU_unitary_jax = None
    TRCDIFF_map_jax = None


@pytest.fixture(
    params=[
        # GOAT
        pytest.param(PSU_state2state, id="PSU state to state (GOAT)"),
        pytest.param(SU_state2state, id="SU state to state (GOAT)"),
        pytest.param(PSU_unitary, id="PSU unitary gate (GOAT)"),
        pytest.param(SU_unitary, id="SU unitary gate (GOAT)"),
        pytest.param(TRCDIFF_map, id="TRACEDIFF map synthesis (GOAT)"),
        # JOPT
        pytest.param(PSU_state2state_jax, id="PSU state to state (JAX)"),
        pytest.param(SU_state2state_jax, id="SU state to state (JAX)"),
        pytest.param(PSU_unitary_jax, id="PSU unitary gate (JAX)"),
        pytest.param(SU_unitary_jax, id="SU unitary gate (JAX)"),
        pytest.param(TRCDIFF_map_jax, id="TRACEDIFF map synthesis (JAX)"),
        # Options
        pytest.param(PSU_state2state_dual_annealing, id="Dual annealing (GOAT)"),
    ]
)
def tst(request):
    return request.param


def test_optimize_pulses(tst):
    if tst is None:
        pytest.skip("JAX not available")

    result = optimize_pulses(
        tst.objectives,
        tst.control_parameters,
        tst.tlist,
        tst.algorithm_kwargs,
        tst.optimizer_kwargs,
    )
    # initial == target <-> infidelity = 0
    assert np.isclose(result.infidelity, 0.0)
    # initial parameter guess is optimal
    assert np.allclose(result.optimized_params, result.guess_params)
