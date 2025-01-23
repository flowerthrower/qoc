"""
Test all algorithms to return a proper Result object.
"""

import pytest
import qutip as qt
import numpy as np
import collections

try:
    import jax
    import jax.numpy as jnp
    _jax_available = True
except ImportError:
    _jax_available = False

try:
    import gymnasium
    import stable_baselines3
    _rl_available = True
except ImportError:
    _rl_available = False

from qutip_qoc.pulse_optim import optimize_pulses
from qutip_qoc.objective import Objective
from qutip_qoc._time import _TimeInterval
from qutip_qoc.result import Result

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


p_guess = q_guess = r_guess = [1, 1, 0]
p_bounds = q_bounds = r_bounds = [(-1, 1), (-1, 1), (-np.pi, np.pi)]

H_d = [qt.sigmaz()]
H_c = [
    [qt.sigmax(), lambda t, p: sin(t, p), {"grad": grad_sin}],
    [qt.sigmay(), lambda t, q: sin(t, q), {"grad": grad_sin}],
    [qt.sigmay(), lambda t, q: sin(t, q), {"grad": grad_sin}],
]

H = H_d + H_c

# ------------------------------- Objective -------------------------------

# state to state transfer
initial = qt.basis(2, 0)
target = qt.basis(2, 1)

state2state_goat = Case(
    objectives=[Objective(initial, H, target)],
    control_parameters={
        "p": {"guess": p_guess, "bounds": p_bounds},
        "q": {"guess": q_guess, "bounds": q_bounds},
        "r": {"guess": r_guess, "bounds": r_bounds},
    },
    tlist=np.linspace(0, 10, 100),
    algorithm_kwargs={
        "alg": "GOAT",
        "fid_err_targ": 0.01,
    },
    optimizer_kwargs={
        "seed": 0,
    },
)
# ----------------------- CRAB --------------------

# state to state transfer with initial parameters (not amplitudes)
state2state_param_crab = state2state_goat._replace(
    objectives=[Objective(initial, H, target)],
    algorithm_kwargs={"alg": "CRAB", "fid_err_targ": 0.01},
)

if _jax_available:
    # ----------------------- JAX ---------------------


    def sin_jax(t, p):
        return p[0] * jnp.sin(p[1] * t + p[2])


    @jax.jit
    def sin_x_jax(t, p, **kwargs):
        return sin_jax(t, p)


    @jax.jit
    def sin_y_jax(t, q, **kwargs):
        return sin_jax(t, q)


    @jax.jit
    def sin_z_jax(t, r, **kwargs):
        return sin_jax(t, r)


    Hc_jax = [[qt.sigmax(), sin_x_jax], [qt.sigmay(), sin_y_jax], [qt.sigmaz(), sin_z_jax]]

    H_jax = H_d + Hc_jax

    # state to state transfer
    state2state_jax = state2state_goat._replace(
        objectives=[Objective(initial, H_jax, target)],
        algorithm_kwargs={"alg": "JOPT", "fid_err_targ": 0.01},
    )

else:
    state2state_jax = None


# ------------------- discrete CRAB / GRAPE  control ------------------------

n_tslots, evo_time = 100, 10
disc_interval = np.linspace(0, evo_time, n_tslots)

p_disc = q_disc = r_disc = np.zeros(n_tslots)
p_bound = q_bound = r_bound = (-1, 1)

Hc_disc = [[qt.sigmax(), p_guess], [qt.sigmay(), q_guess], [qt.sigmaz(), r_guess]]

H_disc = H_d + Hc_disc


state2state_grape = state2state_goat._replace(
    objectives=[Objective(initial, H_disc, target)],
    control_parameters={
        "p": {"guess": p_disc, "bounds": p_bound},
        "q": {"guess": q_disc, "bounds": q_bound},
        "r": {"guess": r_disc, "bounds": r_bound},
    },
    tlist=disc_interval,
    algorithm_kwargs={"alg": "GRAPE", "fid_err_targ": 0.01},
)

state2state_crab = state2state_goat._replace(
    objectives=[Objective(initial, H_disc, target)],
    control_parameters={
        "p": {"guess": p_disc, "bounds": p_bound},
        "q": {"guess": q_disc, "bounds": q_bound},
        "r": {"guess": r_disc, "bounds": r_bound},
    },
    tlist=disc_interval,
    algorithm_kwargs={"alg": "CRAB", "fid_err_targ": 0.01, "fix_frequency": False},
)

if _rl_available:
    # ----------------------- RL --------------------

    # state to state transfer
    initial = qt.basis(2, 0)
    target = (qt.basis(2, 0) + qt.basis(2, 1)).unit()  # |+⟩

    H_c = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]  # control Hamiltonians

    w, d, y = 0.1, 1.0, 0.1
    H_d = 1 / 2 * (w * qt.sigmaz() + d * qt.sigmax())  # drift Hamiltonian

    H = [H_d] + H_c  # total Hamiltonian

    state2state_rl = Case(
        objectives=[Objective(initial, H, target)],
        control_parameters={
            "p": {"bounds": [(-13, 13)]},
        },
        tlist=np.linspace(0, 10, 100),
        algorithm_kwargs={
            "fid_err_targ": 0.01,
            "alg": "RL",
            "max_iter": 20000,
            "shorter_pulses": True,
        },
        optimizer_kwargs={},
    )

    # no big difference for unitary evolution

    initial = qt.qeye(2)  # Identity
    target = qt.gates.hadamard_transform()

    unitary_rl = state2state_rl._replace(
        objectives=[Objective(initial, H, target)],
        control_parameters={
            "p": {"bounds": [(-13, 13)]},
        },
        algorithm_kwargs={
            "fid_err_targ": 0.01,
            "alg": "RL",
            "max_iter": 300,
            "shorter_pulses": True,
        },
    )

else: # skip RL tests
    state2state_rl = None
    unitary_rl = None


@pytest.fixture(
    params=[
        pytest.param(state2state_grape, id="State to state (GRAPE)"),
        pytest.param(state2state_crab, id="State to state (CRAB)"),
        pytest.param(state2state_param_crab, id="State to state (param. CRAB)"),
        pytest.param(state2state_goat, id="State to state (GOAT)"),
        pytest.param(state2state_jax, id="State to state (JAX)"),
        pytest.param(state2state_rl, id="State to state (RL)"),
        pytest.param(unitary_rl, id="Unitary (RL)"),
    ]
)
def tst(request):
    return request.param


def test_optimize_pulses(tst):
    if tst is None:
        pytest.skip("Dependency not available")

    result = optimize_pulses(
        tst.objectives,
        tst.control_parameters,
        tst.tlist,
        tst.algorithm_kwargs,
        tst.optimizer_kwargs,
    )

    assert isinstance(result, Result)
    assert isinstance(result.objectives, list)
    assert isinstance(result.objectives[0], Objective)
    assert isinstance(result.time_interval, _TimeInterval)
    assert isinstance(result.start_local_time, str)
    assert isinstance(result.end_local_time, str)
    assert isinstance(result.total_seconds, float)
    assert isinstance(result.n_iters, int)
    assert isinstance(result.iter_seconds, list)
    assert isinstance(result.iter_seconds[0], float)
    assert isinstance(result.message, str)
    assert isinstance(result.guess_controls, (list, np.ndarray))
    assert isinstance(result.optimized_controls, (list, np.ndarray))
    assert isinstance(result.optimized_H, list)
    assert isinstance(result.optimized_H[0], qt.QobjEvo)
    assert isinstance(result.final_states, list)
    assert isinstance(result.final_states[0], qt.Qobj)
    assert isinstance(result.guess_params, (list, np.ndarray))
    assert isinstance(result.optimized_params, (list, np.ndarray))
    assert isinstance(result.infidelity, float)
    assert isinstance(result.var_time, bool)
