"""
This module contains functions that implement the JOPT algorithm to
calculate optimal parameters for analytical control pulse sequences.
"""
import qutip as qt
from qutip import Qobj, QobjEvo

try:
    import jax
    from jax import custom_jvp
    import jax.numpy as jnp
    import qutip_jax  # noqa: F401

    import jaxlib  # noqa: F401

    from diffrax import Dopri5, PIDController

    _jax_available = True
except ImportError:
    _jax_available = False


if _jax_available:

    @custom_jvp
    def _abs(x):
        return jnp.abs(x)


    def _abs_jvp(primals, tangents):
        """
        Custom jvp for absolute value of complex functions
        """
        (x,) = primals
        (t,) = tangents

        abs_x = _abs(x)
        res = jnp.where(
            abs_x == 0,
            0.0,  # prevent division by zero
            jnp.real(jnp.multiply(jnp.conj(x), t)) / abs_x,
        )

        return abs_x, res


    # register custom jvp for absolut value of complex functions
    _abs.defjvp(_abs_jvp)


class _JOPT:
    """
    Class for storing a control problem and calculating
    the fidelity error function and its gradient wrt the control parameters.
    """

    def __init__(
        self,
        objective,
        time_interval,
        time_options,
        control_parameters,
        alg_kwargs,
        guess_params,
        **integrator_kwargs,
    ):
        if not _jax_available:
            raise ImportError("The JOPT algorithm requires the modules jax, "
                              "jaxlib, and qutip_jax to be installed.")

        self._Hd = objective.H[0]
        self._Hc_lst = objective.H[1:]

        self._control_parameters = control_parameters
        self._guess_params = guess_params
        self._H = self._prepare_generator()

        self._initial = objective.initial.to("jax")
        self._target = objective.target.to("jax")

        self._evo_time = time_interval.evo_time
        self._var_t = "guess" in time_options

        # inferred attributes
        self._norm_fac = 1 / self._target.norm()

        # integrator options
        self._integrator_kwargs = integrator_kwargs
        self._integrator_kwargs["method"] = "diffrax"

        self._rtol = self._integrator_kwargs.get("rtol", 1e-5)
        self._atol = self._integrator_kwargs.get("atol", 1e-5)

        self._integrator_kwargs.setdefault(
            "stepsize_controller", PIDController(rtol=self._rtol, atol=self._atol)
        )
        self._integrator_kwargs.setdefault("solver", Dopri5())

        # choose solver and fidelity type according to problem
        if self._Hd.issuper:
            self._fid_type = alg_kwargs.get("fid_type", "TRACEDIFF")
            self._solver = qt.MESolver(H=self._H, options=self._integrator_kwargs)

        else:
            self._fid_type = alg_kwargs.get("fid_type", "PSU")
            self._solver = qt.SESolver(H=self._H, options=self._integrator_kwargs)

        self.infidelity = jax.jit(self._infid)
        self.gradient = jax.jit(jax.grad(self._infid))

    def _prepare_generator(self):
        """
        prepare Hamiltonian call signature
        to only take one parameter vector 'p' for mesolve like:
        qt.mesolve(H, psi0, tlist, args={'p': p})
        """

        def helper(control, lower, upper):
            # to fix parameter index in loop
            return jax.jit(lambda t, p: control(t, p[lower:upper]))

        H = QobjEvo(self._Hd)
        idx = 0

        for Hc, p_opt in zip(self._Hc_lst, self._control_parameters.values()):
            hc, ctrl = Hc[0], Hc[1]

            guess = p_opt.get("guess")
            M = len(guess)

            evo = QobjEvo(
                [hc, helper(ctrl, idx, idx + M)], args={"p": self._guess_params}
            )
            H += evo
            idx += M

        return H.to("jax")

    def _infid(self, params):
        """
        calculate infidelity to be minimized
        """
        # adjust integration time-interval, if time is parameter
        evo_time = self._evo_time if self._var_t is False else params[-1]

        X = self._solver.run(
            self._initial, [0.0, evo_time], args={"p": params}
        ).final_state

        if self._fid_type == "TRACEDIFF":
            diff = X - self._target
            # to prevent if/else in qobj.dag() and qobj.tr()
            diff_dag = Qobj(diff.data.adjoint(), dims=diff.dims)
            g = 1 / 2 * (diff_dag * diff).data.trace()
            infid = jnp.real(self._norm_fac * g)
        else:
            g = self._norm_fac * self._target.overlap(X)
            if self._fid_type == "PSU":  # f_PSU (drop global phase)
                infid = 1 - _abs(g)  # custom_jvp for abs
            elif self._fid_type == "SU":  # f_SU (incl global phase)
                infid = 1 - jnp.real(g)

        return infid
