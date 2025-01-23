"""
This module contains the Result class for storing and
reporting the results of a full pulse control optimization run.
"""
import pickle
import textwrap
import numpy as np
from inspect import signature
import warnings

import qutip as qt

try:
    import jaxlib
    _jitfun_type = jaxlib.xla_extension.PjitFunction
except ImportError:
    _jitfun_type = None

__all__ = ["Result"]


class _Stats:
    """
    Only for backward compatibility with qtrl.
    """

    def __init__(self, result):
        self._result = result

    def report(self):
        print(self._result)


class Result:
    """
    Class for storing the results of a pulse control optimization run.

    Attributes
    ----------
    objectives : list of :class:`qutip_qoc.Objective`
        List of objectives to be optimized.

    time_interval : :class:`qutip_qoc._TimeInterval`
        Time interval for the optimization.

    start_local_time : struct_time
        Time when the optimization started.

    end_local_time : struct_time
        Time when the optimization ended.

    total_seconds : float
        Total time in seconds the optimization took.
        Equal to the sum of iter_seconds.
        Equal to difference between end_local_time and start_local_time.

    iters : int
        Number of iterations until convergence.
        Equal to the length of iter_seconds.

    iter_seconds : list of float
        Seconds between each iteration.

    message : str
        Reason for termination.

    optimized_params : list of ndarray
        List of optimized parameters.

    guess_controls : list of ndarray
        List of guess control pulses used to initialize the optimization.

    optimized_controls : list of ndarray
        List of optimized control pulses.

    optimized_H : list of :class:`qutip.QobjEvo`
        A specification of the time-depedent quantum object
        one for each objective (see :class:`qutip_qoc.Objective` H attribute).
        with optimized control amplitudes.

    final_states : list of :class:`qutip.Qobj`
        List of final states after the optimization.
        One for each objective.

    infidelity : float
        Final infidelity error after the optimization.

    var_time : bool
        Whether the optimization was performed with variable time.
        If True, the last parameter in optimized_params is the evolution time.
    """

    def __init__(
        self,
        objectives=None,
        time_interval=None,
        start_local_time=None,
        end_local_time=None,
        total_seconds=None,
        n_iters=None,
        iter_seconds=None,
        message=None,
        guess_controls=None,
        optimized_controls=None,
        optimized_H=None,
        final_states=None,
        guess_params=None,
        new_params=None,
        optimized_params=None,
        infidelity=np.inf,
        var_time=False,
        qtrl_optimizers=None,
    ):
        self.time_interval = time_interval
        self.objectives = objectives
        self.start_local_time = start_local_time
        self.end_local_time = end_local_time
        self._total_seconds = total_seconds
        self.n_iters = n_iters
        self.iter_seconds = iter_seconds
        self.message = message
        self._guess_controls = guess_controls
        self._optimized_controls = optimized_controls
        self._optimized_H = optimized_H
        self.guess_params = guess_params
        self.new_params = new_params
        self._optimized_params = optimized_params
        self._final_states = final_states
        self.infidelity = infidelity
        self.var_time = var_time
        self.qtrl_optimizers = qtrl_optimizers

        # qtrl result backward compatibility
        self.stats = _Stats(self)

    def __str__(self):
        time_optim_summary = (
            "- Optimized time parameter: " + str(self.optimized_params[-1])
            if self.var_time
            else ""
        )
        return textwrap.dedent(
            r"""
        Control Optimization Result
        --------------------------
        - Started at {start_local_time}
        - Number of objectives: {n_objectives}
        - Final fidelity error: {final_infid}
        - Final parameters: {final_params}
        - Number of iterations: {n_iters}
        - Reason for termination: {message}
        {time_optim_summary}
        - Ended at {end_local_time} ({time_delta}s)
        """.format(
                start_local_time=self.start_local_time,
                n_objectives=len(self.objectives),
                final_infid=self.infidelity,
                final_params=self.optimized_params,
                n_iters=self.n_iters,
                end_local_time=self.end_local_time,
                time_delta=self.total_seconds,
                time_optim_summary=time_optim_summary,
                message=self.message,
            )
        ).strip()

    def __repr__(self):
        return self.__str__()

    @property
    def total_seconds(self):
        """
        Total time in seconds the optimization took.
        """
        if self._total_seconds is None:
            self._total_seconds = sum(self.iter_seconds)
        return self._total_seconds

    @property
    def optimized_params(self):
        """
        Parameter values after optimization.
        """
        if self._optimized_params is None:
            # reshape (optimized) new_parameters array to match
            # shape and type of the guess_parameters list

            if self.qtrl_optimizers and len(self.guess_params[0]) == len(
                self.time_interval.tslots
            ):  # GRAPE
                amps = self.qtrl_optimizers[0]._get_ctrl_amps(self.new_params)
                opt_params = amps.T
            else:  # GOAT, JOPT, CRAB
                opt_params, idx = [], 0
                for guess in self.guess_params:
                    opt = self.new_params[idx : idx + len(guess)]

                    if isinstance(guess, list):
                        opt = opt.tolist()

                    opt_params.append(opt)
                    idx += len(guess)

            self._optimized_params = opt_params
        return self._optimized_params

    @optimized_params.setter
    def optimized_params(self, params):
        self._optimized_params = params

    @property
    def optimized_controls(self):
        """
        Control pulses after optimization.
        """
        if self._optimized_controls is None:
            opt_ctrl = []

            for j, H in enumerate(zip(self.objectives[0].H[1:], self.optimized_params)):
                Hc, xf = H
                control, cf = Hc[1], []
                if not self.qtrl_optimizers:  # continuous control as in JOPT/GOAT
                    try:
                        tslots = self.time_interval.tslots
                    except Exception:
                        print(
                            "time_interval.tslots not specified "
                            "(probably missing n_tslots), defaulting to 100 "
                            "collocation points for result.optimized_controls"
                        )
                        tslots = np.linspace(0.0, self.time_interval.evo_time, 100)
                    for t in tslots:
                        cf.append(control(t, xf))
                else:  # discrete control as in GRAPE/CRAB
                    if len(xf) == len(self.time_interval.tslots):
                        cf = np.array(xf)
                    else:  # parameterized CRAB
                        pgen = self.qtrl_optimizers[0].pulse_generator[j]
                        pgen.set_optim_var_vals(np.array(self.optimized_params[j]))
                        cf = np.array(pgen.gen_pulse())
                opt_ctrl.append(cf)

            self._optimized_controls = opt_ctrl
        return self._optimized_controls

    @property
    def guess_controls(self):
        """
        Control pulses before the optimization.
        """
        if self._guess_controls is None:
            if self.qtrl_optimizers:
                qtrl_res = self.qtrl_optimizers[0]._create_result()
                gss_ctrl = qtrl_res.initial_amps.T
            else:
                gss_ctrl = []
                for j, H in enumerate(zip(self.objectives[0].H[1:], self.guess_params)):
                    Hc, xi = H
                    control, c0 = Hc[1], []
                    if callable(control):  # continuous control as in JOPT/GOAT
                        try:
                            tslots = self.time_interval.tslots
                        except Exception:
                            print(
                                "time_interval.tslots not specified "
                                "(probably missing n_tslots), defaulting to 100 "
                                "collocation points for result.optimized_controls"
                            )
                            tslots = np.linspace(0.0, self.time_interval.evo_time, 100)
                        for t in tslots:
                            c0.append(control(t, xi))
                    else:  # discrete control as in GRAPE/CRAB
                        if len(xi) == len(self.time_interval.tslots):
                            c0 = xi
                        else:  # parameterized CRAB
                            pgen = self.qtrl_optimizers[0].pulse_generator[j]
                            pgen.set_optim_var_vals(np.array(self.guess_params[j]))
                            c0 = pgen.gen_pulse()
                    gss_ctrl.append(c0)

            self._guess_controls = gss_ctrl
        return self._guess_controls

    @property
    def optimized_H(self):
        """
        Optimized Hamiltonians with optimized controls.
        """
        if self._optimized_H is None:
            opt_H = []

            for obj in self.objectives:
                # Create the optimized Hamiltonian with optimized controls
                if not self.qtrl_optimizers:  # GOAT, JOPT
                    H = obj.H
                else:
                    H = [obj.H[0]]  # drift
                    for Hc, cf in zip(obj.H[1:], self.optimized_controls):
                        if isinstance(Hc, qt.Qobj):  # parameterized CRAB
                            H.append([Hc, cf])
                        else:  # discrete control as in GRAPE, CRAB
                            H.append([Hc[0], cf])

                # Create the corresponding QobjEvo object
                para_keys = []
                args_dict = {}
                if not self.qtrl_optimizers:  # GOAT, JOPT
                    # extract parameter names from control functions f(t, para_key)
                    c_sigs = [signature(Hc[1]) for Hc in self.objectives[0].H[1:]]
                    c_keys = [sig.parameters.keys() for sig in c_sigs]
                    para_keys = [list(keys)[1] for keys in c_keys]
                    for key, val in zip(para_keys, self.optimized_params):
                        args_dict[key] = val

                H_evo = (
                    qt.QobjEvo(H, args=args_dict)
                    if args_dict  # GOAT, JOPT
                    else qt.QobjEvo(H, tlist=self.time_interval.tslots)
                )

                opt_H.append(H_evo)
            self._optimized_H = opt_H
        return self._optimized_H

    @property
    def final_states(self):
        """
        Evolved system states after optimization.
        """
        if self._final_states is None:
            states = []

            if self.var_time:  # last parameter is optimized time
                evo_time = self.optimized_params[-1][0]
            else:
                evo_time = self.time_interval.evo_time

            # choose solver method based on type of control function
            # if jax is installed, _jitfun_type is set to
            # jaxlib.xla_extension.PjitFunction, otherwise it is None
            if _jitfun_type is not None and isinstance(
                self.objectives[0].H[1][1], _jitfun_type
            ):
                method = "diffrax"  # for JAX defined contols
            else:
                method = "adams"

            for obj, opt_H in zip(self.objectives, self.optimized_H):
                if opt_H.issuper:  # choose solver
                    solver = qt.MESolver(
                        opt_H,
                        options={
                            "normalize_output": False,
                            "method": method,
                        },
                    )
                else:
                    solver = qt.SESolver(
                        opt_H,
                        options={
                            "normalize_output": False,
                            "method": method,
                        },
                    )

                states.append(  # compute evolution
                    solver.run(obj.initial, tlist=[0.0, evo_time]).final_state
                )

            self._final_states = states
        return self._final_states

    def _update(self, infidelity, parameters):
        """
        Used to update the result during optimization.
        """
        self.infidelity = infidelity
        self.new_params = parameters

    def dump(self, filename):
        """
        Save the result to a file.
        """
        with open(filename, "wb") as dump_fh:
            pickler = pickle.Pickler(dump_fh)
            pickler.dump(self)

    @classmethod
    def load(cls, filename, objectives=None):
        """
        Load a objective from a file.
        """
        with open(filename, "rb") as dump_fh:
            result = pickle.load(dump_fh)
        result.objectives = objectives
        return result

    @property
    def evo_full_final(self):
        """
        Deprecated, use final_states[0] instead.
        """
        warnings.warn(
            "evo_full_final is deprecated, use final_states[0] instead",
            DeprecationWarning,
        )
        return self.final_states[0]

    @property
    def fid_err(self):
        """
        Deprecated, use infidelity instead.
        """
        warnings.warn(
            "fid_err is deprecated, use infidelity instead", DeprecationWarning
        )
        return self.infidelity

    @property
    def grad_norm_final(self):
        """
        Deprecated, not supported.
        """
        warnings.warn(
            "grad_norm_final is deprecated, it is not supported", DeprecationWarning
        )
        return None  # not supported

    @property
    def termination_reason(self):
        """
        Deprecated, use message instead.
        """
        warnings.warn(
            "termination_reason is deprecated, use message instead", DeprecationWarning
        )
        return self.message

    @property
    def num_iter(self):
        """
        Deprecated, use n_iters instead.
        """
        warnings.warn("num_iter is deprecated, use n_iters instead", DeprecationWarning)
        return self.n_iters

    @property
    def wall_time(self):
        """
        Deprecated, use total_seconds instead.
        """
        warnings.warn(
            "wall_time is deprecated, use total_seconds instead", DeprecationWarning
        )
        return self.total_seconds
