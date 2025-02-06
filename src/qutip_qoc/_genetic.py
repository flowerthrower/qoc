import numpy as np
from joblib import Parallel, delayed
import scipy as sci
import time
from qutip_qoc.result import Result 
import qutip as qt

class _GENETIC():
    ### Template for a genetic algorithm optimizer
    ### Copied from GOAT or RL
    ### Modified to be a genetic algorithm optimizer
    ### Contributed by Jonathan Brown

    def __init__(
        self,
        objectives,
        control_parameters,
        time_interval,
        time_options,
        alg_kwargs,
        optimizer_kwargs,
        minimizer_kwargs,
        integrator_kwargs,
        qtrl_optimizers,
    ):
        """
        Initialize the genetic algorithm with the given parameters.

        Args:
            objectives: The objectives for the optimization.
            control_parameters: Parameters for the control.
            time_interval: The time interval for the optimization.
            time_options: Options related to time discretization.
            alg_kwargs: Additional arguments for the algorithm.
            optimizer_kwargs: Arguments for the optimizer.
            minimizer_kwargs: Arguments for the minimizer.
            integrator_kwargs: Arguments for the integrator.
            qtrl_optimizers: Quantum control optimizers.
        """
        self.objectives = objectives
        self.control_parameters = control_parameters
        self.time_interval = time_interval
        self.time_options = time_options
        self.alg_kwargs = alg_kwargs

        # Initialize the genetic algorithm parameters
        self.N_pop = alg_kwargs.get('population_size', 50)
        self.N_var = alg_kwargs.get('number_variables', 10)
        self.parent_rate = alg_kwargs.get('parent_rate', 0.5)
        self.N_parents = int(np.floor(self.N_pop * self.parent_rate))
        self.N_survivors = max(1, int(self.N_parents * alg_kwargs.get('survival_rate', 1.0)))
        self.N_offspring = self.N_pop - self.N_survivors
        self.mutation_rate = alg_kwargs.get('mutation_rate', 0.1)
        self.workers = alg_kwargs.get('workers', 1)

        # Initialize the Result object
        self._result = Result(
            objectives=objectives,
            time_interval=time_interval,
            start_local_time=time.time(),  # initial optimization time
            n_iters=0,  # Number of iterations(episodes) until convergence
            iter_seconds=[],  # list containing the time taken for each iteration(episode) of the optimization
            var_time=True,  # Whether the optimization was performed with variable time
            guess_params=[],
        )

        self._Hd_lst, self._Hc_lst = [], []
        for objective in objectives:
            # extract drift and control Hamiltonians from the objective
            self._Hd_lst.append(objective.H[0])
            self._Hc_lst.append(
                [H[0] if isinstance(H, list) else H for H in objective.H[1:]]
            )

        def create_pulse_func(idx):
            """
            Create a control pulse lambda function for a given index.
            """
            return lambda t, args: self._pulse(t, args, idx + 1)

        # create the QobjEvo with Hd, Hc and controls(args)
        self._H_lst = [self._Hd_lst[0]]
        dummy_args = {f"alpha{i+1}": 1.0 for i in range(len(self._Hc_lst[0]))}
        for i, Hc in enumerate(self._Hc_lst[0]):
            self._H_lst.append([Hc, create_pulse_func(i)])
        self._H = qt.QobjEvo(self._H_lst, args=dummy_args)

        self.shorter_pulses = alg_kwargs.get(
            "shorter_pulses", False
        )  # lengthen the training to look for pulses of shorter duration, therefore episodes with fewer steps

        # extract bounds for control_parameters
        bounds = []
        for key in control_parameters.keys():
            bounds.append(control_parameters[key].get("bounds"))
        self._lbound = [b[0][0] for b in bounds]
        self._ubound = [b[0][1] for b in bounds]

        self._alg_kwargs = alg_kwargs

        self._initial = objectives[0].initial
        self._target = objectives[0].target
        self._state = None
        self._dim = self._initial.shape[0]

        self._result = Result(
            objectives=objectives,
            time_interval=time_interval,
            start_local_time=time.time(),  # initial optimization time
            n_iters=0,  # Number of iterations(episodes) until convergence
            iter_seconds=[],  # list containing the time taken for each iteration(episode) of the optimization
            var_time=True,  # Whether the optimization was performed with variable time
            guess_params=[],
        )

        self._backup_result = Result(  # used as a backup in case the algorithm with shorter_pulses does not find an episode with infid<target_infid
            objectives=objectives,
            time_interval=time_interval,
            start_local_time=time.time(),
            n_iters=0,
            iter_seconds=[],
            var_time=True,
            guess_params=[],
        )

    def _infid(self, params):
        """
        Calculate infidelity to be minimized
        """
        X = self._solver.run(
            self._initial, [0.0, self._evo_time], args={"p": params}
        ).final_state

        if self._fid_type == "TRACEDIFF":
            diff = X - self._target
            # to prevent if/else in qobj.dag() and qobj.tr()
            diff_dag = qt.Qobj(diff.data.adjoint(), dims=diff.dims)
            g = 1 / 2 * (diff_dag * diff).data.trace()
            infid = np.real(self._norm_fac * g)
        else:
            g = self._norm_fac * self._target.overlap(X)
            if self._fid_type == "PSU":  # f_PSU (drop global phase)
                infid = 1 - np.abs(g)
            elif self._fid_type == "SU":  # f_SU (incl global phase)
                infid = 1 - np.real(g)

        return infid
    
    "Initialize a first population"
    def initial_population(self):
        """Randomly generates an initial popuation to act as generation 0.

        Returns:
            Array: Contains all randomly initialised chromosomes in a single matrix.
        """
        return np.random.uniform(-1,1, (self.N_pop, self.N_var))


    def darwin(self, population, fitness):
        """A function for creating the parent pool which will be used to repopulate the next 
        generation. 
        
        Args:
            population (Array): Array of chromosome vectors for the current generation stacked into a single array.
            fitness (Vector): The corresponding fitness of each chromosome in the population. 

        Returns:
            Array: The pool of parent chromosomes.
            Vector: The corresponding fitness of the parent chromosomes.
            
        """
        # takes the current generation of solutions (population) and kills the weakest
        # Returns the ordered survivors (with the elite survivor at index 0) and their costs
        # np.argsort sorts the indices from smallest at zeroth element to largest thus to get the largest elem at 0
        # we pass -1 X our array to be sorted, then take the first
        indices = np.argsort(-fitness)[:self.N_parents]
        parents = population[indices]
        parent_fitness = fitness[indices]
        return parents, parent_fitness

    def pairing(self, parents, parent_fitness):
        """From the parent pool, probabilistically generates pairs to act as mother and father 
        chromosomes to reproduce, with the probability of pairng determined by the relative fitness
        of each chromosome.
        
        Args:
            parents (Array): Array of stacked chromosome vectors to act as the parent pool.
            fitness (Vector): The corresponding fitness of each chromosome in the parent pool. 

        Returns:
            Array: The mother chromosomes.
            Array: The corresponding father chromosomes.
            
        """
        # Select pair of chromosomes to mate from the survivors
        # ------------ could use a softmax here ------------ #
        # positive_pf = parent_fitness + np.abs(np.min(parent_fitness))
        # prob_dist = (positive_pf)/np.sum(positive_pf)
        prob_dist = sci.special.softmax(parent_fitness)
        # -------------------------------------------------- #
        ma_indices = np.random.choice(np.arange(self.N_parents),
                                   size=int(self.N_offspring), p=prob_dist, replace=True)
        da_indices = np.zeros_like(ma_indices)
        index_ls = np.arange(self.N_parents)
        for i, ma in enumerate(ma_indices):
            inter_prob_dist = parent_fitness.copy()
            inter_prob_dist[ma] = -np.inf
            inter_prob_dist_normd = sci.special.softmax(inter_prob_dist)
            da_indices[i] = np.random.choice(index_ls, size=1, p=inter_prob_dist_normd)
        father_chromes = parents[da_indices]
        mother_chromes = parents[ma_indices]
        return mother_chromes, father_chromes

    def mating_procedure(self, ma, da):
        """Takes the pair of parent chromosomes and combines them in a way to produce offspring.
        (Ma and Da - slang used for mother and father in Ireland, Scotland and northern England)
        
        Args:
            Ma (Array): The list of mother chromosomes.
            Da (Array): The corresponding list of father chromosomes. 

        Returns:
            Array: The pool of parent chromosomes.
            
        """
        # define an array of combination parameters beta to use to combine the parent chromosomes
        # in a continuous way
        beta = np.random.uniform(0,1, size=(self.N_offspring, self.N_var))
        beta_inverse = 1 - beta
        # Randomly select an array of ones to mask the beta array (this also randomly selects)
        # the indices to be swapped
        swap_array = np.random.randint(low=0,high=2,size=(self.N_offspring, self.N_var))
        masked_beta = swap_array * beta
        masked_inverse_beta = swap_array * beta_inverse
        not_swap_array = np.mod((swap_array + 1),2)
        # Implement the b*ma + (1-b)*da on each chosen element
        offspring_array = masked_beta * ma + masked_inverse_beta * da + not_swap_array * ma
        return offspring_array


    def build_next_gen(self, parents, offspring):
        """Takes the parent pool and keeps N_survivor parent chromosomes to survive to the next generation.
        Builds said generation by stacking surviving parents with the offspring chromosomes.
        
        Args:
            parents (Array): The complete parent pool array.
            offspring (Array): The offspring array. 

        Returns:
            Array: The population of unmutated chromosomes which will constitute the next generation.
        """
        # build next generation 
        # get the elite survivors to propagate to the next gen
        survivors = parents[:self.N_survivors]
        return np.concatenate((survivors, offspring), axis=0)


    def mutate(self, population):
        """Takes the unmutated new generation population and randomly applies continuous mutations.
        
        Args:
            population (Array): The unmutated population which will constitute the new generation of chromosomes.

        Returns:
            Array: The population for the new generation.
        """
        # Mutate the new generation
        number_of_mutations = int((population.shape[0] - 1) * population.shape[-1] * self.mutation_rate)
        row_indices = np.random.choice(np.arange(1,int((population.shape[0]))), size=number_of_mutations)
        col_indices = np.random.choice(np.arange(0,int((population.shape[-1]))), size=number_of_mutations)
        mutated_population = np.copy(population)
        # for some reason clipped normal noise works better.
        # mutated_population[row_indices, col_indices] = np.random.uniform(-1,1,size=number_of_mutations) # for uniform random mutation
        mutated_population[row_indices, col_indices] = np.clip(population[row_indices, col_indices] + np.random.normal(loc=0, scale=0.2, size=number_of_mutations), a_min=-1, a_max=1) # for normal mutation
        return mutated_population
    

    def optimize(self, infidelity, iterations = 1000):
        """Implements entire procedure for continuous genetic algorithm optimizer. This is a higher order function
        which accepts a black box fitness function. The fitness function which will accept a single chromosome of
        N_var variables in the range [-1,1], scale and process them returning a figure of merit which describes how useful said parameter 
        values were at acheiving the general task. This figure of merit will act as the fitness of the input chromosome.
        
        Args:
            fitness_func (Function): Black box fitness function to evaluate a chromosome fitness.
            iterations (Int): The number of generations for which to optimize.

        Returns:
            Float: The maximum acheived fitness value from every generation of the optimization.
            Vector: The chromosome that acheived the highest fitness.
            Vector: The history of maximum fitness improvement for the entire optimization.
            
        """
        population = self.initial_population() # initialise a random seroth generation
        bench = -1e10 # the initial benchmark fitness - we assume fitness >= 0
        done = False # terminal flag
        count = 0 # iteration count
        maxfit_hist = [] # to record the maximum fitness at each gen during optimizing
        
        while not done:
            if self.workers > 1:
                fitness_ls = Parallel(n_jobs=self.workers)(delayed(infidelity)(chromosome) for chromosome in population)
            else:
                fitness_ls = []
                for chromosome in population:
                    # The fitness function should be able to take a raw chromosome (whose constituent values will be \in [-1, 1]) 
                    # scale the values to get the true parameter values, then use those parameters to do "stuff" (in our case
                    # build a control pulse and simulate the dynamics) then return a figure of merit which tells us how good
                    # this set of parameters has performed (could be fidelity to a target state for example).
                    # essentially all of the problem under consideration happens inside fitness_func
                    alphas = [
                        ((chromosome + 1) / 2 * (self._ubound[0] - self._lbound[0]))
                        + self._lbound[0]
                        for i in range(len(chromosome))
                    ]
                    args = {f"alpha{i+1}": value for i, value in enumerate(alphas)}
                    infidelity = self._infid(args)
                    f = infidelity(chromosome)
                    fitness_ls.append(f)
            fitness = np.array(fitness_ls)
            max_fit = np.max(fitness)
            if max_fit>bench:
                print(" EPISODE: {}, average cost: {}, Best Cost: {}".format(count, np.round(np.average(fitness), decimals=4), 
                                                                                                    np.max(fitness)))
                # update benchmark and add max fitness to history
                bench = max_fit
                maxfit_hist.append(max_fit)
                best_index = np.argmax(fitness) # get best index
                best_chromosome = population[best_index,:] # so that we can get the best chromosome

                # May want to checkpoint here
            if count == iterations: # if max iterations reached
                done=True
                
            survivors, survivor_fitness = self.darwin(population,fitness)
            mothers, fathers = self.pairing(survivors, survivor_fitness)
            offspring = self.mating_procedure(ma=mothers, da=fathers)
            unmutated_next_gen = self.build_next_gen(survivors,offspring)
            mutated_next_gen = self.mutate(unmutated_next_gen)
            population = mutated_next_gen
            count+=1
            
        return max_fit, best_chromosome, maxfit_hist
    
    def _save_result(self):
        """
        Save the results of the optimization process, including the optimized
        pulse sequences, final states, and performance metrics.
        """
        result_obj = self._backup_result if self._use_backup_result else self._result

        if self._use_backup_result:
            self._backup_result.iter_seconds = self._result.iter_seconds.copy()
            self._backup_result._final_states = self._result._final_states.copy()
            self._backup_result.infidelity = self._result.infidelity

        result_obj.end_local_time = time.time()
        result_obj.n_iters = len(self._result.iter_seconds)
        result_obj.optimized_params = self._actions.copy() + [
            self._result.total_seconds
        ]  # If var_time is True, the last parameter is the evolution time
        result_obj._optimized_controls = self._actions.copy()
        result_obj._guess_controls = []
        result_obj._optimized_H = [self._H]

    def result(self):
        """
        Final conversions and return of optimization results
        """
        if self._use_backup_result:
            self._backup_result.start_local_time = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self._backup_result.start_local_time)
            )
            self._backup_result.end_local_time = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self._backup_result.end_local_time)
            )
            return self._backup_result
        else:
            self._save_result()
            self._result.start_local_time = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self._result.start_local_time)
            )
            self._result.end_local_time = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self._result.end_local_time)
            )
            return self._result