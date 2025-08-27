import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember
from .._general import exponential_decay

class MultiObjectiveSimulatedAnnealingOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Simulated Annealing Optimizer
    
    This algorithm extends the standard Simulated Annealing for multi-objective optimization
    using archive management and Pareto dominance criteria for solution acceptance.
    
    Parameters:
    -----------
    objective_func : Callable
        Objective function that returns a list of fitness values
    lb : Union[float, np.ndarray]
        Lower bounds for variables
    ub : Union[float, np.ndarray]
        Upper bounds for variables
    dim : int
        Problem dimension
    **kwargs
        Additional parameters:
        - max_temperatures: Maximum number of temperature levels (default: 100)
        - tol_fun: Function tolerance for convergence (default: 1e-4)
        - safety_threshold_df: Safety threshold calculation df
        - equilibrium_steps: Number of steps per temperature level (default: 500)
        - initial_temperature: Starting temperature (default: None, auto-calculated)

    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Simulated Annealing Optimizer"
        
        # Set default SA parameters
        self.max_temperatures = kwargs.get('max_temperatures', 100)
        self.tol_fun = kwargs.get('tol_fun', 1e-4)
        self.equilibrium_steps = kwargs.get('equilibrium_steps', 500)
        self.initial_temperature = kwargs.get('initial_temperature', 1)
        self.safety_threshold_df = kwargs.get('safety_threshold_df', 1e-8)
        
        # Initialize temperature scaling parameter
        self.mu_scaling = kwargs.get('mu_scaling', 100.0)

    def solver(self, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Execute the multi-objective simulated annealing optimization algorithm.
        
        Parameters:
        -----------
        max_iter : int
            Maximum number of iterations (temperature levels)
            
        Returns:
        --------
        Tuple[List, List[MultiObjectiveMember]]
            History of archive states and the final archive
        """
        # Override max_iter with max_temperatures if provided
        actual_max_iter = min(max_iter, self.max_temperatures)
        
        # Initialize storage
        history_archive = []
        
        # Initialize current solution from random position within bounds
        current_position = np.random.uniform(self.lb, self.ub, self.dim)
        current_fitness = self.objective_func(current_position)
        current_solution = MultiObjectiveMember(current_position, current_fitness)
        
        # Initialize archive with current solution
        self.archive = [current_solution.copy()]
        
        # Initialize grid for archive
        costs = self._get_fitness(self.archive)
        if costs.size > 0:
            self.grid = self._create_hypercubes(costs)
            for particle in self.archive:
                particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
        
        # Start solver
        self._begin_step_solver(actual_max_iter)
        
        # Main optimization loop (temperature levels)
        for iter in range(actual_max_iter):
            # Calculate temperature parameter (inverse temperature)
            T = self._exponential_decay(self.initial_temperature, self.max_temperatures, iter, actual_max_iter)
            
            # Calculate mu parameter for perturbation scaling
            mu = 10 ** (T * self.mu_scaling)
            
            # Simulate thermal equilibrium at this temperature
            for k in range(self.equilibrium_steps):
                # Generate perturbation vector
                random_vector = 2 * np.random.random(self.dim) - 1  # Range [-1, 1]
                dx = self._mu_inv(random_vector, mu) * (self.ub - self.lb)
                
                # Generate new candidate position
                candidate_position = current_solution.position + dx
                
                # Apply bounds constraint
                candidate_position = np.clip(candidate_position, self.lb, self.ub)
                
                # Evaluate candidate fitness
                candidate_fitness = self.objective_func(candidate_position)
                candidate_solution = MultiObjectiveMember(candidate_position, candidate_fitness)
                
                # Multi-objective acceptance criterion
                accept = self._multi_objective_acceptance(current_solution, candidate_solution, T)
                
                if accept:
                    current_solution = candidate_solution.copy()
                
                # Add candidate to archive for consideration
                self._add_to_archive([candidate_solution])
            
            # Store archive state for history
            history_archive.append([sol.copy() for sol in self.archive])
            
            # Update progress
            self._callbacks(iter, actual_max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
    
    def _multi_objective_acceptance(self, current: MultiObjectiveMember, candidate: MultiObjectiveMember, temperature: float) -> bool:
        """
        Multi-objective acceptance criterion based on Pareto dominance and temperature.
        
        Parameters:
        -----------
        current : MultiObjectiveMember
            Current solution
        candidate : MultiObjectiveMember
            Candidate solution
        temperature : float
            Current temperature level
            
        Returns:
        --------
        bool
            Whether to accept the candidate solution
        """
        # Check if candidate dominates current
        if self._dominates(candidate, current):
            return True
        
        # Check if current dominates candidate
        if self._dominates(current, candidate):
            return False
        
        # If neither dominates the other (non-dominated), use probabilistic acceptance
        # Calculate a composite fitness difference for probabilistic acceptance
        current_avg = np.mean(current.multi_fitness)
        candidate_avg = np.mean(candidate.multi_fitness)

        
        if self.maximize:
            # For maximization: accept if candidate is better on average or with probability
            df = candidate_avg - current_avg
        else:
            # For minimization: accept if candidate is better on average or with probability
            df = current_avg - candidate_avg
            
        scale = np.std([sol.multi_fitness for sol in self.archive], axis=0).mean()
        if scale > self.safety_threshold_df: 
            df /= scale
        accept_prob = np.exp(df / (temperature + np.finfo(float).eps))
        return np.random.random() < accept_prob
    
    def _mu_inv(self, y: np.ndarray, mu: float) -> np.ndarray:
        """
        Generate perturbation vectors according to the mu_inv function.
        
        This function is used to generate new candidate points with perturbations
        that are proportional to the current temperature level.
        
        Parameters:
        -----------
        y : np.ndarray
            Random vector in range [-1, 1]
        mu : float
            Temperature scaling parameter
            
        Returns:
        --------
        np.ndarray
            Scaled perturbation vector
        """
        return (((1 + mu) ** np.abs(y) - 1) / mu) * np.sign(y)

    def _exponential_decay(self, initial_value, final_value, current_iter, max_iter):
        """Calculate exponential decay value."""
        return exponential_decay(initial_value, final_value, current_iter, max_iter)