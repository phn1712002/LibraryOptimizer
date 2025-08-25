import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from utils.general import sort_population

class Particle(Member):
    def __init__(self, position: np.ndarray, fitness: float, velocity: np.ndarray = None):
        super().__init__(position, fitness)
        self.velocity = velocity if velocity is not None else np.zeros_like(position)
    
    def copy(self):
        return Particle(self.position.copy(), self.fitness, self.velocity.copy())
    
    def __str__(self):
        return f"Position: {self.position} - Fitness: {self.fitness} - Velocity: {self.velocity}"

class ParticleSwarmOptimizer(Solver):
    """
    Particle Swarm Optimization (PSO) algorithm implementation.
    
    PSO is a population-based stochastic optimization technique inspired by
    social behavior of bird flocking or fish schooling. Each particle in the
    swarm represents a potential solution and moves through the search space
    by following the current optimum particles.
    
    References:
        Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
        Proceedings of ICNN'95-international conference on neural networks.
    """
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        """
        Initialize the Particle Swarm Optimizer.
        
        Args:
            objective_func (Callable): Objective function to optimize
            lb (Union[float, np.ndarray]): Lower bound for search space
            ub (Union[float, np.ndarray]): Upper bound for search space
            dim (int): Number of dimensions/problem variables
            maximize (bool): Whether to maximize (True) or minimize (False) the objective
            **kwargs: Additional PSO parameters including:
                - w: Inertia weight (default: 1.0)
                - wdamp: Inertia weight damping ratio (default: 0.99)
                - c1: Personal learning coefficient (default: 1.5)
                - c2: Global learning coefficient (default: 2.0)
                - vel_max: Maximum velocity (default: 10% of search space range)
                - vel_min: Minimum velocity (default: -10% of search space range)
        """
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set default PSO parameters
        self.name_solver = "Particle Swarm Optimizer"
        self.w = kwargs.get('w', 1.0)  # Inertia weight
        self.wdamp = kwargs.get('wdamp', 0.99)  # Inertia weight damping ratio
        self.c1 = kwargs.get('c1', 1.5)  # Personal learning coefficient
        self.c2 = kwargs.get('c2', 2.0)  # Global learning coefficient
        
        # Velocity limits (10% of variable range)
        vel_range = 0.1 * (self.ub - self.lb)
        self.vel_max = kwargs.get('vel_max', vel_range)
        self.vel_min = kwargs.get('vel_min', -vel_range)

    def _init_population(self, search_agents_no) -> List:
        """
        Initialize the particle swarm population.
        
        Each particle is initialized with:
        - Random position within bounds
        - Random velocity within velocity limits
        - Fitness evaluation of initial position
        
        Args:
            search_agents_no (int): Number of particles in the swarm
            
        Returns:
            List: List of initialized Particle objects
        """
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            velocity = np.random.uniform(self.vel_min, self.vel_max, self.dim)
            fitness = self.objective_func(position)
            population.append(Particle(position, fitness, velocity))
        return population

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Particle]:
        """
        Execute the Particle Swarm Optimization algorithm.
        Args:
            search_agents_no (int): Number of particles in the swarm
            max_iter (int): Maximum number of iterations
            
        Returns:
            Tuple[List, Particle]: Tuple containing:
                - history_step_solver: List of best solutions at each iteration
                - best_particle: Best particle (solution) found overall
        """
        # Initialize storage variables
        history_step_solver = []
        

        # Initialize the population of particles
        population = self._init_population(search_agents_no)
        
        # Initialize personal best particles (copy of initial particles)
        personal_best = [particle.copy() for particle in population]
        
        # Initialize global best using sort_population method)
        sorted_personal_best, _ = self._sort_population(personal_best)
        global_best = sorted_personal_best[0].copy()
        best_solver = global_best.copy()
        
        # Call the begin function
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Update all particles
            for i in range(search_agents_no):
                # Update velocity
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                
                cognitive_component = self.c1 * r1 * (personal_best[i].position - population[i].position)
                social_component = self.c2 * r2 * (global_best.position - population[i].position)
                
                population[i].velocity = (self.w * population[i].velocity + 
                                        cognitive_component + 
                                        social_component)
                
                # Apply velocity limits
                population[i].velocity = np.clip(population[i].velocity, self.vel_min, self.vel_max)
                
                # Update position
                new_position = population[i].position + population[i].velocity
                
                # Apply position limits and velocity mirror effect
                outside_bounds = (new_position < self.lb) | (new_position > self.ub)
                population[i].velocity[outside_bounds] = -population[i].velocity[outside_bounds]
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Update particle position
                population[i].position = new_position
                
                # Evaluate new fitness
                new_fitness = self.objective_func(new_position)
                population[i].fitness = new_fitness
                
                # Update personal best
                if self._is_better(population[i], personal_best[i]):
                    personal_best[i] = population[i].copy()
                    
                    # Update global best if needed
                    if self._is_better(population[i], global_best):
                        global_best = population[i].copy()
                        
                        # Update best solver immediately
                        if self._is_better(global_best, best_solver):
                            best_solver = global_best.copy()
            
            # Store the best solution at this iteration
            history_step_solver.append(best_solver)
            
            # Update inertia weight
            self.w *= self.wdamp
            
            # Call the callbacks
            self._callbacks(iter, max_iter, best_solver)
        
        # Final evaluation and storage
        self.history_step_solver = history_step_solver
        self.best_solver = best_solver
        
        # Call the end function
        self._end_step_solver()
        return history_step_solver, best_solver
    
    def _sort_population(self, population):
        """
        Sort the population based on fitness.
        """
        return sort_population(population, self.maximize)
