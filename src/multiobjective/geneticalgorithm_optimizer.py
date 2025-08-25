import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember


class MultiObjectiveGeneticAlgorithmOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Genetic Algorithm Optimizer
    
    This algorithm extends the standard GA for multi-objective optimization
    using archive management and grid-based selection for parent selection.
    
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
        - archive_size: Size of the external archive (default: 100)
        - alpha: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta: Leader selection pressure (default: 2)
        - gamma: Archive removal pressure (default: 2)
        - num_groups: Number of independent populations (default: 5)
        - crossover_rate: Probability of crossover (default: 0.8)
        - mutation_rate: Probability of mutation (default: 0.1)
        - tournament_size: Size for tournament selection (default: 3)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Genetic Algorithm Optimizer"
        
        # GA-specific parameters
        self.num_groups = kwargs.get('num_groups', 5)
        self.crossover_rate = kwargs.get('crossover_rate', 0.8)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        self.tournament_size = kwargs.get('tournament_size', 3)
    
    def _uniform_crossover(self, parent1: MultiObjectiveMember, 
                          parent2: MultiObjectiveMember) -> MultiObjectiveMember:
        """
        Perform uniform crossover between two parents
        
        Parameters:
        -----------
        parent1 : MultiObjectiveMember
            First parent
        parent2 : MultiObjectiveMember
            Second parent
            
        Returns:
        --------
        MultiObjectiveMember
            Offspring
        """
        child_position = np.zeros(self.dim)
        
        for d in range(self.dim):
            if np.random.random() < 0.5:  # 50% chance to inherit from parent1
                child_position[d] = parent1.position[d]
            else:
                child_position[d] = parent2.position[d]
        
        # Ensure positions stay within bounds
        child_position = np.clip(child_position, self.lb, self.ub)
        
        # Create child with evaluated fitness
        child_fitness = self.objective_func(child_position)
        return MultiObjectiveMember(child_position, child_fitness)
    
    def _mutation(self, individual: MultiObjectiveMember) -> MultiObjectiveMember:
        """
        Perform mutation on an individual
        
        Parameters:
        -----------
        individual : MultiObjectiveMember
            Individual to mutate
            
        Returns:
        --------
        MultiObjectiveMember
            Mutated individual
        """
        mutated_position = individual.position.copy()
        
        for d in range(self.dim):
            if np.random.random() < self.mutation_rate:
                # Gaussian mutation
                mutation_strength = 0.1 * (self.ub[d] - self.lb[d])
                mutated_position[d] += np.random.normal(0, mutation_strength)
        
        # Ensure positions stay within bounds
        mutated_position = np.clip(mutated_position, self.lb, self.ub)
        
        # Create mutated individual with evaluated fitness
        mutated_fitness = self.objective_func(mutated_position)
        return MultiObjectiveMember(mutated_position, mutated_fitness)
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective GA
        
        Parameters:
        -----------
        search_agents_no : int
            Number of search agents (population size)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, List[MultiObjectiveMember]]
            History of archive states and the final archive
        """
        # Initialize storage
        history_archive = []
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Run multiple populations (as in original GA)
        for pop_idx in range(self.num_groups):
            # Initialize population for this run
            population = self._init_population(search_agents_no)
            
            # Initialize archive with non-dominated solutions from this population
            self._determine_domination(population)
            non_dominated = self._get_non_dominated_particles(population)
            self.archive.extend(non_dominated)
            
            # Initialize grid for archive
            costs = self._get_costs(self.archive)
            if costs.size > 0:
                self.grid = self._create_hypercubes(costs)
                for particle in self.archive:
                    particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
            
            # Main optimization loop for this population
            for iter in range(max_iter):
                # Create new generation
                new_population = []
                
                # Elitism: keep the best individuals
                self._determine_domination(population)
                non_dominated_pop = self._get_non_dominated_particles(population)
                elite_count = min(len(non_dominated_pop), search_agents_no // 4)
                new_population.extend(non_dominated_pop[:elite_count])
                
                # Generate offspring until we reach population size
                while len(new_population) < search_agents_no:
                    # Selection
                    parent1 = self._tournament_selection_multi(population, self.tournament_size)
                    parent2 = self._tournament_selection_multi(population, self.tournament_size)
                    
                    # Crossover with probability
                    if np.random.random() < self.crossover_rate:
                        child = self._uniform_crossover(parent1, parent2)
                    else:
                        # No crossover, select one parent randomly
                        child = parent1.copy() if np.random.random() < 0.5 else parent2.copy()
                    
                    # Mutation
                    if np.random.random() < self.mutation_rate:
                        child = self._mutation(child)
                    
                    new_population.append(child)
                
                # Ensure we have exactly the population size
                population = new_population[:search_agents_no]
                
                # Update archive with current population
                self._add_to_archive(population)
                
                # Store archive state for history
                history_archive.append([member.copy() for member in self.archive])
                
                # Update progress
                self._callbacks(iter + pop_idx * max_iter, self.num_groups * max_iter, 
                               self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
