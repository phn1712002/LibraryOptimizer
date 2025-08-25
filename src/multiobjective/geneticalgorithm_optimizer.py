import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember
from utils.general import roulette_wheel_selection, tournament_selection


class GeneticAlgorithmMultiMember(MultiObjectiveMember):
    def __init__(self, position: np.ndarray, fitness: np.ndarray):
        super().__init__(position, fitness)
    
    def copy(self):
        new_member = GeneticAlgorithmMultiMember(self.position.copy(), self.multi_fitness.copy())
        new_member.dominated = self.dominated
        new_member.grid_index = self.grid_index
        new_member.grid_sub_index = self.grid_sub_index
        return new_member


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
    
    def _init_population(self, search_agents_no) -> List[GeneticAlgorithmMultiMember]:
        """Initialize multi-objective population with custom member class"""
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(GeneticAlgorithmMultiMember(position, fitness))
        return population
    
    def _tournament_selection_multi(self, population: List[GeneticAlgorithmMultiMember]) -> GeneticAlgorithmMultiMember:
        """
        Tournament selection for multi-objective optimization using grid-based diversity
        
        Parameters:
        -----------
        population : List[GeneticAlgorithmMultiMember]
            Population to select from
            
        Returns:
        --------
        GeneticAlgorithmMultiMember
            Selected individual
        """
        if len(population) < self.tournament_size:
            return np.random.choice(population)
        
        # Randomly select tournament participants
        tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament_members = [population[i] for i in tournament_indices]
        
        # For multi-objective, we need a different selection criteria
        # Use non-dominated sorting if possible, otherwise use grid-based selection
        
        # First, check if any members are non-dominated
        non_dominated = [m for m in tournament_members if not m.dominated]
        
        if non_dominated:
            # If we have non-dominated members, select from them using grid-based diversity
            if len(non_dominated) > 1:
                # Use grid index for diversity-based selection
                grid_indices = [m.grid_index for m in non_dominated if m.grid_index is not None]
                if grid_indices:
                    # Select the member from the least crowded cell
                    unique_indices, counts = np.unique(grid_indices, return_counts=True)
                    least_crowded_idx = unique_indices[np.argmin(counts)]
                    least_crowded_members = [m for m in non_dominated if m.grid_index == least_crowded_idx]
                    return np.random.choice(least_crowded_members)
            
            # Fallback: return random non-dominated member
            return np.random.choice(non_dominated)
        
        # If no non-dominated members, use random selection
        return np.random.choice(tournament_members)
    
    def _uniform_crossover(self, parent1: GeneticAlgorithmMultiMember, 
                          parent2: GeneticAlgorithmMultiMember) -> GeneticAlgorithmMultiMember:
        """
        Perform uniform crossover between two parents
        
        Parameters:
        -----------
        parent1 : GeneticAlgorithmMultiMember
            First parent
        parent2 : GeneticAlgorithmMultiMember
            Second parent
            
        Returns:
        --------
        GeneticAlgorithmMultiMember
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
        return GeneticAlgorithmMultiMember(child_position, child_fitness)
    
    def _mutation(self, individual: GeneticAlgorithmMultiMember) -> GeneticAlgorithmMultiMember:
        """
        Perform mutation on an individual
        
        Parameters:
        -----------
        individual : GeneticAlgorithmMultiMember
            Individual to mutate
            
        Returns:
        --------
        GeneticAlgorithmMultiMember
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
        return GeneticAlgorithmMultiMember(mutated_position, mutated_fitness)
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[GeneticAlgorithmMultiMember]]:
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
        Tuple[List, List[GeneticAlgorithmMultiMember]]
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
                    parent1 = self._tournament_selection_multi(population)
                    parent2 = self._tournament_selection_multi(population)
                    
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
