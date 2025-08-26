import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from ._general import sort_population

class ElectromagneticChargedParticlesOptimizer(Solver):
    """
    Electromagnetic Charged Particles Optimization (ECPO) Algorithm.
    
    ECPO is a physics-inspired metaheuristic optimization algorithm that mimics
    the behavior of charged particles in an electromagnetic field. The algorithm
    uses three different strategies for particle movement based on electromagnetic
    forces between particles.
    
    The algorithm features:
    - Three different movement strategies (V[0] parameter)
    - Archive-based selection for maintaining diversity
    - Force-based movement inspired by electromagnetic interactions
    
    Parameters:
    -----------
    objective_func : Callable
        Objective function to optimize
    lb : Union[float, np.ndarray]
        Lower bounds of search space
    ub : Union[float, np.ndarray]
        Upper bounds of search space  
    dim : int
        Number of dimensions in the problem
    maximize : bool, optional
        Optimization direction, default is True (maximize)
    **kwargs
        Additional algorithm parameters:
        - strategy: Movement strategy (1, 2, or 3, default: 1)
        - npi: Number of particles for interaction (default: 2)
        - archive_ratio: Archive size ratio (default: 1.0)
    
    References:
        Original MATLAB implementation by Houssem
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name
        self.name_solver = "Electromagnetic Charged Particles Optimizer"
        
        # Algorithm-specific parameters with defaults
        self.strategy = kwargs.get('strategy', 1)  # Movement strategy (1, 2, or 3)
        self.npi = kwargs.get('npi', 2)  # Number of particles for interaction
        self.archive_ratio = kwargs.get('archive_ratio', 1.0)  # Archive size ratio
        
        # Validate parameters
        if self.strategy not in [1, 2, 3]:
            raise ValueError("Strategy must be 1, 2, or 3")
        if self.npi < 2:
            raise ValueError("NPI must be at least 2")
        if self.archive_ratio <= 0:
            raise ValueError("Archive ratio must be positive")
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Execute the Electromagnetic Charged Particles Optimization Algorithm.
        
        The algorithm uses electromagnetic force-based movement with three different
        strategies for particle interaction and archive-based selection for diversity.
        
        Args:
            search_agents_no (int): Number of charged particles in the population
            max_iter (int): Maximum number of iterations for optimization
            
        Returns:
            Tuple[List, Member]: A tuple containing:
                - history_step_solver: List of best solutions at each iteration
                - best_solver: Best solution found overall
        """
        # Initialize storage variables
        history_step_solver = []
        
        # Calculate archive size
        archive_size = int(search_agents_no / self.archive_ratio)
        
        # Initialize population
        population = self._init_population(search_agents_no)
        
        # Initialize best solution
        sorted_population, _ = self._sort_population(population)
        best_solver = sorted_population[0].copy()
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Sort population and create archive
            sorted_population, _ = self._sort_population(population)
            archive = sorted_population[:archive_size]
            
            # Generate new particles based on strategy
            new_particles = self._generate_new_particles(population, archive, search_agents_no)
            
            # Evaluate new particles
            for particle in new_particles:
                particle.fitness = self.objective_func(particle.position)
            
            # Combine archive and new particles
            combined_population = archive + new_particles
            
            # Sort combined population and select best
            sorted_combined, _ = self._sort_population(combined_population)
            population = sorted_combined[:search_agents_no]
            
            # Update best solution
            current_best = population[0]
            if self._is_better(current_best, best_solver):
                best_solver = current_best.copy()
            
            # Store history
            history_step_solver.append(best_solver.copy())
            
            # Update progress
            self._callbacks(iter, max_iter, best_solver)
        
        # Final processing
        self.history_step_solver = history_step_solver
        self.best_solver = best_solver
        
        # End solver
        self._end_step_solver()
        
        return history_step_solver, best_solver
    
    def _generate_new_particles(self, population: List[Member], archive: List[Member], 
                               search_agents_no: int) -> List[Member]:
        """
        Generate new particles using electromagnetic force-based movement.
        
        Args:
            population (List[Member]): Current population
            archive (List[Member]): Archive of best solutions
            search_agents_no (int): Population size
            
        Returns:
            List[Member]: List of newly generated particles
        """
        new_particles = []
        
        # Calculate population factor based on strategy
        if self.strategy == 1:
            pop_factor = 2 * self._n_choose_2(self.npi)
        elif self.strategy == 2:
            pop_factor = self.npi
        else:  # strategy == 3
            pop_factor = 2 * self._n_choose_2(self.npi) + self.npi
        
        # Number of iterations to generate enough particles
        n_iterations = int(np.ceil(search_agents_no / pop_factor))
        
        for _ in range(n_iterations):
            # Generate Gaussian force with mean=0.7, std=0.2
            force = np.random.normal(0.7, 0.2)
            
            # Select random particles for interaction
            selected_indices = np.random.choice(len(population), self.npi, replace=False)
            selected_particles = [population[i] for i in selected_indices]
            
            if self.strategy == 1:
                # Strategy 1: Pairwise interactions
                new_particles.extend(self._strategy_1(selected_particles, archive, force))
            elif self.strategy == 2:
                # Strategy 2: Combined interactions
                new_particles.extend(self._strategy_2(selected_particles, archive, force))
            else:  # strategy == 3
                # Strategy 3: Hybrid approach
                new_particles.extend(self._strategy_3(selected_particles, archive, force))
        
        # Ensure positions stay within bounds
        for particle in new_particles:
            particle.position = np.clip(particle.position, self.lb, self.ub)
        
        # Apply archive-based mutation
        self._apply_archive_mutation(new_particles, archive)
        
        return new_particles
    
    def _strategy_1(self, selected_particles: List[Member], archive: List[Member], 
                   force: float) -> List[Member]:
        """Strategy 1: Pairwise interactions between particles."""
        new_particles = []
        best_particle = archive[0] if archive else selected_particles[0]
        
        for i in range(self.npi):
            for j in range(self.npi):
                if i == j:
                    continue
                
                # Base movement towards best particle
                new_position = selected_particles[i].position.copy()
                new_position += force * (best_particle.position - selected_particles[i].position)
                
                # Add interaction with other particle
                if j < i:
                    new_position += force * (selected_particles[j].position - selected_particles[i].position)
                else:  # j > i
                    new_position -= force * (selected_particles[j].position - selected_particles[i].position)
                
                new_particles.append(Member(new_position, 0.0))
        
        return new_particles
    
    def _strategy_2(self, selected_particles: List[Member], archive: List[Member], 
                   force: float) -> List[Member]:
        """Strategy 2: Combined interactions from all particles."""
        new_particles = []
        best_particle = archive[0] if archive else selected_particles[0]
        
        for i in range(self.npi):
            new_position = selected_particles[i].position.copy()
            
            # Movement towards best particle (no force for this component)
            new_position += 0 * force * (best_particle.position - selected_particles[i].position)
            
            # Combined interactions with all other particles
            for j in range(self.npi):
                if j < i:
                    new_position += force * (selected_particles[j].position - selected_particles[i].position)
                elif j > i:
                    new_position -= force * (selected_particles[j].position - selected_particles[i].position)
            
            new_particles.append(Member(new_position, 0.0))
        
        return new_particles
    
    def _strategy_3(self, selected_particles: List[Member], archive: List[Member], 
                   force: float) -> List[Member]:
        """Strategy 3: Hybrid approach with two types of movements."""
        new_particles_1 = []
        new_particles_2 = []
        best_particle = archive[0] if archive else selected_particles[0]
        
        for i in range(self.npi):
            # Type 1 movement (similar to strategy 1)
            s1_position = selected_particles[i].position.copy()
            s1_position += force * (best_particle.position - selected_particles[i].position)
            
            # Type 2 movement (full force towards best)
            s2_position = selected_particles[i].position.copy()
            s2_position += 1 * force * (best_particle.position - selected_particles[i].position)
            
            for j in range(self.npi):
                if j < i:
                    s1_position += force * (selected_particles[j].position - selected_particles[i].position)
                    s2_position += force * (selected_particles[j].position - selected_particles[i].position)
                elif j > i:
                    s1_position -= force * (selected_particles[j].position - selected_particles[i].position)
                    s2_position -= force * (selected_particles[j].position - selected_particles[i].position)
            
            new_particles_1.append(Member(s1_position, 0.0))
            new_particles_2.append(Member(s2_position, 0.0))
        
        return new_particles_1 + new_particles_2
    
    def _apply_archive_mutation(self, new_particles: List[Member], archive: List[Member]):
        """Apply archive-based mutation to new particles."""
        if not archive:
            return
        
        for particle in new_particles:
            for j in range(self.dim):
                if np.random.random() < 0.2:  # 20% chance of mutation
                    # Replace dimension with value from random archive member
                    archive_member = np.random.choice(archive)
                    particle.position[j] = archive_member.position[j]
    
    def _n_choose_2(self, n: int) -> int:
        """Calculate n choose 2 (number of combinations)."""
        if n < 2:
            return 0
        return n * (n - 1) // 2
    
    def _sort_population(self, population):
        """
        Sort the population based on fitness.
        """
        return sort_population(population, self.maximize)
