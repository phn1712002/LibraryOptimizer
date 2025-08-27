import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember

class MultiObjectiveTeachingLearningBasedOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Teaching Learning Based Optimizer
    
    This algorithm extends the standard TLBO for multi-objective optimization
    using archive management and grid-based selection for teacher selection.
    
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
        - teaching_factor_range: Range for teaching factor (default: (1, 2))
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Teaching Learning Based Optimizer"
        
        # Set TLBO-specific parameters
        self.teaching_factor_range = kwargs.get('teaching_factor_range', (1, 2))

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective TLBO
        
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
        
        # Initialize population
        population = self._init_population(search_agents_no)
        
        # Initialize archive with non-dominated solutions
        self._determine_domination(population)
        non_dominated = self._get_non_dominated_particles(population)
        self.archive.extend(non_dominated)
        
        # Initialize grid for archive
        costs = self._get_fitness(self.archive)
        if costs.size > 0:
            self.grid = self._create_hypercubes(costs)
            for particle in self.archive:
                particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Partner selection for all students (shuffle indices)
            partner_indices = np.random.permutation(search_agents_no)
            
            # Process each student in the population
            for i in range(search_agents_no):
                # ----------------- Teacher Phase ----------------- #
                # Calculate mean of population
                mean_position = self._calculate_mean_position(population)
                
                # Find the teacher from archive using grid-based selection
                teacher = self._select_teacher()
                
                # If no teacher in archive, use best from current population
                if teacher is None:
                    teacher = self._find_best_in_population(population)
                
                # Generate teaching factor (TF)
                tf = self._generate_teaching_factor()
                
                # Generate new solution in teacher phase
                new_position_teacher = self._teacher_phase_update(
                    population[i].position, teacher.position, mean_position, tf
                )
                
                # Apply bounds and evaluate
                new_position_teacher = np.clip(new_position_teacher, self.lb, self.ub)
                new_fitness_teacher = self.objective_func(new_position_teacher)
                
                # Create new member for comparison
                new_member_teacher = MultiObjectiveMember(new_position_teacher, new_fitness_teacher)
                
                # Greedy selection for teacher phase using Pareto dominance
                if not self._dominates(population[i], new_member_teacher):
                    population[i].position = new_position_teacher
                    population[i].multi_fitness = new_fitness_teacher
                
                # ----------------- Learner Phase ----------------- #
                # Get partner index
                partner_idx = partner_indices[i]
                
                # Skip if partner is the same as current student
                if partner_idx == i:
                    partner_idx = (partner_idx + 1) % search_agents_no
                
                # Generate new solution in learner phase
                new_position_learner = self._learner_phase_update(
                    population[i].position, population[partner_idx].position
                )
                
                # Apply bounds and evaluate
                new_position_learner = np.clip(new_position_learner, self.lb, self.ub)
                new_fitness_learner = self.objective_func(new_position_learner)
                
                # Create new member for comparison
                new_member_learner = MultiObjectiveMember(new_position_learner, new_fitness_learner)
                
                # Greedy selection for learner phase using Pareto dominance
                if not self._dominates(population[i], new_member_learner):
                    population[i].position = new_position_learner
                    population[i].multi_fitness = new_fitness_learner
            
            # Update archive with current population
            self._add_to_archive(population)
            
            # Store archive state for history
            history_archive.append([student.copy() for student in self.archive])
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
    
    def _calculate_mean_position(self, population: List[MultiObjectiveMember]) -> np.ndarray:
        """Calculate the mean position of the population."""
        positions = np.array([member.position for member in population])
        return np.mean(positions, axis=0)
    
    def _select_teacher(self) -> MultiObjectiveMember:
        """Select teacher from archive using grid-based selection."""
        return self._select_leader()
    
    def _find_best_in_population(self, population: List[MultiObjectiveMember]) -> MultiObjectiveMember:
        """Find a good solution in population using grid-based selection."""
        if not population:
            return None
        
        # Use grid-based selection if archive has members
        if self.archive:
            # Select multiple leaders and choose one
            leaders = self._select_multiple_leaders(min(3, len(self.archive)))
            if leaders:
                return np.random.choice(leaders)
        
        # Fallback: return random member from population
        return np.random.choice(population)
    
    def _generate_teaching_factor(self) -> int:
        """Generate a random teaching factor (TF)."""
        tf_min, tf_max = self.teaching_factor_range
        return np.random.randint(tf_min, tf_max + 1)
    
    def _teacher_phase_update(self, current_position: np.ndarray, teacher_position: np.ndarray, 
                             mean_position: np.ndarray, teaching_factor: int) -> np.ndarray:
        """
        Update position in teacher phase.
        
        Formula: X_new = X_old + r * (Teacher - TF * Mean)
        """
        r = np.random.random(self.dim)
        return current_position + r * (teacher_position - teaching_factor * mean_position)
    
    def _learner_phase_update(self, current_position: np.ndarray, partner_position: np.ndarray) -> np.ndarray:
        """
        Update position in learner phase.
        
        Formula: 
        If current dominates partner: X_new = X_old + r * (X_old - X_partner)
        Else: X_new = X_old + r * (X_partner - X_old)
        """
        r = np.random.random(self.dim)
        
        # Create temporary members for dominance comparison
        current_member = MultiObjectiveMember(current_position, self.objective_func(current_position))
        partner_member = MultiObjectiveMember(partner_position, self.objective_func(partner_position))
        
        if self._dominates(current_member, partner_member):
            return current_position + r * (current_position - partner_position)
        else:
            return current_position + r * (partner_position - current_position)
