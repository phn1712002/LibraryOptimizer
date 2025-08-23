import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from utils.general import sort_population


class TeachingLearningBasedOptimizer(Solver):
    """
    Teaching Learning Based Optimization algorithm.
    
    TeachingLearningBased is a population-based optimization algorithm inspired by the teaching-learning
    process in a classroom. It consists of two main phases:
    1. Teacher Phase: Students learn from the teacher (best solution)
    2. Learner Phase: Students learn from each other through interaction
    
    Parameters:
    -----------
    objective_func : Callable
        Objective function to optimize
    lb : Union[float, np.ndarray]
        Lower bounds for variables
    ub : Union[float, np.ndarray]
        Upper bounds for variables  
    dim : int
        Problem dimension
    maximize : bool, optional
        Optimization direction, default is True (maximize)
    **kwargs
        Additional algorithm parameters
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize)
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name
        self.name_solver = "Teaching Learning Based Optimizer"
        
        # Set TeachingLearningBased-specific parameters
        self.teaching_factor_range = kwargs.get('teaching_factor_range', (1, 2))  # TF range (min, max)

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Execute the TeachingLearningBased optimization algorithm.
        
        Parameters:
        -----------
        search_agents_no : int
            Number of search agents (population size)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, Member]
            History of best solutions and the best solution found
        """
        # Initialize storage variables
        history_step_solver = []
        
        # Initialize the population
        population = self._init_population(search_agents_no)
        
        # Initialize best solution
        sorted_population, _ = self._sort_population(population)
        best_solution = sorted_population[0].copy()
        
        # Call the begin function
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
                
                # Find the teacher (best solution in current population)
                teacher = self._find_teacher(population)
                
                # Generate teaching factor (TF)
                tf = self._generate_teaching_factor()
                
                # Generate new solution in teacher phase
                new_position_teacher = self._teacher_phase_update(
                    population[i].position, teacher.position, mean_position, tf
                )
                
                # Apply bounds and evaluate
                new_position_teacher = np.clip(new_position_teacher, self.lb, self.ub)
                new_fitness_teacher = self.objective_func(new_position_teacher)
                
                # Greedy selection for teacher phase
                if self._is_better(Member(new_position_teacher, new_fitness_teacher), population[i]):
                    population[i].position = new_position_teacher
                    population[i].fitness = new_fitness_teacher
                
                # ----------------- Learner Phase ----------------- #
                # Get partner index
                partner_idx = partner_indices[i]
                
                # Generate new solution in learner phase
                new_position_learner = self._learner_phase_update(
                    population[i].position, population[partner_idx].position
                )
                
                # Apply bounds and evaluate
                new_position_learner = np.clip(new_position_learner, self.lb, self.ub)
                new_fitness_learner = self.objective_func(new_position_learner)
                
                # Greedy selection for learner phase
                if self._is_better(Member(new_position_learner, new_fitness_learner), population[i]):
                    population[i].position = new_position_learner
                    population[i].fitness = new_fitness_learner
            
            # Update best solution
            sorted_population, _ = self._sort_population(population)
            current_best = sorted_population[0]
            if self._is_better(current_best, best_solution):
                best_solution = current_best.copy()
            
            # Store the best solution at this iteration
            history_step_solver.append(best_solution.copy())
            
            # Call the callbacks
            self._callbacks(iter, max_iter, best_solution)
        
        # Final evaluation and storage
        self.history_step_solver = history_step_solver
        self.best_solver = best_solution
        
        # Call the end function
        self._end_step_solver()
        
        return history_step_solver, best_solution
    
    def _calculate_mean_position(self, population: List[Member]) -> np.ndarray:
        """Calculate the mean position of the population."""
        positions = np.array([member.position for member in population])
        return np.mean(positions, axis=0)
    
    def _find_teacher(self, population: List[Member]) -> Member:
        """Find the teacher (best solution) in the population."""
        sorted_population, _ = self._sort_population(population)
        return sorted_population[0]
    
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
        If current is better than partner: X_new = X_old + r * (X_old - X_partner)
        Else: X_new = X_old + r * (X_partner - X_old)
        """
        r = np.random.random(self.dim)
        current_fitness = self.objective_func(current_position)
        partner_fitness = self.objective_func(partner_position)
        
        if (self.maximize and current_fitness > partner_fitness) or \
           (not self.maximize and current_fitness < partner_fitness):
            return current_position + r * (current_position - partner_position)
        else:
            return current_position + r * (partner_position - current_position)
    
    def _sort_population(self, population):
        """Sort population using the utility function."""
        return sort_population(population, self.maximize)
