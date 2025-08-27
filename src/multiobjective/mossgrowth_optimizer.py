import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember


class MultiObjectiveMossGrowthOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Moss Growth Optimization (MGO) algorithm.
    
    This algorithm extends the standard MGO for multi-objective optimization
    using archive management and grid-based selection.
    
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
        - w: Inertia weight parameter (default: 2.0)
        - rec_num: Number of positions to record for cryptobiosis (default: 10)
        - divide_num: Number of dimensions to divide (default: dim/4)
        - d1: Probability threshold for spore dispersal (default: 0.2)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Moss Growth Optimizer"
        
        # Set default MGO parameters
        self.w = kwargs.get('w', 2.0)  # Inertia weight parameter
        self.rec_num = kwargs.get('rec_num', 10)  # Number of positions to record
        self.divide_num = kwargs.get('divide_num', max(1, int(dim / 4)))  # Dimensions to divide
        self.d1 = kwargs.get('d1', 0.2)  # Probability threshold for spore dispersal

    def _init_population(self, search_agents_no) -> List[MultiObjectiveMember]:
        """Initialize multi-objective population with custom member class"""
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(MultiObjectiveMember(position, fitness))
        return population

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective MGO
        
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
        
        # Initialize cryptobiosis mechanism
        rM = np.zeros((search_agents_no, self.dim, self.rec_num))  # Record history positions
        rM_cos = np.zeros((search_agents_no, self.n_objectives, self.rec_num))  # Record history multi-fitness
        
        # Initialize record counter
        rec = 0
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Calculate maximum function evaluations
        max_fes = max_iter * search_agents_no
        
        # Main optimization loop
        for iter in range(max_iter):
            # Calculate current function evaluations
            current_fes = iter * search_agents_no
            
            # Calculate progress ratio for adaptive parameters
            progress_ratio = current_fes / max_fes if max_fes > 0 else 0
            
            # Record the first generation of positions for cryptobiosis
            if rec == 0:
                for i in range(search_agents_no):
                    rM[i, :, rec] = population[i].position
                    rM_cos[i, :, rec] = population[i].multi_fitness
                rec += 1
            
            # Process each search agent
            for i in range(search_agents_no):
                # Select leader from archive using grid-based selection
                leader = self._select_leader()
                if leader is None:
                    # If no leader in archive, use random population member
                    leader = np.random.choice(population)
                
                # Select calculation positions based on majority regions
                cal_positions = np.array([member.position for member in population])
                
                # Divide the population and select regions with more individuals
                div_indices = np.random.permutation(self.dim)
                for j in range(min(self.divide_num, self.dim)):
                    th = leader.position[div_indices[j]]
                    index = cal_positions[:, div_indices[j]] > th
                    if np.sum(index) < cal_positions.shape[0] / 2:
                        index = ~index  # Choose the side of the majority
                    cal_positions = cal_positions[index, :]
                    if cal_positions.shape[0] == 0:
                        break
                
                if cal_positions.shape[0] > 0:
                    # Compute distance from individuals to the leader
                    D = leader.position - cal_positions
                    
                    # Calculate the mean of all distances (wind direction)
                    D_wind = np.mean(D, axis=0)
                    
                    # Calculate beta and gamma parameters
                    beta = cal_positions.shape[0] / search_agents_no
                    gamma = 1 / np.sqrt(1 - beta**2) if beta < 1 else 1.0
                    
                    # Calculate step sizes
                    step = self.w * (np.random.random(self.dim) - 0.5) * (1 - progress_ratio)
                    step2 = (0.1 * self.w * (np.random.random(self.dim) - 0.5) * 
                            (1 - progress_ratio) * (1 + 0.5 * (1 + np.tanh(beta / gamma)) * 
                            (1 - progress_ratio)) if gamma > 0 else 0)
                    step3 = 0.1 * (np.random.random() - 0.5) * (1 - progress_ratio)
                    
                    # Calculate activation function
                    act_input = 1 / (1 + (0.5 - 10 * (np.random.random(self.dim) - 0.5)))
                    act = self._act_cal(act_input)
                    
                    # Spore dispersal search
                    if np.random.random() > self.d1:
                        new_position = population[i].position + step * D_wind
                    else:
                        new_position = population[i].position + step2 * D_wind
                    
                    # Dual propagation search
                    if np.random.random() < 0.8:
                        if np.random.random() > 0.5:
                            # Update specific dimension
                            if self.dim > 0:
                                dim_idx = div_indices[0] if len(div_indices) > 0 else 0
                                new_position[dim_idx] = (leader.position[dim_idx] + 
                                                        step3 * D_wind[dim_idx])
                        else:
                            # Update all dimensions with activation
                            new_position = ((1 - act) * new_position + 
                                          act * leader.position)
                    
                    # Boundary absorption
                    new_position = np.clip(new_position, self.lb, self.ub)
                    
                    # Evaluate new fitness
                    new_fitness = self.objective_func(new_position)
                    
                    # Update population member
                    population[i].position = new_position
                    population[i].multi_fitness = new_fitness
                    
                    # Record for cryptobiosis mechanism
                    if rec < self.rec_num:
                        rM[i, :, rec] = new_position
                        rM_cos[i, :, rec] = new_fitness
                else:
                    # If no calculation positions, keep current position
                    pass
            
            # Cryptobiosis mechanism - apply after recording
            if rec >= self.rec_num or iter == max_iter - 1:
                # Find best historical position for each agent using Pareto dominance
                for i in range(search_agents_no):
                    # Create temporary members for historical positions
                    historical_members = []
                    for j in range(rec):
                        hist_member = MultiObjectiveMember(rM[i, :, j], rM_cos[i, :, j])
                        historical_members.append(hist_member)
                    
                    # Determine domination among historical positions
                    self._determine_domination(historical_members)
                    
                    # Get non-dominated historical positions
                    non_dominated_hist = self._get_non_dominated_particles(historical_members)
                    
                    if non_dominated_hist:
                        # Randomly select one non-dominated historical position
                        selected_hist = np.random.choice(non_dominated_hist)
                        population[i].position = selected_hist.position
                        population[i].multi_fitness = selected_hist.multi_fitness
                
                # Reset record counter
                rec = 0
            else:
                rec += 1
            
            # Update archive with current population
            self._add_to_archive(population)
            
            # Store archive state for history
            history_archive.append([moss.copy() for moss in self.archive])
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
    
    def _act_cal(self, X: np.ndarray) -> np.ndarray:
        """
        Activation function calculation.
        
        Parameters:
        -----------
        X : np.ndarray
            Input values
            
        Returns:
        --------
        np.ndarray
            Activated values (0 or 1)
        """
        act = X.copy()
        act[act >= 0.5] = 1
        act[act < 0.5] = 0
        return act
