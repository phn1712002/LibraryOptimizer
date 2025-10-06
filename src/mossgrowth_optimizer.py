import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from ._general import sort_population


class MossGrowthOptimizer(Solver):
    """
    Moss Growth Optimization (MGO) algorithm.
    
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
        Additional algorithm parameters including:
        - w: Inertia weight parameter (default: 2.0)
        - rec_num: Number of positions to record for cryptobiosis (default: 10)
        - divide_num: Number of dimensions to divide (default: dim/4)
        - d1: Probability threshold for spore dispersal (default: 0.2)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name
        self.name_solver = "Moss Growth Optimizer"
        
        # Set default MGO parameters
        self.w = kwargs.get('w', 2.0)  # Inertia weight parameter
        self.rec_num = kwargs.get('rec_num', 10)  # Number of positions to record
        self.divide_num = kwargs.get('divide_num', max(1, int(dim / 4)))  # Dimensions to divide
        self.d1 = kwargs.get('d1', 0.2)  # Probability threshold for spore dispersal

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        # Initialize storage variables
        history_step_solver = []
        
        # Initialize the population
        population = self._init_population(search_agents_no)
        
        # Initialize best solution
        sorted_population, _ = self._sort_population(population)
        best_solver = sorted_population[0].copy()
        
        # Initialize cryptobiosis mechanism
        rM = np.zeros((search_agents_no, self.dim, self.rec_num))  # Record history positions
        rM_cos = np.zeros((search_agents_no, self.rec_num))  # Record history costs
        
        # Initialize record counter
        rec = 0
        
        # Call the begin function
        self._begin_step_solver(max_iter)
        
        # Calculate maximum function evaluations
        max_fes = max_iter * search_agents_no
        
        # Main optimization loop
        for iter in range(max_iter):
            # Calculate current function evaluations
            current_fes = iter * search_agents_no
            
            # Calculate progress ratio for adaptive parameters
            progress_ratio = current_fes / max_fes if max_fes > 0 else 0
            
            # Initialize new population
            new_population = [member.copy() for member in population]
            new_costs = np.zeros(search_agents_no)
            
            # Record the first generation of positions for cryptobiosis
            if rec == 0:
                for i in range(search_agents_no):
                    rM[i, :, rec] = population[i].position
                    rM_cos[i, rec] = population[i].fitness
                rec += 1
            
            # Process each search agent
            for i in range(search_agents_no):
                # Select calculation positions based on majority regions
                cal_positions = np.array([member.position for member in population])
                
                # Divide the population and select regions with more individuals
                div_indices = np.random.permutation(self.dim)
                for j in range(min(self.divide_num, self.dim)):
                    th = best_solver.position[div_indices[j]]
                    index = cal_positions[:, div_indices[j]] > th
                    if np.sum(index) < cal_positions.shape[0] / 2:
                        index = ~index  # Choose the side of the majority
                    cal_positions = cal_positions[index, :]
                    if cal_positions.shape[0] == 0:
                        break
                
                if cal_positions.shape[0] > 0:
                    # Compute distance from individuals to the best
                    D = best_solver.position - cal_positions
                    
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
                                new_position[dim_idx] = (best_solver.position[dim_idx] + 
                                                        step3 * D_wind[dim_idx])
                        else:
                            # Update all dimensions with activation
                            new_position = ((1 - act) * new_position + 
                                          act * best_solver.position)
                    
                    # Boundary absorption
                    new_position = np.clip(new_position, self.lb, self.ub)
                    
                    # Evaluate new fitness
                    new_fitness = self.objective_func(new_position)
                    
                    # Update population member
                    new_population[i].position = new_position
                    new_population[i].fitness = new_fitness
                    new_costs[i] = new_fitness
                    
                    # Record for cryptobiosis mechanism
                    if rec < self.rec_num:
                        rM[i, :, rec] = new_position
                        rM_cos[i, rec] = new_fitness
                else:
                    # If no calculation positions, keep current position
                    new_costs[i] = population[i].fitness
            
            # Update population
            for i in range(search_agents_no):
                population[i] = new_population[i]
            
            # Cryptobiosis mechanism - apply after recording
            if rec >= self.rec_num or iter == max_iter - 1:
                # Find best historical position for each agent
                for i in range(search_agents_no):
                    best_idx = np.argmin(rM_cos[i, :rec]) if not self.maximize else np.argmax(rM_cos[i, :rec])
                    population[i].position = rM[i, :, best_idx]
                    population[i].fitness = rM_cos[i, best_idx]
                
                # Reset record counter
                rec = 0
            else:
                rec += 1
            
            # Update best solution
            sorted_population, _ = self._sort_population(population)
            current_best = sorted_population[0]
            if self._is_better(current_best, best_solver):
                best_solver = current_best.copy()
            
            # Store the best solution at this iteration
            history_step_solver.append(best_solver.copy())
            
            # Call the callbacks
            self._callbacks(iter, max_iter, best_solver)
        
        # Final evaluation and storage
        self.history_step_solver = history_step_solver
        self.best_solver = best_solver
        
        # Call the end function
        self._end_step_solver()
        
        return history_step_solver, best_solver
    
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
    
    def _sort_population(self, population):
        """
        Sort the population based on fitness.
        """
        return sort_population(population, self.maximize)
