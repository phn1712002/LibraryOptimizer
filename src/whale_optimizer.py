import numpy as np
from typing import Callable, Union, Tuple, List
from .core import Solver, Member

class WhaleOptimizer(Solver):
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize)
        # Store additional parameters for later use
        self.kwargs = kwargs
        self.name_solver = "Whale Optimizer"

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        # Initialize the population of search agents and history_step_solver
        population = self._init_population(search_agents_no)
        
        # Initialize storage variables
        history_step_solver = []
        best_solver = self.best_solver
        
        # Call the begin function
        self._begin_step_solver(max_iter)

        # Initialize leader
        _, idx = self._sort_population(population)
        leader = population[idx[0]].copy()

        # Main optimization loop
        for iter in range(max_iter):
            # Update a parameters (decreases linearly)
            a = 2 - iter * (2 / max_iter)
            a2 = -1 + iter * ((-1) / max_iter)
            
            # Update positions of all whales and evaluate fitness
            for i, member in enumerate(population):
                new_position = np.zeros(self.dim)
                
                for j in range(self.dim):
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    A = 2 * a * r1 - a  # Eq. (2.3) in the paper
                    C = 2 * r2          # Eq. (2.4) in the paper
                    
                    b = 1               # parameters in Eq. (2.5)
                    l = (a2 - 1) * np.random.random() + 1  # parameters in Eq. (2.5)
                    
                    p = np.random.random()  # p in Eq. (2.6)
                    
                    if p < 0.5:
                        if abs(A) >= 1:
                            # Search for prey (exploration phase)
                            rand_leader_index = np.random.randint(0, search_agents_no)
                            X_rand = population[rand_leader_index].position
                            D_X_rand = abs(C * X_rand[j] - member.position[j])  # Eq. (2.7)
                            new_position[j] = X_rand[j] - A * D_X_rand  # Eq. (2.8)
                        else:
                            # Encircling prey (exploitation phase)
                            D_leader = abs(C * leader.position[j] - member.position[j])  # Eq. (2.1)
                            new_position[j] = leader.position[j] - A * D_leader  # Eq. (2.2)
                    else:
                        # Bubble-net attacking method (spiral updating position)
                        distance_to_leader = abs(leader.position[j] - member.position[j])
                        # Eq. (2.5) - spiral movement
                        new_position[j] = distance_to_leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + leader.position[j]
                
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Update member position and evaluate fitness
                population[i].position = new_position
                population[i].fitness = self.objective_func(new_position)
                # Update leader immediately if better solution found (MATLAB-style)
                if self._is_better(population[i], best_solver):
                    best_solver = population[i].copy()

           
            # Store the best solution at this iteration
            history_step_solver.append(best_solver)
            #  Call the callbacks 
            self._callbacks(iter, max_iter, best_solver)

        # Final leader is the best solution found
        self.history_step_solver = history_step_solver
        self.best_solver = best_solver

        # Call the end function
        self._end_step_solver()
        return history_step_solver, best_solver
        