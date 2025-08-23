import numpy as np
from typing import Callable, Union, Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm

class Member:
    def __init__(self, position:np.ndarray, fitness:float):
        self.position = position
        self.fitness = fitness
    
    def copy(self):
        return Member(self.position.copy(), self.fitness)
    
    def __gt__(self, other): 
        if isinstance(other, Member):
            return self.fitness > other.fitness
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Member):
            return self.fitness < other.fitness
        return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, Member):
            return self.fitness == other.fitness
        return NotImplemented
    
    def __str__(self):
         return f"Position: {self.position} - Fitness: {self.fitness}"
    
class Solver:
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True):
        self.objective_func = objective_func
        self.dim = dim
        self.lb = np.array(lb) if hasattr(lb, '__iter__') else np.full(dim, lb)
        self.ub = np.array(ub) if hasattr(ub, '__iter__') else np.full(dim, ub)
        self.maximize = maximize
        self.history_step_solver = []
        self.best_solver = Member(np.random.uniform(lb, ub, dim), -np.inf if maximize else np.inf)
        
        self.pbar = None
        self.name_solver = ""

    def solver(self) -> Tuple[List, Member]:
        return self.history_step_solver, self.best_solver

    def _is_better(self, member_1, menber_2) -> bool:
        """Check if fitness1 is better than fitness2 based on optimization direction"""
        if self.maximize:
            return member_1 > menber_2
        else:
            return member_1 < menber_2
    
    def _init_population(self, search_agents_no) -> List:
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(Member(position, fitness))
        return population

    def _sort_population(self, population) -> Tuple[List, List]:
        # Extract fitness values from population
        fitness_values = [member.fitness for member in population]
        
        # Sort indices based on optimization direction
        if self.maximize:
            # Sort in descending order for maximization
            sorted_indices = np.argsort(fitness_values)[::-1]
        else:
            # Sort in ascending order for minimization
            sorted_indices = np.argsort(fitness_values)
        
        # Sort population based on sorted indices
        sorted_population = [population[i] for i in sorted_indices]
        
        return sorted_population, sorted_indices.tolist()
    
    def _callbacks(self, iter, max_iter, best) -> None:
        # Update progress bar with current iteration and best fitness
        self.pbar.update(1)
        self.pbar.set_postfix({
            'Iter': f'{iter+1}/{max_iter}',
            'Best Fitness': f'{best.fitness:.6f}'
        })
    def _begin_step_solver(self, max_iter) -> None:
        # Initialize tqdm progress bar
        self.pbar = tqdm(total=max_iter, desc=self.name_solver, unit="iter")

    def _end_step_solver(self) -> None:
        # Close the progress bar
        self.pbar.close()
        self.plot_history_step_solver()
        
    def plot_history_step_solver(self) -> None:
        """Plot the optimization history showing best fitness over iterations"""
        if self.history_step_solver is None:
            print("No optimization history available. Run the solver first.")
            return
        
        # Extract fitness values from history
        fitness_history = [member.fitness for member in self.history_step_solver]
        iterations = range(len(fitness_history))
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, fitness_history, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.title('Optimization History')
        plt.grid(True)
        
        if self.maximize:
            plt.axhline(y=max(fitness_history), color='r', linestyle='--', 
                        label=f'Max Fitness: {max(fitness_history):.6f}')
        else:
            plt.axhline(y=min(fitness_history), color='r', linestyle='--',
                        label=f'Min Fitness: {min(fitness_history):.6f}')
        
        plt.legend()
        plt.tight_layout()
        plt.show()