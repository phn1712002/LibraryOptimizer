import numpy as np
from typing import Callable, Union, Tuple, List
import matplotlib.pyplot as plt

class Member:
    def __init__(self, position:np.ndarray, fitness:float):
        self.position = position
        self.fitness = fitness
    
    def copy(self):
        return Member(self.position, self.fitness)

class Solver:
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True):
        self.objective_func = objective_func
        self.dim = dim
        self.lb = np.array(lb) if hasattr(lb, '__iter__') else np.full(dim, lb)
        self.ub = np.array(ub) if hasattr(ub, '__iter__') else np.full(dim, ub)
        self.maximize = maximize
        self.history_step_solver = None
        self.best_solver = None
        
    def solver(self) -> Tuple[np.ndarray, float]:
        pass

    def _is_better(self, fitness1: float, fitness2: float) -> bool:
        """Check if fitness1 is better than fitness2 based on optimization direction"""
        if self.maximize:
            return fitness1 > fitness2
        else:
            return fitness1 < fitness2
    
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

    def plot_history_step_solver(self):
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
        plt.grid(True, alpha=0.3)
        
        if self.maximize:
            plt.axhline(y=max(fitness_history), color='r', linestyle='--', alpha=0.7, 
                        label=f'Max Fitness: {max(fitness_history):.6f}')
        else:
            plt.axhline(y=min(fitness_history), color='r', linestyle='--', alpha=0.7,
                        label=f'Min Fitness: {min(fitness_history):.6f}')
        
        plt.legend()
        plt.tight_layout()
        plt.show()

    def summary_solver(self):
        """Print a summary of the optimization results"""
        if self.best_solver is None:
            print("No optimization results available. Run the solver first.")
            return
        
        print("=" * 50)
        print("OPTIMIZATION SUMMARY")
        print("=" * 50)
        print(f"Best solution position: {self.best_solver.position}")
        print(f"Best fitness value: {self.best_solver.fitness}")
        print(f"Optimization type: {'Maximization' if self.maximize else 'Minimization'}")
        
        if self.history_step_solver:
            fitness_history = [member.fitness for member in self.history_step_solver]
            print(f"Number of iterations: {len(fitness_history)}")
            print(f"Final fitness: {fitness_history[-1]}")
            print(f"Best fitness: {max(fitness_history) if self.maximize else min(fitness_history)}")
            print(f"Fitness improvement: {fitness_history[-1] - fitness_history[0]:.6f}")
        
        print("=" * 50)
