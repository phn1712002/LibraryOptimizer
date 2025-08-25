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
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        self.objective_func = objective_func
        self.dim = dim
        self.lb = np.array(lb) if hasattr(lb, '__iter__') else np.full(dim, lb)
        self.ub = np.array(ub) if hasattr(ub, '__iter__') else np.full(dim, ub)
        self.maximize = maximize
        self.show_chart = kwargs.get('show_chart', True)
        self.history_step_solver = []
        self.best_solver = Member(np.random.uniform(lb, ub, dim), -np.inf if maximize else np.inf)
        
        self.pbar = None
        self.name_solver = ""

    def _is_better(self, member_1, menber_2) -> bool:
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
    
    def _callbacks(self, iter, max_iter, best) -> None:
        # Update progress bar with current iteration and best fitness
        self.pbar.update(1)
        self.pbar.set_postfix({
            'Iter': f'{iter+1}/{max_iter}',
            'Best Fitness': f'{best.fitness:.6f}'
        })
        
    def _begin_step_solver(self, max_iter) -> None:
        # Print algorithm start message with parameters
        print("-" * 50)
        print(f"🚀 Starting {self.name_solver} algorithm")
        print(f"📊 Parameters:")
        print(f"   - Problem dimension: {self.dim}")
        print(f"   - Lower bounds: {self.lb}")
        print(f"   - Upper bounds: {self.ub}")
        print(f"   - Optimization direction: {'Maximize' if self.maximize else 'Minimize'}")
        print(f"   - Maximum iterations: {max_iter}")
        if hasattr(self, 'kwargs') and self.kwargs:
            print(f"   - Additional parameters: {self.kwargs}")
        print(f"\n")
        # Initialize tqdm progress bar
        self.pbar = tqdm(total=max_iter, desc=self.name_solver, unit="iter")

    def _end_step_solver(self) -> None:
        # Close the progress bar
        self.pbar.close()
        print(f"\n")
        # Print algorithm completion message with results
        print(f"✅ {self.name_solver} algorithm completed!")
        print(f"🏆 Best solution found:")
        print(f"   - Position: {self.best_solver.position}")
        print(f"   - Fitness: {self.best_solver.fitness:.6f}")
        print("-" * 50)
        if self.show_chart:
            self.plot_history_step_solver()
        
    def plot_history_step_solver(self) -> None:
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

    def solver(self) -> Tuple[List, Member]:
        return self.history_step_solver, self.best_solver