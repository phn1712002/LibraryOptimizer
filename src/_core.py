import numpy as np
from typing import Callable, Union, Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class Member:
    """
    Represents an individual member/solution in the population.
    
    Attributes:
        position (np.ndarray): Current position in the search space
        fitness (float): Fitness value of the current position
    """
    
    def __init__(self, position: np.ndarray, fitness: float):
        """
        Initialize a Member with position and fitness.
        
        Args:
            position (np.ndarray): Position vector in search space
            fitness (float): Fitness value of the position
        """
        self.position = position
        self.fitness = fitness
    
    def copy(self):
        """
        Create a deep copy of the Member.
        
        Returns:
            Member: A new Member object with copied position and fitness
        """
        return Member(self.position.copy(), self.fitness)
    
    def __gt__(self, other): 
        """
        Greater than comparison based on fitness values.
        
        Args:
            other (Member): Another Member to compare with
            
        Returns:
            bool: True if this member's fitness is greater than the other's
        """
        if isinstance(other, Member):
            return self.fitness > other.fitness
        return NotImplemented

    def __lt__(self, other):
        """
        Less than comparison based on fitness values.
        
        Args:
            other (Member): Another Member to compare with
            
        Returns:
            bool: True if this member's fitness is less than the other's
        """
        if isinstance(other, Member):
            return self.fitness < other.fitness
        return NotImplemented
    
    def __eq__(self, other):
        """
        Equality comparison based on fitness values.
        
        Args:
            other (Member): Another Member to compare with
            
        Returns:
            bool: True if this member's fitness equals the other's
        """
        if isinstance(other, Member):
            return self.fitness == other.fitness
        return NotImplemented
    
    def __str__(self):
        """
        String representation of the Member.
        
        Returns:
            str: Formatted string showing position and fitness
        """
        return f"Position: {self.position} - Fitness: {self.fitness}"
    
class Solver:
    """
    Base class for optimization solvers.
    
    This abstract base class provides common functionality for all optimization
    algorithms including population initialization, progress tracking, and
    result visualization.
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        """
        Initialize the Solver base class.
        
        Args:
            objective_func (Callable): Objective function to optimize
            lb (Union[float, np.ndarray]): Lower bounds of search space
            ub (Union[float, np.ndarray]): Upper bounds of search space
            dim (int): Number of dimensions in the problem
            maximize (bool): Whether to maximize (True) or minimize (False) objective
            **kwargs: Additional solver parameters including:
                - show_chart (bool): Whether to display convergence chart (default: True)
        """
        self.objective_func = objective_func
        self.dim = dim
        # Convert bounds to numpy arrays for vectorized operations
        self.lb = np.array(lb) if hasattr(lb, '__iter__') else np.full(dim, lb)
        self.ub = np.array(ub) if hasattr(ub, '__iter__') else np.full(dim, ub)
        self.maximize = maximize
        self.show_chart = kwargs.get('show_chart', True)
        # Initialize optimization history and best solution
        self.history_step_solver = []
        self.best_solver = Member(np.random.uniform(lb, ub, dim), -np.inf if maximize else np.inf)
        
        self.pbar = None  # Progress bar instance
        self.name_solver = ""  # Solver name for display

    def _get_positions(self, population) -> np.ndarray:
        return np.array([member.position for member in population])

    def _get_fitness(self, population) -> np.ndarray:
        return np.array([member.fitness for member in population])

    def _is_better(self, member_1, member_2) -> bool:
        """
        Compare two members to determine which is better based on optimization direction.
        
        Args:
            member_1 (Member): First member to compare
            member_2 (Member): Second member to compare
            
        Returns:
            bool: True if member_1 is better than member_2 according to optimization direction
        """
        if self.maximize:
            return member_1 > member_2
        else:
            return member_1 < member_2
    
    def _init_population(self, search_agents_no) -> List:
        """
        Initialize a population of members with random positions.
        
        Args:
            search_agents_no (int): Number of members to initialize
            
        Returns:
            List[Member]: List of initialized members with random positions and evaluated fitness
        """
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(Member(position, fitness))
        return population
    
    def _callbacks(self, iter, max_iter, best) -> None:
        """
        Update progress tracking during optimization.
        
        Args:
            iter (int): Current iteration number
            max_iter (int): Maximum number of iterations
            best (Member): Current best member
        """
        # Update progress bar with current iteration and best fitness
        self.pbar.update(1)
        self.pbar.set_postfix({
            'Iter': f'{iter+1}/{max_iter}',
            'Best Fitness': f'{best.fitness:.6f}'
        })
        
    def _begin_step_solver(self, max_iter) -> None:
        """
        Initialize solver execution and display startup information.
        
        Args:
            max_iter (int): Maximum number of iterations for the solver
        """
        # Clear console for better visualization
        #os.system('cls' if os.name == 'nt' else 'clear')
        
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
        """
        Finalize solver execution and display results.
        """
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
        """
        Plot the optimization history showing convergence over iterations.
        
        Displays a line plot of the best fitness value found at each iteration.
        Handles different environments including Google Colab.
        """
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
        
        # Add horizontal line showing the best fitness achieved
        if self.maximize:
            plt.axhline(y=max(fitness_history), color='r', linestyle='--', 
                        label=f'Max Fitness: {max(fitness_history):.6f}')
        else:
            plt.axhline(y=min(fitness_history), color='r', linestyle='--',
                        label=f'Min Fitness: {min(fitness_history):.6f}')
        
        plt.legend()
        plt.tight_layout()
        
        # Handle different environments for displaying plots
        try:
            # Check if we're in Google Colab
            import google.colab
            # In Google Colab, we need to use a different approach
            from IPython.display import display
            plt.show()
        except ImportError:
            # Not in Google Colab, use standard plt.show()
            plt.show()
        except Exception as e:
            # Fallback: save to file and inform user
            plt.savefig('optimization_history.png')
            print(f"Plot could not be displayed. Saved as 'optimization_history.png'. Error: {e}")

    def solver(self) -> Tuple[List, Member]:
        """
        Get the optimization results.
        
        Returns:
            Tuple[List, Member]: Tuple containing:
                - history_step_solver: List of best solutions at each iteration
                - best_solver: Best solution found overall
        """
        return self.history_step_solver, self.best_solver
