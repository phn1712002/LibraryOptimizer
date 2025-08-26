import numpy as np
from typing import Callable, Union, Tuple, List
from .._core import Solver, Member
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class MultiObjectiveMember(Member):
    def __init__(self, position: np.ndarray, fitness: np.ndarray):
        # For compatibility, use first fitness value
        super().__init__(position, 0)
        self.multi_fitness = np.array(fitness)  # Store all fitness values
        self.dominated = False
        self.grid_index = None
        self.grid_sub_index = None
    
    def copy(self):
        new_member = MultiObjectiveMember(self.position.copy(), self.multi_fitness.copy())
        new_member.dominated = self.dominated
        new_member.grid_index = self.grid_index
        new_member.grid_sub_index = self.grid_sub_index
        return new_member
    

    def __str__(self):
        return f"Position: {self.position} - Fitness: {self.multi_fitness} - Dominated: {self.dominated}"

class MultiObjectiveSolver(Solver):
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        # Multi-objective doesn't use maximize flag in the same way
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        self.show_chart = kwargs.get('show_chart', True)

        # Multi-objective specific parameters
        self.n_objectives = objective_func(np.random.uniform(self.lb, self.ub, self.dim)).shape[0]
        self.archive_size = kwargs.get('archive_size', 100)
        self.archive = []
        
        # Grid-based selection parameters
        self.alpha = kwargs.get('alpha', 0.1)
        self.n_grid = kwargs.get('n_grid', 7)
        self.beta = kwargs.get('beta', 2)
        self.gamma = kwargs.get('gamma', 2)
        
        # Hypercube grid
        self.grid = None
    
    def _dominates(self, x: MultiObjectiveMember, y: MultiObjectiveMember) -> bool:
        """Check if x dominates y (Pareto dominance)"""
        x_costs = x.multi_fitness
        y_costs = y.multi_fitness
        
        if self.maximize:
            # For maximization: x dominates y if x >= y in all objectives and > y in at least one
            not_worse = np.all(x_costs >= y_costs)
            better = np.any(x_costs > y_costs)
        else:
            # For minimization: x dominates y if x <= y in all objectives and < y in at least one
            not_worse = np.all(x_costs <= y_costs)
            better = np.any(x_costs < y_costs)
        
        return not_worse and better
    
    def _determine_domination(self, population: List[MultiObjectiveMember]) -> None:
        """Determine domination status for all particles"""
        n_pop = len(population)
        
        for i in range(n_pop):
            population[i].dominated = False
            for j in range(n_pop):
                if i != j and not population[j].dominated:
                    if self._dominates(population[j], population[i]):
                        population[i].dominated = True
                        break
    
    def _get_non_dominated_particles(self, population: List[MultiObjectiveMember]) -> List[MultiObjectiveMember]:
        """Get non-dominated particles from population"""
        return [p for p in population if not p.dominated]
    
    def _get_fitness(self, population: List[MultiObjectiveMember]) -> np.ndarray:
        """Get fitness matrix from population"""
        if not population:
            return np.array([])
        return np.array([p.multi_fitness for p in population]).T

    def _get_random_population(self, population: List[MultiObjectiveMember], size: int) -> List[MultiObjectiveMember]:
        """
        Get a random sample of population with specified size
        
        Parameters:
        -----------
        population : List[MultiObjectiveMember]
            Source population to sample from
        size : int
            Number of members to sample
            
        Returns:
        --------
        List[MultiObjectiveMember]
            Random sample of population with specified size
        """
        if not population:
            return []
        
        if size <= 0:
            return []
        
        # If requested size is larger than population, return the entire population
        if size >= len(population):
            return population.copy()
        
        # Randomly select members from population
        selected_indices = np.random.choice(len(population), size, replace=False)
        return [population[i].copy() for i in selected_indices]
    
    def _create_hypercubes(self, costs: np.ndarray) -> List[dict]:
        """Create hypercubes for grid-based selection"""
        if costs.size == 0:
            return []
        
        n_obj = costs.shape[0]
        grid = []
        
        for j in range(n_obj):
            min_cj = np.min(costs[j, :])
            max_cj = np.max(costs[j, :])
            
            dcj = self.alpha * (max_cj - min_cj)
            min_cj = min_cj - dcj
            max_cj = max_cj + dcj
            
            gx = np.linspace(min_cj, max_cj, self.n_grid - 1)
            
            grid.append({
                'lower': np.concatenate([[-np.inf], gx]),
                'upper': np.concatenate([gx, [np.inf]])
            })
        
        return grid
    
    def _get_grid_index(self, particle: MultiObjectiveMember) -> Tuple[int, np.ndarray]:
        """Get grid index for a particle"""
        if self.grid is None:
            return None, None
        
        costs = particle.multi_fitness
        n_obj = len(costs)
        
        sub_index = np.zeros(n_obj, dtype=int)
        for j in range(n_obj):
            # Find the first upper bound that is greater than the cost
            upper_bounds = self.grid[j]['upper']
            idx = np.where(costs[j] < upper_bounds)[0]
            if len(idx) > 0:
                sub_index[j] = idx[0]
        
        # Convert multi-dimensional index to linear index
        grid_dims = [len(self.grid[j]['upper']) for j in range(n_obj)]
        linear_index = np.ravel_multi_index(sub_index, grid_dims, mode='clip')
        
        return linear_index, sub_index
    
    def _select_leader(self) -> MultiObjectiveMember:
        """Select leader from archive using grid-based selection"""
        if not self.archive:
            return None
        
        # Get grid indices of all archive members
        grid_indices = [p.grid_index for p in self.archive if p.grid_index is not None]
        
        if not grid_indices:
            return np.random.choice(self.archive)
        
        # Get occupied cells and their counts
        occupied_cells, counts = np.unique(grid_indices, return_counts=True)
        
        # Selection probabilities (lower density cells have higher probability)
        probabilities = np.exp(-self.beta * counts)
        probabilities = probabilities / np.sum(probabilities)
        
        # Select a cell using roulette wheel
        r = np.random.random()
        cum_probs = np.cumsum(probabilities)
        selected_cell_idx = np.where(r <= cum_probs)[0][0]
        selected_cell = occupied_cells[selected_cell_idx]
        
        # Get members in selected cell
        cell_members = [p for p in self.archive if p.grid_index == selected_cell]
        
        # Randomly select a member from the cell
        return np.random.choice(cell_members)
    
    def _select_multiple_leaders(self, n_leaders: int) -> List[MultiObjectiveMember]:
        """
        Select multiple unique leaders from archive using grid-based selection
        
        Parameters:
        -----------
        n_leaders : int
            Number of leaders to select
            
        Returns:
        --------
        List[MultiObjectiveMember]
            List of selected leaders (may be fewer than n_leaders if archive is small)
        """
        if not self.archive or n_leaders <= 0:
            return []
        
        # Get grid indices of all archive members
        grid_indices = [p.grid_index for p in self.archive if p.grid_index is not None]
        
        if not grid_indices:
            # If no grid indices, return random unique members
            n_available = min(n_leaders, len(self.archive))
            return list(np.random.choice(self.archive, size=n_available, replace=False))
        
        # Get occupied cells and their counts
        occupied_cells, counts = np.unique(grid_indices, return_counts=True)
        n_cells = len(occupied_cells)
        
        # If we need more leaders than available cells, we'll have to reuse cells
        if n_leaders > n_cells:
            # First select one leader from each cell
            leaders = []
            for cell in occupied_cells:
                cell_members = [p for p in self.archive if p.grid_index == cell]
                leaders.append(np.random.choice(cell_members))
            
            # Then fill remaining with random selection from all archive
            remaining = n_leaders - n_cells
            if remaining > 0:
                available_members = [p for p in self.archive if p not in leaders]
                if available_members:
                    additional = list(np.random.choice(available_members, 
                                                     size=min(remaining, len(available_members)), 
                                                     replace=False))
                    leaders.extend(additional)
            
            return leaders
        
        # Selection probabilities (lower density cells have higher probability)
        probabilities = np.exp(-self.beta * counts)
        probabilities = probabilities / np.sum(probabilities)
        
        # Select multiple unique cells without replacement
        selected_cells = []
        temp_probabilities = probabilities.copy()
        temp_cells = occupied_cells.copy()
        
        for _ in range(n_leaders):
            if len(temp_cells) == 0:
                break
                
            # Select a cell using roulette wheel
            r = np.random.random()
            cum_probs = np.cumsum(temp_probabilities)
            selected_cell_idx = np.where(r <= cum_probs)[0][0]
            selected_cell = temp_cells[selected_cell_idx]
            selected_cells.append(selected_cell)
            
            # Remove selected cell from consideration
            mask = temp_cells != selected_cell
            temp_cells = temp_cells[mask]
            temp_probabilities = temp_probabilities[mask]
            if len(temp_probabilities) > 0:
                temp_probabilities = temp_probabilities / np.sum(temp_probabilities)
        
        # Select one leader from each chosen cell
        leaders = []
        for cell in selected_cells:
            cell_members = [p for p in self.archive if p.grid_index == cell]
            leaders.append(np.random.choice(cell_members))
        
        return leaders
    
    def _sort_population(self, population: List[MultiObjectiveMember]) -> List[MultiObjectiveMember]:
        """
        Sort population with multi fitness by selecting leaders and sorting remaining by random fitness
        
        Parameters:
        -----------
        population : List[MultiObjectiveMember]
            Population to sort
            
        Returns:
        --------
        List[MultiObjectiveMember]
            Sorted population
        """
        if not population:
            return []
        
        n_pop = len(population)
        
        # Select multiple leaders from population
        leaders = self._select_multiple_leaders(n_pop)
        population_positions = self._get_positions(population)

        # Remove leaders that don't exist in population
        valid_leaders = [leader for leader in leaders if leader.position in population_positions]
        valid_leaders_positions = self._get_positions(valid_leaders)
        
        # Get population members that are not in leaders
        non_leader_population = [member for member in population if member.position not in valid_leaders_positions]
        
        # Sort non-leader population by random fitness
        # For maximization: higher random fitness is better
        # For minimization: lower random fitness is better
        
        # Sort based on optimization direction
        if self.maximize:
            # For maximization: sort descending by random fitness
            non_leader_population.sort(key=self._get_random_fitness, reverse=True)
        else:
            # For minimization: sort ascending by random fitness
            non_leader_population.sort(key=self._get_random_fitness)
        
        # Combine leaders and sorted non-leaders
        sorted_population = valid_leaders + non_leader_population
        
        # Ensure we return exactly the population size
        return sorted_population[:n_pop]
    
    def _get_random_fitness(self, member):
        """Get fitness value based on random dice roll"""
        dice_roll = np.random.random()
        if dice_roll > 0.5:
            # If dice > 0.5, return average of multi_fitness
            return np.average(member.multi_fitness)
        else:
            # If dice <= 0.5, return a random element from multi_fitness
            return np.random.choice(member.multi_fitness)
    
    def _add_to_archive(self, new_solutions: List[MultiObjectiveMember]) -> None:
        """Add new solutions to archive and maintain archive size"""
        # Determine domination of new solutions
        self._determine_domination(new_solutions)
        
        # Add non-dominated new solutions to archive
        non_dominated_new = self._get_non_dominated_particles(new_solutions)
        self.archive.extend(non_dominated_new)
        
        # Re-determine domination in the combined archive
        self._determine_domination(self.archive)
        
        # Keep only non-dominated solutions
        self.archive = self._get_non_dominated_particles(self.archive)
        
        # Update grid indices
        costs = self._get_fitness(self.archive)
        if costs.size > 0:
            self.grid = self._create_hypercubes(costs)
            for particle in self.archive:
                particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
        
        # Trim archive if it exceeds size limit
        if len(self.archive) > self.archive_size:
            self._trim_archive()
    
    def _trim_archive(self) -> None:
        """Trim archive to maintain size using grid-based removal"""
        extra = len(self.archive) - self.archive_size
        
        for _ in range(extra):
            # Get occupied cells and their counts
            grid_indices = [p.grid_index for p in self.archive if p.grid_index is not None]
            if not grid_indices:
                # Remove random member if no grid indices
                self.archive.pop(np.random.randint(len(self.archive)))
                continue
            
            occupied_cells, counts = np.unique(grid_indices, return_counts=True)
            
            # Selection probabilities (higher density cells have higher probability of removal)
            probabilities = counts ** self.gamma
            probabilities = probabilities / np.sum(probabilities)
            
            # Select a cell using roulette wheel
            r = np.random.random()
            cum_probs = np.cumsum(probabilities)
            selected_cell_idx = np.where(r <= cum_probs)[0][0]
            selected_cell = occupied_cells[selected_cell_idx]
            
            # Get members in selected cell
            cell_members = [p for p in self.archive if p.grid_index == selected_cell]
            
            # Randomly remove one member from the cell
            member_to_remove = np.random.choice(cell_members)
            self.archive.remove(member_to_remove)
    
    def _init_population(self, search_agents_no) -> List[MultiObjectiveMember]:
        """Initialize multi-objective population"""
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(MultiObjectiveMember(position, fitness))
        return population
    
    def _callbacks(self, iter: int, max_iter: int, best: MultiObjectiveMember) -> None:
        """Custom callback for multi-objective optimization"""
        if self.pbar:
            self.pbar.update(1)
            archive_size = len(self.archive)
            if best:
                fitness_str = f"[{', '.join([f'{f:.3f}' for f in best.multi_fitness])}]"
                self.pbar.set_postfix({
                    'Iter': f'{iter+1}/{max_iter}',
                    'Archive': archive_size,
                    'Best Fitness': fitness_str
                })
            else:
                self.pbar.set_postfix({
                    'Iter': f'{iter+1}/{max_iter}',
                    'Archive': archive_size,
                    'Best Fitness': 'N/A'
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
        print(f"   - Objectives dimension: {self.n_objectives}")
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
        """Custom end step for multi-objective optimization"""
        if self.pbar:
            self.pbar.close()
        
        print(f"\n✅ {self.name_solver} algorithm completed!")
        print(f"🏆 Archive contains {len(self.archive)} non-dominated solutions")
        if self.archive:
            print(f"📊 Pareto front statistics:")
            costs = self._get_fitness(self.archive)
            for i in range(costs.shape[0]):
                if self.maximize:
                    print(f"Objective {i+1}: worst={np.min(costs[i]):.6f}, best={np.max(costs[i]):.6f}")
                else:
                    print(f"Objective {i+1}: best={np.min(costs[i]):.6f}, worst={np.max(costs[i]):.6f}")
        print("-" * 50)
        
        # Plot Pareto front if we have at least 2 objectives
        if self.show_chart:
            if self.archive and len(self.archive[0].multi_fitness) >= 2:
                self.plot_pareto_front()

    def _tournament_selection_multi(self, population: List[MultiObjectiveMember], tournament_size: int) -> MultiObjectiveMember:
        """
        Tournament selection for multi-objective optimization using grid-based diversity
        
        Parameters:
        -----------
        population : List[MultiObjectiveMember]
            Population to select from
            
        Returns:
        --------
        MultiObjectiveMember
            Selected individual
        """
        if len(population) < tournament_size:
            return np.random.choice(population)
        
        # Randomly select tournament participants
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_members = [population[i] for i in tournament_indices]
        
        # For multi-objective, we need a different selection criteria
        # Use non-dominated sorting if possible, otherwise use grid-based selection
        
        # First, check if any members are non-dominated
        non_dominated = [m for m in tournament_members if not m.dominated]
        
        if non_dominated:
            # If we have non-dominated members, select from them using grid-based diversity
            if len(non_dominated) > 1:
                # Use grid index for diversity-based selection
                grid_indices = [m.grid_index for m in non_dominated if m.grid_index is not None]
                if grid_indices:
                    # Select the member from the least crowded cell
                    unique_indices, counts = np.unique(grid_indices, return_counts=True)
                    least_crowded_idx = unique_indices[np.argmin(counts)]
                    least_crowded_members = [m for m in non_dominated if m.grid_index == least_crowded_idx]
                    return np.random.choice(least_crowded_members)
            
            # Fallback: return random non-dominated member
            return np.random.choice(non_dominated)
        
        # If no non-dominated members, use random selection
        return np.random.choice(tournament_members)
    
    def plot_pareto_front(self) -> None:
        """Plot Pareto front from archive
        Handles different environments including Google Colab.
        """
        if not self.archive:
            print("No solutions in archive to plot.")
            return
        
        costs = self._get_fitness(self.archive)
        n_objectives = self.n_objectives
        
        if n_objectives == 2:
            # 2D plot
            plt.figure(figsize=(10, 6))
            plt.scatter(costs[0, :], costs[1, :], c='blue', alpha=0.7, s=50)
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
            plt.title('Pareto Front (2D)')
            plt.grid(True)
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
                plt.savefig('pareto_front_2d.png')
                print(f"Plot could not be displayed. Saved as 'pareto_front_2d.png'. Error: {e}")
        
        elif n_objectives == 3:
            # 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(costs[0, :], costs[1, :], costs[2, :], c='blue', alpha=0.7, s=50)
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.set_zlabel('Objective 3')
            ax.set_title('Pareto Front (3D)')
            plt.grid(True)
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
                plt.savefig('pareto_front_3d.png')
                print(f"Plot could not be displayed. Saved as 'pareto_front_3d.png'. Error: {e}")
        
        else:
            print(f"Cannot plot Pareto front for {n_objectives} objectives. Maximum 3D visualization supported.")
            # Optionally, you could plot pairwise scatter plots here
            print("Consider plotting pairwise scatter plots for higher dimensions.")

    def solver(self) -> Tuple[List, List]:
        """
        Get the multi-objective optimization results.
        
        Returns:
            Tuple[List, List]: Tuple containing:
                - history_step_solver: List of best solutions found at each iteration
                - archive: List of non-dominated solutions (Pareto front approximation)
        """
        return self.history_step_solver, self.archive
