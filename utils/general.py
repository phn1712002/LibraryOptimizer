import numpy as np
from typing import Tuple, List
from src.core import Member
import math

def roulette_wheel_selection(probabilities: np.ndarray) -> int:
    r = np.random.random()
    cumulative_sum = np.cumsum(probabilities)
    return np.argmax(r <= cumulative_sum)


def sort_population(population, maximize) -> Tuple[List, List]:
    # Validate that all members are instances of Member class
    for i, member in enumerate(population):
        if not isinstance(member, Member):
            raise TypeError(f"Population member at index {i} is not an instance of Member class. Got {type(member)}")
    
    # Extract fitness values from population
    fitness_values = [member.fitness for member in population]
    
    # Sort indices based on optimization direction
    if maximize:
        # Sort in descending order for maximization
        sorted_indices = np.argsort(fitness_values)[::-1]
    else:
        # Sort in ascending order for minimization
        sorted_indices = np.argsort(fitness_values)
    
    # Sort population based on sorted indices
    sorted_population = [population[i] for i in sorted_indices]
    
    return sorted_population, sorted_indices.tolist()


def tournament_selection(population: List, tournament_size: int = 3, maximize: bool = True) -> Member:
    """
    Tournament selection for choosing individuals from population.
    
    Parameters:
    -----------
    population : List[Member]
        Population to select from
    tournament_size : int, optional
        Number of individuals to compete in each tournament, default is 3
    maximize : bool, optional
        Optimization direction, default is True (maximize)
        
    Returns:
    --------
    Member
        Selected individual
    """
    if len(population) < tournament_size:
        raise ValueError(f"Tournament size ({tournament_size}) cannot be larger than population size ({len(population)})")
    
    # Randomly select tournament participants
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_members = [population[i] for i in tournament_indices]
    
    # Sort tournament participants
    sorted_tournament, _ = sort_population(tournament_members, maximize)
    
    # Return the best individual from the tournament
    return sorted_tournament[0]


def get_best_solution(population: List, maximize: bool = True) -> Member:
    """
    Get the best solution from a population.
    
    Parameters:
    -----------
    population : List[Member]
        Population to evaluate
    maximize : bool, optional
        Optimization direction, default is True (maximize)
        
    Returns:
    --------
    Member
        Best solution in the population
    """
    sorted_population, _ = sort_population(population, maximize)
    return sorted_population[0]


def get_worst_solution(population: List, maximize: bool = True) -> Member:
    """
    Get the worst solution from a population.
    
    Parameters:
    -----------
    population : List[Member]
        Population to evaluate
    maximize : bool, optional
        Optimization direction, default is True (maximize)
        
    Returns:
    --------
    Member
        Worst solution in the population
    """
    sorted_population, _ = sort_population(population, maximize)
    return sorted_population[-1]


def calculate_population_statistics(population: List) -> dict:
    """
    Calculate statistics for a population.
    
    Parameters:
    -----------
    population : List[Member]
        Population to analyze
        
    Returns:
    --------
    dict
        Dictionary containing population statistics
    """
    fitness_values = [member.fitness for member in population]
    
    return {
        'mean_fitness': np.mean(fitness_values),
        'std_fitness': np.std(fitness_values),
        'min_fitness': np.min(fitness_values),
        'max_fitness': np.max(fitness_values),
        'median_fitness': np.median(fitness_values),
        'population_size': len(population)
    }


def levy_flight(dim: int, beta: float = 1.5) -> np.ndarray:
    """
    Generate Levy flight step.
    
    Parameters:
    -----------
    dim : int
        Dimension of the step vector
    beta : float, optional
        Levy exponent, default is 1.5
        
    Returns:
    --------
    np.ndarray
        Levy flight step vector
    """
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
              (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    sigma_v = 1
    
    u = np.random.normal(0, sigma_u, dim)
    v = np.random.normal(0, sigma_v, dim)
    
    step = u / (np.abs(v) ** (1 / beta))
    return step


def exponential_decay(initial_value: float, final_value: float, 
                     current_iter: int, max_iter: int) -> float:
    """
    Calculate exponential decay value.
    
    Parameters:
    -----------
    initial_value : float
        Initial value at iteration 0
    final_value : float
        Final value at max_iter
    current_iter : int
        Current iteration
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    float
        Decayed value
    """
    return final_value + (initial_value - final_value) * np.exp(-current_iter / max_iter)


def linear_decay(initial_value: float, final_value: float, 
                current_iter: int, max_iter: int) -> float:
    """
    Calculate linear decay value.
    
    Parameters:
    -----------
    initial_value : float
        Initial value at iteration 0
    final_value : float
        Final value at max_iter
    current_iter : int
        Current iteration
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    float
        Decayed value
    """
    return initial_value - (initial_value - final_value) * (current_iter / max_iter)
