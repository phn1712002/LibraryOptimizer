import numpy as np
from typing import Tuple, List
from src.core import Member

def roulette_wheel_selection(probabilities: np.ndarray) -> int:
    """
    Roulette wheel selection based on probabilities
    
    Args:
        probabilities: Array of selection probabilities
        
    Returns:
        Index of selected element
    """
    r = np.random.random()
    cumulative_sum = np.cumsum(probabilities)
    return np.argmax(r <= cumulative_sum)


def sort_population(self, population) -> Tuple[List, List]:
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