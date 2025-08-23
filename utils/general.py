import numpy as np
from typing import Tuple, List
from src.core import Member

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
