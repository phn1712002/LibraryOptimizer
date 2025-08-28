import numpy as np
from typing import List, Tuple


def convert_history_archive_to_numpy_xy(history_archive: List[List]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a history archive from multi-objective optimization to numpy arrays.
    
    This function extracts positions and fitness values from a history archive
    containing optimization results and converts them to numpy arrays for
    further analysis or visualization.
    
    Args:
        history_archive: List of lists containing Member objects from 
                        multi-objective optimization. Each inner list represents
                        an archive at a specific iteration.
        
    Returns:
        Tuple containing:
            - X: numpy array of positions (decision variables)
            - Y: numpy array of multi-objective fitness values
            
    Example:
        >>> X, Y = convert_history_archive_to_numpy_xy(history_archive)
        >>> print(f"Positions shape: {X.shape}, Fitnesses shape: {Y.shape}")
    """
    positions = []
    fitnesses = []

    # Extract positions and fitness values from each member in the archive
    for archive in history_archive:
        for member in archive:
            positions.append(member.position)
            fitnesses.append(member.multi_fitness)

    # Convert lists to numpy arrays for efficient computation
    X = np.array(positions)  
    Y = np.array(fitnesses)  
    
    return X, Y
