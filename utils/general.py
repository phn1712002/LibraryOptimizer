import numpy as np
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