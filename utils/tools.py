import numpy as np
import matplotlib.pyplot as plt

def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """
    Identify Pareto efficient points for minimization problems.
    
    This function determines which points in a multi-objective optimization
    problem are Pareto efficient (non-dominated) when minimizing all objectives.
    
    Args:
        costs: numpy array of shape (n_points, n_objectives) containing
               the cost values for each point and objective. Lower values
               are better for minimization problems.
               
    Returns:
        Boolean numpy array of shape (n_points,) where True indicates
        that the corresponding point is Pareto efficient.
        
    Example:
        >>> costs = np.array([[1, 2], [2, 1], [1.5, 1.5], [3, 3]])
        >>> efficient_mask = is_pareto_efficient(costs)
        >>> print(f"Efficient points: {efficient_mask}")
        Efficient points: [ True  True False False]
        
    Note:
        This implementation assumes minimization of all objectives.
        For maximization problems, the costs should be negated first.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    
    for i in range(costs.shape[0]):
        if is_efficient[i]:
            # Remove points dominated by point i
            # A point is dominated if all its objectives are worse or equal,
            # and at least one objective is strictly worse
            is_efficient[is_efficient] = np.any(costs[is_efficient] < costs[i], axis=1) | \
                                         np.all(np.isclose(costs[is_efficient], costs[i]), axis=1)
            is_efficient[i] = True  # Keep point i (it's efficient)
            
    return is_efficient


def plot_2d(X, y, title="2D Data Visualization", xlabel="X", ylabel="y", 
            color=None, cmap='viridis', show_grid=True, save_path=None, 
            figsize=(8, 6), marker='o', alpha=0.7) -> None:
    """
    Plot 2D data (X vs y) using scatter or line plot.
    
    This function visualizes 2D datasets, commonly used for regression, 
    classification, or Pareto front analysis. Supports color mapping 
    based on labels or continuous values.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input feature array of shape (n_samples,) or (n_samples, 1).
    y : numpy.ndarray
        Target/output array of shape (n_samples,) or (n_samples, 1).
    title : str, optional
        Title of the plot. Default is "2D Data Visualization".
    xlabel : str, optional
        Label for the x-axis. Default is "X".
    ylabel : str, optional
        Label for the y-axis. Default is "y".
    color : numpy.ndarray or str, optional
        Array of values to map colors (for scatter). If None, uses default color.
    cmap : str or matplotlib colormap, optional
        Colormap for coloring points. Used when `color` is an array. Default is 'viridis'.
    show_grid : bool, optional
        Whether to display grid. Default is True.
    save_path : str, optional
        If provided, save the plot to this path (e.g., 'plot.png'). Default is None.
    figsize : tuple, optional
        Figure size as (width, height). Default is (8, 6).
    marker : str, optional
        Matplotlib marker style. Default is 'o'.
    alpha : float, optional
        Transparency of points. Default is 0.7.
    
    Returns:
    --------
    None
        Displays the plot and optionally saves it to file.
    
    Example:
    --------
    >>> X = np.linspace(0, 10, 100)
    >>> y = np.sin(X)
    >>> plot_2d(X, y, title="Sine Wave", xlabel="x", ylabel="sin(x)")
    
    >>> # With color mapping
    >>> plot_2d(X, y, color=y, cmap='plasma', title="Sine with colormap")
    """
    # Validate input dimensions
    X = np.array(X).flatten()
    y = np.array(y).flatten()
    
    if X.shape != y.shape:
        raise ValueError(f"Shape mismatch: X shape {X.shape}, y shape {y.shape}")
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    if color is not None:
        color = np.array(color)
        if color.ndim == 0:
            c = None  # scalar color → use default
        else:
            c = color.flatten()
        scatter = plt.scatter(X, y, c=c, cmap=cmap, marker=marker, alpha=alpha, edgecolor='k', linewidth=0.3)
        plt.colorbar(scatter, label='Color value')
    else:
        plt.plot(X, y, marker=marker, linestyle='', alpha=alpha, markersize=6)
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if show_grid:
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()