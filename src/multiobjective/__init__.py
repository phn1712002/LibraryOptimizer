"""
Multi-objective optimization module for LibraryOptimizer

This module provides multi-objective optimization algorithms that extend
the standard single-objective optimizers to handle multiple objectives
using Pareto dominance and archive management.
"""

from ._core import *
from .artificialbeecolony_optimizer import *