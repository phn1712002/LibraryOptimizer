# Contributing to Library Optimizer

Thank you for your interest in contributing to Library Optimizer! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

### 1. Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Environment information (Python version, OS, etc.)
- Code examples if possible

### 2. Suggesting Enhancements

We welcome suggestions for new features or improvements. Please include:
- A clear description of the enhancement
- Use cases and examples
- Any relevant references or research

### 3. Adding New Algorithms

To add a new optimization algorithm:

1. **Follow the Template**: Use the template in `rules/new-algorithm.md`
2. **Inherit from Solver**: Your algorithm must inherit from the base `Solver` class
3. **Implement Required Methods**: Ensure you implement the `solver()` method
4. **Add Tests**: Include comprehensive tests in the `test/` directory
5. **Update Registry**: Add your algorithm to the registry in `src/__init__.py`
6. **Documentation**: Add docstrings and update README if needed

### 4. Improving Documentation

Documentation improvements are always welcome:
- Fix typos and grammatical errors
- Improve clarity and examples
- Add missing documentation
- Translate documentation (if applicable)

## Development Setup

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/LibraryOptimizer.git
   cd LibraryOptimizer
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Coding Standards

### Style Guidelines

- Follow PEP 8 style guide
- Use 4 spaces for indentation
- Maximum line length: 88 characters
- Use type hints for all public functions
- Write comprehensive docstrings

### Naming Conventions

- **Classes**: PascalCase (e.g., `GreyWolfOptimizer`)
- **Functions/Methods**: snake_case (e.g., `create_solver`)
- **Variables**: snake_case (e.g., `objective_func`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_ITERATIONS`)

### Documentation

All public functions and classes must have docstrings following this format:

```python
def function_name(param1: Type, param2: Type = default) -> ReturnType:
    """
    Brief description of the function.
    
    Parameters:
    -----------
    param1 : Type
        Description of param1
    param2 : Type, optional
        Description of param2, default is default
        
    Returns:
    --------
    ReturnType
        Description of return value
        
    Raises:
    -------
    ValueError
        When something goes wrong
    """
```

## Testing

### Writing Tests

- Place tests in the `test/` directory
- Use descriptive test function names starting with `test_`
- Test both minimization and maximization problems
- Include tests for edge cases
- Use standard benchmark functions (Sphere, Rastrigin, etc.)

### Running Tests

```bash
# Run all tests
python -m pytest test/ -v

# Run specific test file
python -m pytest test/test-gwo.py -v

# Run with coverage
python -m pytest test/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Run flake8 for code quality
flake8 src/ --max-line-length=120

# Run black for code formatting
black src/

# Run isort for import sorting
isort src/

# Run mypy for type checking
mypy src/ --ignore-missing-imports
```

## Pull Request Process

1. **Ensure tests pass**: All tests must pass before submitting a PR
2. **Update documentation**: Include relevant documentation updates
3. **Follow the template**: Use the PR template if available
4. **Describe changes**: Provide a clear description of your changes
5. **Reference issues**: Link to any related issues

## Review Process

- PRs will be reviewed by maintainers
- Feedback may be provided for improvements
- Once approved, your PR will be merged

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions about contributing, please:
1. Check the existing documentation
2. Search existing issues
3. Create a new issue if your question hasn't been answered

Thank you for contributing to Library Optimizer!
