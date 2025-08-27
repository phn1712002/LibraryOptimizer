# 📚 Python → MATLAB Conversion Rules

## 📋 Overview

This document defines the standards for **converting an optimization library written in Python into MATLAB**.
It focuses on:

* Naming conventions (kept the same as Python)
* Converting class/function structures
* Commenting & docstring rules
* Array/matrix handling rules (NumPy → MATLAB)
* File & module organization
The goal is to ensure that the MATLAB code is **consistent, readable, maintainable**, and aligned with the original Python library architecture.

---

## 🏷️ Naming Rules

⚠️ **Important:** All naming conventions from Python must be preserved in MATLAB.

### 1. Class Names

* Python: `GreyWolfOptimizer`
* MATLAB: **Keep PascalCase**

```matlab
classdef GreyWolfOptimizer < Solver
    % Implementation here
end
```

### 2. Function/Method Names

* Python: `snake_case` → MATLAB: **keep snake\_case**
* Example:

  * Python: `create_solver` → MATLAB: `create_solver`
  * Python: `_init_population` → MATLAB: `init_population` (private method)

### 3. Variable Names

* Python: `snake_case` → MATLAB: **keep snake\_case**
* Examples:

  * `objective_func` → `objective_func`
  * `search_agents_no` → `search_agents_no`

### 4. Constants

* Python: `UPPER_SNAKE_CASE` → MATLAB: **keep UPPER\_SNAKE\_CASE**

```matlab
properties (Constant)
    MAX_ITERATIONS = 1000;
    DEFAULT_POPULATION_SIZE = 50;
end
```

---

## 💬 Commenting Rules

### 1. Function Docstring

```matlab
%{
create_solver - Create a solver instance by name
Inputs:
    solver_name     - Name of the solver (e.g. 'GreyWolfOptimizer')
    objective_func  - Objective function handle
    lb              - Lower bound (vector or scalar)
    ub              - Upper bound (vector or scalar)
    dim             - Problem dimension
    maximize        - true for maximization, false for minimization
    varargin        - Additional parameters
Outputs:
    solver_obj      - Instance of solver class
%}
```

### 2. Inline Comments

```matlab
% Update parameter (decreases linearly from 2 to 0)
a = 2 - iter * (2 / max_iter);

% Ensure positions stay within bounds
new_position = max(min(new_position, obj.ub), obj.lb);
```

### 3. TODO/FIXME

```matlab
% TODO: Implement adaptive parameter tuning
% FIXME: Handle empty population edge case
```

---

## 📝 Function Writing Rules

### 1. Function Signature

```matlab
function result = function_name(param1, param2, varargin)
    % Documentation here
end
```

### 2. Type Hints

```matlab
arguments
    param1 double
    param2 double = 10
end
```

### 3. Function Length

* Max \~50 lines, split into subfunctions if needed.

### 4. Return Values

* MATLAB supports multiple outputs `[out1, out2]` → keep consistent with Python.

---

## 📁 File Organization Rules

### 1. File Naming

* MATLAB requires: **file = class/function name**
* Keep original naming style (snake\_case for functions, PascalCase for classes).
* Example:

  * `greywolf_optimizer.py` → `greywolf_optimizer.m`
  * `particleswarm_optimizer.py` → `particleswarm_optimizer.m`

---

## 🔄 Python → MATLAB Mapping

| Python Feature             | MATLAB Equivalent                      |
| -------------------------- | -------------------------------------- |
| `def function_name(...)`   | `function result = function_name(...)` |
| `class ClassName(Solver):` | `classdef ClassName < Solver`          |
| `self.param`               | `obj.param`                            |
| `np.array([...])`          | `[ ... ]`                              |
| `np.zeros(dim)`            | `zeros(1, dim)`                        |
| `np.clip(x, lb, ub)`       | `min(max(x, lb), ub)`                  |
| `for i in range(n):`       | `for i = 1:n`                          |
| `if condition:`            | `if condition`                         |
| `return x, y`              | `[x, y] = ...`                         |

---

## 🚀 Best Practices for Conversion

1. **Vectorization**

   * Python (NumPy) → MATLAB (native vector operations).
   * Avoid `for` loops when possible.

2. **Inheritance & Utilities**

   * Use `Solver` as base class.
   * Python utilities in `_general.py` → MATLAB static methods in `general_utils.m`.

3. **Consistency**

   * Keep naming rules from Python.
   * Algorithm logic unchanged, only syntax adapted.

---

## 🔍 Code Review Checklist

* [ ] Class names preserved (PascalCase)
* [ ] Function names preserved (snake\_case)
* [ ] Variable names preserved (snake\_case)
* [ ] Constants preserved (UPPER\_SNAKE\_CASE)
* [ ] NumPy replaced with MATLAB vector/matrix operations
* [ ] Unit tests implemented in MATLAB (`runtests`)
* [ ] Performance checked (vectorization instead of loops)

