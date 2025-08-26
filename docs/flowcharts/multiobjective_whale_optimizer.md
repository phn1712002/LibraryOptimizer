# Multi-Objective Whale Optimization Algorithm Flowchart

```mermaid
flowchart TD
    A["Start"] --> B["Initialize whale population"]
    B --> C["Initialize archive and grid"]
    C --> D["Begin main loop"]
    D --> E["Update parameters a and a2"]
    E --> F["Update all whale positions"]
    F --> G["Select leader from archive"]
    G --> H["Calculate random p"]
    H --> I{"p < 0.5?"}
    I -- "True" --> J{"|A| >= 1?"}
    I -- "False" --> K["Bubble-net attacking"]
    J -- "True" --> L["Search for prey"]
    J -- "False" --> M["Encircling prey"]
    L --> N["Check boundaries and update fitness"]
    M --> N
    K --> N
    N --> O["Update archive with current population"]
    O --> P["Save archive state"]
    P --> Q{"iter >= max_iter?"}
    Q -- "No" --> D
    Q -- "Yes" --> R["End"]
    
    subgraph Initialization
    B
    C
    end
    
    subgraph Main Loop
    E
    F
    G
    H
    I
    J
    L
    M
    K
    N
    O
    P
    end
```

### Detailed Explanation of Steps:

1. **Initialize whale population**:
   - Randomly generate initial positions within the search space
   - Each position X_i ∈ [lb, ub]^dim
   - Calculate multi-objective function values

2. **Initialize archive and grid**:
   - Identify non-dominated solutions from the initial population
   - Initialize archive with non-dominated solutions
   - Create grid to manage archive based on objective space

3. **Main loop** (max_iter times):
   - **Update parameters a and a2**:
     * Decrease linearly with iteration count
     ```python
     a = 2 - iter * (2 / max_iter)
     a2 = -1 + iter * ((-1) / max_iter)
     ```

   - **Select leader from archive**:
     * Use grid-based selection to choose leader from archive
     * If archive is empty, randomly select from population

   - **Update all whale positions**:
     * Each whale updates its position based on hunting behavior

   - **Calculate random p**:
     * p ∈ [0, 1] to determine hunting behavior

   - **If p < 0.5 (Encircling or Searching)**:
     * **If |A| >= 1**: Search for prey (exploration)
       * Randomly select leader from archive or population
       ```python
       if self.archive:
           rand_leader = np.random.choice(self.archive)
           D_X_rand = abs(C * rand_leader.position[j] - whale.position[j])
           new_position[j] = rand_leader.position[j] - A * D_X_rand
       ```
     * **If |A| < 1**: Encircling prey (exploitation)
       ```python
       D_leader = abs(C * leader.position[j] - whale.position[j])
       new_position[j] = leader.position[j] - A * D_leader
       ```

   - **If p >= 0.5 (Bubble-net attacking)**:
     * Spiral movement around prey
     ```python
     distance_to_leader = abs(leader.position[j] - whale.position[j])
     new_position[j] = distance_to_leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + leader.position[j]
     ```

   - **Check boundaries and update fitness**:
     * Ensure positions remain within [lb, ub] boundaries
     * Recalculate multi-objective function values

   - **Update archive with current population**:
     * Add non-dominated solutions to archive
     * Maintain archive size by removing redundant solutions
     * Update grid to reflect new archive

   - **Save archive state**:
     * Store current archive for optimization history

4. **End**:
   - Save final results
   - Return archive (set of Pareto optimal solutions) and history