# Multi-Objective Modified Social Group Optimizer Algorithm Flowchart

```mermaid
flowchart TD
    A["Start"] --> B["Initialize population"]
    B --> C["Initialize archive with non-dominated solutions"]
    C --> D["Create grid for archive"]
    D --> E["Begin main loop"]
    E --> F["Phase 1: Guru Phase"]
    F --> G["Select guru from archive"]
    G --> H["Update position: c*current + rand*(guru - current)"]
    H --> I["Check boundaries and update fitness"]
    I --> J["Greedy selection: accept if better"]
    J --> K["Phase 2: Learner Phase"]
    K --> L["Select global best from archive"]
    L --> M["Select random partner"]
    M --> N{"Check dominance relationship"}
    N -- "Current dominates partner" --> O{"Random exploration probability?"}
    N -- "Partner dominates current or non-dominated" --> P["Learn from partner: current + rand*(partner - current) + rand*(global_best - current)"]
    O -- "High" --> Q["Random exploration"]
    O -- "Low" --> R["Learn from partner: current + rand*(current - partner) + rand*(global_best - current)"]
    Q --> S["Check boundaries and update fitness"]
    R --> S
    P --> S
    S --> T["Greedy selection: accept if better"]
    T --> U["Update archive"]
    U --> V["Save archive history"]
    V --> W{"iter >= max_iter?"}
    W -- "No" --> E
    W -- "Yes" --> X["End"]
    
    subgraph Initialization
    B
    C
    D
    end
    
    subgraph Main Loop
    F
    G
    H
    I
    J
    K
    L
    M
    N
    O
    P
    Q
    R
    S
    T
    U
    V
    end
```

### Detailed Explanation of Steps:

1. **Initialize population**:
   - Randomly generate initial positions within the search space
   - Each position X_i ∈ [lb, ub]^dim
   - Calculate multi-objective fitness values: multi_fitness = objective_func(X_i)

2. **Initialize archive with non-dominated solutions**:
   - Identify non-dominated solutions in the initial population
   - Add these solutions to the external archive

3. **Create grid for archive**:
   - Create hypercube grid to manage the archive
   - Assign grid indices to each solution in the archive

4. **Main loop** (max_iter times):
   - **Phase 1: Guru Phase** (learning from the best):
     * Select guru from archive using grid-based selection
     * Update position:
       ```python
       new_position[j] = c * current.position[j] + random * (guru.position[j] - current.position[j])
       ```
     * Check boundaries and update fitness
     * Greedy selection: accept new solution if it dominates the current solution or is non-dominated
   
   - **Phase 2: Learner Phase** (mutual learning with random exploration):
     * Select global best from archive
     * Select random partner different from current individual
     * Check dominance relationship between current and partner:
       - If current dominates partner and is not dominated by partner:
         * If random > sap (0.7): learn from partner
           ```python
           new_position = current + random*(current - partner) + random*(global_best - current)
           ```
         * Else: random exploration
       - Else (partner dominates current or non-dominated): learn from partner
         ```python
         new_position = current + random*(partner - current) + random*(global_best - current)
         ```
     * Check boundaries and update fitness
     * Greedy selection: accept new solution if better
   
   - **Update archive**: Add new non-dominated solutions to archive
   
   - **Save archive history**: Record current archive state

5. **End**:
   - Save final results
   - Return archive history and final archive