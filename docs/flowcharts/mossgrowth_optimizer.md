# Moss Growth Optimizer Algorithm Flowchart

```mermaid
flowchart TD
    A["Start"] --> B["Initialize Moss Population"]
    B --> C["Sort Population and Initialize Best Solution"]
    C --> D["Initialize Cryptobiosis Mechanism"]
    D --> E["Begin Main Loop"]
    E --> F["Calculate Progress Ratio"]
    F --> G["Record Initial Generation for Cryptobiosis"]
    G --> H["Process Each Search Agent"]
    H --> I["Select Calculation Position Based on Majority Region"]
    I --> J["Divide Population and Select Region with Most Individuals"]
    J --> K["Calculate Distance from Individual to Best"]
    K --> L["Calculate Wind Direction (Average Distance)"]
    L --> M["Calculate Beta and Gamma Parameters"]
    M --> N["Calculate Movement Step Size"]
    N --> O{"Spore Dispersion Probability > d1?"}
    O -- "Yes" --> P["Spore Dispersion Search: position + step2 * D_wind"]
    O -- "No" --> Q["Spore Dispersion Search: position + step * D_wind"]
    P --> R{"Double Propagation Probability < 0.8?"}
    Q --> R
    R -- "Yes" --> S{"Probability > 0.5?"}
    R -- "No" --> T["Check Boundaries and Evaluate Fitness"]
    S -- "Yes" --> U["Update Specific Dimension: best + step3 * D_wind[dim]"]
    S -- "No" --> V["Update All Dimensions with Activation Function: (1-act)*new + act*best"]
    U --> T
    V --> T
    T --> W["Update Population and Record Cryptobiosis"]
    W --> X{"rec >= rec_num or Final Iteration?"}
    X -- "Yes" --> Y["Cryptobiosis Mechanism: Restore Historical Best Position"]
    X -- "No" --> Z["Increment rec Counter"]
    Y --> AA["Update Best Solution"]
    Z --> AA
    AA --> AB["Store Best Solution"]
    AB --> AC{"iter >= max_iter?"}
    AC -- "Not Yet" --> E
    AC -- "Yes" --> AD["End"]
    
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
    W
    X
    Y
    Z
    AA
    AB
    end
```

### Detailed Step-by-Step Explanation:

1. **Initialize Moss Population**:
   - Randomly generate initial positions within the search space
   - Each position X_i ∈ [lb, ub]^dim
   - Calculate objective function value objective_func(X_i)

2. **Sort Population and Initialize Best Solution**:
   - Sort population based on fitness values
   - Select initial best solution

3. **Initialize Cryptobiosis Mechanism**:
   - Initialize memory to record position and fitness history
   - rM: Store position history
   - rM_cos: Store fitness history

4. **Main Loop** (max_iter times):
   - **Calculate Progress Ratio**:
     ```python
     progress_ratio = current_fes / max_fes
     ```

   - **Record Initial Generation for Cryptobiosis**:
     * Store initial positions and fitness values

   - **Select Calculation Position Based on Majority Region**:
     * Divide search space and select region with most individuals

   - **Calculate Distance and Wind Direction**:
     ```python
     D = best_solution.position - cal_positions
     D_wind = np.mean(D, axis=0)
     ```

   - **Calculate Beta and Gamma Parameters**:
     ```python
     beta = cal_positions.shape[0] / search_agents_no
     gamma = 1 / np.sqrt(1 - beta**2) if beta < 1 else 1.0
     ```

   - **Spore Dispersion Search**:
     * **If probability > d1**: Use step2
       ```python
       new_position = population[i].position + step2 * D_wind
       ```
     * **If probability <= d1**: Use step
       ```python
       new_position = population[i].position + step * D_wind
       ```

   - **Double Propagation Search**:
     * **Update Specific Dimension**:
       ```python
       new_position[dim_idx] = best_solution.position[dim_idx] + step3 * D_wind[dim_idx]
       ```
     * **Update All Dimensions with Activation Function**:
       ```python
       new_position = (1 - act) * new_position + act * best_solution.position
       ```

   - **Check Boundaries and Evaluate Fitness**:
     * Ensure positions remain within bounds [lb, ub]
     * Calculate objective function value for new position

   - **Cryptobiosis Mechanism**:
     * When sufficient records or final iteration, restore historical best position
     * Reset rec counter

   - **Update Best Solution**:
     * Compare and update if better solution is found

   - **Store Best Solution**:
     * Save best solution at each iteration

5. **End**:
   - Store final results
   - Display optimization history
   - Return best solution and history