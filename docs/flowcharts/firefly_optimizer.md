# Firefly Optimizer Algorithm Flowchart

```mermaid
flowchart TD
    A["Start"] --> B["Initialize firefly population"]
    B --> C["Sort population and initialize best solution"]
    C --> D["Calculate scale for random movement"]
    D --> E["Begin main loop"]
    E --> F["Evaluate fitness of all fireflies"]
    F --> G["Sort fireflies by brightness"]
    G --> H["Update best solution"]
    H --> I["Move fireflies toward brighter ones"]
    I --> J["Calculate distance and attractiveness"]
    J --> K["Update position with random movement"]
    K --> L["Check boundaries and update position"]
    L --> M["Store best solution"]
    M --> N{"Reduce alpha over time?"}
    N -- "Yes" --> O["Reduce alpha parameter"]
    N -- "No" --> P{"iter >= max_iter?"}
    O --> P
    P -- "Not yet" --> E
    P -- "Yes" --> Q["End"]
    
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
    end
```

### Detailed Explanation of Steps:

1. **Initialize firefly population**:
   - Randomly generate initial positions within the search space
   - Each position X_i ∈ [lb, ub]^dim
   - Calculate objective function value objective_func(X_i)

2. **Sort population and initialize best solution**:
   - Sort population based on fitness values (brightness)
   - Select initial best solution

3. **Calculate scale for random movement**:
   ```python
   scale = np.abs(self.ub - self.lb)
   ```

4. **Main loop** (max_iter times):
   - **Evaluate fitness of all fireflies**:
     * Recalculate objective function value for each firefly

   - **Sort fireflies by brightness**:
     * Sort population based on fitness to identify brightest fireflies

   - **Update best solution**:
     * Compare and update if better solution is found

   - **Move fireflies toward brighter ones**:
     * Each firefly moves toward fireflies brighter than itself
     ```python
     if self._is_better(population[j], population[i]):
         # Firefly i moves toward firefly j
     ```

   - **Calculate distance and attractiveness**:
     ```python
     r = np.sqrt(np.sum((population[i].position - population[j].position)**2))
     beta = self._calculate_attractiveness(r)
     ```

   - **Update position with random movement**:
     ```python
     random_move = self.alpha * (np.random.random(self.dim) - 0.5) * scale
     new_position = (population[i].position * (1 - beta) + 
                    population[j].position * beta + 
                    random_move)
     ```

   - **Check boundaries and update position**:
     * Ensure position stays within boundaries [lb, ub]
     * Update firefly position

   - **Store best solution**:
     * Save the best solution at each iteration

   - **Reduce alpha parameter** (if enabled):
     ```python
     if self.alpha_reduction:
         self.alpha = self._reduce_alpha(self.alpha, self.alpha_delta)
     ```

5. **End**:
   - Store final results
   - Display optimization history
   - Return best solution and history