# Cuckoo Search Optimizer Algorithm Flowchart

```mermaid
flowchart TD
    A["Start"] --> B["Initialize bird nest population"]
    B --> C["Find initial best solution"]
    C --> D["Initialize optimization history"]
    D --> E["Begin main loop"]
    E --> F["Generate new solution using Levy flights"]
    F --> G["Evaluate and update population"]
    G --> H["Detect and replace abandoned nests"]
    H --> I["Evaluate and update population"]
    I --> J["Update best solution"]
    J --> K["Store current best solution"]
    K --> L{"iter >= max_iter?"}
    L -- "No" --> E
    L -- "Yes" --> M["Store final results"]
    M --> N["End"]
    
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
    end
    
    subgraph Generate new solution using Levy flights
    F1["Generate Levy flight step using beta = 1.5"]
    F2["Calculate step size: 0.01 * step * (x - best_x)"]
    F3["Create new position: x + step_size * random"]
    F4["Check boundaries and update position"]
    F --> F1 --> F2 --> F3 --> F4
    end
    
    subgraph Detect and replace abandoned nests
    H1["Generate discovery status: random > pa (pa = 0.25)"]
    H2{"Nest detected?"}
    H2 -- "Yes" --> H3["Create new nest using random walk"]
    H2 -- "No" --> H4["Keep current nest"]
    H3 --> H5["Check boundaries and update position"]
    H4 --> H5
    H --> H1 --> H2
    end
```

### Detailed Step Explanations:

1. **Initialize bird nest population**:
   - Randomly create initial positions within the search space
   - Each bird nest has a position and fitness value
   - Calculate objective function value: objective_func(position)

2. **Find initial best solution**:
   - Sort the population and select the best solution as best_solution

3. **Initialize optimization history**:
   - Initialize a list to store the history of best solutions

4. **Main loop** (max_iter times):
   - **Generate new solution using Levy flights**:
     * Each bird nest creates a new solution using Levy flight
     * Use beta coefficient = 1.5 for Levy flight
     ```python
     step = self._levy_flight()  # Using beta = 1.5
     step_size = 0.01 * step * (member.position - best_solution.position)
     new_position = member.position + step_size * np.random.randn(self.dim)
     ```
     * Check boundaries to ensure positions remain within [lb, ub]

   - **Evaluate and update population**:
     * Compare new population with old population
     * Keep better solutions

   - **Detect and replace abandoned nests**:
     * With probability pa = 0.25, bird nests are detected and abandoned
     ```python
     discovery_status = np.random.random(n) > self.pa  # pa = 0.25
     ```
     * Create new nests using random walk:
     ```python
     idx1, idx2 = np.random.choice(n, 2, replace=False)
     step_size = np.random.random() * (population[idx1].position - population[idx2].position)
     new_position = population[i].position + step_size
     ```
     * Keep nests that are not detected

   - **Evaluate and update population**:
     * Compare new population with old population
     * Keep better solutions

   - **Update best solution**:
     * Compare and update if a better solution is found

   - **Store current best solution**:
     * Store best_solution in history

5. **End**:
   - Store final results
   - Display optimization history
   - Return the best solution and history