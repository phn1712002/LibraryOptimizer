# Multi-Objective Cuckoo Search Optimizer Algorithm Flowchart

```mermaid
flowchart TD
    A["Start"] --> B["Initialize Multi-Objective Cuckoo Population"]
    B --> C["Identify Non-Dominated Solutions"]
    C --> D["Initialize Archive with Non-Dominated Solutions"]
    D --> E["Initialize Grid for Archive"]
    E --> F["Begin Main Loop"]
    F --> G["Select Leader from Archive using Grid-based Selection"]
    G --> H["Generate New Solutions via Levy Flights with Leader Guidance"]
    H --> I["Update Population using Pareto Dominance"]
    I --> J["Detect and Replace Abandoned Nests with Archive Guidance"]
    J --> K["Update Population using Pareto Dominance"]
    K --> L["Update Archive with Current Population"]
    L --> M["Archive State Storage"]
    M --> N{"iter >= max_iter?"}
    N -- "No" --> F
    N -- "Yes" --> O["Save Final Results"]
    O --> P["End"]
    
    subgraph Initialization
    B
    C
    D
    E
    end
    
    subgraph Main Loop
    G
    H
    I
    J
    K
    L
    M
    end
    
    subgraph Generate New Solutions via Levy Flights
    H1["Select Leader from Archive"]
    H2["Generate Levy Flight Step using beta_levy = 1.5"]
    H3["Calculate Step Size: 0.01 * step * (x - leader_x)"]
    H4["Generate New Position: x + step_size"]
    H5["Check Boundaries and Update Position"]
    H --> H1 --> H2 --> H3 --> H4 --> H5
    end
    
    subgraph Detect and Replace Abandoned Nests
    J1["Generate Discovery Status: random > pa (pa = 0.25)"]
    J2{"Nest Discovered?"}
    J2 -- "Yes" --> J3["Select 2 Random Archive Members"]
    J3 --> J4["Create New Nest via Random Walk with Archive Guidance"]
    J2 -- "No" --> J5["Keep Current Nest"]
    J4 --> J6["Check Boundaries and Update Position"]
    J5 --> J6
    J --> J1 --> J2
    end
```

### Detailed Step-by-Step Explanation:

1. **Initialize Multi-Objective Cuckoo Population**:
   - Randomly generate initial positions within the search space
   - Each nest has a position and multi_fitness value
   - Calculate multi-objective function values: objective_func(position)

2. **Identify Non-Dominated Solutions**:
   - Analyze the population to identify solutions not dominated by others
   - Use Pareto dominance relationships

3. **Initialize Archive**:
   - Initialize archive with initial non-dominated solutions
   - Archive stores the set of Pareto optimal solutions

4. **Initialize Grid**:
   - Create grid system to manage the archive
   - Divide objective space into hypercubes
   ```python
   self.grid = self._create_hypercubes(costs)
   ```

5. **Main Loop** (max_iter iterations):
   - **Select Leader**:
     * Choose leader from archive using grid-based selection
     * Prioritize less crowded grids
     ```python
     leader = self._select_leader()
     ```

   - **Generate New Solutions via Levy Flights**:
     * Each nest creates new solutions using Levy flight guided by the leader
     * Use beta_levy = 1.5 coefficient for Levy flight
     ```python
     step = self._levy_flight()  # Using beta_levy = 1.5
     step_size = 0.01 * step * (member.position - leader.position)
     new_position = member.position + step_size
     ```
     * Check boundaries to ensure positions remain within [lb, ub]

   - **Update Population using Pareto Dominance**:
     * Compare new and current populations using Pareto dominance
     * Retain non-dominated solutions
     ```python
     if self._dominates(new, current):
         updated_population.append(new)
     elif self._dominates(current, new):
         updated_population.append(current)
     else:
         # Random selection if no dominance relationship
         if np.random.random() < 0.5:
             updated_population.append(new)
         else:
             updated_population.append(current)
     ```

   - **Detect and Replace Abandoned Nests**:
     * With probability pa = 0.25, nests are discovered and abandoned
     ```python
     discovery_status = np.random.random(n) > self.pa  # pa = 0.25
     ```
     * Create new nests via random walk with archive guidance:
     ```python
     if self.archive and len(self.archive) >= 2:
         idx1, idx2 = np.random.choice(len(self.archive), 2, replace=False)
         step_size = np.random.random() * (self.archive[idx1].position - self.archive[idx2].position)
     ```
     * Keep undiscovered nests unchanged

   - **Update Population using Pareto Dominance**:
     * Compare new and current populations using Pareto dominance
     * Retain non-dominated solutions

   - **Update Archive**:
     * Add new non-dominated solutions to the archive
     * Maintain archive size and update grid
     ```python
     self._add_to_archive(population)
     ```

   - **Archive State Storage**:
     * Save current archive state to history

6. **Termination**:
   - Save final results
   - Return archive containing the set of Pareto optimal solutions