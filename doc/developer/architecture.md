# Software Architecture

OptiCut is designed with a modular architecture that strictly separates the physical problem definition, the numerical solvers, and the optimization algorithms. This modularity ensures that researchers can add new physics without altering the core optimization loop or the CutFEM integration logic.

## High-Level Overview

The framework is built around three primary pillars:
1. **The Physics (`config/`)**: Defines the cost functions, constraints, and their analytical shape derivatives using Unified Form Language (UFL).
2. **The PDE Solvers (`solvers/`)**: Handles the finite element assembly and linear system resolution using FEniCSx and CutFEMx.
3. **The Optimizer (`optimization/` & `levelset/`)**: Manages the Augmented Lagrangian Method (ALM) and the geometric evolution of the boundary via the Hamilton-Jacobi equation.


## Software Stack & Dependencies

OptiCut is designed as a high-performance Python package relying on a robust scientific stack. It abstracts away the heavy lifting of parallel linear algebra and fictitious domain integrations.

```{mermaid}

flowchart TD
    classDef user fill:#eef2ff,stroke:#6366f1,stroke-width:2px,color:#3730a3,rx:8px,ry:8px;
    classDef opti fill:#fdf4ff,stroke:#c026d3,stroke-width:2px,color:#86198f,rx:8px,ry:8px;
    classDef ext fill:#f1f5f9,stroke:#64748b,stroke-width:2px,color:#334155,rx:8px,ry:8px;
    
    subgraph User Space
        Config[param.txt]:::user --> Script[main.py]:::user
    end

    subgraph OptiCut
        Script --> O_Opti[Optimization & ALM]:::opti
        Script --> O_Prob[Physics & BaseProblem]:::opti
        O_Opti --> O_Solv[Ersatz & CutFEM Solvers]:::opti
        O_Opti --> O_LS[Level Set Advection & Reinit]:::opti
    end

    subgraph External Libraries
        O_Solv --> FEniCSx[FEniCSx / dolfinx]:::ext
        O_Solv --> CutFEMx[cutfemx]:::ext
        O_LS --> FEniCSx
        FEniCSx --> PETSc[PETSc / petsc4py]:::ext
        CutFEMx --> PETSc
        PETSc --> MPI[MPI / mpi4py]:::ext
    end
    
    style User Space fill:none,stroke:#cbd5e1,stroke-width:2px,stroke-dasharray: 5 5
    style OptiCut fill:#faf5ff,stroke:#d946ef,stroke-width:2px,rx:10px
    style External Libraries fill:none,stroke:#cbd5e1,stroke-width:2px,stroke-dasharray: 5 5
```

## Directory Structure
To help developers navigate the codebase, here is how the architectural concepts map to the physical directory structure:
| Module | Directory | Responsibilities |
|:---|:---|:---|
| **Configuration** | `src/config/` | Physics formulation (cost, constraints) using UFL |
| **PDE Solvers** | `src/solvers/` | CutFEM and Ersatz primal/adjoint PDEs resolution |
| **Level Set** | `src/levelset/` | Transport (Hamilton-Jacobi) & Reinitialization |
| **Optimization** | `src/optimization/`| Augmented Lagrangian Method and heuristic parameters |
| **Finite Elements**| `src/fem/` | Mesh generation, I/O, and boundary conditions setup |
| **Utilities** | `src/utils/` | I/O, visualization (ParaView/XDMF), and math tools |
| **Entry Point** | `src/main.py` | Command-line interface and optimization loop orchestration |


## Core Modules

### `src/config/` (Configuration & Physics)
This module acts as the user interface for defining the optimization setup.
- **`parameters.py`**: A centralized dataclass holding all runtime parameters (mesh size, elasticity limits, ALM penalties).
- **`problem.py`**: Contains the physical formulations. Each problem class defines the UFL integrands for the cost function, the constraint, and their shape derivatives. For non-self-adjoint problems, the adjoint operator is assembled automatically via `ufl.derivative`, FEniCSx's symbolic automatic differentiation engine, directly from the constraint UFL expression — without any manual derivation of the adjoint equations.

```{mermaid}
%%{init: {'theme': 'neutral'}}%%
classDiagram
    %% --- Physics & Config ---
    class Parameters {
        +mesh_case
        +cost_func
        +target_constraint
        +...()
    }
    class BaseProblem {
        <<abstract>>
        +cost_integrand()*
        +constraint_integrand()*
        +shape_derivative_integrand()*
    }
    class Compliance_Problem
    class VMLp_Problem
    class AreaProblem

    BaseProblem <|-- AreaProblem : inherits
    BaseProblem <|-- Compliance_Problem : inherits
    BaseProblem <|-- VMLp_Problem : inherits

    %% --- PDE Solvers ---
    class ErsatzElasticSolver {
        +primal_problem()
        +adjoint_problem()
        +descent_direction()
    }
    class CutFEMElasticSolver {
        +primal_problem()
        +adjoint_problem()
        +descent_direction()
    }

    %% --- Level Set Evolution ---
    class LevelSet {
        +level_set
        +update_level_set()
    }
    class Advection {
        +cut_fem_adv()
    }
    class Reinitialization {
        +reinitializationPC_inplace()
        +predictor()
        +corrector()
    }

    LevelSet <|-- Advection : inherits
    LevelSet <|-- Reinitialization : inherits

    %% Relationships
    ErsatzElasticSolver ..> Parameters : uses
    CutFEMElasticSolver ..> Parameters : uses
    BaseProblem ..> Parameters : uses

    %% Clickable Links
    click Parameters href "../parameters.html#config.parameters.Parameters" "View Parameters"
    click BaseProblem href "../problem.html" "View BaseProblem"
    click Compliance_Problem href "../problem.html#config.problem.Compliance_Problem" "View Compliance"
    click VMLp_Problem href "../problem.html#config.problem.VMLp_Problem" "View Von Mises"
    click AreaProblem href "../problem.html#config.problem.AreaProblem" "View Area"
    
    click ErsatzElasticSolver href "../ersatz_method.html#solvers.ersatz_elastic_solver.ErsatzElasticSolver" "View Ersatz Solver"
    click CutFEMElasticSolver href "../cutfem_method.html#solvers.cutfem_elastic_solver.CutFEMElasticSolver" "View CutFEM Solver"
    
    click LevelSet href "../reinitialization.html#levelset.levelSet_tool.LevelSet" "View LevelSet"
    click Advection href "../reinitialization.html#levelset.levelSet_tool.Advection" "View Advection"
    click Reinitialization href "../reinitialization.html#levelset.levelSet_tool.Reinitialization" "View Reinitialization"
```

### `src/solvers/` (Numerical Resolution)
This module is responsible for solving the primal and dual states based on the current domain geometry.
- **`cutfem_elastic_solver.py`**: Solves the elasticity equations on a background mesh intersected by the zero-contour of the level set function, applying Nitsche's method for boundary conditions and Ghost Penalties for stability.
- **`ersatz_elastic_solver.py`**: An alternative solver using the classical fictitious material approach for comparison purposes.

### `src/levelset/` (Geometry Evolution)
Manages the implicit representation of the structural boundaries.
- **`velocity_tools.py`**: Computes the descent direction (velocity field) by combining the shape derivatives of the cost and the constraints. It extends and regularizes this boundary velocity across the full mesh using the Riesz representation theorem.
- **`levelSet_tool.py`**: Contains the `Advection` and `Reinitialization` classes. It manages the evolution of the level set function by solving the Hamilton-Jacobi advection PDE, and periodically re-distancing the level set to maintain numerical stability.

### `src/optimization/` (ALM Framework)
- **`almMethod.py`**: Updates the Lagrange multipliers and penalty parameters based on constraint violations.
- **`opti_tool.py`**: Provides adaptive heuristics, such as dynamic CFL step sizing (Courant-Friedrichs-Lewy) for the Hamilton-Jacobi advection.

## Data Flow (The Optimization Loop)

The main execution flow in `src/main.py` follows a standard shape optimization iterative process:

<div style="max-width: 350px; margin: 0 auto;">

```{mermaid}

flowchart TD
    classDef init fill:#dcfce7,stroke:#22c55e,stroke-width:2px,color:#166534,rx:8px,ry:8px;
    classDef pde fill:#e0f2fe,stroke:#0ea5e9,stroke-width:2px,color:#075985,rx:8px,ry:8px;
    classDef opt fill:#fef9c3,stroke:#eab308,stroke-width:2px,color:#854d0e,rx:8px,ry:8px;
    classDef geom fill:#fce7f3,stroke:#ec4899,stroke-width:2px,color:#9d174d,rx:8px,ry:8px;
    classDef cond fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#1f2937,rx:8px,ry:8px;
    classDef endnode fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#991b1b,rx:8px,ry:8px;

    Init([Initialization]):::init --> Primal[Primal problem]:::pde
    Primal --> Dual[Adjoint problem]:::pde
    Dual --> Sensitivity[Sensitivity Analysis]:::opt
    Sensitivity --> ALM[Augmented Lagrangian Update]:::opt
    ALM --> Advection[Interface Transport]:::geom
    Advection --> Reinit[Level-set Regularization]:::geom
    Reinit --> Check{Evaluate Convergence}:::cond
    Check -- "Criteria unmet" --> Primal
    Check -- "Converged" --> End([Optimal Shape<br>Reached]):::endnode
```

</div>
