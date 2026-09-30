# Problem Definition

Defining an optimization problem in OptiCut involves three main components: setting up the geometry (mesh and initial level-set), defining the boundary conditions (Dirichlet and Neumann), and configuring the physical and algorithmic parameters.

## 1. Defining the Initial Geometry

OptiCut uses an implicit boundary representation (see the {ref}`Level set method <level-set-method>` theoretical section). The structural boundary is given by the zero contour of a scalar field $\phi(x)$.

The initial geometry is fully configurable from your parameter file (e.g., `param_compliance.txt`). By defining the `mesh_case` variable, you select the geometric scenario. The pre-built geometric primitives and their initial level-set functions are programmed in `src/utils/ls_utils.py` (e.g., `rectangle`, `L_shape`, or `3D`).

```bash
# In src/parameters/param_....txt (e.g., `param_compliance.txt`)
mesh_case L_shape
```
OptiCut dynamically loads the requested mesh and level-set function during initialization.

## 2. Boundary Conditions

OptiCut distinguishes between solid mechanics boundary conditions (clamped regions) and load conditions (applied forces). Because the mesh is fixed (Eulerian approach) and the structure moves through it, the boundary conditions are applied using robust mathematical formulations like {ref}`Nitsche's method <nitsche-method>` for CutFEM.

You define these conditions by writing marker functions in `src/fem/boundary_conditions.py`.

### Dirichlet Conditions (Clamped Boundaries)

The `clamped_boundary_...` functions identify where the structure is attached to the wall. For instance, in a cantilever rectangle:

```python
# In src/fem/boundary_conditions.py
def clamped_boundary_cantilever(x):
    return np.isclose(x[0], 0)
```

### Neumann Conditions (Applied Forces)

Similarly, `load_marker` identifies where external forces are applied in the domain.

```python
# In src/fem/boundary_conditions.py
def load_marker(test_case, parameters):
    def marker(x):
        # ...
        elif test_case == "rectangle":
            return np.logical_and(np.isclose(x[0], parameters.lx), np.logical_and(x[1] < (0.55), x[1] > (0.45)))
    return marker
```

**Force Magnitude:** The magnitude of the traction force is **not hardcoded**. It is directly driven by the `strenght_x`, `strenght_y`, and `strenght_z` (for 3D) parameters defined in your parameter file. The code internally creates a force vector (called `shift`) applying this magnitude to the localized nodes:

```python
# In src/fem/boundary_conditions.py
shift = fem.Constant(msh, ScalarType((parameters.strenght_x, parameters.strenght_y))) # (2D case)
```

## 3. Material Properties (Linear Elasticity)

OptiCut assumes the structure behaves according to the laws of **isotropic linear elasticity** (see the {eq}`eqn:elasticity_weak_form` theoretical formulation). 

The material properties are fully defined in your parameter file (e.g., `param_compliance.txt`). You only need to set the Young's modulus $E$ and the Poisson's ratio $\nu$:

```bash
# In src/parameters/param_....txt
young_modulus 1.0
poisson 0.3
```
These properties are used internally to compute the Lamé parameters ($\lambda, \mu$) for the elasticity solver.

## 4. Choosing the Physics (Cost and Constraints)

The optimization objective is fully determined by the parameter file (e.g., `param_compliance.txt`) (see [Parameters](parameters.md)). By setting `cost_func`, you can switch between different physics:

- `"compliance"`: Minimizes the strain energy (maximizes stiffness) for a given volume fraction.
- `"volume"`: Minimizes the total volume subject to a constraint.
- `"VonMises"`: Minimizes the Lp norm of the Von Mises stress field.

Once your boundaries and physics are selected, you simply run `main.py` to start the optimization!

## Summary Workflow for New Problems

If you are a novice user and want to simulate a completely new mechanical case (e.g., a bridge instead of a cantilever beam), follow these simple steps:

```{mermaid}
flowchart TD
    A([1. Geometry]) -->|Build mesh or load .msh & define Level-Set| B[src/utils/ls_utils.py]
    C([2. Boundaries]) -->|Define Dirichlet & Neumann markers| D[src/fem/boundary_conditions.py]
    E([3. Material & 4. Parameters]) -->|Set E, nu, physics, & discretization params| F[param_bridge.txt]

    B -.-> G
    D -.-> G
    F -.-> G
    
    G((fa:fa-play Run main.py))

    style A fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px
    style C fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px
    style E fill:#fce4ec,stroke:#d81b60,stroke-width:2px
    style G fill:#ede7f6,stroke:#5e35b1,stroke-width:3px
```

1. **Geometry**: Build a new mesh scenario or load a pre-existing mesh (using `mesh_file mesh.xdmf` in `param.txt`), and define a new initial level-set function block in `src/utils/ls_utils.py` (within `init_level_set`).
2. **Boundaries**: Add your new Dirichlet (`clamped_boundary_...`) and Neumann (`load_marker`) coordinates to `src/fem/boundary_conditions.py`.
3. **Parameters (Material & Physics)**: Create a new `param_bridge.txt` file. Define all your material properties (`young_modulus`, `poisson`), mechanical, algorithmic, discretization, and geometric parameters (including `mesh_case` and `mesh_file` if needed).
4. **Execute**: Run the code!
