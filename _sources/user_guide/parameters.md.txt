# Parameters Configuration

The `Parameters` class (located in `src/config/parameters.py`) acts as the central control panel for your optimization run. OptiCut reads its settings from a specified text file (like `param_compliance.txt`), which is parsed dynamically.

Here is the exhaustive list of all configurable parameters, grouped by category.

## 1. Structural, Geometric and Mesh Properties

These parameters define the physical domain, the initial shape, and the mesh.

- `mesh_case`: The bounding box geometry type (`"rectangle"`, `"L_shape"`, or `"3D"`).
- `mesh_file`: Optional. The name of the Gmsh file to load from the `mesh/` folder (e.g. `my_mesh.msh`). If `0`, the mesh is generated algorithmically.
- `lx`, `ly`, `lz`: The physical dimensions of the background bounding box (if algorithmically generated).
- `h`: The characteristic size of the finite elements. A smaller `h` provides better resolution but increases computational time.
- `young_modulus`: Young's modulus of the solid material ($E$) (see {eq}`eqn:elasticity_weak_form`).
- `poisson`: Poisson's ratio ($\nu$) (see {eq}`eqn:elasticity_weak_form`).
- `elasticity_limit`: The maximum allowed stress before yielding (used for stress constraints).
- `strenght`: Global traction force magnitude fallback if components are not explicitly set.
- `strenght_x`, `strenght_y`, `strenght_z`: The components of the applied force vector (Neumann boundary conditions).
- `linear_solver`: The KSP solver type for the elasticity PDE (`"lu"` for direct MUMPS, `"amg"` for iterative Algebraic Multigrid).
- `eta`: The smoothing scale parameter for the Heaviside function (used only when `cutFEM` is `0` for the Ersatz material method).

## 2. Optimization Settings (ALM & Cost)

These parameters govern the objective function and the Augmented Lagrangian Method (ALM).

- `cost_func`: The objective function to minimize (`"compliance"`, `"volume"`, or `"VonMises"`).
- `p_const`: The integer $p$ exponent used for the $L^p$ norm approximation when `cost_func` is `"VonMises"` (see {eq}`eq:JuVM`).
- `target_constraint`: The target value for the constraint (e.g., `0.4` for 40% of the initial volume).
- `max_incr`: The maximum number of optimization iterations (e.g., 200).
- `tol_cost_func`: The tolerance/stopping criterion for convergence.
- `ALM`: Toggle the Augmented Lagrangian Method (`1` to enable, `0` to disable).
- `augmented_lagrangian`: Toggle specific Augmented Lagrangian penalty formulation (`1` to enable, `0` to disable).
- `ALM_lagrangian_multiplicator` ($\lambda$): Initial value of the Lagrange multiplier (see {ref}`Augmented Lagrangian Method <augmented-lagrangian-method>`).
- `ALM_penalty_parameter` ($\mu$): Initial penalty parameter enforcing the constraint.
- `ALM_penalty_coef_multiplicator`: Factor by which the penalty parameter is multiplied at each iteration.
- `ALM_penalty_limit`: Maximum allowed value for the penalty parameter to prevent ill-conditioning.
- `ALM_slack_variable`: The slack variable used for handling inequality constraints in the ALM formulation.
- `uzawa`: Toggle the Uzawa method (`1` to enable, `0` to disable).
- `constraint`: The type of constraint to apply (`"volume"` or `"VonMises"`).
- `type_constraint`: The nature of the constraint (`"equal"` for equality).

## 3. Level-Set, Advection, and Regularization

These parameters control how the boundary evolves and how the gradient is regularized.

- `cutFEM`: Main toggle between the CutFEM method (`1`, see {eq}`eq:20`) and the Ersatz material method (`0`, see {eq}`eqn:elasticity_weak_form`).
- `cut_fem_advection`: Advection scheme (`0` for SUPG stabilization, `1` for pure CutFEM advection).
- `dt`: The time step for the Hamilton-Jacobi advection equation (see {eq}`eqn:HJ_equation`).
- `adapt_time_step`: If set to `1`, the solver dynamically adjusts `dt` based on the Courant-Friedrichs-Lewy (CFL) condition.
- `j_max`: Maximum number of sub-iterations for the advection phase.
- `freq_reinit`: Frequency of the level-set reinitialization (e.g., `1` = every step).
- `step_reinit`: Fictitious time step size for the reinitialization PDE.
- `l_reinit`: Total fictitious time length for the reinitialization phase.
- `extend_velocity`: Toggle the velocity extension across the domain (`1` to enable).
- `vel_normalization`: Normalization of the velocity field (`1` to enable).
- `alpha_reg_velocity`: The $\alpha$ parameter used to regularize the velocity field during the Riesz representation (see {eq}`eqn:reg_velocity`).
- `descent_direction_Riesz`: Toggle for calculating the descent direction via the Riesz representation (`1` to enable).

## Loading Custom Parameters

OptiCut relies on text files (e.g., `param_compliance.txt`) to load these settings dynamically via the `set__paramFolder(filename)` method. The syntax is simply `[parameter_name] [value]`.

**Example `param.txt`:**
```text
mesh_case rectangle
lx 2.0
ly 1.0
h 0.01
cost_func compliance
target_constraint 0.4
cutFEM 1
adapt_time_step 1
freq_reinit 5
strenght_y -10.0
linear_solver amg
```
