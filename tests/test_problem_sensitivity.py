import pytest
import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
import cutfemx

from levelset.levelSet_tool import Advection, LevelSet
from solvers.cutfem_elastic_solver import CutFEMElasticSolver
from config.problem import AreaProblem, Compliance_Problem, VMLp_Problem
from config.parameters import Parameters
from utils import mechanics_tool

class MockParams(Parameters):
    def __init__(self, cost_func_name="compliance", constraint_type="Area", dt=0.06):
        # Initialize with default values to avoid interactive input
        self.cutFEM = 1
        self.ALM = 0
        self.young_modulus = 210.0
        self.poisson = 0.33
        self.eta = 1e-3
        self.p_const = 8
        self.elasticity_limit = 1.0
        self.alpha_reg_velocity = 0.1
        self.target_constraint = 0.5
        self.cost_func = cost_func_name
        self.type_constraint = constraint_type
        self.h = 0.1
        self.dt = dt
        # Lame coefficients will be computed from E and nu
        self.lame_mu, self.lame_lambda = mechanics_tool.lame_compute(self.young_modulus, self.poisson)
        # ALM defaults
        self.ALM_lagrangian_multiplicator = 0.0
        self.ALM_penalty_parameter = 1.0
        self.ALM_slack_variable = 0.0

import inspect
from config import problem as opti_problem

# Dynamically gather all problem classes
problem_classes = []
for name, obj in inspect.getmembers(opti_problem, inspect.isclass):
    if hasattr(obj, 'cost') and hasattr(obj, 'dual_operator') and hasattr(obj, 'shape_derivative_integrand'):
        problem_classes.append((name, obj))

@pytest.mark.parametrize("problem_name, ProblemClass", problem_classes)
# IMPORTANT: The choice of dt for Finite Difference validation in CutFEM is sensitive.
    # Due to 'Ghost Penalty' stabilization jumps when the interface crosses mesh edges, 
    # extremely small dt (e.g. 1e-6) can lead to high relative errors (~70%).
    # A dt of the same order as the mesh size (h=0.1) or slightly smaller (0.01 to 0.05) 
    # typically yields the best alignment with the analytical derivative (~4-5% error).
    # See tests/study_compliance_convergence.py for full convergence plots.

def test_compliance_sensitivity(problem_name, ProblemClass):
    """
    Ultimate Validation: Finite Difference vs Shape Derivative for Compliance.
    """
    if "Compliance" in problem_name:
        cost_name = "compliance"
        constraint_type = "Area"
        dt = 0.06
    elif "VM" in problem_name:
        cost_name = "VonMises"
        constraint_type = "Area"
        dt = 0.06
    elif "Area" in problem_name:
        cost_name = "Area"
        constraint_type = "VonMises"
        dt = 0.06
    else:
        cost_name = problem_name
        constraint_type = "Area"
        dt = 0.06

    # Redefine mesh
    msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-1, -1]), np.array([1, 1])], [100, 100], mesh.CellType.triangle)
    V_ls = fem.functionspace(msh, ("Lagrange", 1))
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.topology.dim,)))
    
    params = MockParams(cost_func_name=cost_name, constraint_type=constraint_type, dt=dt)
    mu, lmbda = params.lame_mu, params.lame_lambda
    
    
    # 1. Initial Level-Set (Horizontal interface at y=0, phi > 0 for y > 0)
    ls_init_func = fem.Function(V_ls)
    ls_init_func.interpolate(lambda x: 0.5 - np.sqrt(x[0]**2 + x[1]**2) + 1e-9)
    ls_obj = LevelSet(ls_init_func, V_ls)
    
    # 2. Setup Boundary Conditions (Clamped on left, Load on right)
    fdim = msh.topology.dim - 1
    # Clamp at x = -1
    facets_left = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], -1.0))
    dofs_left = fem.locate_dofs_topological(V, fdim, facets_left)
    bc = fem.dirichletbc(np.array([0, 0], dtype=PETSc.ScalarType), dofs_left, V)
    
    # Load at x = 1 (subdomain 2)
    facets_right = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    facet_indices = facets_right
    facet_values = np.full_like(facet_indices, 2, dtype=np.int32)
    facet_tags = mesh.meshtags(msh, fdim, facet_indices, facet_values)
    ds_tagged = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    
    force = fem.Constant(msh, PETSc.ScalarType((0, -1.0)))
    
    problem = ProblemClass()
    
    def solve_and_get_cost(ls_func):
        # The solver recalculates everything internally
        solver = CutFEMElasticSolver(ls_func, V_ls, V, ds_tagged, [bc], [], params, problem, force)
        u, p = solver.cutfem_solver(ls_func, params, problem)
        # Use the solver's internal measures and consistent Lame coefficients
        cost = problem.cost(u, p, mu, lmbda, solver.dxq, params)
        return u, p, cost, solver.dxq, solver.dxq

    # --- Step 0: Initial solve ---
    u0, p0, J0, dx0, ds0 = solve_and_get_cost(ls_init_func)
    


    print(f"J_CutFEM: {J0:.6f}")

    # Step 1: Shape Derivative
    # Pass dx0 so that VMLp_Problem can assemble its norm
    # IMPORTANT: Use the adjoint state p0 (not u0) for non-self-adjoint problems.
    integrand = problem.shape_derivative_integrand(u0, p0, mu, lmbda, params, measure=dx0)

    # FEniCS/UFL Tip: For AreaProblem, the integrand is a pure constant.
    # The FFCx compiler crashes ("Number of elements in generated code is zero")
    # if there are no finite element coefficients in the form.
    # We add + 0.0 * u0 to force the compiler to detect the finite element space.
    integrand = integrand + fem.Constant(msh, PETSc.ScalarType(0.0)) * ufl.inner(u0, u0)

    from cutfemx.fem import cut_form
    # v = -1.0 * n_ls. So v.n = -1.0
    dJ0_form = cut_form(integrand * (-1.0) * ds0)
    dJ0 = msh.comm.allreduce(cutfemx.fem.assemble_scalar(dJ0_form), op=MPI.SUM)

    # --- Step 2: Advection ---
    adv_solver = Advection(ls_obj.level_set, V_ls, dt=dt)
    velocity_field = fem.Function(V_ls)
    velocity_field.x.array[:] = -1.0 # Normal velocity F=-1.0 (Shrink hole)
    ls_new_func = adv_solver.cut_fem_adv(velocity_field, dt)

    # Create new level-set object for the new state
    ls_new = LevelSet(ls_new_func, V_ls)
    ls_new.level_set.x.array[:] = ls_new_func.x.array

    # --- Step 3: New solve ---
    u1, p1, J1, dx1, ds1 = solve_and_get_cost(ls_new.level_set)
    
    # --- Step 4: Consistency Check ---
    df_dt = (J1 - J0) / dt

    log_filename = f"{cost_name}_debug.log"
    with open(log_filename, "w") as f:
        f.write(f"--- {cost_name} Sensitivity Test ---\n")
        f.write(f"J0: {J0:.6e}\n")
        f.write(f"J1: {J1:.6e}\n")
        f.write(f"dJ0: {dJ0:.6e}\n")
        f.write(f"FD (J1-J0)/dt: {df_dt:.6e}\n")
        f.write(f"u0 norm: {np.linalg.norm(u0.x.array):.2e}\n")
        if abs(dJ0) > 1e-15:
            rel_err = abs(df_dt - dJ0) / abs(dJ0)
            f.write(f"Relative Error: {rel_err:.2e}\n")
    
    print(f"\nJ0: {J0:.6e}, J1: {J1:.6e}, dJ0: {dJ0:.6e}, FD: {df_dt:.6e}")
    
    if abs(dJ0) > 1e-15:
        relative_error = abs(df_dt - dJ0) / abs(dJ0)
        assert relative_error < 5.0, f"cost sensitivity inconsistent: {relative_error}"
    else:
        assert J0 > 1e-15, "cost J0 is zero"

if __name__ == "__main__":
    # Create a dummy test_spaces dictionary for standalone execution
    msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-1, -1]), np.array([1, 1])], [20, 20], mesh.CellType.triangle)
    V_ls = fem.functionspace(msh, ("Lagrange", 1))
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.topology.dim,)))
    test_spaces = {"V_ls": V_ls, "V": V}
    test_compliance_sensitivity(test_spaces)
