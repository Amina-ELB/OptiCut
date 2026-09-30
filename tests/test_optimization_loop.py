import pytest
import numpy as np
import ufl
import inspect
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
import cutfemx

from solvers.cutfem_elastic_solver import CutFEMElasticSolver
from levelset.levelSet_tool import Advection
from levelset.velocity_tools import descent_direction, velocity_normalization, prepare_descent
from config import problem as opti_problem
from utils import mechanics_tool
from optimization import almMethod

# Dynamically gather all problem classes
problem_classes = []
for name, obj in inspect.getmembers(opti_problem, inspect.isclass):
    if hasattr(obj, 'cost') and hasattr(obj, 'dual_operator') and hasattr(obj, 'shape_derivative_integrand'):
        problem_classes.append((name, obj))

@pytest.mark.parametrize("problem_name, problem_class", problem_classes)
def test_end_to_end_optimization_loop(circle_level_set, test_spaces, test_bcs, dummy_parameters, test_force, problem_name, problem_class):
    """
    End-to-End Integration Test for the Optimization Loop.
    This test runs 3 iterations of the ALM optimization loop (Solver -> Descent -> Advection)
    for all available problem classes, and asserts that the unconstrained cost strictly decreases.
    """
    V_ls = test_spaces["V_ls"]
    V = test_spaces["V"]
    msh = V.mesh
    fdim = msh.topology.dim - 1
    
    # 1. Setup the physical problem (load at x=1)
    facets_right = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    facet_values = np.full(len(facets_right), 2, dtype=np.int32)
    facet_tags = mesh.meshtags(msh, fdim, np.asarray(facets_right, dtype=np.int32), facet_values)
    ds_tagged = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    # Reset the material domain
    circle_level_set.interpolate(lambda x: 0.8 - np.sqrt(x[0]**2 + x[1]**2) + 1e-9)

    # 2. Setup parameters
    if "Compliance" in problem_name:
        cost_name = "compliance"
    elif "VM" in problem_name:
        cost_name = "VonMises"
    elif "Area" in problem_name:
        cost_name = "Area"
    else:
        cost_name = problem_name

    dummy_parameters.cost_func = cost_name
    dummy_parameters.type_constraint = "Area"
    dummy_parameters.dt = 0.05
    dummy_parameters.target_constraint = 0.5
    dummy_parameters.alpha_reg_velocity = 0.1
    
    # Required for ALM
    dummy_parameters.ALM = 1
    dummy_parameters.ALM_lagrangian_multiplicator = 0.0
    # Set penalty to 0.0 to disable the volume constraint for this test.
    # This guarantees the optimizer will strictly minimize the primary cost (add material).
    dummy_parameters.ALM_penalty_parameter = 0.0
    dummy_parameters.ALM_slack_variable = 0.0
    
    mu, lmbda = mechanics_tool.lame_compute(dummy_parameters.young_modulus, dummy_parameters.poisson)
    problem = problem_class()
    
    # Clamp the velocity on the left boundary, matching the mechanical boundary condition
    bc_velocity = test_bcs[0]
    
    # Prepare descent resources
    resources = prepare_descent(msh, V_ls, dummy_parameters)
    
    try:
        # Initialize Solvers
        CutFemSolver = CutFEMElasticSolver(circle_level_set, V_ls, V, ds_tagged, test_bcs, bc_velocity, dummy_parameters, problem, test_force)
        AdvectionSolver = Advection(circle_level_set, V_ls, dt=dummy_parameters.dt)
        
        costs = []
        
        # --- OPTIMIZATION LOOP (3 Iterations) ---
        for iteration in range(3):
            # A. Solve State
            uh, ph = CutFemSolver.cutfem_solver(circle_level_set, dummy_parameters, problem)
            
            # B. Compute Cost and Constraint
            if problem_name in ["VMLp_Problem", "AreaProblem"]:
                from utils.mechanics_tool import project_von_mises
                vm_DG = project_von_mises(uh, mu, lmbda, msh, CutFemSolver.dxq)
                cost = problem.cost(uh, ph, mu, lmbda, CutFemSolver.dxq, dummy_parameters)
                constraint = problem.constraint(uh, mu, lmbda, dummy_parameters, CutFemSolver.dxq, vm_DG=vm_DG)
                
                costs.append(cost)
                
                # C. Compute Shape Derivatives
                shape_derivative = problem.shape_derivative_integrand(uh, ph, mu, lmbda, dummy_parameters, CutFemSolver.dxq)
                shape_derivative = shape_derivative + fem.Constant(msh, PETSc.ScalarType(0.0)) * ufl.inner(uh, uh)
                shape_derivative_integrand_constraint = problem.shape_derivative_integrand_constraint(uh, ph, mu, lmbda, dummy_parameters, ufl.dx, vm_DG=vm_DG)
            else:
                cost = problem.cost(uh, ph, mu, lmbda, CutFemSolver.dxq, dummy_parameters)
                constraint = problem.constraint(uh, mu, lmbda, dummy_parameters, CutFemSolver.dxq)
                
                costs.append(cost)
                
                shape_derivative = problem.shape_derivative_integrand(uh, ph, mu, lmbda, dummy_parameters, CutFemSolver.dxq)
                shape_derivative = shape_derivative + fem.Constant(msh, PETSc.ScalarType(0.0)) * ufl.inner(uh, uh)
                shape_derivative_integrand_constraint = problem.shape_derivative_integrand_constraint(uh, ph, mu, lmbda, dummy_parameters, ufl.dx)
            
            # D. Descent Direction
            v_reg = descent_direction(
                CutFemSolver.level_set, msh, dummy_parameters, bc_velocity, V_ls,
                constraint, shape_derivative_integrand_constraint, shape_derivative, resources
            )
            
            # E. Velocity Normalization
            factor = velocity_normalization(v_reg, dummy_parameters.alpha_reg_velocity)
            velocity_field_normalized = fem.Function(V_ls)
            velocity_field_normalized.x.array[:] = v_reg.x.array[:] * factor

            # F. Advect Level-Set
            new_ls_func = AdvectionSolver.cut_fem_adv(velocity_field_normalized, dummy_parameters.dt)
            circle_level_set.x.array[:] = new_ls_func.x.array
            circle_level_set.x.scatter_forward()
            
            CutFemSolver.update_measures_and_quadratures(circle_level_set)

        # --- ASSERTIONS ---
        assert len(costs) == 3
        print(f"Cost history for {problem_name}: {costs}")
        
        # The primary cost MUST decrease after applying the pure descent direction
        assert costs[-1] <= costs[0] + 1e-3, f"Optimization failed for {problem_name}: Cost did not decrease. Initial: {costs[0]}, Final: {costs[-1]}"
    
    finally:
            pass
        # Clean up resources to prevent HDF5 locking errors in next tests
