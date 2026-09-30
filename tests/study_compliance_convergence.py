import sys
import os
sys.path.append(os.path.abspath("./src"))

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc
import cutfemx
import matplotlib.pyplot as plt

from levelset.levelSet_tool import Advection, LevelSet
from solvers.cutfem_elastic_solver import CutFEMElasticSolver
from config.problem import AreaProblem, Compliance_Problem, VMLp_Problem
from config.parameters import Parameters
from utils import mechanics_tool

class MockParams(Parameters):
    def __init__(self, cost_func_name="compliance"):
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
        self.type_constraint = "Area"
        self.h = 0.1
        self.lame_mu, self.lame_lambda = mechanics_tool.lame_compute(self.young_modulus, self.poisson)
        
        # ALM parameters required by some Problem classes (like AreaProblem)
        self.ALM_lagrangian_multiplicator = 0.0
        self.ALM_penalty_parameter = 1.0
        self.ALM_slack_variable = 0.0

def run_convergence_study(ProblemClass, cost_name):
    print(f"\n{'='*50}\nStarting Convergence Study for: {cost_name}\n{'='*50}")
    msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-1, -1]), np.array([1, 1])], [50, 50], mesh.CellType.triangle)
    V_ls = fem.functionspace(msh, ("Lagrange", 1))
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.topology.dim,)))
    
    params = MockParams(cost_func_name=cost_name)
    mu, lmbda = params.lame_mu, params.lame_lambda
    
    ls_init_func = fem.Function(V_ls)
    # Expand the material domain so it reaches x=1.0 where the force is applied
    ls_init_func.interpolate(lambda x: 0.8 - np.sqrt(x[0]**2 + x[1]**2) + 1e-9)
    
    fdim = msh.topology.dim - 1
    facets_left = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], -1.0))
    dofs_left = fem.locate_dofs_topological(V, fdim, facets_left)
    bc = fem.dirichletbc(np.array([0, 0], dtype=PETSc.ScalarType), dofs_left, V)
    
    facets_right = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    facet_tags = mesh.meshtags(msh, fdim, facets_right, np.full_like(facets_right, 2, dtype=np.int32))
    ds_tagged = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    
    force = fem.Constant(msh, PETSc.ScalarType((0, -1.0)))
    problem = ProblemClass()
    
    def solve_and_get_cost(ls_func):
        solver = CutFEMElasticSolver(ls_func, V_ls, V, ds_tagged, [bc], [], params, problem, force)
        u, p = solver.cutfem_solver(ls_func, params, problem)
        cost = problem.cost(u, p, mu, lmbda, solver.dxq, params)
        return u, p, cost, solver.dxq, solver.dsq

    # Analytical derivative
    u0, p0, J0, dx0, ds0 = solve_and_get_cost(ls_init_func)
    
    integrand = problem.shape_derivative_integrand(u0, p0, mu, lmbda, params, measure=dx0)
    # Force FFCx to recognize the finite element space (especially for AreaProblem where integrand is purely scalar)
    integrand = integrand + fem.Constant(msh, PETSc.ScalarType(0.0)) * ufl.inner(u0, u0)
    
    from cutfemx.fem import cut_form
    dJ0_form = cut_form(integrand * (-1.0) * ds0)
    dJ0 = msh.comm.allreduce(cutfemx.fem.assemble_scalar(dJ0_form), op=MPI.SUM)
    print(f"Analytical Shape Derivative dJ0: {dJ0:.6e}")
    
    # If derivative is exactly zero (e.g. constant cost), skip FD
    if abs(dJ0) < 1e-15:
        print(f"Derivative is near zero ({dJ0}), skipping convergence plot.")
        return

    # For CutFEM, the mesh size h is 2.0 / 50 = 0.04.
    # Small dt values (e.g., 1e-4) cause numerical jumps due to Ghost Penalties crossing cells.
    # The appropriate range to validate the gradient using finite differences is dt ~ O(h).
    dt_values = np.linspace(0.005, 0.05, 50)
    errors = []
    
    for dt in dt_values:
        ls_obj = LevelSet(ls_init_func, V_ls)
        adv_solver = Advection(ls_obj.level_set, V_ls, dt=dt)
        velocity_field = fem.Function(V_ls)
        velocity_field.x.array[:] = -1.0
        ls_new_func = adv_solver.cut_fem_adv(velocity_field, dt)
        
        _, _, J1, _, _ = solve_and_get_cost(ls_new_func)
        df_dt = (J1 - J0) / dt
        
        rel_error = abs(df_dt - dJ0) / abs(dJ0)
        errors.append(rel_error)
        print(f"dt: {dt:.1e}, FD: {df_dt:.6e}, Error: {rel_error:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(dt_values, errors, 'o-', label=f'{cost_name} Error')
    plt.xlabel('dt')
    plt.ylabel('Relative Error')
    plt.title(f'Convergence of FD derivative for {cost_name}')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    filename = f'{cost_name}_convergence.png'
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

if __name__ == "__main__":
    import inspect
    import config.problem as opti_problem
    
    problems = []
    # Iterate over all classes defined in config.problem
    for name, obj in inspect.getmembers(opti_problem, inspect.isclass):
        # Check that the class implements the Optimization Problem interface
        if (hasattr(obj, 'cost') and 
            hasattr(obj, 'dual_operator') and 
            hasattr(obj, 'shape_derivative_integrand')):
            
            # Infer the cost name (to initialize parameters correctly)
            if "Compliance" in name:
                cost_name = "compliance"
            elif "VM" in name:
                cost_name = "VonMises"
            elif "Area" in name:
                cost_name = "Area"
            else:
                cost_name = name.replace('_Problem', '').replace('Problem', '')
                
            problems.append((obj, cost_name))
            
    print(f"Dynamically discovered {len(problems)} problem classes to test: {[n for _, n in problems]}")
    
    for cls, name in problems:
        run_convergence_study(cls, name)
