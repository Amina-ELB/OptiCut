import pytest
from mpi4py import MPI
from dolfinx import fem, mesh
import ufl
import numpy as np
import cutfemx
import dolfinx.fem.petsc
from solvers.cutfem_elastic_solver import CutFEMElasticSolver
from solvers.ersatz_elastic_solver import ErsatzElasticSolver
from config.problem import Compliance_Problem, VMLp_Problem
from utils import mechanics_tool

def test_regression_ersatz_compliance(circle_level_set, test_spaces, test_measures, test_bcs, dummy_parameters):
    """Regression test for Ersatz solver: displacement L2 norm."""
    V_ls = test_spaces["V_ls"]
    V = test_spaces["V"]
    msh = V.mesh
    fdim = msh.topology.dim - 1
    
    facets_right = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    # Handle parallel execution: some ranks might have no facets
    facet_values = np.full(len(facets_right), 2, dtype=np.int32)
    facet_tags = mesh.meshtags(msh, fdim, np.asarray(facets_right, dtype=np.int32), facet_values)
    ds_tagged = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    
    circle_level_set.interpolate(lambda x: 0.8 - np.sqrt(x[0]**2 + x[1]**2) + 1e-9)
    
    params = dummy_parameters
    params.cost_func = "compliance"
    force = fem.Constant(msh, np.array([0.0, -1.0], dtype=np.float64))
    
    solver = ErsatzElasticSolver(circle_level_set, V_ls, V, ds_tagged, test_bcs, [], params, force)
    u = solver.primal_problem(circle_level_set, params)
    
    norm = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u, u) * ufl.dx(domain=msh))), op=MPI.SUM))
    
    expected_norm = 0.005907625143527931
    print(f"Computed Ersatz Norm: {norm:.8f}, Expected: {expected_norm:.8f}")
    assert np.isclose(norm, expected_norm, rtol=1e-5)

def test_regression_cutfem_compliance(circle_level_set, test_spaces, test_measures, test_bcs, dummy_parameters):
    """Regression test for CutFEM solver: displacement L2 norm."""
    V_ls = test_spaces["V_ls"]
    V = test_spaces["V"]
    msh = V.mesh
    fdim = msh.topology.dim - 1
    
    facets_right = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    facet_values = np.full(len(facets_right), 2, dtype=np.int32)
    facet_tags = mesh.meshtags(msh, fdim, np.asarray(facets_right, dtype=np.int32), facet_values)
    ds_tagged = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    
    circle_level_set.interpolate(lambda x: 0.8 - np.sqrt(x[0]**2 + x[1]**2) + 1e-9)
    
    params = dummy_parameters
    params.cost_func = "compliance"
    force = fem.Constant(msh, np.array([0.0, -1.0], dtype=np.float64))
    
    problem = Compliance_Problem()
    solver = CutFEMElasticSolver(circle_level_set, V_ls, V, ds_tagged, test_bcs, [], params, problem, force)
    u = solver.primal_problem(circle_level_set, params)
    
    from cutfemx.fem import cut_form
    norm_form = cut_form(ufl.inner(u, u) * solver.dxq)
    norm = np.sqrt(msh.comm.allreduce(cutfemx.fem.assemble_scalar(norm_form), op=MPI.SUM))
    
    expected_norm = 0.004602241883348144
    print(f"Computed CutFEM Norm: {norm:.8f}, Expected: {expected_norm:.8f}")
    assert np.isclose(norm, expected_norm, rtol=1e-5)

def test_regression_cutfem_lp_adjoint(circle_level_set, test_spaces, test_measures, test_bcs, dummy_parameters):
    """Regression test for CutFEM Adjoint (Lp)."""
    V_ls = test_spaces["V_ls"]
    V = test_spaces["V"]
    msh = V.mesh
    fdim = msh.topology.dim - 1
    
    facets_right = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    facet_values = np.full(len(facets_right), 2, dtype=np.int32)
    facet_tags = mesh.meshtags(msh, fdim, np.asarray(facets_right, dtype=np.int32), facet_values)
    ds_tagged = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    
    circle_level_set.interpolate(lambda x: 0.8 - np.sqrt(x[0]**2 + x[1]**2) + 1e-9)
    
    params = dummy_parameters
    params.cost_func = "VonMises"
    params.p_const = 8
    params.elasticity_limit = 1.0
    force = fem.Constant(msh, np.array([0.0, -1.0], dtype=np.float64))
    
    problem = VMLp_Problem()
    solver = CutFEMElasticSolver(circle_level_set, V_ls, V, ds_tagged, test_bcs, [], params, problem, force)
    
    u = solver.primal_problem(circle_level_set, params)
    mu, lmbda = mechanics_tool.lame_compute(params.young_modulus, params.poisson)
    dual_op = problem.dual_operator(u, mu, lmbda, params, msh, solver.dxq)
    p = solver.adjoint_problem(u, dual_op)
    
    from cutfemx.fem import cut_form
    norm_p_form = cut_form(ufl.inner(p, p) * solver.dxq)
    norm_p = np.sqrt(msh.comm.allreduce(cutfemx.fem.assemble_scalar(norm_p_form), op=MPI.SUM))
    
    expected_norm_p = 1158981064.17731786
    print(f"Computed Adjoint Norm: {norm_p:.8f}, Expected: {expected_norm_p:.8f}")
    assert np.isclose(norm_p, expected_norm_p, rtol=1e-5)
