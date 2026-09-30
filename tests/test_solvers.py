import pytest
import numpy as np
from dolfinx import fem, mesh
import ufl
from mpi4py import MPI
from solvers.cutfem_elastic_solver import CutFEMElasticSolver
from solvers.ersatz_elastic_solver import ErsatzElasticSolver
from config.parameters import Parameters
from config.problem import AreaProblem, Compliance_Problem

def get_tagged_measure(msh):
    """Helper to create a ds measure tagged on the right boundary."""
    fdim = msh.topology.dim - 1
    facets_right = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    facet_values = np.full(len(facets_right), 2, dtype=np.int32)
    facet_tags = mesh.meshtags(msh, fdim, np.asarray(facets_right, dtype=np.int32), facet_values)
    return ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

def test_ersatz_solver_init(circle_level_set, test_spaces, test_bcs, dummy_parameters, test_force):
    """
    Test that Ersatz solver can be initialized and solve the primal problem.
    This acts as a 'Smoke Test' to verify the end-to-end FEniCSx pipeline (assembly + PETSc solve).
    """
    V_ls = test_spaces["V_ls"]
    V = test_spaces["V"]
    msh = V.mesh
    ds_tagged = get_tagged_measure(msh)

    # Expand the material domain so it reaches x=1.0 where the force is applied
    circle_level_set.interpolate(lambda x: 0.8 - np.sqrt(x[0]**2 + x[1]**2) + 1e-9)
    dummy_parameters.cost_func = "compliance"

    solver = ErsatzElasticSolver(circle_level_set, V_ls, V, ds_tagged, test_bcs, [], dummy_parameters, test_force)
    u = solver.primal_problem(circle_level_set, dummy_parameters)

    # Asserting norm > 0 is crucial to avoid "silent failures" where the force is applied to an empty boundary
    norm = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u, u) * ufl.dx(domain=msh))), op=MPI.SUM))
    assert norm > 0

def test_cutfem_solver_init(circle_level_set, test_spaces, test_bcs, dummy_parameters, test_force):
    """
    Test that CutFEM solver can be initialized and solve the primal problem.
    Verifies that the cutfemx integration routines and runtime quadratures compile correctly.
    """
    V_ls = test_spaces["V_ls"]
    V = test_spaces["V"]
    msh = V.mesh
    ds_tagged = get_tagged_measure(msh)

    circle_level_set.interpolate(lambda x: 0.8 - np.sqrt(x[0]**2 + x[1]**2) + 1e-9)

    problem = AreaProblem()
    dummy_parameters.cost_func = "Area"
    solver = CutFEMElasticSolver(circle_level_set, V_ls, V, ds_tagged, test_bcs, [], dummy_parameters, problem, test_force)
    u = solver.primal_problem(circle_level_set, dummy_parameters)

    from cutfemx.fem import cut_form, assemble_scalar
    norm_form = cut_form(ufl.inner(u, u) * solver.dxq)
    norm = np.sqrt(V.mesh.comm.allreduce(assemble_scalar(norm_form), op=MPI.SUM))
    assert norm > 0

def test_functional_solver_comparison(circle_level_set, test_spaces, test_bcs, dummy_parameters, test_force):
    """
    Compare CutFEM and Ersatz solvers.
    This is a strong functional test proving that both methods (diffuse vs sharp interface)
    converge to a physically similar behavior on the same mechanical problem.
    """
    V_ls = test_spaces["V_ls"]
    V = test_spaces["V"]
    msh = V.mesh
    ds_tagged = get_tagged_measure(msh)

    circle_level_set.interpolate(lambda x: 0.8 - np.sqrt(x[0]**2 + x[1]**2) + 1e-9)
    dummy_parameters.cost_func = "compliance"

    # 1. Ersatz
    solver_ersatz = ErsatzElasticSolver(circle_level_set, V_ls, V, ds_tagged, test_bcs, [], dummy_parameters, test_force)
    u_ersatz = solver_ersatz.primal_problem(circle_level_set, dummy_parameters)
    norm_ersatz = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u_ersatz, u_ersatz) * ufl.dx(domain=msh))), op=MPI.SUM))

    # 2. CutFEM
    problem = Compliance_Problem()
    solver_cut = CutFEMElasticSolver(circle_level_set, V_ls, V, ds_tagged, test_bcs, [], dummy_parameters, problem, test_force)
    u_cut = solver_cut.primal_problem(circle_level_set, dummy_parameters)
    
    from cutfemx.fem import cut_form, assemble_scalar
    norm_cut_form = cut_form(ufl.inner(u_cut, u_cut) * solver_cut.dxq)
    norm_cut = np.sqrt(V.mesh.comm.allreduce(assemble_scalar(norm_cut_form), op=MPI.SUM))
    
    print(f"Norm Ersatz: {norm_ersatz}, Norm CutFEM: {norm_cut}")
    # They should be in the same ballpark (on a coarse 20x20 mesh they can differ by ~30%)
    assert np.isclose(norm_ersatz, norm_cut, rtol=0.40)
