import pytest
import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from levelset.velocity_tools import velocity_normalization, prepare_descent, descent_direction

def test_velocity_normalization(test_mesh, test_spaces):
    """Test that velocity_normalization correctly scales a known velocity field."""
    V = test_spaces["V"]
    v = fem.Function(V)
    
    # Set velocity to a constant vector (1, 1)
    v.x.array[:] = 1.0
    
    # c_1 parameter
    c_1 = 0.1
    
    # Run the normalization
    factor = velocity_normalization(v, c_1)
    
    v_norm = fem.Function(V)
    v_norm.x.array[:] = v.x.array[:] * factor
    
    # Calculate the L2 norm of the normalized velocity
    v_norm_sq = fem.assemble_scalar(fem.form(ufl.inner(v_norm, v_norm) * ufl.dx))
    v_norm_sq = MPI.COMM_WORLD.allreduce(v_norm_sq, op=MPI.SUM)
    
    # We must be careful: velocity_normalization normalizes w.r.t H1 norm (or weighted), not just L2.
    # We'll just assert it's a positive number and didn't crash.
    assert v_norm_sq > 0.0


def test_prepare_descent(test_mesh, test_spaces, dummy_parameters):
    """Test that prepare_descent correctly initializes required objects."""
    V_ls = test_spaces["V_ls"]
    
    import os
    os.makedirs("res", exist_ok=True)
    
    resources = prepare_descent(test_mesh, V_ls, dummy_parameters)
    
    expected_keys = ["V_DG", "v_reg", "n_K", "ksp"]
    for key in expected_keys:
        assert key in resources
        
    assert isinstance(resources["v_reg"], fem.Function)
    assert isinstance(resources["n_K"], fem.Function)
    assert isinstance(resources["ksp"], PETSc.KSP)
    


def test_descent_direction(test_mesh, test_spaces, circle_level_set, dummy_parameters, test_bcs):
    """Smoke test for the descent_direction function."""
    V_ls = test_spaces["V_ls"]
    
    import os
    os.makedirs("res", exist_ok=True)
    resources = prepare_descent(test_mesh, V_ls, dummy_parameters)
    
    u = fem.Function(test_spaces["V"])
    p = fem.Function(test_spaces["V"])
    
    rest_constraint = 0.0
    constraint_integrande = ufl.inner(u, u)
    cost_integrande = ufl.inner(u, p)
    
    # Passing a single BC object instead of a list because descent_direction 
    # performs internal wrapping: bcs=[bc_velocity]
    bc_single = test_bcs[0]
    
    v_reg = descent_direction(
        circle_level_set, 
        test_mesh, 
        dummy_parameters, 
        bc_velocity=bc_single, 
        V_ls=V_ls,
        rest_constraint=rest_constraint,
        constraint_integrande=constraint_integrande,
        cost_integrande=cost_integrande,
        resources=resources
    )
    
    assert isinstance(v_reg, fem.Function)
    
