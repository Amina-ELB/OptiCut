import pytest
import numpy as np
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import cutfemx
from cutfemx.fem import cut_form
from levelset.levelSet_tool import LevelSet, Advection, Reinitialization

def test_cut_fem_adv_geometric(test_spaces, circle_level_set):
    """
    Validation test: Advect a circle of radius r=0.5 with constant velocity F=0.1.
    After dt=0.1, the new radius should be r_new = r + F*dt = 0.51.
    We compare the measured perimeter with the theoretical perimeter 2*pi*r_new.
    """
    V_ls = test_spaces["V_ls"]
    mesh = V_ls.mesh
    r_init = 0.5
    dt = 0.1
    F = 0.1
    
    # Initialize Advection solver
    adv_solver = Advection(circle_level_set, V_ls, dt=dt)
    
    # Constant velocity field F (normal velocity)
    velocity_field = fem.Function(V_ls)
    velocity_field.x.array[:] = F
    
    # Run the advection step
    ls_new = adv_solver.cut_fem_adv(velocity_field, dt)
    
    # Compute PERIMETER of the advected level-set
    ls_new.name = "phi"
    cut_data = cutfemx.cut(ls_new)
    interface_quadrature = cutfemx.runtime_quadrature(cut_data, "phi=0", 3)
    
    dc_measure = ufl.Measure("dx", domain=mesh, subdomain_data=interface_quadrature)
    
    # In CutFEM standalone assembly, cut_form requires at least one Function/Trial/Test 
    # to deduce the function space.
    v_one = fem.Function(V_ls)
    v_one.x.array[:] = 1.0
    
    # Assemble on the interface for perimeter
    perimeter_form = cut_form(v_one * dc_measure)
    perimeter_measured = cutfemx.fem.assemble_scalar(perimeter_form)
    perimeter_measured = mesh.comm.allreduce(perimeter_measured, op=MPI.SUM)
    
    # Theoretical value
    r_expected = r_init + F * dt
    perimeter_theoretical = 2 * np.pi * r_expected
    
    # Output for debugging
    print(f"\n--- Geometric Validation (Perimeter) ---")
    print(f"Radius initial: {r_init}")
    print(f"Radius expected: {r_expected}")
    print(f"Perimeter Measured: {perimeter_measured:.4f}")
    print(f"Perimeter Theoretical: {perimeter_theoretical:.4f}")
    
    # Check accuracy (3% tolerance on coarse 20x20 mesh)
    assert np.isclose(perimeter_measured, perimeter_theoretical, rtol=0.03)

def test_reinit_PC_precision(test_spaces, circle_level_set):
    """
    Validation test: Reinitialization should preserve the interface position.
    We measure the L2 error of phi on the initial interface after 5 steps.
    """
    V_ls = test_spaces["V_ls"]
    mesh = V_ls.mesh
    l_param = 1.0
    
    # Work on a local copy to preserve circle_level_set fixture
    phi_test = fem.Function(V_ls)
    phi_test.x.array[:] = circle_level_set.x.array
    phi_test.x.scatter_forward()

    # 1. Define the measure on the INITIAL interface (phi=0)
    phi_test.name = "phi"
    cut_data_init = cutfemx.cut(phi_test)
    interface_quad_init = cutfemx.runtime_quadrature(cut_data_init, "phi=0", 3)
    dc_init = ufl.Measure("dx", domain=mesh, subdomain_data=interface_quad_init)
    
    # 2. Run reinitialization
    reinit_solver = Reinitialization(phi_test, V_ls, l=l_param)
    reinit_solver.reinitializationPC_inplace(phi_test, step_reinit=5)
    
    # 3. Compute L2 error: integral of (phi_new^2) on the initial interface
    error_interface_form = cut_form(phi_test**2 * dc_init)
    error_interface_l2 = (mesh.comm.allreduce(cutfemx.fem.assemble_scalar(error_interface_form), op=MPI.SUM))**0.5
    
    # 4. Compute Eikonal error: L2 norm of (|grad phi| - 1) on the whole domain
    grad_phi = ufl.grad(phi_test)
    norm_grad = ufl.sqrt(ufl.dot(grad_phi, grad_phi))
    # We use a standard dolfinx form here as it's a global integral
    eikonal_error_form = fem.form((norm_grad - 1.0)**2 * ufl.dx)
    eikonal_error_l2 = (mesh.comm.allreduce(fem.assemble_scalar(eikonal_error_form), op=MPI.SUM))**0.5
    
    # Normalization by domain area (2x2 = 4.0)
    domain_area = 4.0
    normalized_eikonal_error = eikonal_error_l2 / domain_area
    
    print(f"\n--- Reinitialization Validation ---")
    print(f"Interface preservation (L2 error): {error_interface_l2:.2e}")
    print(f"Eikonal property (|grad phi| = 1) error: {normalized_eikonal_error:.2e}")
    
    # Assertions
    assert error_interface_l2 < 1e-3, f"Interface shifted too much: {error_interface_l2}"
    assert normalized_eikonal_error < 0.05, f"Eikonal property not satisfied: {normalized_eikonal_error}"
