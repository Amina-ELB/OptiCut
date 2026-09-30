import pytest
import numpy as np
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc
import ufl
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
    import cutfemx
    from cutfemx.fem import cut_form
    import cutfemx

    tdim = mesh.topology.dim
    dim = mesh.geometry.dim

    # 3. Quadrature and Measures
    cut_data = cutfemx.cut(ls_new)
    cutfemx.update(cut_data)
    inside_quadrature = cutfemx.runtime_quadrature(cut_data, "phi<0", 3)
    interface_quadrature = cutfemx.runtime_quadrature(cut_data, "phi=0", 3)
    quad_domains = interface_quadrature
    
    dc_measure = ufl.Measure("dx", domain=mesh, subdomain_data=quad_domains)
    
    # In CutFEM standalone assembly, cut_form requires at least one Function/Trial/Test 
    # to deduce the function space. fem.Constant(1.0) leads to RuntimeError.
    v_one = fem.Function(V_ls)
    v_one.x.array[:] = 1.0
    
    # Assemble on subdomain 1 (the interface) for perimeter
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
    
    # Check accuracy (1% tolerance)
    assert np.isclose(perimeter_measured, perimeter_theoretical, rtol=0.01)

def test_reinit_PC_precision(test_spaces, circle_level_set):
    """
    Validation test: Reinitialization should preserve the interface position.
    We measure the L2 error of phi on the initial interface after 5 steps.
    """
    V_ls = test_spaces["V_ls"]
    mesh = V_ls.mesh
    l_param = 1.0
    
    import cutfemx
    from cutfemx.fem import cut_form
    import cutfemx
    
    # 1. Define the measure on the INITIAL interface (phi=0)
    cut_data_init = cutfemx.cut(circle_level_set)
    cutfemx.update(cut_data_init)
    interface_quad_init = cutfemx.runtime_quadrature(cut_data_init, "phi=0", 3)
    quad_domains = interface_quad_init
    dc_init = ufl.Measure("dx", domain=mesh, subdomain_data=quad_domains)
    
    # 2. Run reinitialization for 5 steps
    reinit_solver = Reinitialization(circle_level_set, V_ls, l=l_param)
    reinit_solver.reinitializationPC_inplace(circle_level_set, step_reinit=5)
    
    # 3. Compute L2 error: integral of (phi_new^2) on the initial interface
    phi_new = reinit_solver.level_set
    error_interface_form = cut_form(phi_new**2 * dc_init(1))
    error_interface_l2 = (mesh.comm.allreduce(cutfemx.fem.assemble_scalar(error_interface_form), op=MPI.SUM))**0.5
    
    # 4. Compute Eikonal error: L2 norm of (|grad phi| - 1) on the whole domain
    grad_phi = ufl.grad(phi_new)
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
