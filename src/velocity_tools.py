# Copyright (c) 2025 ONERA and MINES Paris, France 
#
# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari 

# The modules that will be used are imported:
import numpy as np
import math
import ufl
 
# mathematical language for FEM, auto differentiation, python
from dolfinx import fem, io, mesh
import matplotlib.pyplot as plt
# meshes, assembly, c++, ython, pybind
from ufl import dx, grad, inner, dS
import dolfinx
from petsc4py import PETSc
from typing import TYPE_CHECKING

from ufl import (FacetNormal,dx, grad, inner, dc, FacetNormal, CellDiameter)

from cutfemx.level_set import locate_entities, cut_entities, ghost_penalty_facets, facet_topology
from cutfemx.level_set import compute_normal
from cutfemx.mesh import create_cut_mesh, create_cut_cells_mesh
from cutfemx.quadrature import runtime_quadrature, physical_points
from cutfemx.fem import cut_form, cut_function

from cutfemx.petsc import assemble_vector, assemble_matrix, deactivate, locate_dofs


from mpi4py import MPI

import gc
import os
import psutil




def prepare_descent(msh, V_ls, parameters):
    r"""
    Prepare reusable objects for the descent direction problem.
    This should be called ONCE outside the iteration loop.

    :param dolfinx.mesh.Mesh msh: Computational mesh.
    :param dolfinx.fem.FunctionSpace V_ls: Level-set function space.
    :param object parameters: Object containing problem constants.

    :returns: Dictionary with reusable objects:
        - ``V_DG``: DG0 vector space for normals
        - ``v_reg``: fem.Function, solution of the extension subproblem
        - ``n_K``: fem.Function, normal vector function
        - ``ksp``: PETSc.KSP solver (pre-configured)
        - ``xdmf_ls``: XDMF file for results
    :rtype: dict
    """
    # DG space created only once (vectorial)
    V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim, ))) 
    
    # Reusable functions
    v_reg = fem.Function(V_ls)   # solution of the subproblem
    n_K = fem.Function(V_DG)     # normal vector, updated every iteration
    
    # Pre-create PETSc KSP solver to avoid costly recreation
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setType("cg")            # iterative solver (use "gmres" if needed)
    pc = ksp.getPC()
    pc.setType("hypre")          # AMG preconditioner (MPI-safe)
    ksp.setFromOptions()
    
    # Main XDMF file opened once (parallel safe)
    xdmf_ls = io.XDMFFile(msh.comm, "res/results.xdmf", "w")
    xdmf_ls.write_mesh(msh)
    
    return {
        "V_DG": V_DG,
        "v_reg": v_reg,
        "n_K": n_K,
        "ksp": ksp,
        "xdmf_ls": xdmf_ls
    }


def descent_direction(level_set, msh, parameters, bc_velocity, V_ls,
                      rest_constraint, constraint_integrande, cost_integrande,
                      resources):
    r"""
    Perform one iteration of the descent direction computation.
    Everything that depends on the current level set is recomputed here.

    :param fem.Function level_set: Current level-set function.
    :param dolfinx.mesh.Mesh msh: Computational mesh.
    :param object parameters: Problem parameters.
    :param list bc_velocity: List of velocity boundary conditions.
    :param fem.FunctionSpace V_ls: Level-set function space.
    :param fem.Form rest_constraint: Rest constraint form.
    :param fem.Form constraint_integrande: Constraint integrand form.
    :param fem.Form cost_integrande: Cost integrand form.
    :param dict resources: Pre-prepared objects from `prepare_descent`.

    :returns: Velocity field for advection (v_reg).
    :rtype: fem.Function
    """
    V_DG = resources["V_DG"]
    v_reg = resources["v_reg"]
    n_K = resources["n_K"]
    ksp = resources["ksp"]
    xdmf_ls = resources["xdmf_ls"]
    
    tdim = msh.topology.dim
    dim = msh.geometry.dim
    
    intersected_entities = locate_entities(level_set, dim, "phi=0")
    inside_entities = locate_entities(level_set, dim, "phi<0")
    
    # Update normal function
    compute_normal(n_K, level_set, intersected_entities)
    
    dof_coordinates = V_ls.tabulate_dof_coordinates()
    
    # Cut and interface meshes
    cut_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi<0")
    cut_mesh = create_cut_mesh(msh.comm, cut_cells, msh, inside_entities)
    interface_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi=0")
    interface_mesh = create_cut_cells_mesh(msh.comm, interface_cells)
    
    # Quadrature rules
    order = 2
    inside_quadrature = runtime_quadrature(level_set, "phi<0", order)
    interface_quadrature = runtime_quadrature(level_set, "phi=0", order)
    quad_domains = [(0, inside_quadrature), (1, interface_quadrature)]
    
    # Measures
    dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities),(2, intersected_entities)], domain=msh)
    dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)
    dxq = dx_rt(0) + dx(0)
    dsq = dx_rt(1)
    
    # Trial/test functions
    u_r = ufl.TrialFunction(V_ls)
    v_r = ufl.TestFunction(V_ls)
    
    # Bilinear form
    a_reg  = parameters.alpha_reg_velocity * ufl.inner(ufl.grad(u_r), ufl.grad(v_r)) * dx
    a_reg += u_r * v_r * dx
    
    # Linear form
    C_Omega_value = (rest_constraint + parameters.ALM_slack_variable)
    temp = ufl.as_ufl(cost_integrande)  # avoid DAG growth
    temp_ALM = parameters.ALM * (
        parameters.ALM_lagrangian_multiplicator * constraint_integrande +
        parameters.ALM_penalty_parameter * C_Omega_value * constraint_integrande +
        2 * constraint_integrande * parameters.ALM_slack_variable
    )
    temp_ALM += (1 - parameters.ALM) * parameters.target_constraint
    temp += temp_ALM
    L_reg = -(ufl.inner(temp * v_r * n_K, n_K) * dsq)
    
    # Cut forms (temporary, destroyed after solve)
    a_cut_reg = cut_form(a_reg, jit_options={"cache_dir" : "ffcx-forms"})
    L_cut_reg = cut_form(L_reg)
    
    # Assembly
    b_reg = assemble_vector(L_cut_reg)
    A_reg = assemble_matrix(a_cut_reg, bcs=[bc_velocity])
    A_reg.assemble()
    
    # Solve
    ksp.setOperators(A_reg)
    ksp.setUp()  # prevent PETSc internal memory retention
    ksp.setTolerances(rtol=1e-8, atol=1e-12)
    ksp.solve(b_reg, v_reg.x.petsc_vec)
    
    # Explicit cleanup of PETSc objects
    b_reg.destroy()
    A_reg.destroy()
    
    # Drop references to heavy temporary Python/C++ objects
    del a_cut_reg, L_cut_reg, cut_mesh, interface_mesh, cut_cells, interface_cells
    gc.collect()
    
    # Optional: track memory usage
    process = psutil.Process(os.getpid())
    print(f"[Rank {msh.comm.rank}] RAM used: {process.memory_info().rss / 1e9:.3f} GB")
    
    return v_reg


def velocity_normalization(v, c_1):
    r"""
    Normalization of the velocity field according to the following equation:

    .. math::

        \overline{v} = \frac{v}{\sqrt{c \left\Vert \nabla v \right\Vert_{L^2(D)}^2 + \left\Vert v \right\Vert_{L^2(D)}^2 }}

    with :math:`c>0` and :math:`\left\Vert . \right\Vert_{L^2(D)}` norm defined as:

    .. math::

        \left\Vert f \right\Vert_{L^2(D)}^2 = \int_{D} f \cdot f \, dx

    :param fem.Function v: Velocity field (vector) to normalize.
    :param float c_1: Smoothing parameter used in the extension PDE.
    :param ufl.Measure dx: Measure for volume integration.

    :returns: Normalized velocity field
    :rtype: fem.Function
    """
    # Compute volume integrals in parallel
    b_grad = fem.form(ufl.inner(ufl.grad(v), ufl.grad(v)) * ufl.dx)
    b_v = fem.form(ufl.inner(v, v) * ufl.dx)
    
    denom = MPI.COMM_WORLD.allreduce(fem.assemble_scalar(b_grad) * c_1 + fem.assemble_scalar(b_v), op=MPI.SUM)
    
    res = v / ufl.sqrt(denom)  # returns a UFL expression
    return res
