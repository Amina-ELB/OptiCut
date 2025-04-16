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

import cutfemx 
import mechanics_tool

from mpi4py import MPI

def descent_direction_3D_test(phi):
    return phi/phi

def descent_direction(level_set, msh,parameters,bc_velocity,V_ls,V_DG,\
                    rest_constraint,constraint_integrand,cost_integrand,xsi):

    u_r = ufl.TrialFunction(V_ls)
    v_r = ufl.TestFunction(V_ls)
    v_reg = fem.Function(V_ls)
    
    tdim = msh.topology.dim
    dim = msh.topology.dim

    intersected_entities = locate_entities(level_set,dim,"phi=0")
    inside_entities = locate_entities(level_set,dim,"phi<0")

    V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim,)))
    n_K = fem.Function(V_DG)
    compute_normal(n_K,level_set,intersected_entities)
    
    dof_coordinates = V_ls.tabulate_dof_coordinates()

    cut_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi<0")
    cut_mesh = create_cut_mesh(msh.comm,cut_cells,msh,inside_entities)
    interface_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi=0")
    interface_mesh = create_cut_cells_mesh(msh.comm,interface_cells)

    order = 2
    inside_quadrature = runtime_quadrature(level_set,"phi<0",order)
    interface_quadrature = runtime_quadrature(level_set,"phi=0",order)

    quad_domains = [(0,inside_quadrature), (1,interface_quadrature)]

    gp_ids =  ghost_penalty_facets(level_set, "phi<0")
    gp_topo = facet_topology(msh,gp_ids)

    #dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities),(2, intersected_entities)], domain=msh)
    dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)

    dxq = dx_rt(0) + dx(0)
    dsq = dx_rt(1)

    a_reg  =   parameters.alpha_reg_velocity *ufl.inner(grad(u_r), grad(v_r))*dx
    a_reg += u_r*v_r*dx

    # #C_Omega_value = fem.Constant(msh, PETSc.ScalarType(rest_constraint + parameters.ALM_slack_variable ))
    C_Omega_value = (rest_constraint + parameters.ALM_slack_variable )
    # # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    # # print("value vel c = ",parameters.ALM*(parameters.ALM_lagrangian_multiplicator * constraint_integrand + parameters.ALM_penalty_parameter * rest_constraint * constraint_integrand \
    # #         + 2 * constraint_integrand*parameters.ALM_slack_variable)+ (1-parameters.ALM)*parameters.target_constraint)

    # temp = -0.5*(2.0*lame_mu * ufl.inner(mecanics_tool.strain(sol_primal),\
    #         mecanics_tool.strain(sol_primal))  + lame_lambda *  ufl.inner(ufl.nabla_div(sol_primal), \
    #             ufl.nabla_div(sol_primal)))




    temp = cost_integrand
    temp_ALM = parameters.ALM*(parameters.ALM_lagrangian_multiplicator * constraint_integrand + parameters.ALM_penalty_parameter * C_Omega_value * constraint_integrand \
            + 2 * constraint_integrand*parameters.ALM_slack_variable)
    temp_ALM += (1-parameters.ALM)*parameters.target_constraint
    temp += temp_ALM


    #L_reg = -(inner(temp*v_r*n_K,n_K)*dsq)
    L_reg = -(ufl.dot(ufl.grad(xsi), -n_K)*inner(temp*v_r*n_K,n_K)*dx)

    a_cut_reg = cut_form(a_reg, jit_options={"cache_dir" : "ffcx-forms" })
    L_cut_reg = cut_form(L_reg)

    b_reg = assemble_vector(L_cut_reg)
    A_reg = assemble_matrix(a_cut_reg, bcs = [bc_velocity])
    A_reg.assemble()

    solver_reg = PETSc.KSP().create(msh.comm)
    solver_reg.setOperators(A_reg)
    solver_reg.setType(PETSc.KSP.Type.PREONLY)
    solver_reg.getPC().setType(PETSc.PC.Type.LU)
    solver_reg.solve(b_reg, v_reg.x.petsc_vec)
    b_reg.destroy()

    return v_reg

