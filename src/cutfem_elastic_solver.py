# Copyright (c) 2025 ONERA and MINES Paris, France 
#
# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari 

import ufl
from ufl import (FacetNormal, Measure, SpatialCoordinate, TestFunction, TrialFunction, 
                 div, dot, dx, grad, inner, lhs, rhs, dc, dS, FacetNormal, CellDiameter, dot, avg, jump)

###

import cutfemx
from dolfinx import cpp as _dolfinxcpp
from dolfinx import fem, io, mesh

from mpi4py import MPI

import math
import collections
import functools
import typing

import numpy as np

from dolfinx import la

#  from dolfinx.cpp.la.petsc import create_vector # error 4 
from dolfinx.cpp.fem import pack_coefficients as dolfinx_pack_coefficients
# from dolfinx.fem.forms import form_types# error 5 
from dolfinx.fem.assemble import pack_constants as dolfinx_pack_constants
# from dolfinx.fem.bcs import DirichletBCMetaClass # error 6 
from dolfinx.fem.function import Function, FunctionSpace
###

from dolfinx import fem
from dolfinx.mesh import locate_entities, meshtags
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
import numpy as np
#from mecanics_problem import *
from dolfinx.mesh import locate_entities, meshtags, locate_entities_boundary

from cutfemx.level_set import locate_entities, cut_entities, ghost_penalty_facets, facet_topology
from cutfemx.level_set import compute_normal
from cutfemx.mesh import create_cut_mesh, create_cut_cells_mesh
from cutfemx.quadrature import runtime_quadrature, physical_points
from cutfemx.fem import cut_form, cut_function

from cutfemx.petsc import assemble_vector, assemble_matrix, deactivate, locate_dofs


from matplotlib import pyplot as plt
import mechanics_tool

class CutFEMElasticSolver:
    r"""This is the CutFEM class.

    Some details about the initialization of linear elasticity problem with CutFEM method.

    Definition of Primal problem
    =================================================

    Linear elasticity problem is given by: 
    Find :math:`u:\Omega \rightarrow \mathbb{R}^{d}`

    .. math::

            \begin{align}
            \begin{cases}
                -\text{div}( \sigma(u)) & \!\!\!\!=0 \text{ in }\Omega\\
                u& \!\!\!\!=0\text{ on }\Gamma_{D}\\
                \sigma(u)\cdot n & \!\!\!\!=g\text{ on }\Gamma_{N}
            \end{cases}
            \end{align}


    Where :math:`d` the dimension of the problem.
    *We assume small deformations and zero volumetric forces.* 
        
    This yields to the following weak formulation: 
    Find :math:`u \in V`, such that for all :math:`v \in V` we have


    .. _bilinearFormCutfem:

    .. math::

        a\left(u,v\right)=l\left(v\right)

    with:

    .. _bilinearFormCutfemDetails:

    .. math::

        \begin{align} 
        a\left(u,v\right) &= 2\mu\left(\varepsilon(u),\varepsilon(v)\right)_{L^{2}(\Omega)} + \lambda\left(\nabla\cdot u,\nabla\cdot v\right)_{L^{2}(\Omega)} \\
        l\left(v\right) &= \left(g,v\right)_{L^{2}\left(\Gamma_{N}\right)},
        \end{align}

    .. _bilinearFormCutfemCode:

    Bilinear form (primal):
    -------------------------------
    .. code-block:: python

        import ufl

        u =ufl.TrialFunction(self.space_displacement)
        v =ufl.TestFunction(self.space_displacement)

        self.gamma = 1e-5*(self.lame_mu + self.lame_lambda)

        self.h = CellDiameter(self.mesh)

        self.bc = bc

        self.a_primal =  2.0*self.lame_mu  * ufl.inner(mecanics_tool.strain(u), mecanics_tool.strain(v)) * self.dxq \
            + self.lame_lambda *  ufl.inner(ufl.nabla_div(u), ufl.nabla_div(v)) * self.dxq
        # Stabilization:
        self.a_primal += avg(self.gamma) * avg(self.h)**3*ufl.inner(ufl.jump(ufl.grad(u),self.n),\
            ufl.jump(ufl.grad(v),self.n))*self.dS(0)
            
    .. _linearForm:

    Linear form (primal):
    --------------------------
        
    .. code-block:: python

        self.L_primal = ufl.dot(self.shift,v) * self.ds(2)


    Definition of Dual problem
    =================================================

    Some details about the initialization of adjoint problem with CutFEM.

    .. _bilinearFormDual:
            
    Bilinear form (dual):
    ------------------------
    .. code-block:: python

        import ufl

        u =ufl.TrialFunction(self.space_displacement)
        v =ufl.TestFunction(self.space_displacement)

        self.gamma = 1e-5*(self.lame_mu + self.lame_lambda)

        self.h = CellDiameter(self.mesh)

        self.bc = bc

        self.a_adjoint =  2.0*self.lame_mu  * ufl.inner(mecanics_tool.strain(u), mecanics_tool.strain(v)) * self.dxq \
            + self.lame_lambda *  ufl.inner(ufl.nabla_div(u), ufl.nabla_div(v)) * self.dxq
        # Stabilization:
        self.a_adjoint += avg(self.gamma) * avg(self.h)**3*ufl.inner(ufl.jump(ufl.grad(u),self.n),\
            ufl.jump(ufl.grad(v),self.n))*self.dS(0)


    .. _linearFormLpnorm:

    Linear form (dual):
    ---------------------------

    The dual operator is compute using the automatic differentiation :

       
    .. code-block:: python

        ## Exemple for Lp nom of VonMises constraint minimization:

        self.J = ((mechanics_tool.von_mises(self.uh,self.lame_mu,self.lame_lambda,self.dim)/parameters.elasticity_limit)**self.p_const)*self.dxq

        self.L_adj = ufl.derivative(self.J,self.uh,v_adj)


        
        
    """

    def __init__(self, level_set, level_set_space, space_displacement,ds, bc, bc_velocity, parameters,problem_topo, shift):
        
        self.level_set = fem.Function(level_set_space)
        self.level_set.x.array[:] = level_set.x.array

        self.mesh = self.level_set.function_space.mesh
        self.space_displacement = space_displacement
        self.cutFEM = parameters.cutFEM

        lame_mu,lame_lambda = mechanics_tool.lame_compute(parameters.young_modulus,parameters.poisson)

        self.V_ls = level_set_space
        self.cost_func = parameters.cost_func

        self.lame_mu = lame_mu
        self.lame_lambda = lame_lambda
        self.dim = self.mesh.topology.dim

        self.bc_velocity = bc_velocity

        ################
        #Mecanic Problem
        ################
        self.tdim = self.dim 

        self.shift = shift
        
        self.intersected_entities = locate_entities(self.level_set,self.dim,"phi=0")
        self.inside_entities = locate_entities(self.level_set,self.dim,"phi<0")

        V_DG = fem.functionspace(self.mesh, ("DG", 0, (self.dim,)))
        self.n = fem.Function(V_DG)
        self.n = FacetNormal(self.mesh)
        
        self.dof_coordinates = self.V_ls.tabulate_dof_coordinates()
        
        self.cut_cells = cut_entities(self.level_set, self.dof_coordinates, self.intersected_entities, self.dim, "phi<0")
        self.cut_mesh = create_cut_mesh(self.mesh.comm,self.cut_cells,self.mesh,self.inside_entities)
        self.interface_cells = cut_entities(self.level_set, self.dof_coordinates, self.intersected_entities, self.tdim, "phi=0")
        self.interface_mesh = create_cut_cells_mesh(self.mesh.comm,self.interface_cells)

        
        self.order = 2
        self.inside_quadrature = runtime_quadrature(self.level_set,"phi<0",self.order)
        self.interface_quadrature = runtime_quadrature(self.level_set,"phi=0",self.order)

        self.quad_domains = [(0,self.inside_quadrature), (1,self.interface_quadrature)]

        self.gp_ids =  ghost_penalty_facets(self.level_set, "phi<0")
        self.gp_topo = facet_topology(self.mesh,self.gp_ids)
                    
        self.ds = ds

        self.dx =ufl.Measure("dx", subdomain_data=[(0, self.inside_entities)], domain=self.mesh)

        self.dx_rt = ufl.Measure("dC", subdomain_data=self.quad_domains, domain=self.mesh)
        self.dS = ufl.Measure("dS", subdomain_data=[(0, self.gp_topo)], domain=self.mesh)
        
        self.dxq = self.dx_rt(0) + self.dx(0)
        self.dsq = self.dx_rt(1)


       
        u =ufl.TrialFunction(self.space_displacement)
        v =ufl.TestFunction(self.space_displacement)
        self.uh = fem.Function(self.space_displacement)
        self.ph = fem.Function(self.space_displacement)
        
        self.gamma_N = 1e3
        self.gamma = 1e-5*(self.lame_mu + self.lame_lambda)

        self.h = CellDiameter(self.mesh)

        self.bc = bc

        self.a_primal =  2.0*self.lame_mu  * ufl.inner(mechanics_tool.strain(u), mechanics_tool.strain(v)) * self.dxq \
            + self.lame_lambda *  ufl.inner(ufl.nabla_div(u), ufl.nabla_div(v)) * self.dxq
        self.a_primal += 0.0 * ufl.inner(u, v) * self.dsq
        #Stabilization:
        self.a_primal += avg(self.gamma) * avg(self.h)**3*ufl.inner(ufl.jump(ufl.grad(u),self.n),\
            ufl.jump(ufl.grad(v),self.n))*self.dS(0)
        
        
        self.L_primal = ufl.dot(self.shift,v) * self.ds(2)

        self.a_cut_primal = cut_form(self.a_primal)
        self.L_cut_primal = cut_form(self.L_primal)

        
        self.A_primal = assemble_matrix(self.a_cut_primal, bcs=self.bc)
        
        self.A_primal.assemble()

        self.A_primal.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
        self.A_primal.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)


        
        self.b_primal = fem.petsc.create_vector(self.L_cut_primal) #assemble_vector(cut_form(self.L_primal))
        self.b_primal.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.b_primal.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
 

        self.solver_primal = PETSc.KSP().create(self.mesh.comm)
        self.solver_primal.setOperators(self.A_primal)

        self.solver_primal.setType(PETSc.KSP.Type.PREONLY)
        pc = self.solver_primal.getPC()
        pc.setType(PETSc.PC.Type.LU)        
        pc.setFactorSolverType("mumps")
        self.solver_primal.setUp()

        if parameters.cost_func != 'compliance':
            ################
            # Dual problem
            ################

            self.p_const = parameters.p_const

            p =  ufl.TrialFunction(self.space_displacement)
            v_adj = ufl.TestFunction(self.space_displacement)
            self.a_adj = 2.0*self.lame_mu  * ufl.inner(mechanics_tool.strain(u), mechanics_tool.strain(v)) * self.dxq \
            + self.lame_lambda *  ufl.inner(ufl.nabla_div(u), ufl.nabla_div(v)) * self.dxq
            
            #Stabilization:
            self.a_adj += avg(self.gamma) * avg(self.h)**3*ufl.inner(ufl.jump(ufl.grad(u),self.n),\
            ufl.jump(ufl.grad(v),self.n))*self.dS(0)
                
            self.a_adj_test = 2.0*self.lame_mu  * ufl.inner(mechanics_tool.strain(p), mechanics_tool.strain(v)) * ufl.dx +  self.lame_lambda*ufl.inner(ufl.nabla_div(p), ufl.nabla_div(v)) * ufl.dx
            self.a_adjoint_test = fem.form(self.a_adj_test, jit_options={"cache_dir" : "ffcx-forms" })

            self.L_adj = problem_topo.dual_operator(self.uh,self.lame_mu,self.lame_lambda,parameters,self.mesh,self.dxq) #ufl.derivative(self.J,self.uh,v_adj)

            self.a_cut_adjoint = cut_form(self.a_adj)
            self.L_cut_adjoint = cut_form(self.L_adj)
            
            self.b_adjoint = assemble_vector(cut_form(self.L_adj))
            fem.apply_lifting(self.b_adjoint, [self.a_adjoint_test], [self.bc])
            self.b_adjoint.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self.b_adjoint.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            fem.petsc.set_bc(self.b_adjoint, self.bc)
            self.A_adjoint = assemble_matrix(self.a_cut_adjoint, bcs = self.bc)
            
            self.A_adjoint.assemble()
            self.A_adjoint.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
            self.A_adjoint.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)

            self.solver_adjoint = PETSc.KSP().create(self.mesh.comm)
            self.solver_adjoint.setOperators(self.A_adjoint)
            self.solver_adjoint.setType(PETSc.KSP.Type.PREONLY)
            pc =  self.solver_adjoint.getPC()
            pc.setType(PETSc.PC.Type.LU)
            pc.setFactorSolverType("mumps")
            self.solver_adjoint.setUp()
            
    def primal_problem(self, level_set):
        """Resolution of the primal problem with the CutFEM method.

        :param fem.Function level_set: The level set field wich defined implicitely the domain :math:`\Omega`.
        :param Parameters parameters: The object parameters.
        
        :returns: The primal solution.
        :rtype: fem.Function
        
        """
        self.level_set.x.array[:] = level_set.x.array
        self.update_measures_and_quadratures(level_set)
    
        # Update forms domains if needed
        if hasattr(self, "subdomain_data"):
            self.a_cut_primal.update_integration_domains(self.subdomain_data)
            self.a_cut_primal.update_runtime_domains(self.quad_domains)

        # Assemble vector
        with self.b_primal.localForm() as loc:
            loc.set(0.0)
        b_tmp = assemble_vector(self.L_cut_primal)
        self.b_primal.axpy(1.0, b_tmp)
        b_tmp.destroy()
        self.b_primal.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.b_primal.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # Assemble matrix
        if hasattr(self, "A_primal"):
            self.A_primal.destroy()
        self.A_primal = assemble_matrix(self.a_cut_primal, self.bc)
        deactivate(self.A_primal, "phi>0", self.level_set, [self.space_displacement])
        self.A_primal.assemble()

        # Solve
        self.solver_primal.setOperators(self.A_primal)
        self.solver_primal.setUp()
        self.solver_primal.solve(self.b_primal, self.uh.x.petsc_vec)
        self.level_set.x.scatter_forward()
        self.uh.x.scatter_forward()

        return self.uh
    
    def adjoint_problem(self, u, level_set, adjoint=0):
        """Resolution of the dual problem with the CutFEM method.

        :param fem.Function u: The displacement field function, :math:`u_{h}`.
        :param Parameters parameters: The object parameters.
        :param ufl.Expression adjoint: The adjoint operator if needed.
        
        :returns: The dual solution, :math:`p_{h}`.
        :rtype: fem.Function
        
        """
        # Update the level set with the provided value
        self.level_set.x.array[:] = level_set.x.array
        # Update measures and quadratures associated with the level set
        self.update_measures_and_quadratures(level_set)

        # Store the displacement field and the adjoint operator
        self.uh = u
        self.L_adj = adjoint

        # Update form domains if needed
        if hasattr(self, "subdomain_data"):
            self.a_cut_adjoint.update_integration_domains(self.subdomain_data)
            self.a_cut_adjoint.update_runtime_domains(self.quad_domains)
            self.L_cut_adjoint.update_integration_domains(self.subdomain_data)
            self.L_cut_adjoint.update_runtime_domains(self.quad_domains)

        # Assemble the adjoint matrix
        if hasattr(self, "A_adjoint"):
            self.A_adjoint.destroy()  # Destroy the old matrix if it exists
        self.A_adjoint = assemble_matrix(self.a_cut_adjoint, self.bc)  # Assemble the new matrix
        deactivate(self.A_adjoint, "phi>0", self.level_set, [self.space_displacement])  # Deactivate DOFs outside the domain
        self.A_adjoint.assemble()  # Assemble the matrix
        self.A_adjoint.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)  # Begin final assembly
        self.A_adjoint.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)  # End final assembly

        # Initialize the adjoint vector
        with self.b_adjoint.localForm() as loc:
            loc.set(0.0)  # Reset the vector to zero
        self.b_adjoint = assemble_vector(self.L_cut_adjoint)  # Assemble the vector
        fem.apply_lifting(self.b_adjoint, [self.a_adjoint_test], [self.bc])  # Apply boundary conditions
        self.b_adjoint.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # Update the vector
        self.b_adjoint.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # Update the vector
        fem.petsc.set_bc(self.b_adjoint, self.bc)  # Apply boundary conditions to the vector

        # Solve the adjoint problem
        self.solver_adjoint.setOperators(self.A_adjoint)  # Set the operator for the solver
        self.solver_adjoint.solve(self.b_adjoint, self.ph.x.petsc_vec)  # Solve the system

        self.ph.x.scatter_forward()  # Update the adjoint solution
        self.level_set.x.scatter_forward()  # Update the level set

        return self.ph  # Return the adjoint solution


    def cutfem_solver(self, level_set, parameters, problem_topo=0):
        r"""Resolution of the primal and dual problem.

        :param fem.Function level_set: The level set field which defined implicitly the domain :math:`\Omega`.
        :param Parameters parameters: The object parameters.
        :param ufl.Expression adjoint: The adjoint operator if needed.
        
        :returns: The values of the primal and dual solution.
        :rtype: fem.Function, fem.Function
        
        """
        self.level_set.x.array[:] = level_set.x.array
        self.uh = self.primal_problem(level_set)
        if parameters.cost_func == "compliance":
            self.ph.x.array[:] = self.uh.x.array
        else:
            adjoint = problem_topo.dual_operator(
                self.uh, self.lame_mu, self.lame_lambda, parameters, self.mesh, self.dxq
            )
            self.ph = self.adjoint_problem(self.uh, level_set, adjoint)
        return self.uh, self.ph


    def _assemble_matrix(self, form, bcs):
        mat = assemble_matrix(form, bcs)
        mat.assemble()
        mat.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
        mat.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)
        return mat

    def _assemble_vector(self, form, bcs=None):
        vec = assemble_vector(form)
        if bcs is not None:
            fem.petsc.set_bc(vec, bcs)
        vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        return vec
    
    def __del__(self):
        if hasattr(self, "A_primal"):
            self.A_primal.destroy()
        if hasattr(self, "A_adjoint"):
            self.A_adjoint.destroy()
        if hasattr(self, "solver_primal"):
            self.solver_primal.destroy()
        if hasattr(self, "solver_adjoint"):
            self.solver_adjoint.destroy()
        if hasattr(self, "b_primal"):
            self.b_primal.destroy()
        if hasattr(self, "b_adjoint"):
            self.b_adjoint.destroy()
    
    def get_cut_mesh(self, level_set):
        """
        Updates the cut entities and returns the cut mesh associated with the given level set.
        """
        self.update_measures_and_quadratures(level_set)
        return self.cut_mesh

    def update_measures_and_quadratures(self, level_set, order=2):
        """
        Updates measures, quadratures, and entities associated with the level set.
        """
        
        self.level_set.x.array[:] = level_set.x.array

        if level_set != 0:
            self.level_set.x.array[:] = level_set.x.array
        order = 2

        self.intersected_entities = locate_entities(self.level_set, self.dim, "phi=0")
        self.inside_entities = locate_entities(self.level_set, self.dim, "phi<0")
        
        self.inside_quadrature = runtime_quadrature(self.level_set, "phi<0", order)
        self.interface_quadrature = runtime_quadrature(self.level_set, "phi=0", order)

        self.quad_domains = {"cutcell": [(0,self.inside_quadrature)]} 
        self.subdomain_data = {"cell": [(0, self.inside_entities)]}
        

        self.gp_ids = ghost_penalty_facets(self.level_set, "phi<0")
        self.gp_topo = facet_topology(self.mesh, self.gp_ids)
        
        self.dx = ufl.Measure("dx", subdomain_data=[(0, self.inside_entities)], domain=self.mesh)
        self.dx_rt = ufl.Measure("dC", subdomain_data=self.quad_domains, domain=self.mesh)

        self.dxq = self.dx_rt(0) + self.dx(0)
        self.dS = ufl.Measure("dS", subdomain_data=[(0, self.gp_topo)], domain=self.mesh)




        