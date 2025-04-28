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

class CutFemMethod:
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

    def __init__(self, level_set, level_set_space, space_displacement,ds, bc, bc_velocity, parameters, shift):
        
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
        #Stabilization:
        self.a_primal += avg(self.gamma) * avg(self.h)**3*ufl.inner(ufl.jump(ufl.grad(u),self.n),\
            ufl.jump(ufl.grad(v),self.n))*self.dS(0)
        
        self.L_primal = ufl.dot(self.shift,v) * self.ds(2)

        self.a_cut_primal = cut_form(self.a_primal)
        self.L_cut_primal = cut_form(self.L_primal)

        self.A_primal = assemble_matrix(self.a_cut_primal, bcs=[self.bc])

        self.A_primal.assemble()
        self.A_primal.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
        self.A_primal.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)

        self.b_primal = assemble_vector(self.L_cut_primal)
        self.b_primal.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.b_primal.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


        self.solver_primal = PETSc.KSP().create(self.mesh.comm)

        self.solver_primal.setOperators(self.A_primal)
        self.solver_primal.setType(PETSc.KSP.Type.PREONLY)
        self.solver_primal.getPC().setType(PETSc.PC.Type.LU)


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

            self.J = ((mechanics_tool.von_mises(self.uh,self.lame_mu,self.lame_lambda,self.dim)/parameters.elasticity_limit)**self.p_const)*self.dxq

            self.L_adj = ufl.derivative(self.J,self.uh,v_adj)
            self.a_cut_adjoint = cut_form(self.a_adj)
            self.L_cut_adjoint = cut_form(self.L_adj)
            
            self.b_adjoint = assemble_vector(self.L_cut_adjoint)
            fem.apply_lifting(self.b_adjoint, [self.a_adjoint_test], [[self.bc]])
            self.b_adjoint.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self.b_adjoint.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            fem.petsc.set_bc(self.b_adjoint, [self.bc])
            self.A_adjoint = assemble_matrix(self.a_cut_adjoint, bcs = [self.bc])
            
            self.A_adjoint.assemble()
            self.A_adjoint.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
            self.A_adjoint.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)

            self.solver_adjoint = PETSc.KSP().create(self.mesh.comm)
            self.solver_adjoint.setOperators(self.A_adjoint)
            self.solver_adjoint.setType(PETSc.KSP.Type.PREONLY)
            self.solver_adjoint.getPC().setType(PETSc.PC.Type.LU)

        self.velocity_field = fem.Function(level_set_space)
        self.velocity_field.x.array[:] = 0 * level_set.x.array

        self.dt = parameters.dt 

        self.phi_n = ufl.TrialFunction(self.V_ls)
        self.phi_test = ufl.TestFunction(self.V_ls)
        self.const = 1e-3
        self.n_adv = FacetNormal(self.mesh)
        self.a_adv  =  ufl.dot(self.phi_n,self.phi_test) * ufl.dx 
        self.a_adv +=  avg(self.const) * avg(self.h)**3*ufl.inner(ufl.jump(ufl.grad(self.phi_n),self.n_adv),ufl.jump(ufl.grad(self.phi_test),self.n_adv))*self.dS   
            
        self.L_adv = ufl.dot(self.level_set,self.phi_test)*ufl.dx + ufl.dot(-self.dt*self.velocity_field*self.euclidean_norm_grad(self.level_set),self.phi_test)*ufl.dx
            
        self.a_cut_adv = cut_form(self.a_adv)
        self.L_cut_adv = cut_form(self.L_adv)

        self.b_adv = assemble_vector(self.L_cut_adv)
        self.b_adv.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.b_adv.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.A_adv = assemble_matrix(self.a_cut_adv)
        self.A_adv.assemble()


        self.solver_adv = PETSc.KSP().create(self.mesh.comm)
        self.solver_adv.setOperators(self.A_adv)
        self.solver_adv.setType(PETSc.KSP.Type.PREONLY)
        self.solver_adv.getPC().setType(PETSc.PC.Type.LU)

        self.sol_adv = fem.Function(self.V_ls)
        self.solver_adv.solve(self.b_adv, self.sol_adv.x.petsc_vec)
        
        
        
    def primal_problem(self,level_set,parameters):
        r"""Resolution of the primal problem with the CutFEM method.

        :param fem.Function level_set: The level set field which defined implicitly the domain :math:`\Omega`.
        :param Parameters parameters: The object parameters.
        
        :returns: The primal solution.
        :rtype: fem.Function
        
        """

        self.level_set.x.array[:] = level_set.x.array

        self.set_measure_dxq(level_set) # actualization of the measure on \Omega

        self.intersected_entities = locate_entities(self.level_set,self.dim,"phi=0")
        self.inside_entities = locate_entities(self.level_set,self.dim,"phi<0")

        self.gp_ids =  ghost_penalty_facets(self.level_set, "phi<0")
        self.gp_topo = facet_topology(self.mesh,self.gp_ids)


        self.inside_quadrature = runtime_quadrature(self.level_set,"phi<0",self.order)
        self.interface_quadrature = runtime_quadrature(self.level_set,"phi=0",self.order)

        self.subdomain_data={"cell": [(0, self.inside_entities)]} #, "interior_facet": [(0, self.gp_topo)]}
        self.quad_domains = {"cutcell": [(0,self.inside_quadrature)]} #, (1,self.interface_quadrature)]}

        self.a_cut_primal.update_integration_domains(self.subdomain_data)
        self.a_cut_primal.update_runtime_domains(self.quad_domains)

        # compute_normal(self.n,self.level_set,self.intersected_entities)
        #self.b_primal = assemble_vector(self.L_cut_primal)

        self.b_primal.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.b_primal.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        # MPI.COMM_WORLD.Barrier()

        self.A_primal = assemble_matrix(self.a_cut_primal,  [self.bc])

        deactivate(self.A_primal,"phi>0",self.level_set,[self.space_displacement])
        self.A_primal.assemble()

        self.solver_primal.setOperators(self.A_primal)
        self.solver_primal.setType(PETSc.KSP.Type.PREONLY)
        self.solver_primal.getPC().setType(PETSc.PC.Type.LU)

        self.solver_primal.solve(self.b_primal, self.uh.x.petsc_vec)
        self.level_set.x.scatter_forward() 
        
        self.uh.x.scatter_forward()
        
        return self.uh
    
    def adjoint_problem(self,u,parameters,level_set,adjoint=0):
        r"""Resolution of the dual problem with the CutFEM method.

        :param fem.Function u: The displacement field function, :math:`u_{h}`.
        :param Parameters parameters: The object parameters.
        :param ufl.Expression adjoint: The adjoint operator if needed.
        
        :returns: The dual solution, :math:`p_{h}`.
        :rtype: fem.Function
        
        """
        # temporaire
        self.set_measure_dxq(level_set)

        self.uh = u
        self.L_adj = adjoint

        self.a_cut_adjoint.update_integration_domains(self.subdomain_data)
        self.a_cut_adjoint.update_runtime_domains(self.quad_domains)
        # if adjoint ==0 
        self.L_cut_adjoint.update_integration_domains(self.subdomain_data)
        self.L_cut_adjoint.update_runtime_domains(self.quad_domains)
        
        self.set_measure_dxq(level_set)

        self.A_adjoint = assemble_matrix(self.a_cut_adjoint,  [self.bc])
        deactivate(self.A_adjoint,"phi>0",self.level_set,[self.space_displacement])
        self.A_adjoint.assemble()

        # if djoint ==0
        self.b_adjoint = assemble_vector(self.L_cut_adjoint)
        fem.apply_lifting(self.b_adjoint, [self.a_adjoint_test], [[self.bc]])
        self.b_adjoint.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.b_adjoint.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        fem.petsc.set_bc(self.b_adjoint, [self.bc])

        self.A_adjoint.assemble()
        self.A_adjoint.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
        self.A_adjoint.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)

        self.solver_adjoint.setOperators(self.A_adjoint)
        self.solver_adjoint.setType(PETSc.KSP.Type.PREONLY)
        self.solver_adjoint.getPC().setType(PETSc.PC.Type.LU)

        self.solver_adjoint.solve(self.b_adjoint, self.ph.x.petsc_vec)

        # A_adjoint.destroy()
        # b_adjoint.destroy()
        self.level_set.x.scatter_forward() 

        return self.ph
    
    def set_measure_dxq(self,level_set):
        r"""Set the measure dxq on :math:`\Omega`.

        :param fem.Function level_set: The level_set field function, :math:`\phi`.
        
        """

        if level_set != 0:
            self.level_set.x.array[:] = level_set.x.array
        order = 2

        intersected_entities = locate_entities(self.level_set,self.dim,"phi=0")
        inside_entities = locate_entities(self.level_set,self.dim,"phi<0")
        
        inside_quadrature = runtime_quadrature(self.level_set,"phi<0",order)
        interface_quadrature = runtime_quadrature(self.level_set,"phi=0",order)

        quad_domains = {"cutcell": [(0,inside_quadrature), (1,interface_quadrature)]}

        gp_ids =  ghost_penalty_facets(self.level_set, "phi<0")
        gp_topo = facet_topology(self.mesh,gp_ids)

        
        self.dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities),(2, intersected_entities)], domain=self.mesh)
        self.dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=self.mesh)
        
        self.dxq = self.dx_rt(0) + self.dx(0)
        
    def cutfem_solver(self,level_set,parameters,problem_topo=0):
        r"""Resolution of the primal and dual problem.

        :param fem.Function level_set: The level set field which defined implicitly the domain :math:`\Omega`.
        :param Parameters parameters: The object parameters.
        :param ufl.Expression adjoint: The adjoint operator if needed.
        
        :returns: The values of the primal and dual solution.
        :rtype: fem.Function, fem.Function
        
        """
        self.level_set.x.array[:] = level_set.x.array
        print("before primal solve")
        self.uh = self.primal_problem(level_set,parameters)
        print("after primal solve")
        self.ph.x.array[:] = self.uh.x.array
        adjoint = 0
        if (parameters.cost_func != "compliance"):
            #almMethod.maj_param_constrainst_optim_slack(parameters,rest_constraint)
            adjoint = problem_topo.dual_operator(self.uh,self.lame_mu,self.lame_lambda,parameters,self.mesh,self.dxq)
            
            self.ph =  self.adjoint_problem(self.uh,parameters,level_set,adjoint) #self.adjoint_problem(self.uh,parameters,level_set,adjoint)
        return self.uh, self.ph    
    
    def euclidean_norm_grad(self,func):
        r"""Calculation of the integrand of the L2-norm of the gradient of the function provided, given by the following equality:

        .. math::

            \left|\nabla\phi\right| = \sqrt{\nabla\phi\cdot\nabla\phi}


        :param fem.Function func: Function field.
        
        :returns: The values of integrand of the L2-norm of the gradient.
        :rtype: fem.Expression
        
        """
        euclidean_norm = ufl.sqrt(ufl.inner(ufl.grad(func),ufl.grad(func)))
        return euclidean_norm
        


    def cut_fem_adv(self,level_set, dt, velocity_field):
        r"""Resolution of advection equation with CutFEM stabilization.

        :param fem.Function level_set: The level set field which defined implicitly the domain :math:`\Omega`.
        :param float dt: The dt time parameters.
        :param ufl.Expression velocity_field: The value of advection velocity_field, in normal direction of the interface :math:`\partial\Omega`.
        
        :returns: The values of the advected level set.
        :rtype: fem.Function
        
        """
        self.level_set.x.array[:] = level_set.x.array
        self.dt = dt
        velocity_expr = fem.Expression(velocity_field, self.V_ls.element.interpolation_points())
        self.velocity_field.interpolate(velocity_expr)
        #self.velocity_field.x.array[:] = velocity_field.x.array
        self.intersected_entities = locate_entities(self.level_set,self.dim,"phi=0")
        
        self.gp_ids =  ghost_penalty_facets(self.level_set, "phi<0")
        self.gp_topo = facet_topology(self.mesh,self.gp_ids)

        self.inside_quadrature = runtime_quadrature(self.level_set,"phi<0",self.order)
        self.interface_quadrature = runtime_quadrature(self.level_set,"phi=0",self.order)

        #self.subdomain_data={"interior_facet": [(0, self.gp_topo)]}
        #self.quad_domains = {"cutcell": [(1,self.interface_quadrature)]}


        #self.a_cut_adv.update_integration_domains(self.subdomain_data)
        #self.a_cut_adv.update_runtime_domains(self.quad_domains)

        # compute_normal(self.n_K,self.level_set,self.intersected_entities)

        self.L_adv = ufl.dot(self.level_set,self.phi_test)*ufl.dx + ufl.dot(-self.dt*self.velocity_field*self.euclidean_norm_grad(self.level_set),self.phi_test)*ufl.dx
        self.L_cut_adv = cut_form(self.L_adv)

        self.b_adv = assemble_vector(self.L_cut_adv)
        self.b_adv.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.b_adv.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.A_adv = assemble_matrix(self.a_cut_adv)
        self.A_adv.assemble()

        self.solver_adv.setOperators(self.A_adv)
        self.solver_adv.setType(PETSc.KSP.Type.PREONLY)
        self.solver_adv.getPC().setType(PETSc.PC.Type.LU)

        self.solver_adv.solve(self.b_adv, self.level_set.x.petsc_vec)
        self.level_set.x.scatter_forward() 
        
        
        
        self.level_set.x.scatter_forward()
       
        return self.level_set
        
    def cut_fem_adv(self,level_set, dt, velocity_field):
        r"""Resolution of advection equation with CutFEM stabilization.

        :param fem.Function level_set: The level set field wich defined implicitely the domain :math:`\Omega`.
        :param float dt: The :math: `dt` time parameters.
        :param ufl.Expression velocity_field: The value of advection velocity_field, in normal direction of the interface :math:`\partial\Omega`.
        
        :returns: The values of the advected level set.
        :rtype: fem.Function
        
        """

        tdim = self.mesh.topology.dim
        dim = self.mesh.topology.dim

        intersected_entities = locate_entities(level_set,dim,"phi=0")
        inside_entities = locate_entities(level_set,dim,"phi<0")

        V_DG = fem.functionspace(self.mesh, ("DG", 0, (self.mesh.geometry.dim,)))
        n_K = fem.Function(V_DG)
        compute_normal(n_K,level_set,intersected_entities)


        phi_n = ufl.TrialFunction(self.V_ls)
        phi_test = ufl.TestFunction(self.V_ls)

        # Compute n_K on local cells and ghost layer (all cells)
        const = 1e-3
        h = CellDiameter(self.mesh)
        n_adv = FacetNormal(self.mesh)
        a_adv  =  ufl.dot(phi_n,phi_test) * ufl.dx 
        a_adv +=  avg(const) * avg(h)**3*ufl.inner(ufl.jump(ufl.grad(phi_n),n_adv),ufl.jump(ufl.grad(phi_test),n_adv))*self.dS   
            
        L_adv = ufl.dot(level_set,phi_test)*ufl.dx + ufl.dot(-dt*velocity_field*self.euclidean_norm_grad(level_set),phi_test)*ufl.dx
            
        a_cut_adv = cut_form(a_adv)
        L_cut_adv = cut_form(L_adv)

        b_adv = assemble_vector(L_cut_adv)
        b_adv.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b_adv.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        A_adv = assemble_matrix(a_cut_adv)
        A_adv.assemble()


        solver_adv = PETSc.KSP().create(self.mesh.comm)
        solver_adv.setOperators(A_adv)
        solver_adv.setType(PETSc.KSP.Type.PREONLY)
        solver_adv.getPC().setType(PETSc.PC.Type.LU)

        sol_adv = fem.Function(self.V_ls)
        solver_adv.solve(b_adv, sol_adv.x.petsc_vec)
        b_adv.destroy()
        return sol_adv
    
    def velocity_normalization(self,v,c):
        r"""Normalization of the Velocity field according to the following equation:

         .. math::

            \overline{v} = \frac{v}{\sqrt{c\left\Vert \nabla\phi\right\Vert _{L^{2}\left(D\right)}^{2}+\left\Vert \phi\right\Vert _{L^{2}\left(D\right)}^{2}}}
        
            
        with :math:`c>0` and :math:`\left\Vert . \right\Vert _{L^{2}\left(D\right)}` norm defined as: 
        
         .. math::

            \left\Vert f \right\Vert _{L^{2}\left(D\right)}^{2} = \int_{D} f \cdot f \text{ }dx.

        :param fem.Expression or fem.Function v: The scalar velocity field which defined the value of advection in direction of the normal to :math:`\partial\Omega`.
        :param float c: Value of the smoothing for the velocity normalization. Topically, this value is equal to the smoothing value in the extension equation.
        
        :returns: The normalized velocity field, defined in `D`.
        :rtype: fem.Expression
        
        """
        b_grad = fem.form(inner(ufl.grad(v),ufl.grad(v))*dx)
        b_v = fem.form(inner(v,v)*dx)
        denom = MPI.COMM_WORLD.allreduce((fem.assemble_scalar(b_grad)*c+fem.assemble_scalar(b_v)),op=MPI.SUM)
        denom_temp = MPI.COMM_WORLD.allreduce((fem.assemble_scalar(b_v)),op=MPI.SUM)
        res = v / ufl.sqrt(denom)
        return res
    
    def descent_direction(self,level_set,parameters,rest_constraint,constraint_integrand,cost_integrand,xsi=0):
        r"""Determine the descent direction by solving the following equation:

        Find :math:`v'_{\text{reg}}\in H_{\Gamma_{D}}^{1}=\left\{ v\in H^{1}\left(D\right)\text{ such that }v=0\text{ on }\Gamma_{D}\right\}` such that :math:`\forall w\in H_{\Gamma_{D}}^{1}`
        
        .. math::

                \alpha\left(\nabla v'_{\text{reg}},\nabla w\right)_{L^{2}\left(D\right)}+\left(v'_{\text{reg}},w\right)_{L^{2}\left(D\right)}=-J'(\Omega)\left(w\right)
       
        with :math:`J` the cost function and :math:`\alpha>0` is a smoothing parameter instantiated in the Parameter class. 
            
        :param fem.Function level_set: The level set field which defined implicitly the domain :math:`\Omega`.
        :param Parameters parameters: The object parameters.
        :param float rest_constraint: The value of the constraint function :math:`C(\Omega)`.
        :param fem.Expression constraint_integrand: The integrand of the constraint function.
        :param fem.Expression cost_integrand: The integrand of the cost function.

        :returns: The velocity field, defined in `D`.
        :rtype: fem.Function
        
        """

        u_r = ufl.TrialFunction(self.V_ls)
        v_r = ufl.TestFunction(self.V_ls)
        v_reg = fem.Function(self.V_ls)

        intersected_entities = locate_entities(level_set,self.dim,"phi=0")
        
        V_DG = fem.functionspace(self.mesh, ("DG", 0, (self.mesh.geometry.dim,)))
        n_K = fem.Function(V_DG)
        compute_normal(n_K,level_set,intersected_entities)

        order = 2
        inside_quadrature = runtime_quadrature(level_set,"phi<0",order)
        interface_quadrature = runtime_quadrature(level_set,"phi=0",order)

        quad_domains = [(0,inside_quadrature), (1,interface_quadrature)]

        dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=self.mesh)

        dsq = dx_rt(1)

        a_reg  =   parameters.alpha_reg_velocity *ufl.inner(grad(u_r), grad(v_r))*dx
        a_reg += u_r*v_r*dx

        C_Omega_value = (rest_constraint + parameters.ALM_slack_variable )


        temp = cost_integrand
        temp_ALM = parameters.ALM*(parameters.ALM_lagrangian_multiplicator * constraint_integrand + parameters.ALM_penalty_parameter * C_Omega_value * constraint_integrand \
                + 2 * constraint_integrand*parameters.ALM_slack_variable)
        temp_ALM += (1-parameters.ALM)*parameters.target_constraint
        temp += temp_ALM

        L_reg = -(inner(temp*v_r*n_K,n_K)*dsq)
        
        a_cut_reg = cut_form(a_reg, jit_options={"cache_dir" : "ffcx-forms" })
        L_cut_reg = cut_form(L_reg)

        b_reg = assemble_vector(L_cut_reg)
        b_reg.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b_reg.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        A_reg = assemble_matrix(a_cut_reg, bcs = [self.bc_velocity])
        A_reg.assemble()

        solver_reg = PETSc.KSP().create(self.mesh.comm)
        solver_reg.setOperators(A_reg)
        solver_reg.setType(PETSc.KSP.Type.PREONLY)
        solver_reg.getPC().setType(PETSc.PC.Type.LU)
        solver_reg.solve(b_reg, v_reg.x.petsc_vec)
        b_reg.destroy()

        return v_reg

    

