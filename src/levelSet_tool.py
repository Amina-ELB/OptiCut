# Copyright (c) 2025 ONERA and MINES Paris, France 
#
# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari 

# The modules that will be used are imported:

# +
import numpy as np

import ufl
# mathematical language for FEM, auto differentiation, python
from dolfinx import fem, io, mesh, plot
import matplotlib.pyplot as plt
# meshes, assembly, c++, ython, pybind
from ufl import ds, dx, grad, inner, tr, dS

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc

from typing import TYPE_CHECKING
import pyvista


from dolfinx.fem import (Constant,  Function, FunctionSpace, assemble_scalar, 
                         dirichletbc, form, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square, locate_entities, meshtags
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import (FacetNormal, Measure, SpatialCoordinate, TestFunction, TrialFunction, 
                 div, dot, dx, grad, inner, lhs, rhs, dc, FacetNormal, CellDiameter, dot, avg, jump)
from dolfinx.io import XDMFFile
#  from dolfinx.plot import create_vtk_mesh # error  


from cutfemx.level_set import locate_entities, cut_entities, ghost_penalty_facets, facet_topology
from cutfemx.level_set import compute_normal
from cutfemx.mesh import create_cut_mesh, create_cut_cells_mesh
from cutfemx.quadrature import runtime_quadrature, physical_points
from cutfemx.fem import cut_form, cut_function

from cutfemx.petsc import assemble_vector, assemble_matrix, deactivate, locate_dofs


import mechanics_tool

############################################
# HJ Reinitialization
############################################

class LevelSet:
    def __init__(self, level_set, space):
        """
        LevelSet class to define a level-set function.

        Parameters
        ----------
        level_set : fem.Function
            Initial level-set function.
        space : fem.FunctionSpace
            Function space of the level-set.
        dt : float, optional
            Time step for advection.
        """
        self.level_set = fem.Function(space)
        self.level_set.x.array[:] = level_set.x.array
        self.space = space

    def set_level_set(self, level_set):
        """Update the level-set function."""
        self.level_set.x.array[:] = level_set.x.array


class Advection(LevelSet):
    """
    Class to perform level-set advection using CutFEM.
    Inherits from LevelSet.
    """

    def __init__(self, level_set, V_ls, dt=1e-3):
        """
        Initialize the Advection solver for a level-set.

        Parameters
        ----------
        level_set : fem.Function
            Initial level-set function.
        V_ls : fem.FunctionSpace
            Function space of the level-set.
        dt : float, optional
            Time step for advection (default: 0.0).
        l : float, optional
            Stabilization parameter (default: 1.0).
        """
        # Initialise LevelSet
        super().__init__(level_set, V_ls)

        self.dt = dt
        self.V_ls = V_ls
        self.mesh = V_ls.mesh

        # Trial/test functions
        self.phi_n = ufl.TrialFunction(self.V_ls)
        self.phi_test = ufl.TestFunction(self.V_ls)

        # Velocity field initialized to zero
        self.velocity_field = fem.Function(self.V_ls)
        self.velocity_field.x.array[:] = 1e-3

        # Mesh-dependent quantities
        self.h = ufl.CellDiameter(self.mesh)
        self.dS = ufl.dS
        self.n = ufl.FacetNormal(self.mesh)
        self.const = 1e-3

        # Assemble matrix and RHS
        self._assemble_matrix()
        self._assemble_rhs()

        # Solver
        self.solver_adv = PETSc.KSP().create(self.mesh.comm)
        self.solver_adv.setOperators(self.A_adv)
        self.solver_adv.setType(PETSc.KSP.Type.PREONLY)
        pc = self.solver_adv.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
        self.solver_adv.setUp()

        self.sol_adv = fem.Function(self.V_ls)

    def _assemble_matrix(self):
        """Assemble the advection CutFEM matrix."""
        a_adv = ufl.dot(self.phi_n, self.phi_test) * ufl.dx
        a_adv += avg(self.const) * avg(self.h)**3 * ufl.inner(
            ufl.jump(ufl.grad(self.phi_n), self.n),
            ufl.jump(ufl.grad(self.phi_test), self.n)
        ) * self.dS

        a_cut_adv = cut_form(a_adv)
        if hasattr(self, "A_adv") and self.A_adv is not None:
            self.A_adv.destroy()
        self.A_adv = assemble_matrix(a_cut_adv)
        self.A_adv.assemble()

    def _assemble_rhs(self):
        """Assemble the RHS vector for CutFEM advection."""
        L_adv = ufl.dot(self.level_set, self.phi_test) * ufl.dx + \
                ufl.dot(-self.dt * self.velocity_field * ufl.sqrt(ufl.inner(ufl.grad(self.level_set), ufl.grad(self.level_set))), self.phi_test) * ufl.dx

        L_cut_adv = cut_form(L_adv)
        if hasattr(self, "b_adv") and self.b_adv is not None:
            with self.b_adv.localForm() as loc:
                loc.set(0.0)
        self.b_adv = assemble_vector(L_cut_adv)
        self.b_adv.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.b_adv.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def cut_fem_adv(self, velocity_field, dt):
        r"""Perform one advection step of the level-set function.

        :param fem.Function level_set: The level set field wich defined implicitely the domain :math:`\Omega`.
        :param float dt: The :math: `dt` time parameters.
        :param ufl.Expression velocity_field: The value of advection velocity_field, in normal direction of the interface :math:`\partial\Omega`.
        
        :returns: The values of the advected level set.
        :rtype: fem.Function
        
        """
        velocity_expr = fem.Expression(velocity_field, self.V_ls.element.interpolation_points())
        self.velocity_field.interpolate(velocity_expr)

        self.dt = dt

        self._assemble_matrix()
        self._assemble_rhs()

        # Solve system
        self.solver_adv.solve(self.b_adv, self.sol_adv.x.petsc_vec)
        self.sol_adv.x.scatter_forward()

        # Update level-set in-place
        self.level_set.x.array[:] = self.sol_adv.x.array
        self.level_set.x.scatter_forward()

        return self.sol_adv

        
class Reinitialization(LevelSet):
    r"""This is the Reinitialization class.

    In the subsection we give some details about the Reinitialization method used. 
    We use the Prediction Correction scheme proposed in .....
    Initialization of predictor problem.

    Definition of the Predictor variational problem
    =================================================

    prediction problem is given by: 
    Find :math:`\phi_{p}:D \rightarrow \mathbb{R}`

    .. math::

            \begin{cases}
            -\Delta\phi_{p}  & \!\!\!\!
            = \psi(x)\text{ in D}\\
            \phi_{p}  & \!\!\!\!
            = 0\text{ on }\Gamma \\ 
            \nabla \phi_{p} \cdot n & \!\!\!\!
            = \psi(x) \text{ on } \partial D .
            \end{cases}


    With Nitsche method, this yields to the following weak formulation: 
    Find :math:`\phi^{0} \in V`, such that for all :math:`v \in V` 

    .. math::

        a\left(\phi_{p},v\right)=l\left(v\right),
        
    where:

    .. math::
    
        \begin{align}
        a\left(\phi_{p},v\right) &= \int_{D}\nabla \phi_{p}\cdot\nabla v\,\text{ }dx{\color{red}{ -\int_{\Gamma}\nabla \phi_{p}\cdot n_{\Gamma} \, v\text{ }ds-\int_{\Gamma}\nabla v\cdot n_{\Gamma} \, \phi_{p}\text{ }ds}} \\
            &{\color{red}{ +\gamma_{D}\int_{\Gamma} \phi_{p} \, v\text{ }ds}}\\
        l\left(v\right) &= \int_{D}\psi(x) v\text{ }dx + \int_{\partial D}\psi(x) v\text{ }ds
        \end{align}

    with :math:`\gamma_{D}>0` is the Nistche parameter.

    *The text in red is the Nitsche terms.*

    Intialization of normal field to isocontour:
    ---------------------------------------------
    .. code-block:: python

        V_DG = fem.functionspace(self.mesh, ("DG", 0, (self.dim,)))
        self.n_K = fem.Function(V_DG)
        self.norm_euclidienne = ufl.sqrt(inner(ufl.grad(self.level_set),ufl.grad(self.level_set)))
        self.n_K = ufl.grad(self.level_set)/ self.norm_euclidienne


    Bilinear form (Predictor):
    --------------------------
    .. code-block:: python

        import ufl

        u_r = ufl.TrialFunction(self.V_ls) # Trial function
        v_r = ufl.TestFunction(self.V_ls) # Test function

        self.gamma_r = 1e+4 # Value of Nitsche parameter

        self.h_r = CellDiameter(self.mesh) # mesh size

        self.a_predict  = ufl.inner(grad(u_r), grad(v_r))*self.dx
        self.a_predict += - dot(grad(u_r), self.n_K)*v_r*self.dsq
        self.a_predict += - dot(grad(v_r), self.n_K)*u_r*self.dsq
        self.a_predict += self.gamma_r*1.0/self.h_r*u_r*v_r*self.dsq

    Linear form (Predictor):
    ------------------------------
    Approximation of the signed indicator function :math:`\psi`:
        
    .. code-block:: python

        self.eps = 1e-6
        self.sign = self.level_set / (ufl.sqrt(self.level_set**2+self.eps**2))


    .. code-block:: python

        self.L_predict = inner(self.l**2*self.sign,v_r)*self.dx
        nuemann_bc = ufl.conditional(ufl.le(self.sign,0),-1,0)
        nuemann_bc = ufl.conditional(ufl.ge(self.sign,0),1,nuemann_bc)
        self.L_predict += ufl.dot(nuemann_bc,v_r)*ufl.ds

    Be aware to impose correct Nuemann condition to guaranty the order of convergence.


    Definition of the Corrector variational problem
    =================================================

    prediction problem is given by: 
    Find :math:`\phi:D \rightarrow \mathbb{R}`

    .. math::
            
            \begin{cases}
            \nabla\cdot\left(\nabla\phi-\frac{\nabla\phi}{\left|\nabla\phi\right|}\right)& \!\!\!\!
            =0\text{ in }D\\
            \phi&\!\!\!\!
            =0\text{ on }\Gamma\\
            \left(\nabla\phi-\frac{\nabla\phi}{\left|\nabla\phi\right|}\right)\cdot n&
            \!\!\!\! =0\text{ on }\partial D.
            \end{cases}
            

            

    With Nitsche method, this yields to the following weak formulation: 
    Find :math:`\phi^{n+1} \in V`, such that for all :math:`v \in V`:

    .. math::  
        a\left(\phi^{n+1},v\right)=l\left(v,\phi^{n}\right),
        
    where :

    .. math::

        \begin{align}
        a\left(\phi^{n+1},v\right)&=\int_{D}\nabla\phi^{n+1}\cdot\nabla v\text{ }dx -\int_{\Gamma}\nabla\phi^{n+1}\cdot n_{\Gamma}v\text{ }ds-\int_{\Gamma}\nabla v\cdot n_{\Gamma}\phi^{n+1}\text{ }ds \notag \\ 
        &+\gamma_{D}\int_{\Gamma} \phi^{n+1}v\text{ }ds\label{corrector_bilin}\\
        l\left(v,\phi^{n}\right)&=\int_{D}\frac{\nabla\phi^{n}}{\max\left(\left|\nabla\phi^{n}\right|,\epsilon\right)}\cdot\nabla v\text{ }dx \quad\text{ with } \epsilon >0 \text{, very small} \label{corrector_lin}
        \end{align}

        
    with :math:`\phi^0=\phi_{p}` given by the Predictor problem and :math:`\gamma_{D}>0` is the Nistche parameter.

        
    
    Bilinear form (Corrector):
    -----------------------------
    .. code-block:: python

        import ufl

        u_r = ufl.TrialFunction(self.V_ls) # Trial function
        v_r = ufl.TestFunction(self.V_ls) # Test function

        self.gamma_r = 1e+4 # Value of Nitsche parameter

        self.h_r = CellDiameter(self.mesh) # mesh size

        self.a_correct  = ufl.inner(grad(u_r), grad(v_r))*self.dx
        self.a_correct += - dot(grad(u_r), self.n_K)*v_r*self.dsq
        self.a_correct += - dot(grad(v_r), self.n_K)*u_r*self.dsq
        self.a_correct += self.gamma_r*1.0/self.h_r*u_r*v_r*self.dsq

    Linear form (Corrector):
    -------------------------

    .. code-block:: python

        self.L_correct = inner(self.n_K, grad(v_r))*self.dx

    The approximation of the normal to isocontour is automatically updated for each iteration.

    """

    def __init__(self, level_set, V_ls, l):

        super().__init__(level_set,V_ls)  

        self.mesh = level_set.function_space.mesh
        self.l = l
        self.V_ls = V_ls
        
        ### Reinitialisation schema predicteur correcteur
        ## Predictor :
        
        self.dim = self.mesh.topology.dim
        self.tdim = self.dim
        
        self.intersected_entities = locate_entities(self.level_set,self.dim,"phi=0")
        self.inside_entities = locate_entities(self.level_set,self.dim,"phi<0")

        V_DG = fem.functionspace(self.mesh, ("DG", 0, (self.dim,)))

        self.n_K = fem.Function(V_DG)

        self.norm_euclidienne = ufl.sqrt(inner(ufl.grad(self.level_set),ufl.grad(self.level_set)))
        
        self.n_K = ufl.grad(self.level_set)/ self.norm_euclidienne
        self.dof_coordinates = self.V_ls.tabulate_dof_coordinates()
        
        self.order = 1
        self.inside_quadrature = runtime_quadrature(self.level_set,"phi<0",self.order)
        self.interface_quadrature = runtime_quadrature(self.level_set,"phi=0",self.order)

        self.quad_domains = [(0,self.inside_quadrature), (1,self.interface_quadrature)]

        self.dx = ufl.Measure("dx", subdomain_data=[(0, self.inside_entities),(2, self.intersected_entities)], domain=self.mesh)
        self.dx_rt = ufl.Measure("dC", subdomain_data=self.quad_domains, domain=self.mesh)
        
        self.dsq = self.dx_rt(1)
        
        u_r = ufl.TrialFunction(self.V_ls)
        v_r = ufl.TestFunction(self.V_ls)
        self.gamma_r = 1e+4

        self.n_r = FacetNormal(self.mesh)
        self.h_r = CellDiameter(self.mesh)

        self.a_predict  = ufl.inner(grad(u_r), grad(v_r))*self.dx
        self.a_predict += - dot(grad(u_r), self.n_K)*v_r*self.dsq
        self.a_predict += - dot(grad(v_r), self.n_K)*u_r*self.dsq
        self.a_predict += self.gamma_r*1.0/self.h_r*u_r*v_r*self.dsq


        self.eps = 1e-6
        self.sign = self.level_set / (ufl.sqrt(self.level_set**2+self.eps**2))
        self.L_predict = inner(self.l**2*self.sign,v_r)*self.dx
        self.L_predict += ufl.dot(self.sign,v_r)*ds

        self.a_cut_predict = cut_form(self.a_predict,jit_options={"cache_dir" : "ffcx-forms" })
        self.L_cut_predict = cut_form(self.L_predict)
        
        
        self.A_predict = assemble_matrix(self.a_cut_predict)
        self.A_predict.assemble()

        self.b_predict = assemble_vector(self.L_cut_predict)
        self.b_predict.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.b_predict.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        

        self.solver_predict = PETSc.KSP().create(self.mesh.comm)
        self.solver_predict.setOperators(self.A_predict)
        self.solver_predict.setType(PETSc.KSP.Type.PREONLY)
        self.solver_predict.getPC().setType(PETSc.PC.Type.LU)
        self.solver_predict.getPC().setFactorSolverType("mumps")

        self.a_correct  = ufl.inner(grad(u_r), grad(v_r))*self.dx
        self.a_correct += - dot(grad(u_r), self.n_K)*v_r*self.dsq
        self.a_correct += - dot(grad(v_r), self.n_K)*u_r*self.dsq
        self.a_correct += self.gamma_r*1.0/self.h_r*u_r*v_r*self.dsq

        self.L_correct = inner(self.n_K, grad(v_r))*self.dx

        self.a_cut_correct = cut_form(self.a_correct, jit_options={"cache_dir" : "ffcx-forms" })
        self.L_cut_correct = cut_form(self.L_correct,jit_options={"cache_dir" : "ffcx-forms" })


        self.A_correct = assemble_matrix(self.a_cut_correct)
        self.A_correct.assemble()

        self.b_correct = assemble_vector(self.L_cut_correct)
        self.b_correct.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.b_correct.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


        self.solver_corrector = PETSc.KSP().create(self.mesh.comm)
        self.solver_corrector.setOperators(self.A_correct)
        self.solver_corrector.setType(PETSc.KSP.Type.PREONLY)
        self.solver_corrector.getPC().setType(PETSc.PC.Type.LU)
        self.solver_corrector.getPC().setFactorSolverType("mumps")
        

    def predictor(self, level_set):
        r""" Returns the solution of the prediction problem, denoted :math:`\phi_{p}`.

        :param fem.Expression level_set: The level_set function :math:`\phi`.

        :returns: The solution to prediction problem.
        :rtype: fem.Expression

        """

        v_r = ufl.TestFunction(self.V_ls)
        self.level_set.x.array[:] = level_set.x.array
        self.level_set.x.scatter_forward()
        self.n_K = ufl.grad(self.level_set)/ self.norm_euclidienne

        self.intersected_entities = locate_entities(self.level_set,self.dim,"phi=0")
        self.inside_entities = locate_entities(self.level_set,self.dim,"phi<0")
        self.interface_quadrature = runtime_quadrature(self.level_set,"phi=0",self.order)


        self.quad_domains = {"cutcell": [(1,self.interface_quadrature)]}

        self.a_cut_predict.update_runtime_domains(self.quad_domains)

        self.L_predict = inner(self.l**2*self.sign,v_r)*self.dx
        self.L_predict += ufl.dot(self.sign,v_r)*ds
        self.L_cut_predict = cut_form(self.L_predict)

        with self.b_predict.localForm() as loc:
            loc.set(0.0)

        b_tmp = assemble_vector(self.L_cut_predict)
        b_tmp.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b_tmp.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        b_tmp.copy(self.b_predict)
        b_tmp.destroy()

        if hasattr(self,"A_predict"):
            self.A_predict.destroy()
        self.A_predict = assemble_matrix(self.a_cut_predict)
        self.A_predict.assemble()

        self.solver_predict.setOperators(self.A_predict)

        self.solver_predict.solve(self.b_predict, self.level_set.x.petsc_vec)
        self.level_set.x.scatter_forward() 
        

        return self.level_set


    def corrector(self,level_set):
        r""" Returns the solution of the correction problem, denoted :math:`\phi_{i}`.

        :param fem.Expression level_set: The level_set function :math:`\phi`.

        :returns: The solution to correction problem.
        :rtype: fem.Expression

        """
        self.level_set.x.array[:] = level_set.x.array
        self.n_K = ufl.grad(self.level_set)/ self.norm_euclidienne
        
        interface_quadrature = runtime_quadrature(self.level_set,"phi=0",self.order)

        quad_domains = {"cutcell": [ (1,interface_quadrature)]}

        self.a_cut_correct.update_runtime_domains(quad_domains)

        with self.b_correct.localForm() as loc:
            loc.set(0.0)

        b_tmp = assemble_vector(self.L_cut_correct)
        b_tmp.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b_tmp.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        b_tmp.copy(self.b_correct)
        b_tmp.destroy()

        if hasattr(self,"A_correct"):
            self.A_correct.destroy()
        self.A_correct = assemble_matrix(self.a_cut_correct)
        self.A_correct.assemble()

        self.solver_corrector.setOperators(self.A_correct)

        self.solver_corrector.solve(self.b_correct, self.level_set.x.petsc_vec)
        self.level_set.x.scatter_forward() 
        return self.level_set
    
    def reinitializationPC(self,level_set,step_reinit):
        r""" Returns the solution of the PC reinitialization method for step_reinit iteration of the correction problem.
        
        :param fem.Expression level_set: The level_set function :math:`\phi`.
        :param int step_reinit: The number of iteration for correction problem.

        :returns: The solution to P.C. reinitialization problem.
        :rtype: fem.Expression

        """
        level_set = self.predictor(level_set)
        num_step = 0

        while (num_step < step_reinit):
            num_step += 1                    
            level_set = self.corrector(level_set)
        return level_set
