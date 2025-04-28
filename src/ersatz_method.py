# Copyright (c) 2025 ONERA and MINES Paris, France 
#
# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari 


import ufl
import cutfemx
from dolfinx import fem
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from ufl import (FacetNormal, Measure, SpatialCoordinate, TestFunction, TrialFunction, 
                 div, dot, dx, grad, inner, lhs, rhs, dc, dS, FacetNormal, CellDiameter, dot, avg, jump)
import mechanics_tool

#from mecanics_problem import *
from dolfinx.mesh import locate_entities, meshtags, locate_entities_boundary
from cutfemx.level_set import locate_entities, cut_entities, ghost_penalty_facets, facet_topology
from cutfemx.level_set import compute_normal
from cutfemx.mesh import create_cut_mesh, create_cut_cells_mesh
from cutfemx.quadrature import runtime_quadrature, physical_points
from cutfemx.fem import cut_form, cut_function

from cutfemx.petsc import assemble_vector, assemble_matrix, deactivate, locate_dofs
import os

from mechanics_tool import *

class ErsatzMethod:
   
    r"""  This is the Ersatz class.

    Some details about the initialization of linear elasticity problem with Ersatz method.

    Definition of the linear elasticity PDE 
    =================================================

    Linear elasticity problem is given by: 
    Find :math:`u:\Omega \rightarrow \mathbb{R}^{d}`

    .. math::

            \begin{align}
            \begin{cases}
                -\text{div} \sigma(u) & \!\!\!\!=0 \text{ in }\Omega\\
                u& \!\!\!\!=0\text{ on }\Gamma_{D}\\
                \sigma(u)\cdot n & \!\!\!\!=g\text{ on }\Gamma_{N}
            \end{cases}
            \end{align}


    Where :math:`d` the dimension of the problem.
    *We assume small deformations and zero volumetric forces.* 
        


    The fictitious material method uses a soft material in the domain :math:`D \ \overline{\Omega}`.
    Thus, we denote the adapted Lamé coefficients as follows: :math:`\widetilde{\lambda}` and :math:`\widetilde{\mu}`. 
    For more details on this method, we refer you to the explanation page :doc:`demo_cutfem` in section :ref:`demoErsatz`.

        
    The following weak formulation: 
    Find :math:`u \in V`, such that for all :math:`v \in V` we have

    .. math::

            a\left(u,v\right)=l\left(v\right)


    with :
    
    .. math::

            \begin{align} 
            a\left(u,v\right)&=2\widetilde{\mu}\left(\varepsilon(u),\varepsilon\left(v\right)\right)_{L^{2}(\Omega)}+\widetilde{\lambda}\left(\nabla\cdot u,\nabla\cdot v\right)_{L^{2}(\Omega)}\\
            l\left(v\right)&=\left(g,v\right)_{L^{2}\left(\Gamma_{N}\right)},
            \end{align}

            
    Bilinear form:
    --------------
    .. code-block:: python

        import ufl

        self.u = ufl.TrialFunction(self.space_displacement)   
        self.v = ufl.TestFunction(self.space_displacement)  

        self.uh = fem.Function(self.space_displacement)
        self.ph = fem.Function(self.space_displacement)


        self.lame_mu_fic = self.compute_lame_mu(self.xsi)
        self.lame_lambda_fic = self.compute_lame_lambda(self.xsi)
        
        self.a = 2.0*self.lame_mu_fic*ufl.inner(mechanics_tool.strain(self.u), mechanics_tool.strain(self.v)) * ufl.dx +\
            self.lame_lambda_fic * ufl.inner(ufl.nabla_div(self.u), ufl.nabla_div(self.v)) * ufl.dx
    

    Linear form:
    --------------
        
    .. code-block:: python

        self.L = ufl.dot(self.shift,self.v) * self.ds(2)


    """


    def __init__(self,level_set,V_ls, space_displacement, ds, bc,bc_velocity, parameters,shift):

        self.V_ls = V_ls

        self.lame_mu, self.lame_lambda = mechanics_tool.lame_compute(parameters.young_modulus,parameters.poisson)
        self.mesh = level_set.function_space.mesh
        self.dim = self.mesh.topology.dim

        self.t_paraview = 1
        
        self.space_displacement = space_displacement
        self.xsi = 0.5*(1-level_set/ufl.sqrt(level_set**2+parameters.eta**2))

        self.norm_vm = 1
        self.eta = parameters.eta
        
        self.ds = ds
        self.bc = bc
        self.bc_velocity = bc_velocity

        self.shift = shift
        


        self.u = ufl.TrialFunction(self.space_displacement)   
        self.v = ufl.TestFunction(self.space_displacement)  

        self.uh = fem.Function(self.space_displacement)
        self.ph = fem.Function(self.space_displacement)


        self.lame_mu_fic = self.compute_lame_mu(self.xsi)
        self.lame_lambda_fic = self.compute_lame_lambda(self.xsi)
        
        ## Primal problem
        self.a = 2.0*self.lame_mu_fic*ufl.inner(mechanics_tool.strain(self.u), mechanics_tool.strain(self.v)) * ufl.dx +\
            self.lame_lambda_fic * ufl.inner(ufl.nabla_div(self.u), ufl.nabla_div(self.v)) * ufl.dx
        self.L = ufl.dot(self.shift,self.v) * self.ds(2)

        self.a_primal = fem.form(self.a)
        self.L_primal = fem.form(self.L)
        
        self.b_primal = fem.petsc.create_vector(self.L_primal)
        
        self.A_primal = fem.petsc.assemble_matrix(self.a_primal, bcs=[self.bc])
        self.A_primal.assemble()

        self.primal_solver = PETSc.KSP().create(self.mesh.comm)
        self.primal_solver.setOperators(self.A_primal)
        self.primal_solver.setType(PETSc.KSP.Type.PREONLY)
        self.primal_solver.getPC().setType(PETSc.PC.Type.LU)
        
        ## Dual problem
        p =  ufl.TrialFunction(self.space_displacement)

        self.J_cost = (1/self.norm_vm)*((mechanics_tool.von_mises(self.uh,self.lame_mu_fic,self.lame_lambda_fic,self.dim)/parameters.elasticity_limit)**parameters.p_const)*ufl.dx
        self.L_adj = ufl.derivative(self.J_cost,self.uh,self.v)
        
        self.a_adjoint = fem.form(self.a)
        self.L_adjoint = fem.form(self.L_adj)

        self.A_adjoint = fem.petsc.assemble_matrix(self.a_adjoint, bcs=[self.bc])
        self.A_adjoint.assemble()
        
        self.adjoint_solver = PETSc.KSP().create(self.mesh.comm)
        self.adjoint_solver.setOperators(self.A_adjoint)
        self.adjoint_solver.setType(PETSc.KSP.Type.PREONLY)
        self.adjoint_solver.getPC().setType(PETSc.PC.Type.LU)

    def heaviside(self,level_set):
        r""" Returns the smoothing function  :math:`\chi`.

        :param fem.Expression level_set: The level_set function :math:`\phi`.

        :returns: The fictitious :math:`\mu` Lamé coefficient update with the smoothing function.
        :rtype: fem.Expression

        """
        xsi = 0.5*(1-level_set/ufl.sqrt(level_set**2+self.eta**2))
        return xsi
        
    def compute_lame_mu(self, xsi): 
        r""" Returns the Lamé coefficient :math:`\mu` according to the smoothing xsi.

        :param fem.Expression xsi: The heaviside function :math:`\chi`.

        :returns: The fictitious :math:`\mu` Lamé coefficient update with the smoothing function.
        :rtype: fem.Expression

        """
        Q = fem.functionspace(self.mesh, ("DG",0))
        epsilon = 0.001
        vect_ones = xsi/xsi
        res = (vect_ones-xsi)*epsilon* self.lame_mu + xsi * self.lame_mu
        lame_mu_expr = fem.Expression(res, Q.element.interpolation_points())
        lame_mu_fic = fem.Function(Q)
        lame_mu_fic.interpolate(lame_mu_expr)
        return res
        
    def compute_lame_lambda(self, xsi):
        r""" Returns the Lamé coefficient  :math:`\lambda`  according to the smoothing xsi.

        :param fem.Expression xsi: The heaviside function :math:`\chi`.

        :returns: The fictitious :math:`\lambda` Lamé coefficient update with the smoothing function.
        :rtype: fem.Expression
        
        """
        
        Q = fem.functionspace(self.mesh, ("DG",0))

        epsilon = 0.001 
        vect_ones = xsi/xsi
        res = (vect_ones-xsi) * epsilon * self.lame_lambda + xsi * self.lame_lambda	
        lame_mu_expr = fem.Expression(res, Q.element.interpolation_points())
        lame_lambda_fic = fem.Function(Q)
        lame_lambda_fic.interpolate(lame_mu_expr)
        return res
        
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
    
    def advection(self,level_set,dt, velocity_field):
        """Resolution of advection equation without any stabilization.

        :param fem.Function level_set: The level set field which defined implicitly the domain :math:`\Omega`.
        :param float dt: The :math: `dt` time parameters.
        :param ufl.Expression velocity_field: The value of advection velocity_field, in normal direction of the interface :math:`\partial\Omega`.
        
        :returns: The values of the advected level set.
        :rtype: fem.Function
        
        """

        phi_n = ufl.TrialFunction(self.V_ls)
        phi_test = ufl.TestFunction(self.V_ls)

        a_adv  =  ufl.dot(phi_n,phi_test) * ufl.dx 
       
            
        L_adv = ufl.dot(level_set,phi_test)*ufl.dx + ufl.dot(-dt*velocity_field*self.euclidean_norm_grad(level_set),phi_test)*ufl.dx
            
        a_cut_adv = fem.form(a_adv)
        L_cut_adv = fem.form(L_adv)

        b_adv = assemble_vector(L_cut_adv)



        b_adv = fem.petsc.create_vector(L_cut_adv)
        
        A_adv = fem.petsc.assemble_matrix(a_cut_adv, bcs=[self.bc])
        A_adv.assemble()


        solver_adv = PETSc.KSP().create(self.mesh.comm)
        solver_adv.setOperators(A_adv)
        solver_adv.setType(PETSc.KSP.Type.PREONLY)
        solver_adv.getPC().setType(PETSc.PC.Type.LU)

        sol_adv = fem.Function(self.V_ls)
        solver_adv.solve(b_adv, sol_adv.x.petsc_vec)
        b_adv.destroy()
        return sol_adv

    def set_bilin_form(self):
        """ Returns the update of the primal bilinear form with the smoothed value of the Lamé coefficients.

        :returns: The bilinear form of the primal and dual problems.
        :rtype: fem.form
        
        """
        self.a = 2.0*self.lame_mu_fic*ufl.inner(mechanics_tool.strain(self.u), mechanics_tool.strain(self.v)) * ufl.dx +\
            self.lame_lambda_fic * ufl.inner(ufl.nabla_div(self.u), ufl.nabla_div(self.v)) * ufl.dx 
        return fem.form(self.a)
    
    def primal_problem(self,level_set,parameters):        
        """Resolution of the primal problem with the Ersatz method.

        :param fem.Function level_set: The level set field which defined implicitly the domain :math:`\Omega`.
        :param Parameters parameters: The object parameters.
        
        :returns: The primal solution.
        :rtype: fem.Function
        
        """
        self.xsi = self.heaviside(level_set)
        self.lame_mu_fic = self.compute_lame_mu(self.xsi)
        self.lame_lambda_fic= self.compute_lame_lambda(self.xsi)

        self.a_primal = self.set_bilin_form()
        self.A_primal.zeroEntries()
        fem.petsc.assemble_matrix(self.A_primal, self.a_primal, bcs=[self.bc])
        self.A_primal.assemble()

        with self.b_primal.localForm() as loc:
            loc.set(0)
        fem.petsc.assemble_vector(self.b_primal, self.L_primal)
        fem.petsc.apply_lifting(self.b_primal, [self.a_primal], [[self.bc]])
        self.b_primal.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.b_primal.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        fem.petsc.set_bc(self.b_primal, [self.bc])

        
        self.primal_solver.setOperators(self.A_primal)

        self.primal_solver.solve(self.b_primal, self.uh.x.petsc_vec)
        self.uh.x.scatter_forward() 

        return self.uh
    
    def adjoint_problem(self,uh,parameters,level_set,adjoint=0):  
        
        self.uh = uh
        self.xsi = self.heaviside(level_set)

        self.lame_mu_fic = self.compute_lame_mu(self.xsi)
        self.lame_lambda_fic = self.compute_lame_lambda(self.xsi)

        v = ufl.TestFunction(self.space_displacement)  
        self.A_adjoint.zeroEntries()
        fem.petsc.assemble_matrix(self.A_adjoint, self.a_adjoint, bcs=[self.bc])
        self.A_adjoint.assemble()

        b_adjoint = fem.petsc.create_vector(self.L_adjoint)

        fem.petsc.assemble_vector(b_adjoint, self.L_adjoint)
        
        fem.apply_lifting(b_adjoint, [self.a_adjoint], [[self.bc]])
        b_adjoint.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b_adjoint.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        fem.petsc.set_bc(b_adjoint, [self.bc])
        
        self.adjoint_solver.setOperators(self.A_adjoint)

        self.adjoint_solver.solve(b_adjoint, self.ph.x.petsc_vec)
        self.ph.x.scatter_forward() 
        b_adjoint.destroy()
        
        return self.ph
    
    def ersatz_solver(self,level_set,parameters,adjoint=0):
        r"""Resolution of the primal and dual problem with Ersatz method.

        :param fem.Function level_set: The level set field which defined implicitly the domain :math:`\Omega`.
        :param Parameters parameters: The object parameters.
        :param ufl.Expression adjoint: The adjoint operator if needed.
        
        :returns: The values of the primal and dual solution.
        :rtype: fem.Function, fem.Function
        
        """
        self.uh = self.primal_problem(level_set,parameters)
        self.ph.x.array[:] = self.uh.x.array
        if (parameters.cost_func != "compliance"):
            self.ph = self.adjoint_problem(self.uh,parameters,level_set,adjoint)
        return self.uh, self.ph    
    
    def descent_direction(self,level_set,parameters,rest_constraint,constraint_integrand,cost_integrand,xsi_temp):

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

        a  =   parameters.alpha_reg_velocity *ufl.inner(grad(u_r), grad(v_r))*dx
        a += u_r*v_r*dx

        C_Omega_value = (rest_constraint + parameters.ALM_slack_variable )

        temp = cost_integrand
        temp_ALM = parameters.ALM*(parameters.ALM_lagrangian_multiplicator * constraint_integrand + parameters.ALM_penalty_parameter * C_Omega_value * constraint_integrand \
                + 2 * constraint_integrand*parameters.ALM_slack_variable)
        temp_ALM += (1-parameters.ALM)*parameters.target_constraint
        temp += temp_ALM

        L = -(ufl.dot(ufl.grad(xsi_temp), -n_K)*inner(temp*v_r*n_K,n_K)*dx)

        a_form = fem.form(a)
        L_form = fem.form(L)

        b = fem.petsc.create_vector(L_form)

        fem.petsc.assemble_vector(b, L_form)
        
        A = fem.petsc.assemble_matrix(a_form,  bcs = [self.bc_velocity])
        A.assemble()



        solver_reg = PETSc.KSP().create(self.mesh.comm)
        solver_reg.setOperators(A)
        solver_reg.setType(PETSc.KSP.Type.PREONLY)
        solver_reg.getPC().setType(PETSc.PC.Type.LU)
        solver_reg.solve(b, v_reg.x.petsc_vec)
        b.destroy()

        return v_reg
