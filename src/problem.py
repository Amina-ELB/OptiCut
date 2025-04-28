# Copyright (c) 2025 ONERA and MINES Paris, France 

# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari 

import mechanics_tool
import ufl
# mathematical language for FEM, auto differentiation, python
from dolfinx import fem, io, mesh
import ufl
from ufl import (FacetNormal, Measure, SpatialCoordinate, TestFunction, TrialFunction, 
                 div, dot, dx, grad, inner, lhs, rhs, dc, dS, FacetNormal, CellDiameter, dot, avg, jump)

###

import cutfemx
from dolfinx import cpp as _dolfinxcpp
from dolfinx import fem, io, mesh


import math
import collections
import functools
import typing

import numpy as np

from dolfinx import la

from mpi4py import MPI

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


import data_manipulation 

class Compliance_Problem:
    """This is the Compliance problem class.

    :param str name: The Compliance  shape optimization problem's name.
    """
    def __init__(self):
        """The constructor."""
        pass

    def cost_integrand(self,u,lame_mu,lame_lambda,Parameters=0):
        r"""Compute the integrand of the cost function, defined as :

        .. math::
                
                j\left( u \right) = 0.5 \times \left( 2\mu \sigma(u):\sigma(u) + \lambda \nabla \cdot \sigma(u):\nabla \cdot \sigma(u) \right)
               

        :param fem.Function u: The displacement field, solution of the primal problem :math:`\\u_{h}`.
        :param float lame_mu: The :math:`\mu` Lame coefficient.
        :param float lame_lambda: The :math:`\lambda` Lame coefficient.

        :returns: The integrand of the cost function.
        :rtype: ufl.Expression
        """
        integrand = 0.5*(2.0*lame_mu  * ufl.inner(mechanics_tool.strain(u), mechanics_tool.strain(u))  + lame_lambda *  ufl.inner(ufl.nabla_div(u), ufl.nabla_div(u)) )
        
        return integrand

    def cost(self,u=0,p=0,lame_mu=0,lame_lambda=0,measure=0,Parameters=0):
        r"""Compute the cost function, defined as : 

        .. math::
                
                J\left( \Omega \right) = \int_{\Omega} 0.5 \times \left( 2\mu \sigma(u_{h}):\sigma(u_{h}) + \lambda \nabla \cdot \sigma(u_{h}):\nabla \cdot \sigma(u_{h}) \right) \text{ }dx
                      

        :param fem.Function u: The displacement field, solution of the primal problem :math:`\\u_{h}`.
        :param float lame_mu: The :math:`\mu` Lame coefficient.
        :param float lame_lambda: The :math:`\lambda` Lame coefficient.

        :returns: The integrand of the cost function.
        :rtype: ufl.Expression
        """
        if Parameters.cutFEM == 1:
            cost = cutfemx.fem.assemble_scalar(cut_form(self.cost_integrand(u,lame_mu,lame_lambda)*measure))
            cost = MPI.COMM_WORLD.allreduce(cost,op=MPI.SUM)
        else:
            cost = fem.assemble_scalar(fem.form(self.cost_integrand(u,lame_mu,lame_lambda)*ufl.dx))
            cost = MPI.COMM_WORLD.allreduce(cost,op=MPI.SUM)
        return cost 
    
    def shape_derivative_integrand(self,u,p,lame_mu,lame_lambda,parameters,measure=0):
        r"""Compute the following shape derivative integrand: 

        .. math::

                \begin{align}
                x =& j\left( u_{h} \right) - 2\mu \sigma(u_{h}):\sigma(p_{h}) + \lambda \nabla \cdot \sigma(u_{h}):\nabla \cdot \sigma(p_{h}) \\
                   = & - 0.5 \times \left( 2\mu \sigma(u_{h}):\sigma(p_{h}) + \lambda \nabla \cdot \sigma(u_{h}):\nabla \cdot \sigma(p_{h}) \right) 
                \end{align}     
        
                
        Remark:
        -------
        For the special case of the compliance minimization, we have :math:`\\u_{h} = \\p_{h}`

        
        :param fem.Function u: The solution of the primal problem :math:`\\u_{h}`.
        

        :returns: The shape derivative of the cost function minus linear elasticity.
        :rtype: ufl.Expression
        """
        res = self.cost_integrand(u,lame_mu,lame_lambda) - (2.0*lame_mu  * ufl.inner(mechanics_tool.strain(u), mechanics_tool.strain(u))  + lame_lambda *  ufl.inner(ufl.nabla_div(u), ufl.nabla_div(u)))
        return res
        
    def shape_derivative_integrand_constraint(self,u,p,lame_mu,lame_lambda,parameters,measure=0,vm_DG = 0,c_k=0):
        r"""Compute the shape derivative integrand of the area constraint. 

        :returns: The integrand of the constraint function.
        :rtype: ufl.Expression
        """
        res = 1
        return res
    
    def constraint_integrand(self,u,lame_mu,lame_lambda,parameters):
        r"""Compute the integrand of the area constraint. 

        :returns: The integrand of the constraint function.
        :rtype: ufl.Expression
        """
        mesh = u.function_space.mesh
        res = fem.Constant(mesh, ScalarType(1.0))
        return res 

    def constraint(self,u,lame_mu,lame_lambda,parameters,measure,xsi=0,vm_DG = 0,c_k=0):
        r"""Compute the value of constraint. 

        :param fem.Function u: The displacement field, solution of the primal problem :math:`\\u_{h}`.
        :param float lame_mu: The :math:`\mu` Lame coefficient.
        :param float lame_lambda: The :math:`\lambda` Lame coefficient.
        :param Parameter parameters: The parameter object.
        :param Measure measure: The measure on the domain :math:`\Omega`.

        :returns: The integrand of the norm Lp of the Von Mises criteria.
        :rtype: ufl.Expression

        """

        if parameters.cutFEM == 1:
            res = cutfemx.fem.assemble_scalar(cut_form(self.constraint_integrand(u,lame_mu,lame_lambda,parameters)*measure)) 
            print("area = ", res)
            res = MPI.COMM_WORLD.allreduce(res,op=MPI.SUM) - parameters.target_constraint
            print("area = ", res)
        else:
            res = fem.assemble_scalar(fem.form(xsi*measure)) 
            res = MPI.COMM_WORLD.allreduce(res,op=MPI.SUM) - parameters.target_constraint
        return res
    
    def dual_operator(self,u,lame_mu,lame_lambda,parameters,mesh,measure=0,vm_DG = 0,c_k=0):
        r"""Compute the dual operator. 
        
        :param fem.Function u: The solution of the primal problem :math:`\\u_{h}`.
        :param float lame_mu: The :math:`\mu` Lame coefficient.
        :param float lame_lambda: The :math:`\lambda` Lame coefficient.
        :param Parameter parameters: The parameter object.
        :param Mesh mesh: The mesh of the domaine :math:`D`.
        :param Measure measure: The measure on the domain :math:`\Omega`.

        :returns: 0.
        :rtype: float.
        """
        dual_operator = 0
        shape_derivative_constraint_integrand = 1
        shape_derivative_multiplicator = 1
        return dual_operator 
    
    
class VMLp_Problem:
    """This is the Von Mises Lp norm problem class.

    :param str name: The Von Mises Lp norm shape optimization problem's name.
    """
    def __init__(self):
        """The constructor."""
        pass
    def cost_integrand(self,u,lame_mu,lame_lambda,Parameters):
        r"""Compute the integrand of the cost function given by the following formula:
        
        .. math::
                
                x = \left[\frac{\sigma_{VM}\left(u_{h}\right)}{\overline{\sigma_{VM}}}\right]^{p}

        with :math:`\sigma_{VM}\left(u_{h}\right)` compute with the function :func:`mecanics_tool.von_mises` and :math:`\overline{\sigma_{VM}}` a constant attribut of Parameters. 
        
        :param fem.Function u: The displacement field, solution of the primal problem :math:`\\u_{h}`.
        :param float lame_mu: The :math:`\mu` Lame coefficient.
        :param float lame_lambda: The :math:`\lambda` Lame coefficient.

        :returns: The integrand of the norm Lp of the Von Mises criteria.
        :rtype: ufl.Expression

        """
        dim = 2
        integrand = (mechanics_tool.von_mises(u,lame_mu,lame_lambda,dim)/Parameters.elasticity_limit)**Parameters.p_const
        
        return   integrand

    def cost(self,u=0,p=0,lame_mu=0,lame_lambda=0,measure=0,Parameters=0):
        r"""Compute the cost function, defined as : 

        .. math::
                
                J\left( \Omega \right) = & \left[ \int_{\Omega} \left[\frac{\sigma_{VM}\left(u_{h}\right)}{\overline{\sigma_{VM}}}\right]^{p}\text{ }dx \right]^{\frac{1}{p}} \\

                
        with :math:`\sigma_{VM}\left(u_{h}\right)` compute with the function :func:`mecanics_tool.von_mises` and :math:`\overline{\sigma_{VM}}` a constant attribut of Parameters. 
        

        :param fem.Function u: The displacement field, solution of the primal problem :math:`\\u_{h}`.
        :param float lame_mu: The :math:`\mu` Lame coefficient.
        :param float lame_lambda: The :math:`\lambda` Lame coefficient.
        :param Parameter parameters: The parameter object.
        :param Measure measure: The measure on the domain :math:`\Omega`.


        :returns: The integrand of the cost function.
        :rtype: ufl.Expression
        """
        
        cost = cutfemx.fem.assemble_scalar(cut_form(self.cost_integrand(u,lame_mu,lame_lambda,Parameters)*measure))**(1/Parameters.p_const)
        cost = MPI.COMM_WORLD.allreduce(cost,op=MPI.SUM)
        return cost 
    

    def constraint_integrand(self,u,lame_mu,lame_lambda,parameters):
        r"""Compute the integrand of the area constraint. 

        :returns: The integrand of the constraint function.
        :rtype: ufl.Expression
        """
        mesh = u.function_space.mesh
        res = fem.Constant(mesh, ScalarType(1.0))
        return res 
    

    def constraint(self,u,lame_mu,lame_lambda,parameters,measure,xsi=0,vm_DG = 0,c_k=0):
        r"""Compute the value of constraint. 

        :param fem.Function u: The displacement field, solution of the primal problem :math:`\\u_{h}`.
        :param float lame_mu: The :math:`\mu` Lame coefficient.
        :param float lame_lambda: The :math:`\lambda` Lame coefficient.
        :param Parameter parameters: The parameter object.
        :param Measure measure: The measure on the domain :math:`\Omega`.

        :returns: The integrand of the norm Lp of the Von Mises criteria.
        :rtype: ufl.Expression

        """
        if parameters.cutFEM == 1:
            res = cutfemx.fem.assemble_scalar(cut_form(self.constraint_integrand(u,lame_mu,lame_lambda,parameters)*measure)) 
            res = MPI.COMM_WORLD.allreduce(res,op=MPI.SUM) - parameters.target_constraint
        else:
            res = fem.assemble_scalar(fem.form(xsi*measure)) 
            res = MPI.COMM_WORLD.allreduce(res,op=MPI.SUM) - parameters.target_constraint
        return res
    

    def shape_derivative_integrand(self,u,p,lame_mu,lame_lambda,parameters,measure=0):
        r"""Compute the following shape derivative integrand: 

        .. math::

                \begin{align}
                x = & \frac{1}{p} \left[ \int_{\Omega} \left[\frac{\sigma_{VM}\left(u_{h}\right)}{\overline{\sigma_{VM}}}\right]^{p} \text{ }dx \right]^{\frac{1}{p}-1} \\
                    & \left[ \left( \frac{\sigma_{VM}\left(u_{h}\right)}{\overline{\sigma_{VM}}}\right)^{p}- 2\mu \sigma(u_{h}):\sigma(p_{h}) + \lambda \nabla \cdot \sigma(p_{h}):\nabla \cdot \sigma(u_{h}) \right]
                \end{align}
                
        
        :param fem.Function u: The solution of the primal problem :math:`\\u_{h}`.
        :param fem.Function p: The solution of the dual problem :math:`\\p_{h}`.
        :param float lame_mu: The :math:`\mu` Lame coefficient.
        :param float lame_lambda: The :math:`\lambda` Lame coefficient.
        :param Parameter parameters: The parameter object.
        :param Measure measure: The measure on the domain :math:`\Omega`.

    
        :returns: The shape derivative of the cost function minus linear elasticity.
        :rtype: ufl.Expression
        """
        bilin_integrand = (2.0*lame_mu * ufl.inner(mechanics_tool.strain(u),\
            mechanics_tool.strain(p))  + lame_lambda *  ufl.inner(ufl.nabla_div(u), \
                ufl.nabla_div(p)))
        shapeDerivativeintegrand = self.cost_integrand(u,lame_mu,lame_lambda,parameters)
        J =((shapeDerivativeintegrand))*measure
        multiplyShapeDerivative = (1/parameters.p_const)*cutfemx.fem.assemble_scalar(cut_form(J))**(1/parameters.p_const-1)
        multiplyShapeDerivative = MPI.COMM_WORLD.allreduce(multiplyShapeDerivative,op=MPI.SUM)
        shape_derivative = multiplyShapeDerivative * (shapeDerivativeintegrand - bilin_integrand)
        return shape_derivative
    
    
    def shape_derivative_integrand_constraint(self,u,p,lame_mu,lame_lambda,parameters,measure=0,vm_DG = 0,c_k=0):
        r"""Compute the shape derivative integrand of the area constraint. 

        :returns: The integrand of the constraint function.
        :rtype: ufl.Expression
        """
        res = 1
        return res  
    

    def dual_operator(self,u,lame_mu,lame_lambda,parameters,mesh,measure=0,vm_DG = 0,c_k =0):
        r"""Compute the dual operator. 
        
        :param fem.Function u: The solution of the primal problem :math:`\\u_{h}`.
        :param float lame_mu: The :math:`\mu` Lame coefficient.
        :param float lame_lambda: The :math:`\lambda` Lame coefficient.
        :param Parameter parameters: The parameter object.
        :param Mesh mesh: The mesh of the domaine :math:`D`.
        :param Measure measure: The measure on the domain :math:`\Omega`.

        :returns: The dual linear form.
        :rtype: ufl.Expression
        """
        dim = mesh.geometry.dim
        V = fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, )))
        shapeDerivativeConstraintintegrand = self.cost_integrand(u,lame_mu,lame_lambda,parameters)
        J =((shapeDerivativeConstraintintegrand))*measure
        v_adj = ufl.TestFunction(V)
        dual_operator = ufl.derivative(J,u,v_adj)
        multiplyShapeDerivative = (1/parameters.p_const)*cutfemx.fem.assemble_scalar(cut_form(((mechanics_tool.von_mises(u,lame_mu,lame_lambda,dim)/parameters.elasticity_limit)**parameters.p_const)*measure))**(1/parameters.p_const-1)
        multiplyShapeDerivative = MPI.COMM_WORLD.allreduce(multiplyShapeDerivative,op=MPI.SUM)
        return dual_operator
    

class AreaProblem:
    """This is the Area problem class.

    :param str name: The Area shape optimization problem's name.
    """
    def __init__(self):
        """The constructor."""
        pass

    def cost_integrand(self,u,lame_mu,lame_lambda,Parameters):
        r"""Compute the integrand of the cost function.
        
        :param fem.Function u: The displacement field, solution of the primal problem :math:`\\u_{h}`.
        :param float lame_mu: The :math:`\mu` Lame coefficient.
        :param float lame_lambda: The :math:`\lambda` Lame coefficient.

        :returns: The integrand of the norm Lp of the Von Mises criteria.
        :rtype: ufl.Expression
        """
        mesh = u.function_space.mesh
        res = fem.Constant(mesh, ScalarType(1.0))
        return res 
    
    def cost(self,u=0,p=0,lame_mu=0,lame_lambda=0,measure=0,Parameters=0):
        r"""Compute the cost function, defined as : 

        .. math::
                
                J\left( \Omega \right) = \int_{\Omega}  \text{ }dx

                
        :param fem.Function u: The displacement field, solution of the primal problem :math:`\\u_{h}`.
        :param float lame_mu: The :math:`\mu` Lame coefficient.
        :param float lame_lambda: The :math:`\lambda` Lame coefficient.
        :param Parameter parameters: The parameter object.
        :param Measure measure: The measure on the domain :math:`\Omega`.

                      
        :returns: The the cost value.
        :rtype: ufl.Expression
        """
        cost = cutfemx.fem.assemble_scalar(cut_form(self.cost_integrand(u,lame_mu,lame_lambda,Parameters)*measure))
        cost = MPI.COMM_WORLD.allreduce(cost,op=MPI.SUM)
        return cost 
    
    def constraint_integrand(self,u,lame_mu,lame_lambda,parameters):
        r"""Compute the integrand of the Lp norm of the Von Mises constraint, defined as:
        
        .. math::

            x = \left[\frac{\sigma_{VM}\left(u_{h}\right)}{\overline{\sigma_{VM}}}\right]^{p}

        with :math:`\sigma_{VM}\left(u_{h}\right)` compute with the function :func:`mecanics_tool.von_mises` and :math:`\overline{\sigma_{VM}}` a constant attribut of Parameters. 
        
        :param fem.Function u: The displacement field, solution of the primal problem :math:`\\u_{h}`.
        :param float lame_mu: The :math:`\mu` Lame coefficient.
        :param float lame_lambda: The :math:`\lambda` Lame coefficient.
        :param Parameter parameters: The parameter object.

        :returns: The integrand of the constraint fonction.
        :rtype: ufl.Expression
        """
        mesh = u.function_space.mesh
        dim = mesh.geometry.dim
        integrand = (mechanics_tool.von_mises(u,lame_mu,lame_lambda,dim)/parameters.elasticity_limit)**parameters.p_const
        return integrand 
    
    def constraint(self,u,lame_mu,lame_lambda,parameters,measure,xsi=0,vm_DG = 0,c_k=1):
        r"""Compute the value of constraint. 

        :param fem.Function u: The displacement field, solution of the primal problem :math:`\\u_{h}`.
        :param float lame_mu: The :math:`\mu` Lame coefficient.
        :param float lame_lambda: The :math:`\lambda` Lame coefficient.
        :param Parameter parameters: The parameter object.
        :param Measure measure: The measure on the domain :math:`\Omega`.

        :returns: The integrand of the norm Lp of the Von Mises criteria.
        :rtype: ufl.Expression

        """
        if parameters.cutFEM == 1:
            """ P mean function """
            # mesh = u.function_space.mesh
            # constant_one = fem.Constant(mesh, ScalarType(1.0))
            # area = cutfemx.fem.assemble_scalar(cut_form((constant_one*measure)))
            # res = (cutfemx.fem.assemble_scalar(cut_form(self.constraint_integrand(u,lame_mu,lame_lambda,parameters)*measure)))**(1/parameters.p_const) / area
            # res = MPI.COMM_WORLD.allreduce(res,op=MPI.SUM) - parameters.target_constraint
            """ P norm function """
            #res = (cutfemx.fem.assemble_scalar(cut_form(self.constraint_integrand(u,lame_mu,lame_lambda,parameters)*measure)))**(1/parameters.p_const) 
            mesh = u.function_space.mesh
            dim = mesh.geometry.dim
            
            array_vm  = vm_DG.x.array[:]
    
            divided = array_vm / parameters.elasticity_limit
            power = np.power(divided, parameters.p_const)
            sum_array  = np.sum(power)
            sum_array = MPI.COMM_WORLD.allreduce(sum_array,op=MPI.SUM) 
            res = sum_array**(1/parameters.p_const)
            print("in constraint result", res)
            #res = (fem.assemble_scalar(fem.form((vm_list/parameters.elasticity_limit)**8*ufl.dx)))**(1/8) 
            res = (MPI.COMM_WORLD.allreduce(res,op=MPI.SUM)) - parameters.target_constraint
            print("res = ",res)
            #res = MPI.COMM_WORLD.allreduce(res,op=MPI.SUM) - parameters.target_constraint
        else:
            """ P mean function """
            # mesh = u.function_space.mesh
            # constant_one = fem.Constant(mesh, ScalarType(1.0))
            # area = cutfemx.fem.assemble_scalar(cut_form((constant_one*measure)))
            # res = fem.assemble_scalar(fem.form(self.constraint_integrand(u,lame_mu,lame_lambda,parameters)*measure))**(1/parameters.p_const) / area
            # res = MPI.COMM_WORLD.allreduce(res,op=MPI.SUM) - parameters.target_constraint
            """ P norm function """
            res = fem.assemble_scalar(fem.form(self.constraint_integrand(u,lame_mu,lame_lambda,parameters)*measure))**(1/parameters.p_const) 
            res = MPI.COMM_WORLD.allreduce(res,op=MPI.SUM) - parameters.target_constraint
        return res
    
    def shape_derivative_integrand_constraint(self,u,p,lame_mu,lame_lambda,parameters,measure=0,vm_DG = 0,c_k=1):
        r"""Compute the shape derivative integrand of the constraint. 

        :param fem.Function u: The solution of the primal problem :math:`\\u_{h}`.
        :param fem.Function p: The solution of the dual problem :math:`\\p_{h}`.
        :param float lame_mu: The :math:`\mu` Lame coefficient.
        :param float lame_lambda: The :math:`\lambda` Lame coefficient.
        :param Parameter parameters: The parameter object.
        :param Measure measure: The measure on the domain :math:`\Omega`.

        :returns: The integrand of the constraint function.
        :rtype: ufl.Expression
        """
        mesh = u.function_space.mesh
        dim = mesh.geometry.dim
        Q = fem.functionspace(mesh, ("DG", 0))

        constraint_integrand = (mechanics_tool.von_mises(u,lame_mu,lame_lambda,dim)/parameters.elasticity_limit)**parameters.p_const
        
        array_vm  = vm_DG.x.array[:]
    
        divided = array_vm / parameters.elasticity_limit
        power = np.power(divided, parameters.p_const)
        sum_array  = np.sum(power)
        sum_array = MPI.COMM_WORLD.allreduce(sum_array,op=MPI.SUM) 
        multiplyShapeDerivative = (1/parameters.p_const)* sum_array**(1/parameters.p_const-1)
        constraint = sum_array**(1/parameters.p_const)
        print("in constraint result", multiplyShapeDerivative)
        print("constraint = ",constraint)
        #res = (fem.assemble_scalar(fem.form((vm_list/parameters.elasticity_limit)**8*ufl.dx)))**(1/8) 
        # constraint_integrand = MPI.COMM_WORLD.allreduce(res,op=MPI.SUM) - 1
        # print("res = ",res)
        # constraint = self.constraint(u,lame_mu,lame_lambda,parameters,measure,vm_DG)#+ compliance
        # multiplyShapeDerivative = (1/parameters.p_const)*(cutfemx.fem.assemble_scalar(cut_form(((mecanics_tool.von_mises(u,lame_mu,lame_lambda,dim)/parameters.elasticity_limit)**parameters.p_const)*measure)))**(1/parameters.p_const-1)
        # multiplyShapeDerivative = MPI.COMM_WORLD.allreduce(multiplyShapeDerivative,op=MPI.SUM)
        print("multiplyShapeDerivative = ",multiplyShapeDerivative)
        """P means"""
        #res =  multiplyShapeDerivative * constraint_integrand / area - constraint / area
        """P norm """
        # constant_one = fem.Constant(mesh, ScalarType(1.0))
        # area = cutfemx.fem.assemble_scalar(cut_form((constant_one*measure)))

        compliance = - (2.0*lame_mu  * ufl.inner(mechanics_tool.strain(u), mechanics_tool.strain(p))  + lame_lambda *  ufl.inner(ufl.nabla_div(u), ufl.nabla_div(p)))
        shape_derivative_integrand_constraint_expr = fem.Expression(constraint_integrand, Q.element.interpolation_points())
        shape_derivative_integrand_constraint = fem.Function(Q)
        shape_derivative_integrand_constraint.interpolate(shape_derivative_integrand_constraint_expr)
        res =  multiplyShapeDerivative * (shape_derivative_integrand_constraint)  + compliance
        
        return res

    def shape_derivative_integrand(self,u,p,lame_mu,lame_lambda,parameters,measure=0):
        r"""Compute the shape derivative integrand of the area constraint. 

        :param fem.Function u: The solution of the primal problem :math:`\\u_{h}`.
        :param fem.Function p: The solution of the dual problem :math:`\\p_{h}`.
        :param float lame_mu: The :math:`\mu` Lame coefficient.
        :param float lame_lambda: The :math:`\lambda` Lame coefficient.
        :param Parameter parameters: The parameter object.
        :param Measure measure: The measure on the domain :math:`\Omega`.

        :returns: The integrand of the cost function.
        :rtype: ufl.Expression
        """
        compliance = - (2.0*lame_mu  * ufl.inner(mechanics_tool.strain(u), mechanics_tool.strain(p))  + lame_lambda *  ufl.inner(ufl.nabla_div(u), ufl.nabla_div(p)))
        mesh = u.function_space.mesh
        one = fem.Constant(mesh, ScalarType(1.0))
        return one 
    
    def dual_operator(self,u,lame_mu,lame_lambda,parameters,mesh,measure=0,vm_DG=0,c_k=0):
        r"""Compute the dual operator. 
        
        :param fem.Function u: The solution of the primal problem :math:`\\u_{h}`.
        :param float lame_mu: The :math:`\mu` Lame coefficient.
        :param float lame_lambda: The :math:`\lambda` Lame coefficient.
        :param Parameter parameters: The parameter object.
        :param Mesh mesh: The mesh of the domaine :math:`D`.
        :param Measure measure: The measure on the domain :math:`\Omega`.

        :returns: The dual linear form.
        :rtype: ufl.Expression
        """
        dim = mesh.geometry.dim
        V = fem.functionspace(mesh, ("Lagrange", 1, (dim, )))
        shapeDerivativeConstraintintegrand = self.constraint_integrand(u,lame_mu,lame_lambda,parameters)
        J =(shapeDerivativeConstraintintegrand)*measure
        constant_one = fem.Constant(mesh, ScalarType(1.0))
        area = cutfemx.fem.assemble_scalar(cut_form((constant_one*measure)))
        v_adj = ufl.TestFunction(V)
        array_vm  = vm_DG.x.array[:]
        sum_array = (np.sum(np.power(array_vm / parameters.elasticity_limit,parameters.p_const)))
        sum_array = MPI.COMM_WORLD.allreduce(sum_array,op=MPI.SUM) 
        multiplyShapeDerivative = (1/parameters.p_const)*sum_array**(1/parameters.p_const-1)
        
        multiplyShapeDerivative = MPI.COMM_WORLD.allreduce(multiplyShapeDerivative,op=MPI.SUM) 
        """ P mean """
        # dual_operator = ufl.derivative((multiplyShapeDerivative/area)*J,u,v_adj)
        # C_Omega_value = self.constraint(u,lame_mu,lame_lambda,parameters,measure) 
        """P norm """
        dual_operator = ufl.derivative(multiplyShapeDerivative*J,u,v_adj)
        C_Omega_value = self.constraint(u,lame_mu,lame_lambda,parameters,measure,vm_DG) 
        res = parameters.ALM_lagrangian_multiplicator * dual_operator + parameters.ALM_penalty_parameter *\
              (C_Omega_value+ parameters.ALM_slack_variable)* dual_operator
        #dual_operator = ufl.derivative(J,u,v_adj)
        return  res
    
