# The modules that will be used are imported:
import numpy as np
import math
import ufl
 
# mathematical language for FEM, auto differentiation, python
from dolfinx import fem
import matplotlib.pyplot as plt
# meshes, assembly, c++, ython, pybind
from ufl import ds, dx, inner


from typing import TYPE_CHECKING
import pyvista


from dolfinx.fem import Function
import dolfinx.fem.petsc 
from dolfinx.mesh import locate_entities, meshtags
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import ( dx, inner)

from Parameters import *
from create_mesh import *
from ersatz_method import *
from cutfem_method import *
from levelSet_tool import *
from extension_regularization import *
from geometry_initialization import *
from dolfinx.mesh import locate_entities, meshtags, locate_entities_boundary
from cutfemx.level_set import locate_entities, cut_entities, ghost_penalty_facets, facet_topology
from cutfemx.level_set import compute_normal
from cutfemx.mesh import create_cut_mesh, create_cut_cells_mesh
import mechanics_tool
import shutil
import os


def create_list_vm(msh,uh,parameters,lame_mu,lame_lambda,i,level_set,V_ls,Q, domain_marker = 0):
    
    dim = msh.topology.dim
    xsi_vm = ufl.conditional(ufl.le(level_set,0),1,0)
    xsi_expr = fem.Expression(xsi_vm, V_ls.element.interpolation_points())
    xsi_vm_func = fem.Function(V_ls)
    xsi_vm_func.interpolate(xsi_expr)
    xsi_vm = ufl.conditional(ufl.eq(xsi_vm_func,0),0,1)
    xsi_expr = fem.Expression(xsi_vm, Q.element.interpolation_points())
    xsi_vm_func = fem.Function(Q)
    xsi_vm_func.interpolate(xsi_expr)
    
    vm = mechanics_tool.von_mises(uh,lame_mu, lame_lambda, dim)
    vm_expr = fem.Expression(vm, Q.element.interpolation_points())
    vm = fem.Function(Q)
    vm.interpolate(vm_expr)
    
    vm_expr = fem.Expression(xsi_vm_func*vm, Q.element.interpolation_points())
    vm_test = fem.Function(Q)
    vm_test.interpolate(vm_expr)
    # with io.XDMFFile(msh.comm, "out_ls/level_set"+str(i)+".xdmf", "w") as file:
    #         file.write_mesh(msh)
    #         file.write_function(cutfem_method.level_set)
    #         file.write_meshtags(domain_marker)
    return vm_test

def save_data_compliance(folder, ls_func, ls_space, xsi, velocity, velocity_space, primal_sol, primal_space, \
              dual_sol, vonMises, vonMises_space, time):
      
            xsi_expr = fem.Expression(xsi, ls_space.element.interpolation_points())
            xsi = fem.Function(ls_space)
            xsi.interpolate(xsi_expr)

            velocity_expr = fem.Expression(velocity, velocity_space.element.interpolation_points())
            velocity = fem.Function(velocity_space)
            velocity.interpolate(velocity_expr)
        
            uh_expr = fem.Expression(primal_sol, primal_space.element.interpolation_points())
            primal_sol = fem.Function(primal_space)
            primal_sol.interpolate(uh_expr)
            ph_expr = fem.Expression(dual_sol, primal_space.element.interpolation_points())
            dual_sol = fem.Function(primal_space)
            dual_sol.interpolate(ph_expr)

            ls_func.name = "ls_func"
            xsi.name = "xsi"
            velocity.name = "velocity"
            primal_sol.name = "displacement"
            dual_sol.name = "dualsol"
            
            folder.write_function(ls_func, time)
            folder.write_function(xsi, time)
            folder.write_function(velocity, time)
            folder.write_function(primal_sol, time)

            folder.write_function(dual_sol, time)

def save_data(folder, ls_func, ls_space, xsi, velocity, velocity_space, primal_sol, primal_space, \
              dual_sol, vonMises, vonMises_space, time):
            
            xsi_func = fem.Function(velocity_space)
            xsi_func.interpolate(xsi)

            velocity_func = fem.Function(velocity_space)
            velocity_func.interpolate(velocity)

            
            # xsi_vm = ufl.conditional(ufl.le(ls_func,0),1,0)
            # xsi_expr = fem.Expression(xsi_vm, vonMises_space.element.interpolation_points())
            # xsi_vm_func = fem.Function(vonMises_space)
            # xsi_vm_func.interpolate(xsi_expr)
            # xsi_vm = ufl.conditional(ufl.eq(xsi_vm_func,0),0,1)
            # xsi_expr = fem.Expression(xsi_vm, vonMises_space.element.interpolation_points())
            # xsi_vm_func = fem.Function(vonMises_space)
            # xsi_vm_func.interpolate(xsi_expr)
            
            
            # vonMises_func = fem.Function(vonMises_space)
            # vonMises_func.interpolate(vonMises)

            # vm_expr = fem.Expression(xsi_vm_func*vm, vonMises_space.element.interpolation_points())
            # vm_cut = fem.Function(vonMises_space)
            # vm_cut.interpolate(vm_expr)
            
            ls_func.name = "ls_func"
            xsi_func.name = "xsi"
            velocity_func.name = "velocity"
            primal_sol.name = "displacement"
            dual_sol.name ="sol_dual"
            # vonMises_func.name = "vm"
            # vonMises_func.name = "heaviside_vm"
            
            folder.write_function(ls_func, time)
            folder.write_function(xsi_func, time)
            folder.write_function(velocity_func, time)
            folder.write_function(primal_sol, time)
            folder.write_function(dual_sol, time)
            # folder.write_function(vonMises_func, time)
            # folder.write_function(vonMises_func,time)

def histogram(array,bins,iteration):
    val_max = np.max(array)
    print("val max = ",np.max(array))
    plt.hist(array, range = (0, val_max),  bins = bins)
    plt.xlabel('Von Mises criteria')
    plt.savefig("histo_ite"+str(iteration)+".png")
    plt.close()
 
def histogram_final_1(array1,array2,bins,parameters):
    max_1 = np.max(array1)
    max_2 = np.max(array2)
    val_max = max(max_1,max_2)
    plt.hist(array1,range = (parameters.elasticity_limit, val_max), bins = bins,alpha =0.5, label = 'initial distribution of Von Mises criteria')
    plt.hist(array2,range = (parameters.elasticity_limit, val_max), bins = bins, alpha=0.5, label = 'optimized final distribution of Von Mises criteria' )
    plt.legend(loc='upper right')
    plt.xlabel('Von Mises criteria')
    plt.savefig("histo_final_1.png")
    plt.close()
def histogram_final_2(array1,array2,bins,parameters):
    max_1 = np.max(array1)
    max_2 = np.max(array2)
    val_max = min(max_1,max_2)
    plt.hist(array1,range = (0, parameters.elasticity_limit), bins = bins,alpha =0.5, label = 'initial distribution of Von Mises criteria')
    plt.hist(array2,range = (0, parameters.elasticity_limit), bins = bins, alpha=0.5, label = 'optimized final distribution of Von Mises criteria')
    plt.legend(loc='upper right')
    plt.xlabel('Von Mises criteria')
    plt.savefig("histo_final_2.png")
    plt.close()
