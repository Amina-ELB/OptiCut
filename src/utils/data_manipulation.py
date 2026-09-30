# Copyright (c) 2025 ONERA and MINES Paris, France
#
# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari


import numpy as np
import math
import ufl

from dolfinx import fem
import matplotlib.pyplot as plt


from typing import TYPE_CHECKING


import dolfinx.fem.petsc
from mpi4py import MPI


from config.parameters import *
from utils.ls_utils import *
from solvers.ersatz_elastic_solver import *
from solvers.cutfem_elastic_solver import *
from levelset.levelSet_tool import *
from levelset.velocity_tools import *


from utils import mechanics_tool


def _get_points(space_or_element):
    elem = getattr(space_or_element, "element", space_or_element)
    pts = getattr(elem, "interpolation_points", None)
    return pts() if callable(pts) else pts


class VonMisesCalculator:
    def __init__(self, msh, V_ls, Q, uh, lame_mu, lame_lambda, level_set):
        self.msh = msh
        self.V_ls = V_ls
        self.Q = Q

        # Pre-allocate functions to avoid memory leaks
        self.xsi_vm_func_ls = fem.Function(V_ls)
        self.xsi_vm_func_Q = fem.Function(Q)
        self.vm = fem.Function(Q)
        self.vm_test = fem.Function(Q)

        dim = self.msh.topology.dim

        # Cache expressions to avoid FFI pointer corruption / memory leaks over many iterations
        xsi_vm = ufl.conditional(ufl.le(level_set, 0), 1, 0)
        self.xsi_expr = fem.Expression(xsi_vm, _get_points(self.V_ls))

        xsi_vm_Q = ufl.conditional(ufl.eq(self.xsi_vm_func_ls, 0), 0, 1)
        self.xsi_expr_Q = fem.Expression(xsi_vm_Q, _get_points(self.Q))

        vm_val = mechanics_tool.von_mises(uh, lame_mu, lame_lambda, dim)
        self.vm_expr = fem.Expression(vm_val, _get_points(self.Q))

        vm_masked = self.xsi_vm_func_Q * self.vm
        self.vm_masked_expr = fem.Expression(vm_masked, _get_points(self.Q))

    def compute(self):
        self.xsi_vm_func_ls.interpolate(self.xsi_expr)
        self.xsi_vm_func_Q.interpolate(self.xsi_expr_Q)
        self.vm.interpolate(self.vm_expr)
        self.vm_test.interpolate(self.vm_masked_expr)

        return self.vm_test


def create_list_vm(
    msh, uh, parameters, lame_mu, lame_lambda, i, level_set, V_ls, Q, domain_marker=0
):
    # Legacy function - use VonMisesCalculator for iterative calls
    calculator = VonMisesCalculator(msh, V_ls, Q, uh, lame_mu, lame_lambda, level_set)
    return calculator.compute()


def save_data_compliance(
    folder,
    ls_func,
    ls_space,
    xsi,
    velocity,
    velocity_space,
    primal_sol,
    primal_space,
    dual_sol,
    vonMises,
    vonMises_space,
    time,
):

    xsi_expr = fem.Expression(xsi, _get_points(ls_space))
    xsi = fem.Function(ls_space)
    xsi.interpolate(xsi_expr)

    velocity_expr = fem.Expression(velocity, _get_points(velocity_space))
    velocity = fem.Function(velocity_space)
    velocity.interpolate(velocity_expr)

    uh_expr = fem.Expression(primal_sol, _get_points(primal_space))
    primal_sol = fem.Function(primal_space)
    primal_sol.interpolate(uh_expr)
    ph_expr = fem.Expression(dual_sol, _get_points(primal_space))
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


def save_data(
    folder,
    ls_func,
    ls_space,
    xsi,
    velocity,
    velocity_space,
    primal_sol,
    primal_space,
    dual_sol,
    vonMises,
    vonMises_space,
    time,
):

    xsi_func = fem.Function(velocity_space)
    xsi_func.interpolate(xsi)

    velocity_func = fem.Function(velocity_space)
    velocity_func.interpolate(velocity)

    # vm_expr = fem.Expression(xsi_vm_func*vm, vonMises_space.element.interpolation_points())
    # vm_cut = fem.Function(vonMises_space)
    # vm_cut.interpolate(vm_expr)

    ls_func.name = "ls_func"
    xsi_func.name = "xsi"
    velocity_func.name = "velocity"
    primal_sol.name = "displacement"
    dual_sol.name = "sol_dual"

    folder.write_function(ls_func, time)
    folder.write_function(xsi_func, time)
    folder.write_function(velocity_func, time)
    folder.write_function(primal_sol, time)
    folder.write_function(dual_sol, time)


def histogram(array, bins, iteration):
    val_max = np.max(array)
    print("val max = ", np.max(array))
    plt.hist(array, range=(0, val_max), bins=bins)
    plt.xlabel("Von Mises criteria")
    plt.savefig("histo_ite" + str(iteration) + ".png")
    plt.close()


def histogram_final_1(array1, array2, bins, parameters):
    max_1 = np.max(array1)
    max_2 = np.max(array2)
    val_max = max(max_1, max_2)
    plt.hist(
        array1,
        range=(parameters.elasticity_limit, val_max),
        bins=bins,
        alpha=0.5,
        label="initial distribution of Von Mises criteria",
    )
    plt.hist(
        array2,
        range=(parameters.elasticity_limit, val_max),
        bins=bins,
        alpha=0.5,
        label="optimized final distribution of Von Mises criteria",
    )
    plt.legend(loc="upper right")
    plt.xlabel("Von Mises criteria")
    plt.savefig("histo_final_1.png")
    plt.close()


def histogram_final_2(array1, array2, bins, parameters):
    max_1 = np.max(array1)
    max_2 = np.max(array2)
    val_max = min(max_1, max_2)
    plt.hist(
        array1,
        range=(0, parameters.elasticity_limit),
        bins=bins,
        alpha=0.5,
        label="initial distribution of Von Mises criteria",
    )
    plt.hist(
        array2,
        range=(0, parameters.elasticity_limit),
        bins=bins,
        alpha=0.5,
        label="optimized final distribution of Von Mises criteria",
    )
    plt.legend(loc="upper right")
    plt.xlabel("Von Mises criteria")
    plt.savefig("histo_final_2.png")
    plt.close()
