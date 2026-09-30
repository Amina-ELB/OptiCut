# Copyright (c) 2025 ONERA and MINES Paris, France
#
# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari
#
# Migration note: updated to CutFEMx 0.2 (DOLFINx 0.11)
#   - Reinitialization class replaced by cutfemx.distance.reinitialize()
#   - Advection class: removed unused CutFEMx imports (cut_entities, create_cut_mesh, etc.)


import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
import matplotlib.pyplot as plt
from ufl import ds, dx, grad, inner, tr, dS

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc

from typing import TYPE_CHECKING

from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square, meshtags
from petsc4py.PETSc import ScalarType
from ufl import (
    FacetNormal,
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    div,
    dot,
    dx,
    grad,
    inner,
    lhs,
    rhs,
    dc,
    FacetNormal,
    CellDiameter,
    dot,
    avg,
    jump,
)
from dolfinx.io import XDMFFile

# CutFEMx 0.2 imports
import cutfemx
from cutfemx.distance import reinitialize

from utils import mechanics_tool

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
        """
        self.level_set = fem.Function(space)
        self.level_set.x.array[:] = level_set.x.array
        self.space = space

    def set_level_set(self, level_set):
        """Update the level-set function."""
        self.level_set.x.array[:] = level_set.x.array
        self.level_set.x.scatter_forward()


class Advection(LevelSet):
    """
    Class to perform level-set advection using a standard FEM scheme (no CutFEM).
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
            Time step for advection (default: 1e-3).
        """
        super().__init__(level_set, V_ls)

        self.dt = dt
        self.V_ls = V_ls
        self.mesh = V_ls.mesh

        self.phi_n = ufl.TrialFunction(self.V_ls)
        self.phi_test = ufl.TestFunction(self.V_ls)

        self.velocity_field = fem.Function(self.V_ls)
        self.velocity_field.x.array[:] = 1e-3

        self.h = ufl.CellDiameter(self.mesh)
        self.dS = ufl.dS
        self.n = ufl.FacetNormal(self.mesh)
        self.const = 1e-3

        self.fast_jit = {"cffi_extra_compile_args": ["-O0", "-w"]}
        self.dt_const = fem.Constant(self.mesh, PETSc.ScalarType(self.dt))

        a_adv = ufl.dot(self.phi_n, self.phi_test) * ufl.dx
        a_adv += (
            avg(self.const)
            * avg(self.h) ** 3
            * ufl.inner(
                ufl.jump(ufl.grad(self.phi_n), self.n),
                ufl.jump(ufl.grad(self.phi_test), self.n),
            )
            * self.dS
        )
        a_std_adv = fem.form(a_adv, jit_options=self.fast_jit)
        self.A_adv = fem.petsc.assemble_matrix(a_std_adv)
        self.A_adv.assemble()

        L_adv = (
            ufl.dot(self.level_set, self.phi_test) * ufl.dx
            + ufl.dot(
                -self.dt_const
                * self.velocity_field
                * ufl.sqrt(
                    ufl.inner(ufl.grad(self.level_set), ufl.grad(self.level_set))
                ),
                self.phi_test,
            )
            * ufl.dx
        )
        self.L_std_adv = fem.form(L_adv, jit_options=self.fast_jit)
        self.b_adv = fem.petsc.create_vector([self.V_ls])
        fem.petsc.assemble_vector(self.b_adv, self.L_std_adv)
        self.b_adv.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        self.b_adv.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        self.solver_adv = PETSc.KSP().create(self.mesh.comm)
        self.solver_adv.setOperators(self.A_adv)
        self.solver_adv.setType(PETSc.KSP.Type.PREONLY)
        pc = self.solver_adv.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
        self.solver_adv.setUp()

        self.sol_adv = fem.Function(self.V_ls)

    def cut_fem_adv(self, velocity_field, dt):
        r"""Perform one advection step of the level-set function."""
        self.velocity_field.x.array[:] = velocity_field.x.array
        self.velocity_field.x.scatter_forward()

        self.dt = dt
        self.dt_const.value = dt

        with self.b_adv.localForm() as loc:
            loc.set(0.0)
        fem.petsc.assemble_vector(self.b_adv, self.L_std_adv)
        self.b_adv.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        self.b_adv.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        self.solver_adv.solve(self.b_adv, self.sol_adv.x.petsc_vec)
        self.sol_adv.x.scatter_forward()

        self.level_set.x.array[:] = self.sol_adv.x.array
        self.level_set.x.scatter_forward()

        return self.sol_adv


############################################
# Reinitialization — CutFEMx 0.2 API
############################################


class Reinitialization(LevelSet):
    r"""Reinitialization wrapper using cutfemx.distance.reinitialize().

    Replaces the old Predictor-Corrector scheme that relied on the now-obsolete
    cut_form / update_runtime_domains API. The new CutFEMx 0.2 exposes a single
    ``cutfemx.distance.reinitialize(phi)`` call that handles signed-distance
    reinitialization internally.

    The public interface (``reinitializationPC``, ``reinitializationPC_inplace``)
    is kept identical so that callers in main.py do not need to change.
    """

    def __init__(self, level_set, V_ls, l):
        """
        Parameters
        ----------
        level_set : fem.Function
            Initial level-set function.
        V_ls : fem.FunctionSpace
            Function space of the level-set.
        l : float
            Characteristic length (kept for interface compatibility; not used
            by the new reinitialize() backend).
        """
        super().__init__(level_set, V_ls)
        self.mesh = level_set.function_space.mesh
        self.l = l
        self.V_ls = V_ls
        self.dim = self.mesh.topology.dim

    # ------------------------------------------------------------------
    # Public interface (same as before)
    # ------------------------------------------------------------------

    def reinitializationPC(self, level_set, step_reinit):
        r"""Return a reinitialized level-set (signed-distance field).

        :param fem.Function level_set: The level-set function :math:`\phi`.
        :param int step_reinit: Ignored (kept for API compatibility with old PC scheme).

        :returns: Reinitialized level-set.
        :rtype: fem.Function
        """
        self.level_set.x.array[:] = level_set.x.array
        self.level_set.x.scatter_forward()

        # CutFEMx 0.2: one call handles everything
        reinitialize(self.level_set)
        self.level_set.x.scatter_forward()

        # Propagate result back to the caller's function
        level_set.x.array[:] = self.level_set.x.array
        level_set.x.scatter_forward()
        return level_set

    def reinitializationPC_inplace(self, level_set, step_reinit):
        """Reinitialize in-place (modifies level_set directly).

        :param fem.Function level_set: The level-set function to reinitialize.
        :param int step_reinit: Ignored (kept for API compatibility).
        """
        reinitialize(level_set)
        level_set.x.scatter_forward()
