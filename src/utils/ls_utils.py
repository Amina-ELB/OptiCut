# Copyright (c) 2025 ONERA and MINES Paris, France
#
# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari

import os
from typing import Optional, Tuple

from mpi4py import MPI
from dolfinx import mesh, io
import ufl
from dolfinx import fem
import numpy as np
import math
import ufl

from fem import create_mesh

try:
    from levelset_expressions import level_set_L_shape, level_set, level_set_3D
except ImportError:

    def level_set_L_shape(x):
        """
        Calculate the level set for an L-shaped domain.

        This function computes the level set for a geometric shape consisting of a series of
        overlapping circles. The L-shape is represented by a set of level sets for these
        circles, and the function uses the `ufl.max_value` and `ufl.sqrt` functions to
        compute the level set values for a given point `x` in the domain.

        Parameters
        ----------
        x : tuple of float
            A tuple containing the coordinates (x[0], x[1]) of the point where the level set
            is evaluated. The point is expected to be in 2D space.

        Returns
        -------
        float
            The computed level set value at the point `x`. The value represents the distance
            from the point `x` to the nearest boundary of the L-shaped domain formed by the
            overlapping circles. A negative value indicates the point is inside the domain,
            and a positive value indicates the point is outside the domain.

        Notes
        -----
        The function uses a series of circles with radius `r_cercle` and an adjusted radius
        `r_cercle_modif` to form the shape. The level set is computed by iteratively
        taking the maximum of the distances to the circle boundaries.

        The function is designed to work with the `ufl` module, typically used in finite
        element methods for computational geometry and simulations.

        Example
        -------
        >>> x = (1.0, 2.0)
        >>> level_set_L_shape(x)
        -0.0037388523762586055

        See Also
        --------
        ufl.max_value, ufl.sqrt
        """
        r_cercle = 0.05
        r_cercle_modif = 0.06
        d0 = ufl.max_value(
            -(ufl.sqrt((x[0] - 0) ** 2 + (x[1] - 0) ** 2) - r_cercle_modif),
            -(ufl.sqrt((x[0] - 4 * r_cercle) ** 2 + (x[1] - 0) ** 2) - r_cercle_modif),
        )
        d1 = ufl.max_value(
            -(
                ufl.sqrt((x[0] - 8 * r_cercle) ** 2 + (x[1] - 0.0) ** 2)
                - r_cercle_modif
            ),
            d0,
        )
        d2 = ufl.max_value(
            d1,
            -(
                ufl.sqrt((x[0] - 12 * r_cercle) ** 2 + (x[1] - 0.0) ** 2)
                - r_cercle_modif
            ),
        )
        d3 = ufl.max_value(
            d2, -(ufl.sqrt((x[0] - 1) ** 2 + (x[1] - 0.0) ** 2) - r_cercle_modif)
        )

        d4 = ufl.max_value(
            d3,
            -(
                ufl.sqrt((x[0] - 0.0) ** 2 + (x[1] - 4 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )
        d5 = ufl.max_value(
            d4,
            -(
                ufl.sqrt((x[0] - 4 * r_cercle) ** 2 + (x[1] - 4 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )
        d7 = ufl.max_value(
            d5,
            -(
                ufl.sqrt((x[0] - 8 * r_cercle) ** 2 + (x[1] - 4 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )
        d8 = ufl.max_value(
            d7,
            -(
                ufl.sqrt((x[0] - 12 * r_cercle) ** 2 + (x[1] - 4 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )
        d9 = ufl.max_value(
            d8,
            -(
                ufl.sqrt((x[0] - 16 * r_cercle) ** 2 + (x[1] - 4 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )

        d10 = ufl.max_value(
            d9,
            -(
                ufl.sqrt((x[0] - 0.0) ** 2 + (x[1] - 8 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )
        d12 = ufl.max_value(
            d10,
            -(
                ufl.sqrt((x[0] - 8 * r_cercle) ** 2 + (x[1] - 8 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )
        d13 = ufl.max_value(
            d12,
            -(
                ufl.sqrt((x[0] - 12 * r_cercle) ** 2 + (x[1] - 8 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )
        d14 = ufl.max_value(
            d13,
            -(
                ufl.sqrt((x[0] - 1.0) ** 2 + (x[1] - 4 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )

        d15 = ufl.max_value(
            d14,
            -(
                ufl.sqrt((x[0] - 0.0) ** 2 + (x[1] - 12 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )
        d16 = ufl.max_value(
            d15,
            -(
                ufl.sqrt((x[0] - 4 * r_cercle) ** 2 + (x[1] - 12 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )
        d17 = ufl.max_value(
            d16,
            -(
                ufl.sqrt((x[0] - 8 * r_cercle) ** 2 + (x[1] - 12 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )
        d18 = ufl.max_value(
            d17,
            -(
                ufl.sqrt((x[0] - 4 * r_cercle) ** 2 + (x[1] - 1.0) ** 2)
                - r_cercle_modif
            ),
        )
        d19 = ufl.max_value(
            d18,
            -(
                ufl.sqrt((x[0] - 8 * r_cercle) ** 2 + (x[1] - 1.0) ** 2)
                - r_cercle_modif
            ),
        )

        d20 = ufl.max_value(
            d19,
            -(
                ufl.sqrt((x[0] - 16 * r_cercle) ** 2 + (x[1] - 8 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )
        d21 = ufl.max_value(
            d20,
            -(
                ufl.sqrt((x[0] - 16 * r_cercle) ** 2 + (x[1] - 0 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )
        d22 = ufl.max_value(
            d21,
            -(
                ufl.sqrt((x[0] - 4 * r_cercle) ** 2 + (x[1] - 16 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )
        d23 = ufl.max_value(
            d22,
            -(
                ufl.sqrt((x[0] - 8 * r_cercle) ** 2 + (x[1] - 16 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )
        d24 = ufl.max_value(
            d23,
            -(
                ufl.sqrt((x[0] - 0.0 * r_cercle) ** 2 + (x[1] - 16 * r_cercle) ** 2)
                - r_cercle_modif
            ),
        )
        return d24

    def level_set(x, parameters):
        r"""Computes a 2D level set function based on cosine functions.

        This function returns a level set value based on a combination of cosine functions
        depending on the coordinates of the input point `x = (x_0, x_1)` and a parameter `lx`.

        .. math::

            \text{level set}(x) = -\cos\left(\frac{6\pi x_0}{l_x}\right)\cos(4\pi x_1) - 0.6

        :param tuple x: A tuple representing the point coordinates (x_0, x_1).
        :param parameters: The parameters object containing the domain length `lx`.
        :type parameters: object with attribute `lx`.
        :returns: The computed level set value.
        :rtype: float
        """
        res = (
            -2
            * ufl.cos(6.0 / parameters.lx * math.pi * x[0])
            * ufl.cos(4.0 * math.pi * x[1])
            - 0.6
        )
        return res / 2

    def level_set_3D(x, parameters):
        r"""Computes a 3D level set function based on cosine functions.

        This function returns a level set value based on a combination of cosine functions
        depending on the coordinates of the input point `x = (x_0, x_1, x_2)` and a parameter `lx`, 'ly' and 'lz', the dimensions of the box containing :math:`\Omega`.

        .. math::

            \text{level set}(x) = -\cos\left(\frac{6\pi x_0}{l_x}\right)\cos(4\pi x_1)\cos(4\pi x_2)  - 0.6

        :param tuple x: A tuple representing the point coordinates (x_0, x_1, x_2).
        :param parameters: The parameters object containing the domain length `lx`, 'ly' and 'lz'.
        :type parameters: object with attribute `lx`.
        :returns: The computed level set value.
        :rtype: float
        """
        res = (
            -2
            * ufl.cos(6.0 / parameters.lx * math.pi * x[0])
            * ufl.cos(4.0 * math.pi * x[1])
            * ufl.cos(4.0 / parameters.lz * math.pi * x[2])
            - 0.6
        )
        return res / 2


def load_mesh(test_case, parameters, mesh_folder="mesh"):
    """
    Load or create a mesh based on the test case or provided mesh file.

    :param str test_case: The test case name ("rectangle", "L_shape", or "3D").
    :param Parameters parameters: The object parameters.
    :param str mesh_folder: The folder where mesh files are stored.

    :returns: The mesh object.
    :rtype: dolfinx.mesh.Mesh
    """
    mesh_filename = getattr(parameters, "mesh_file", "0")

    if mesh_filename != "0" and mesh_filename != 0:
        # User specified a custom .msh file in param.txt
        mesh_path = f"{mesh_folder}/{mesh_filename}"
        if not mesh_path.endswith(".msh"):
            mesh_path += ".msh"

        # Determine gdim based on test case (default to 2D unless explicitly 3D)
        gdim = 3 if test_case == "3D" else 2
        mesh_data = io.gmsh.read_from_msh(mesh_path, MPI.COMM_WORLD, 0, gdim=gdim)
        msh = mesh_data.mesh
        ct = mesh_data.cell_tags

        # Save XDMF for visualization
        with io.XDMFFile(
            MPI.COMM_WORLD, f"{mesh_folder}/loaded_mesh.xdmf", "w"
        ) as xdmf:
            xdmf.write_mesh(msh)

    else:
        # Generate mesh algorithmically based on dimensions
        if test_case == "rectangle":
            # Set default parameters if not provided
            if parameters.lx == 0:
                parameters.lx = 2
            if parameters.ly == 0:
                parameters.ly = 1
            parameters.lz = 0

            # Create 2D mesh
            msh = create_mesh.create_mesh_2D(
                parameters.lx,
                parameters.ly,
                int(parameters.lx / parameters.h),
                int(parameters.ly / parameters.h),
            )

        elif test_case == "L_shape":
            # Set parameters for L-shape test case
            if parameters.lx == 0:
                parameters.lx = 1
            if parameters.ly == 0:
                parameters.ly = 1
            parameters.lz = 0

            import gmsh
            import numpy as np
            import basix.ufl as bufl
            from dolfinx.io.gmsh import model_to_mesh

            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            factory = gmsh.model.geo
            lc = parameters.h
            L = parameters.lx
            H = 0.4 * L

            p1 = factory.addPoint(0, 0, 0, lc, 1)
            p2 = factory.addPoint(L, 0, 0, lc, 2)
            p3 = factory.addPoint(L, H, 0, lc, 3)
            p4 = factory.addPoint(H, H, 0, lc, 4)
            p5 = factory.addPoint(H, L, 0, lc, 5)
            p6 = factory.addPoint(0, L, 0, lc, 6)

            l1 = factory.addLine(1, 2)
            l2 = factory.addLine(2, 3)
            l3 = factory.addLine(3, 4)
            l4 = factory.addLine(4, 5)
            l5 = factory.addLine(5, 6)
            l6 = factory.addLine(6, 1)

            loop = factory.addCurveLoop([l1, l2, l3, l4, l5, l6])
            surface = factory.addPlaneSurface([loop])

            gmsh.model.geo.synchronize()
            gmsh.model.addPhysicalGroup(2, [surface], 1)
            gmsh.model.setPhysicalName(2, 1, "Domain")
            gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
            gmsh.option.setNumber("Mesh.VolumeEdges", 0)
            gmsh.model.mesh.generate(2)

            mesh_data = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
            msh_gmsh = mesh_data.mesh
            gmsh.finalize()

            # Rebuild as native FEniCSx mesh to avoid equispaced coordinate
            # element mismatch with CutFEMx runintgen JIT compilation.
            cell_name = msh_gmsh.topology.cell_name()
            c_el = bufl.element("P", cell_name, 1, shape=(msh_gmsh.geometry.dim,))
            domain = ufl.Mesh(c_el)
            cells = msh_gmsh.topology.connectivity(2, 0).array.reshape(-1, 3).astype(np.int64)
            x_coords = np.array(
                msh_gmsh.geometry.x[:, :msh_gmsh.geometry.dim],
                dtype=np.float64,
                copy=True,
            )
            msh = mesh.create_mesh(MPI.COMM_WORLD, cells, domain, x_coords)


        elif test_case == "3D":
            # Set parameters for 3D test case
            if getattr(parameters, "lx", 0) == 0:
                parameters.lx = 2
            if getattr(parameters, "ly", 0) == 0:
                parameters.ly = 1
            if getattr(parameters, "lz", 0) == 0:
                parameters.lz = 1
            # Create 3D mesh
            msh = create_mesh.create_mesh_3D(
                parameters.lx,
                parameters.ly,
                parameters.lz,
                int(parameters.lx / parameters.h),
                int(parameters.ly / parameters.h),
                int(parameters.lz / parameters.h),
            )

        else:
            raise ValueError(f"Test case '{test_case}' not implemented")

    # Create connectivity for the mesh
    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)

    return msh


def init_level_set(msh: mesh.Mesh, parameters, test_case: str) -> fem.Function:
    r"""
    Initialize the level-set function defined on the mesh.

    The function returns a :class:`dolfinx.fem.Function` defined in a standard
    Lagrange space (degree 1). The level-set expression is interpolated onto
    the function space.

    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
        Mesh where the level-set is defined.
    parameters : Parameters
        Parameters object used by analytic expressions (if required).
    test_case : str
        Identifier of the test case to choose the analytic level-set.

    Returns
    -------
    dolfinx.fem.Function
        The level-set function interpolated on V_ls.
    """
    V_ls = fem.FunctionSpace(msh, ("Lagrange", 1))
    x = ufl.SpatialCoordinate(msh)

    if test_case == "L_shape":
        expr_ufl = level_set_L_shape(x)
    elif test_case == "rectangle":
        expr_ufl = level_set(x, parameters)
    elif test_case == "3D":
        expr_ufl = level_set_3D(x, parameters)
    else:
        raise RuntimeError(
            f"Test case '{test_case}' not implemented for level set initialization."
        )

    expr = fem.Expression(expr_ufl, V_ls.element.interpolation_points())
    ls_func = fem.Function(V_ls)
    ls_func.interpolate(expr)
    return ls_func
