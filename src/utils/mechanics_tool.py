# Copyright (c) 2025 ONERA and MINES Paris, France
#
# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari


import ufl


def strain(v):
    r"""Computes the strain tensor for the elasticity law:

    .. math::

        \varepsilon(u) = \frac{1}{2}(\nabla\cdot u + \nabla^{T} \cdot u).


    :param fem.Function u: The displacement field.

    :returns: Expression of the strain tensor.
    :rtype: fem.Expression

    """
    return ufl.sym(ufl.grad(v))


def stress(u, lame_mu, lame_lambda, dim):
    r"""Computes the stress tensor for the elasticity law

    .. math::

        \sigma(u) = \lambda  \nabla\cdot u * \text{Id} + 2\mu * \varepsilon(u)



    with :math:`\varepsilon(u)` computed with the function :func:`mechanics_tool.strain`.

    :param fem.Function u: The displacement field.
    :param float lame_mu: the :math:`\mu` Lame coefficient.
    :param float lame_lambda: the :math:`\lambda` Lame coefficient.
    :param int dim: the dimension of the displacement field.

    :returns: Expression of the stress tensor.
    :rtype: fem.Expression

    """
    return lame_lambda * ufl.nabla_div(u) * ufl.Identity(dim) + 2 * lame_mu * strain(u)


def lame_compute(E, v):
    r"""Computes the Lamé coefficients from Young modulus and Poisson coefficient:

    .. math::
        \begin{aligned}
        \mu &= \frac{E}{2(1+\nu)}  \\
        \lambda &= \frac{E\nu}{(1+\nu)(1-2\nu)}
        \end{aligned}

    with :math:`\sigma(u)` computed with the function :func:`mechanics_tool.stress`.
        

    :param float E: The Young modulus.
    :param float v: The Poisson coefficient.
    
    :returns: Lame :math:`\mu` and Lame :math:`\lambda` coefficients.
    :rtype: float, float
        
    """
    lame_mu = E / (2.0 * (1.0 + v))
    lame_lambda = E * v / ((1.0 + v) * (1.0 - 2.0 * v))
    return lame_mu, lame_lambda


def von_mises(u, lame_mu, lame_lambda, dim):
    r"""Computes the Von Mises stress:

    .. math::

        \sigma_{VM} = \sigma(u) - \frac{1}{3}\text{Tr}(\sigma(u))\text{Id}


    with :math:`\sigma(u)` compute with the function :func:`mechanics_tool.stress`.


    :param fem.Function u: The displacement field function.
    :param float lame_mu: The Lame :math:`\mu` coefficient.
    :param float lame_lambda: The lame :math:`\lambda` coefficient.
    :param float dim: The dimension of the displacement field.

    :returns: The value of the Von Mises stress constraint.
    :rtype: fem.Function

    """
    s = stress(u, lame_mu, lame_lambda, dim) - (1.0 / 3) * ufl.tr(
        stress(u, lame_mu, lame_lambda, dim)
    ) * ufl.Identity(dim)
    r = (2.0 / 3) * ufl.inner(s, s)
    return ufl.sqrt(r)


def project_von_mises(u, lame_mu, lame_lambda, mesh, measure=None):
    import dolfinx.fem as fem

    dim = mesh.geometry.dim
    V_DG = fem.functionspace(mesh, ("DG", 0))
    vm_expr = von_mises(u, lame_mu, lame_lambda, dim)

    expr = fem.Expression(vm_expr, V_DG.element.interpolation_points)
    vm_func = fem.Function(V_DG)
    vm_func.interpolate(expr)
    return vm_func
