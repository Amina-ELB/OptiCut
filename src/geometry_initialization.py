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


###################################
####  Level set's definition   ####
###################################
def circle(x):
    r"""Computes the signed distance from a point to a circle.

    This function calculates the signed distance from a point `x = (x_0, x_1)` to a circle 
    centered at `(1, 0.5)` with radius `0.3`. The function returns a negative value if 
    the point is inside the circle and a positive value if it is outside.

    .. math::

        \text{distance}(x) = -\left(\sqrt{(x_0 - 1)^2 + (x_1 - 0.5)^2} - 0.3\right)

    :param tuple x: A tuple representing the point coordinates (x_0, x_1).
    :returns: The signed distance from the point to the circle.
    :rtype: float
    """
    return -(np.sqrt((x[0]- 1)**2 + (x[1] - 0.5)**2) - 0.3)


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
    res = -2*ufl.cos(6.0/parameters.lx*math.pi*x[0]) * ufl.cos(4.0*math.pi*x[1]) -0.6
    return res/2

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
    res = -2*ufl.cos(6.0/parameters.lx*math.pi*x[0]) * ufl.cos(4.0*math.pi*x[1]) -0.6
    return res/2

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
    d0 = ufl.max_value(-(ufl.sqrt((x[0]- 0)**2 + (x[1] - 0)**2) - r_cercle_modif),-(ufl.sqrt((x[0]- 4*r_cercle)**2 + (x[1] - 0)**2) - r_cercle_modif))
    d1 = ufl.max_value(-(ufl.sqrt((x[0]- 8*r_cercle)**2 + (x[1] - 0.0)**2) - r_cercle_modif),d0)
    d2 = ufl.max_value(d1,-(ufl.sqrt((x[0]- 12*r_cercle)**2 + (x[1] - 0.)**2) - r_cercle_modif))
    d3 = ufl.max_value(d2,-(ufl.sqrt((x[0]- 1)**2 + (x[1] - 0.0)**2) - r_cercle_modif))
    
    d4 = ufl.max_value(d3,-(ufl.sqrt((x[0]- 0.)**2 + (x[1] - 4*r_cercle)**2) - r_cercle_modif))
    d5 = ufl.max_value(d4,-(ufl.sqrt((x[0]- 4*r_cercle)**2 + (x[1] - 4*r_cercle)**2) - r_cercle_modif))
    d7 = ufl.max_value(d5,-(ufl.sqrt((x[0]- 8*r_cercle)**2 + (x[1] - 4*r_cercle)**2) - r_cercle_modif))
    d8 = ufl.max_value(d7,-(ufl.sqrt((x[0]- 12*r_cercle)**2 + (x[1] - 4*r_cercle)**2) - r_cercle_modif))
    d9 = ufl.max_value(d8,-(ufl.sqrt((x[0]- 16*r_cercle)**2 + (x[1] - 4*r_cercle)**2) - r_cercle_modif))
    
    d10 = ufl.max_value(d9,-(ufl.sqrt((x[0]- 0.)**2 + (x[1] - 8*r_cercle)**2) - r_cercle_modif))
    #d11 = ufl.max_value(d10,-(ufl.sqrt((x[0]- 4*r_cercle)**2 + (x[1] - 8*r_cercle)**2) - r_cercle_modif))
    d12 = ufl.max_value(d10,-(ufl.sqrt((x[0]- 8*r_cercle)**2 + (x[1] - 8*r_cercle)**2) - r_cercle_modif))
    d13 = ufl.max_value(d12,-(ufl.sqrt((x[0]- 12*r_cercle)**2 + (x[1] - 8*r_cercle)**2) - r_cercle_modif))
    d14 = ufl.max_value(d13,-(ufl.sqrt((x[0]- 1.)**2 + (x[1] - 4*r_cercle)**2) - r_cercle_modif))

    d15 = ufl.max_value(d14,-(ufl.sqrt((x[0]- 0.)**2 + (x[1] - 12*r_cercle)**2) - r_cercle_modif))
    d16 = ufl.max_value(d15,-(ufl.sqrt((x[0]- 4*r_cercle)**2 + (x[1] - 12*r_cercle)**2) - r_cercle_modif))
    d17 = ufl.max_value(d16,-(ufl.sqrt((x[0]- 8*r_cercle)**2 + (x[1] - 12*r_cercle)**2) - r_cercle_modif))
    d18 = ufl.max_value(d17,-(ufl.sqrt((x[0]- 4*r_cercle)**2 + (x[1] - 1.)**2) - r_cercle_modif))
    d19 = ufl.max_value(d18,-(ufl.sqrt((x[0]- 8*r_cercle)**2 + (x[1] - 1.)**2) - r_cercle_modif))
    
    d20 = ufl.max_value(d19,-(ufl.sqrt((x[0]- 16*r_cercle)**2 + (x[1] - 8*r_cercle)**2) - r_cercle_modif))
    d21 = ufl.max_value(d20,-(ufl.sqrt((x[0]- 16*r_cercle)**2 + (x[1] - 0*r_cercle)**2) - r_cercle_modif))
    d22 = ufl.max_value(d21,-(ufl.sqrt((x[0]- 4*r_cercle)**2 + (x[1] - 16*r_cercle)**2) - r_cercle_modif))
    d23 = ufl.max_value(d22,-(ufl.sqrt((x[0]- 8*r_cercle)**2 + (x[1] - 16*r_cercle)**2) - r_cercle_modif))
    d24 = ufl.max_value(d23,-(ufl.sqrt((x[0]- 0.*r_cercle)**2 + (x[1] - 16*r_cercle)**2) - r_cercle_modif))
    return d24
