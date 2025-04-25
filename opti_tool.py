# Copyright (c) 2025 ONERA and MINES Paris, France 

# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari 


import numpy as np
from mpi4py import MPI

def lagrangian_cost(cost_value,constraint_value,parameters):
    r""" Compute the Lagrangian cost, which is the sum of the cost function and the constraint term using the Augmented Lagrangian Method (ALM).
    
    With the ALM method, it is defined as: 

    .. math::

            \mathcal{L}(\Omega^{n}) = J(\Omega^{n}) + \lambda_{ALM}^{n} \left( C(\Omega^{n}) + s_{ALM}^{n} \right) + \frac{\mu_{ALM}^{n}}{2} \left( C(\Omega^{n}) s_{ALM}^{n} \right)^{2}.
        
    :param float cost_value: The cost value, :math:`J(\Omega^{n})`.
    :param float constraint_value: The value of :math:`C(\Omega^{n})`.
    :param Parameters parameters: The parameter object.

    :returns: The Lagrangian cost.
    :rtype: float.

    """

    lagrangian = cost_value \
        + parameters.ALM*(parameters.ALM_lagrangian_multiplicator * (constraint_value + parameters.ALM_slack_variable)\
        + parameters.ALM_penalty_parameter/2 * (constraint_value + parameters.ALM_slack_variable)**2) \
        + (1-parameters.ALM)*parameters.target_constraint
    return lagrangian

def adapt_c_HJ(c,crit,tol,lagrangian):
    r""" Automatically compute the parameter :math:`c` for time step :math:`dt` adaptation  
        using the cost values from the previous three iterations, the three previous Lagrangian cost values,  
        and a tolerance value denoted as `tol`.  
            
        :param float c: The previous value of the parameter `c`.  
        :param np.array crit: The cost values from the last three iterations.  
        :param np.array lagrangian: The Lagrangian cost values from the last three iterations.  

        :returns: The updated value of `c`.  
        :rtype: float.  
    """
    tol_temp = tol*10
    if crit[0]<tol_temp and crit[1]<tol_temp and crit[2]<tol_temp :
        new_c = 0.25
        return new_c
    elif abs((lagrangian[0]-lagrangian[2])/lagrangian[0])<tol:
        new_c = 0.25
        return new_c
    else:
        return 0.5
    
def adapt_dt(_lagrangian_cost,lagrangian_cost_previous,max_velocity,parameters,c):
    """
    Compute the adaptive time step (dt) based on compliance with the Lagrangian cost evolution.

    The time step is scaled using a factor `c` and the ratio of `parameters.h` to `max_velocity`.
    The minimum value of `dt` is taken to ensure numerical stability.

    :param float _lagrangian_cost: The current Lagrangian cost (not used in the function but likely relevant for future modifications).
    :param float lagrangian_cost_previous: The Lagrangian cost from the previous iteration.
    :param float max_velocity: The maximum velocity in the system.
    :param Parameters parameters: An object containing various simulation parameters, including `h`.
    :param float c: A scaling factor to control the time step size.

    :returns: The adapted time step `dt`.
    :rtype: float
    """
    dt = c * (parameters.h/max_velocity)
    dt = np.min(dt)
    return dt

def adapt_HJ(_lagrangian_cost,lagrangian_cost_previous,j_max,dt,parameters):
    """
    Compute an adaptive j_max parameter for number iteration of advection equation.

    The function calculates the shape derivative using the difference in Lagrangian costs
    and adjusts the j_max value within a bounded range.

    .. note::
            For Non linear problem like the minimization of Lp norm of Von Mises constraint the bounded range is more restrictive. 

    :param float _lagrangian_cost: The current Lagrangian cost.
    :param float lagrangian_cost_previous: The Lagrangian cost from the previous iteration.
    :param int j_max: The maximum iteration step index.
    :param float dt: The time step size.
    :param dict parameters: Additional parameters (not used in function but included for extensibility).
    :param any c: Additional argument (not used in function but included for extensibility).
    
    :returns: A computed adaptation value between 1 and 10, based on the shape derivative.
    :rtype: int
    """

    if parameters.cost_func == "compliance":
        shape_derivative = (_lagrangian_cost-lagrangian_cost_previous)/(j_max*dt)
        res = max(int(-(shape_derivative))/100,1)
        res = min(res,10)
    else: 
        shape_derivative = (_lagrangian_cost-lagrangian_cost_previous)/(j_max*dt)
        res = max(int(-(shape_derivative)),1)
        res = min(res,1)
    return int(res)

def catch_NAN(cost,lagrangian_cost,rest_constraint,dt,adv_bool):
    """
    Handle and check for potential NaN (Not a Number) or very small values in the input parameters.

    The function checks if the cost, Lagrangian cost, and rest constraint are all close to zero, 
    indicating a potential numerical issue (NaN or very small values). If the conditions are met 
    and the `adv_bool` parameter is greater than or equal to 1, the function returns the time step `dt` 
    and a zero value. Otherwise, it returns `dt` and twice the value of `adv_bool`.

    :param float cost: The current cost value.
    :param float lagrangian_cost: The current Lagrangian cost.
    :param float rest_constraint: The rest constraint value.
    :param float dt: The time step size.
    :param int adv_bool: A boolean-like value (1 or 0) indicating whether to perform an adaptation.

    :returns: A tuple with the time step `dt` and either 0 or `2 * adv_bool`, based on the conditions.
    :rtype: tuple
    """
    tol = 10**32
    if (cost<tol and abs(lagrangian_cost)<tol and abs(rest_constraint)<tol) and adv_bool>=1:
        return dt, 0
    else:
        return dt, 2*adv_bool

