import numpy as np

from math import *

from decimal import Decimal

def maj_param_constraint_optim_slack(parameters,rest_constraint):
    r"""Update the slack parameter for inequality constraint, as the following equation:

    .. math::

            s^{n+1} = \text{max}(0,-( \frac{\lambda_{ALM}^{n}}{\mu_{ALM}^{n}}+C(\Omega^{n}) ) )
        
                
    :param Parameter parameters: The parameter object.
    :param float rest_constraint: The value of :math:`C(\Omega^{n})`.

    """
    if parameters.ALM == 1:
        if parameters.type_constraint == 'inequality':
            # 1// update of slack parameter:
            parameters.ALM_slack_variable = np.maximum(0,-(parameters.ALM_lagrangian_multiplicator/parameters.ALM_penalty_parameter + float(rest_constraint)))
            

def maj_param_constraint_optim(parameters,rest_constraint):
    r"""Update the Augmented Lagrangian parameters of the Parameter object.

        1. *Update the Lagrange multiplier :* 
        
        .. math::

                \lambda_{ALM}^{n+1} = \lambda_{ALM}^{n} + \mu_{ALM}^{n}(C(\Omega^{n}) + s^{n+1})


        2. *Update the Lagrange penalty parameter :*
        
        
        .. math::

            \mu_{ALM}^{n+1} = \text{min}(\overline{\mu},c \mu_{ALM}^{n})

            

        where :math:`\overline{\mu}` is the limit of the penalty parameter and :math:`c` is the penalty multiplier coefficient.

           
        :param Parameter parameters: The parameter object.
        :param float rest_constraint: The value of :math:`C(\Omega)`.

        """
    if parameters.ALM == 1:
        # 1// update of lagrangian mutiplicator
        parameters.ALM_lagrangian_multiplicator = parameters.ALM_lagrangian_multiplicator + \
            parameters.ALM_penalty_parameter*(rest_constraint + parameters.ALM_slack_variable)
        # 2// update of penalization parameter
        parameters.ALM_penalty_parameter = min(parameters.ALM_penalty_limit,parameters.ALM_penalty_coef_multiplicator*parameters.ALM_penalty_parameter)

def init_param_constraint_optim(constraint,parameters,cost,denom=100):
    r"""Initialized the Augmented Lagrangian parameters:

        1. *Initialization of the Lagrange multiplier :*

        .. math::

                \lambda_{ALM}^{0} = \frac{10^{k}C(\Omega^{0})}{D}

        with :math:`k=10^{\text{round}(\log_{10}(\text{cost}))}` 
        
        2. *Initialization of the Lagrange penalty parameter :*

        .. math::

                \mu_{ALM}^{0} = \frac{10^{k}C(\Omega^{0})}{D}

        with :math:`k=10^{\text{round}(\log_{10}(\text{cost}))}` 
        
        3. *Initialization of the penalty limit :*

        .. math::

                \overline{\mu}_{ALM}^{0} = \text{denom}*\frac{10^{k}C(\Omega^{0})}{D}

        with :math:`k=10^{\text{round}(\log_{10}(\text{cost}))}` 

        
        .. note::
            The set of simulations we conducted allowed us to determine a value for D of approximately 100.


        :param Parameter parameters: The parameter object.
        :param float cost: The cost value :math:`J(\Omega^{0})` at the initialization.
        :param float denom: The D value, with a default value of :math:`100`.

        """
    cost_power = round(Decimal(cost).log10())  
    cost_decimal =  float(Decimal(10) ** cost_power)
    if parameters.ALM == 1:
        parameters.ALM_lagrangian_multiplicator = (cost_decimal*constraint)/(denom)
        parameters.ALM_penalty_parameter = (cost_decimal*constraint)/(denom)
        parameters.ALM_penalty_limit = 1000*(cost_decimal*constraint)/(denom)


