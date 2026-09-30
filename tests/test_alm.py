import pytest
import numpy as np
from optimization import almMethod
from config.parameters import Parameters

class MockParameters:
    """A simple mock to avoid interactive inputs from the real Parameters class if any."""
    def __init__(self):
        self.ALM = 1
        self.type_constraint = 'inequality'
        self.ALM_lagrangian_multiplicator = 0.0
        self.ALM_penalty_parameter = 1.0
        self.ALM_slack_variable = 0.0
        self.ALM_penalty_limit = 100.0
        self.ALM_penalty_coef_multiplicator = 1.1

def test_alm_slack_update():
    params = MockParameters()
    params.ALM = 1
    params.type_constraint = 'inequality'
    params.ALM_lagrangian_multiplicator = 1.0
    params.ALM_penalty_parameter = 10.0
    
    # Case 1: lambda/mu + C > 0 -> slack = 0
    # lambda/mu = 0.1, C = 0.5 -> 0.1 + 0.5 = 0.6 > 0 -> max(0, -0.6) = 0
    almMethod.maj_param_constraint_optim_slack(params, 0.5)
    assert params.ALM_slack_variable == 0.0
    
    # Case 2: lambda/mu + C < 0 -> slack > 0
    # lambda/mu = 0.1, C = -0.5 -> 0.1 - 0.5 = -0.4 < 0 -> max(0, -(-0.4)) = 0.4
    almMethod.maj_param_constraint_optim_slack(params, -0.5)
    assert np.isclose(params.ALM_slack_variable, 0.4)

def test_alm_param_update():
    params = MockParameters()
    params.ALM = 1
    params.ALM_lagrangian_multiplicator = 1.0
    params.ALM_penalty_parameter = 10.0
    params.ALM_slack_variable = 0.5
    params.ALM_penalty_limit = 100.0
    params.ALM_penalty_coef_multiplicator = 2.0
    
    rest_constraint = 0.1
    # lambda_new = 1.0 + 10.0 * (0.1 + 0.5) = 1.0 + 6.0 = 7.0
    # mu_new = min(100.0, 2.0 * 10.0) = 20.0
    almMethod.maj_param_constraint_optim(params, rest_constraint)
    
    assert np.isclose(params.ALM_lagrangian_multiplicator, 7.0)
    assert np.isclose(params.ALM_penalty_parameter, 20.0)

def test_alm_init():
    params = MockParameters()
    params.ALM = 1
    
    cost_derivative = 1000.0
    constraint_derivative = 10.0
    k = 10  # This will be used to set the limit for the penalty parameter
    # lambda = cost_der / cons_der = 1000 / 10 = 100
    # mu = cost_der / cons_der^2 = 1000 / 100 = 10
    # limit = k * mu = 100
    almMethod.init_param_constraint_optim(constraint_derivative, params, cost_derivative, k = 10)
    
    assert np.isclose(params.ALM_lagrangian_multiplicator, 100.0)
    assert np.isclose(params.ALM_penalty_parameter, 10.0)
    assert np.isclose(params.ALM_penalty_limit, 100.0)

def test_alm_disabled():
    """Verify that no changes occur if ALM is disabled."""
    params = MockParameters()
    params.ALM = 0
    params.ALM_lagrangian_multiplicator = 1.0
    
    almMethod.maj_param_constraint_optim(params, 0.1)
    assert params.ALM_lagrangian_multiplicator == 1.0
