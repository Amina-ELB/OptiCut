# Extending OptiCut

OptiCut is designed to be easily extensible. To implement a new physical problem or a new optimization objective, you do not need to modify the core solvers or the Augmented Lagrangian Method (ALM) framework. Instead, you only need to create a new class that represents your formulation.

## The BaseProblem Concept

The problem classes in OptiCut handle all standard computational operations, including:
- Switching between standard FEniCSx forms and `cutfemx` boundary integrals.
- MPI parallel reductions.
- Evaluation of complex constraints (e.g., the $L^p$ norm of the Von Mises stress).

As a developer, you only need to provide the pure mathematical formulation of your problem using UFL (Unified Form Language).

## Step-by-step Guide

### 1. Create a New Class

Create a new class in `src/config/problem.py`. For this example, let's create a problem that minimizes the **Area** under a **Compliance** constraint:

```python
import ufl
from dolfinx import fem
from petsc4py import PETSc
from utils import mechanics_tool

class AreaMin_ComplianceConstraint:
    def __init__(self):
        pass
```

### 2. Define the Cost and Constraint Integrands

You must implement the methods `cost_integrand` and `constraint_integrand`. These methods return UFL expressions representing the integrands of your cost function and structural constraint.

For our example:
- **Cost (Area):** $J(\Omega) = \int_{\Omega} 1 \, dx$
- **Constraint (Compliance):** $C(\Omega) = \int_{\Omega} \sigma(u) : arepsilon(u) \, dx$

```python
    def cost_integrand(self, u, lame_mu, lame_lambda, parameters):
        # Area cost integrand: f(x) = 1.0
        mesh = u.function_space.mesh
        return fem.Constant(mesh, PETSc.ScalarType(1.0))
        
    def constraint_integrand(self, u, lame_mu, lame_lambda, parameters):
        # Compliance constraint integrand: sigma(u) : eps(u)
        return 0.5 * (2.0*lame_mu * ufl.inner(mechanics_tool.strain(u), mechanics_tool.strain(u)) + \
                      lame_lambda * ufl.inner(ufl.nabla_div(u), ufl.nabla_div(u)))
```

### 3. Define the Shape Derivatives

Next, provide the analytical shape derivatives using Céa's method or the Lagrangian approach. Implement `shape_derivative_integrand` and `shape_derivative_integrand_constraint`.

For our example:
- **Shape derivative of Area:** $dJ(\Omega; \theta) = \int_{\partial \Omega} 1 (	heta \cdot n) \, ds$
- **Shape derivative of Compliance:** $dC(\Omega; \theta) = \int_{\partial \Omega} -\sigma(u) : arepsilon(u) (	heta \cdot n) \, ds$

```python
    def shape_derivative_integrand(self, u, p, lame_mu, lame_lambda, parameters, measure=0):
        # Integrand for the Area shape derivative (constant 1.0)
        mesh = u.function_space.mesh
        return fem.Constant(mesh, PETSc.ScalarType(1.0))

    def shape_derivative_integrand_constraint(self, u, p, lame_mu, lame_lambda, parameters, measure=0, vm_DG=0, c_k=0):
        # Integrand for the Compliance shape derivative (-sigma:eps)
        # Note: Since compliance is self-adjoint, p = -u, so the sign adapts accordingly.
        return self.constraint_integrand(u, lame_mu, lame_lambda, parameters) - \
               (2.0*lame_mu * ufl.inner(mechanics_tool.strain(u), mechanics_tool.strain(p)) + \
                lame_lambda * ufl.inner(ufl.nabla_div(u), ufl.nabla_div(p)))
```

### 4. Provide the Dual Operator

If your problem is not self-adjoint (e.g., stress constraints), you must define the `dual_operator` to allow the solver to compute the adjoint state. This generally involves computing the Fréchet derivative of the constraint with respect to the state variable $u$.

```python
    def dual_operator(self, u, lame_mu, lame_lambda, parameters, mesh, measure=0, vm_DG=0, c_k=0):
        dim = mesh.geometry.dim
        V = fem.functionspace(mesh, ("Lagrange", 1, (dim, )))
        J = self.constraint_integrand(u, lame_mu, lame_lambda, parameters) * measure
        v_adj = ufl.TestFunction(V)
        
        # Fréchet derivative of J with respect to u
        dual_operator = ufl.derivative(J, u, v_adj)
        return dual_operator
```

### 5. Activate the Problem in the Configuration

Once your class is written, you need to map it in the main executable so that it can be triggered via the configuration file.

1. Open `src/main.py`.
2. Import your new class and add it to the problem selection logic:

```python
from config.problem import AreaMin_ComplianceConstraint

if parameters.cost_func == "CustomAreaCompliance":
    problem_topo = AreaMin_ComplianceConstraint()
```

3. Finally, use this new key in your `param.txt`:
```text
cost_func CustomAreaCompliance
```

## Integration with the Test Suite

The OptiCut test suite utilizes the `inspect` module to dynamically discover problem classes. Once your new class is added to `config/problem.py` and implements the required methods, the finite difference verification tests (`tests/test_problem_sensitivity.py`) will automatically execute and validate your analytical shape derivative against numerical approximations. No modifications to the testing infrastructure are required.
