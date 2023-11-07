[![PyPI Package latest release](https://img.shields.io/pypi/v/numba-integrators.svg)][1]
[![PyPI Wheel](https://img.shields.io/pypi/wheel/numba-integrators.svg)][1]
[![Supported versions](https://img.shields.io/pypi/pyversions/numba-integrators.svg)][1]
[![Supported implementations](https://img.shields.io/pypi/implementation/numba-integrators.svg)][1]

# Numba Integrators <!-- omit in toc -->

Numba Integrators is collection numerical integrators based on the ones in [SciPy][2]. Aim is to make them faster and much more compatible with [Numba][3].

## Table of Contents <!-- omit in toc -->

- [Quick start guide](#quick-start-guide)
    - [The first steps](#the-first-steps)
        - [Installing](#installing)
        - [Importing](#importing)
        - [Example](#example)
        - [Example of the advanced function](#example-of-the-advanced-function)

# Quick start guide

Here's how you can start

## The first steps

### Installing

Install Numba Integrators with pip

```
pip install numba-integrators
```

### Importing

Import name is not the same as install name, `numba-integrators`.

```python
import numba_integrators
```

### Example

```python
import numba as nb
import numba_integrators as ni
import numpy as np

@nb.njit(nb.float64[:](nb.float64, nb.float64[:]))
def f(t, y):
    '''Differential equation for sine wave'''
    return np.array((y[1], -y[0]))

y0 = np.array((0., 1.))

solver = ni.RK45(f, 0.0, y0,
                 t_bound = 1, atol = 1e-8, rtol = 1e-8)

t = []
y = []

while ni.step(solver):
    t.append(solver.t)
    y.append(solver.y)

print(t)
print(y)

```

### Example of the advanced function

```python
import numba as nb
import numba_integrators as ni
import numpy as np

@nb.njit
def f(t, y, parameters):
    '''Differential equation for sine wave'''
    auxiliary = parameters[0] * y[1]
    dy = np.array((auxiliary, -y[0])) + parameters[1]
    return dy, auxiliary

t0 = 0.
y0 = np.array((0., 1.))
parameters = (2., np.array((-1., 1.)))

# Numba type signatures
parameters_signature = nb.types.Tuple((nb.float64, nb.float64[:]))
auxiliary_signature = nb.float64
solver_type = ni.RK45

Solver = ni.Advanced(parameters_signature, auxiliary_signature, solver_type)
solver = Solver(f, t0, y0, parameters,
                t_bound = 1, atol = 1e-8, rtol = 1e-8)

t = []
y = []
auxiliary = []

while ni.step(solver):
    t.append(solver.t)
    y.append(solver.y)
    auxiliary.append(solver.auxiliary)

print(t)
print(y)
print(auxiliary)

```

# Changelog <!-- omit in toc -->

## 0.3.1 2023-11-07 <!-- omit in toc -->

- Fast forward
- Refactoring

## 0.2.2 2023-11-07 <!-- omit in toc -->

- Dev tools update

## 0.2.1 2023-08-11 <!-- omit in toc -->

- Advanced mode solver to handle functions with parameters and auxiliary output

## 0.1.2 2023-08-06 <!-- omit in toc -->

- Fixes

## 0.1.1 2023-08-05 <!-- omit in toc -->

- Initial working version

## 0.0.3 2023-05-14 <!-- omit in toc -->

- Inital working state

[1]: <https://pypi.org/project/numba-integrators> "Project PyPI page"
[2]: <https://scipy.org/> "SciPy organisation homepage"
[3]: <https://numba.pydata.org> "Numba organisation homepage"
