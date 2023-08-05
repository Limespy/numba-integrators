[![PyPI Package latest release](https://img.shields.io/pypi/v/numba_integrators.svg)][1]
[![PyPI Wheel](https://img.shields.io/pypi/wheel/numba_integrators.svg)][1]
[![Supported versions](https://img.shields.io/pypi/pyversions/numba_integrators.svg)][1]
[![Supported implementations](https://img.shields.io/pypi/implementation/numba_integrators.svg)][1]

# Numba Integrators <!-- omit in toc -->

Numba Integrators is collection numerical integrators based on the ones in [SciPy][2]. Aim is to make them faster and much more compatible with [Numba][3].

## Table of Contents <!-- omit in toc -->

- [Quick start guide](#quick-start-guide)
    - [The first steps](#the-first-steps)
        - [Installing](#installing)
        - [Importing](#importing)
        - [Example](#example)

# Quick start guide

Here's how you can start numerically

## The first steps

### Installing

Install Numba Integrators with pip

```
pip install numba_integrators
```

### Importing

Import name is the same as install name, `numba_integrators`.

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

# Changelog <!-- omit in toc -->

## 0.1.2 2023-08-06 <!-- omit in toc -->

- Fixes

## 0.1.1 2023-08-05 <!-- omit in toc -->

- Initial working version

## 0.0.3 2023-05-14 <!-- omit in toc -->

- Inital working state

[1]: <https://pypi.org/project/numba_integrators> "Project PyPI page"
[2]: <https://scipy.org/> "SciPy organisation homepage"
[3]: <https://numba.pydata.org> "Numba organisation homepage"
