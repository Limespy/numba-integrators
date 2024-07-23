"""This file generates the module for."""
import pathlib

PATH_BASE = pathlib.Path(__file__).parent

#         name   settable
names = (('fun', False),
         ('x', False),
         ('y', False),
         ('rtol', False),
         ('atol', False),
         ('x_bound', True),
         ('max_step', False),
         ('h_abs', False),
         ('direction', False),
         ('step_size', False),
         ('nfev', False),
         ('error_exponent', False),
         ('n_stages', False),
         ('A', False),
         ('C', False),
         ('K', True))

clsname = 'RK'
header = f'''\
# pylint: skip-file
"""This file has been generated automatically"""
import numba as nb

from numba.experimental import structref

# Define a StructRef.
# `structref.register` associates the type with the default data model.
# This will also install getters and setters to the fields of
# the StructRef.
@structref.register
class {clsname}Type(nb.types.StructRef):
    def preprocess_fields(self, fields):
        # This method is called by the type constructor for additional
        # preprocessing on the fields.
        # Here, we don't want the struct to take Literal types.
        return tuple((name, nb.types.unliteral(typ)) for name, typ in fields)
'''

moduleparts = [header,
                f'class {clsname}(structref.StructRefProxy):']
# New
new = '    def __new__('
new += ',\n                '.join(name for name, _ in (('cls', False), *names))
new += '):'
# new += '''
#         K = np.zeros((n_stages + 1, len(y)), dtype = y.dtype)
#         K[-1] = fun(x, y)'''
#         direction = 1. if x_bound == x else np.sign(x_bound - x)

#         if not first_step:
#             h_abs = select_initial_step(
#                 fun, x0, y0, K[-1], direction,
#                 error_exponent, rtol, atol)
#         else:
#             h_abs = np.abs(first_step)
#         step_size = direction * h_abs'''
new += '\n        return structref.StructRefProxy.__new__('
new += (',\n'+ 48* ' ').join(name for name, _ in (('cls', False), *names))
new += ')'
moduleparts.append(new)

# Attributes
for name, is_setter in names:
    moduleparts.append( '    @property\n'
                    f'    def {name}(self):\n'
                    f'        return {clsname}_get_{name}(self)')
    if is_setter:
        moduleparts.append(f'    @{name}.setter\n'
                        f'    def {name}(self, value):\n'
                        f'        {clsname}_set_{name}(self, value)')

# getters and setters
for name, is_setter in names:
    moduleparts.append( '@nb.njit\n'
                f'def {clsname}_get_{name}(self):\n'
                f'    return self.{name}')
    if is_setter:
        moduleparts.append('@nb.njit\n'
                        f'def {clsname}_set_{name}(self, value):\n'
                        f'    self.{name} = value')

# proxy

proxy_start = f'structref.define_proxy({clsname}, {clsname}Type, ['

proxy_start += (',\n' + ' '* len(proxy_start)).join(f"'{name}'" for name, _ in names)
proxy_start += '])'
moduleparts.append(proxy_start)
moduletext = '\n\n'.join(moduleparts)
(PATH_BASE / '_structref_generated.py').write_text(moduletext)

from . import _structref_generated
