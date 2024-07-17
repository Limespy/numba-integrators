import pathlib

import numba_integrators as ni
import yamdog as md
from limedev import readme
#=======================================================================
PATH_BASE = pathlib.Path(__file__).parent
#=======================================================================
def quick_start() -> md.Document:
    example_text = (PATH_BASE / 'example.py').read_text()

    guide = md.Document([md.Heading('Example', 3),
                         md.CodeBlock(example_text, 'python')])
    return guide
#=======================================================================
def advanced() -> md.Document:
    example_advanced = (PATH_BASE / 'example_advanced.py').read_text()
    return md.Document([md.Heading('Example of the advanced function', 3),
                         md.CodeBlock(example_advanced, 'python')])
#=======================================================================
def main(pyproject: readme.PyprojectType) -> md.Document:

    link_numba = md.Link('https://numba.pydata.org', 'Numba',
                         'Numba organisation homepage')
    link_scipy = md.Link('https://scipy.org/', 'SciPy',
                         'SciPy organisation homepage')
    name = pyproject['tool']['limedev']['full_name']
    semi_description = md.Document([md.Paragraph([
        f'{name} is collection numerical integrators based on the ones in ',
       link_scipy,
       '. Aim is to make them faster and much more compatible with ',
       link_numba,
       '.'
    ])])

    return readme.make(ni, semi_description,
                       name = name,
                       abbreviation = 'ni',
                       quick_start = quick_start() + advanced())
