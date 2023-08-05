import pathlib

import numba_integrators as ni
import yamdog as md
from limedev import readme

PATH_BASE = pathlib.Path(__file__).parent
PATH_README = PATH_BASE / 'README.md'
PATH_PYPROJECT = PATH_BASE / 'pyproject.toml'

#=======================================================================
NAME = 'Numba Integrators'
def quick_start():
    example_text = (PATH_BASE / '.readme' / 'example.py').read_text()

    guide = md.Document([md.Heading('Example', 3), md.CodeBlock(example_text, 'python')])
    return guide
#=======================================================================
def make(project_info):

    link_numba = md.Link('https://numba.pydata.org', 'Numba',
                         'Numba organisation homepage')
    link_scipy = md.Link('https://scipy.org/', 'SciPy',
                         'SciPy organisation homepage')
    semi_description = md.Document([md.Paragraph([
        f'{NAME} is collection numerical integrators based on the ones in ',
       link_scipy,
       '. Aim is to make them faster and much more compatible with ',
       link_numba,
       '.'
    ])])

    return readme.make(ni, semi_description,
                       name = NAME,
                       quick_start = quick_start())
#=======================================================================
def main():

    import pathlib
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib # type: ignore

    PATH_README.write_text(str(make(tomllib.loads(PATH_PYPROJECT.read_text())['project']))
                           + '\n')
    return 0
#=======================================================================
if __name__ == '__main__':
    raise SystemExit(main())
