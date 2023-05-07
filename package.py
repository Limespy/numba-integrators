#!/usr/bin/env python
# type: ignore
#%%═════════════════════════════════════════════════════════════════════
# IMPORT
import pathlib
import re
import sys
import time

import tomli_w
from build import __main__ as build

import readme

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

PATH_BASE = pathlib.Path(__file__).parent
PATH_LICENCE = next(PATH_BASE.glob('LICENSE*'))
PATH_README = PATH_BASE / 'README.md'
PATH_PYPROJECT = PATH_BASE / 'pyproject.toml'

def main(args = sys.argv[1:]):
    if '--print' in args:
        import pprint
        args.pop(args.index('--print'))
        is_verbose = True
    else:
        is_verbose = False

    if not '--notests' in args or is_verbose:
        import colorama as col
        RESET = col.Style.RESET_ALL
        GREEN = col.Fore.GREEN
        RED = col.Fore.RED
    #%%═════════════════════════════════════════════════════════════════════
    # SETUP GLOBALS
    #%%═════════════════════════════════════════════════════════════════════
    # Run tests first
    if '--no-tests' in args:
        args.pop(args.index('--no-tests'))
    else:
        import tests
        print('Running typing checks')
        typing_test_result = tests.typing(shell = False)
        failed = not typing_test_result[0].startswith('Success')
        failed |= bool(typing_test_result[1])
        print(f'{RED if failed else GREEN}{typing_test_result[0]}{RESET}')
        if typing_test_result[1]:
            print(typing_test_result[1])

        print('Running unit tests')
        from tox import run
        failed |= bool(run.main([]))
        if failed:
            raise Exception('Tests did not pass, read above')
    #%%═════════════════════════════════════════════════════════════════════
    # BUILD INFO

    # Loading the pyproject TOML file
    pyproject = tomllib.loads(PATH_PYPROJECT.read_text())
    master_info = pyproject['master-info']
    build_info = pyproject['project']

    VERSION = re.search(r"(?<=__version__ = ').*(?=')",
                    next((PATH_BASE / 'src').rglob('__init__.py')).read_text()
                    )[0]

    if '--build-number' in args:
        VERSION += f'.{time.time():.0f}'

    build_info['version'] = VERSION
    package_name = master_info["package_name"]
    full_name = master_info.get("full_name",
                                package_name.replace('-', ' ').capitalize())
    description = master_info['description']

    build_info["name"] = package_name
    build_info['description'] = description
    #───────────────────────────────────────────────────────────────────────
    # URL
    URL = f'{master_info["organisation"]}/{package_name}'
    GITHUB_MAIN_URL = f'{URL}/blob/main/'
    #───────────────────────────────────────────────────────────────────────
    # Long Description
    readme_text = str(readme.make(full_name, package_name, description)) + '\n'
    readme_text_pypi = readme_text.replace('./', GITHUB_MAIN_URL)
    #───────────────────────────────────────────────────────────────────────
    # Classifiers
    # complete classifier list:
    #   http://pypi.python.org/pypi?%3Aaction=list_classifiers
    #───────────────────────────────────────────────────────────────────────
    # Project URLs
    build_info['urls'] = {
        'Homepage': URL,
        'Changelog': f'{GITHUB_MAIN_URL}{PATH_README.name}#Changelog',
        'Issue Tracker': f'{URL}/issues'}
    #%%═════════════════════════════════════════════════════════════════════
    # RUNNING THE BUILD

    pyproject['project'] = build_info
    pyproject['master-info'] = master_info
    PATH_PYPROJECT.write_text(tomli_w.dumps(pyproject))

    for path in (PATH_BASE / 'dist').glob('*'):
        path.unlink()

    PATH_README.write_text(readme_text_pypi)

    if not '--no-build' in args:
        build.main([])

    PATH_README.write_text(readme_text)

if __name__ =='__main__':
    raise SystemExit(main())
