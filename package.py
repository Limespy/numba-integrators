#!/usr/bin/env python
# type: ignore
#%%═════════════════════════════════════════════════════════════════════
# IMPORT
import pathlib
import sys

import tomli_w
import tomllib
from build import __main__ as build

import readme

if '--print' in sys.argv:
    import pprint
    sys.argv.pop(sys.argv.index('--print'))
    is_verbose = True
else:
    is_verbose = False

if not '--notests' in sys.argv or is_verbose:
    import colorama as col
    RESET = col.Style.RESET_ALL
    BLACK = col.Fore.BLACK
    BLUE = col.Fore.BLUE
    CYAN = col.Fore.CYAN
    GREEN = col.Fore.GREEN
    MAGENTA = col.Fore.MAGENTA
    RED = col.Fore.RED
    YELLOW = col.Fore.YELLOW
    WHITE = col.Fore.WHITE
    WHITE_BG = col.Back.WHITE
#%%═════════════════════════════════════════════════════════════════════
# SETUP GLOBALS
BASE_DIR = pathlib.Path(__file__).parent
PYTHON_VERSION = '>=3.9'
PATH_LICENCE = next(BASE_DIR.glob('LICENSE*'))
PATH_SRC = BASE_DIR / 'src'
PATH_INIT = next(PATH_SRC.rglob('__init__.py'))
PATH_README = BASE_DIR / 'README.md'
PATH_PYPROJECT = BASE_DIR / 'pyproject.toml'
#%%═════════════════════════════════════════════════════════════════════
# Run tests first
if not '--notests' in sys.argv:
    import tests
    print('Running typing checks')
    typing_test_result = tests.typing(shell = False)
    failed = not typing_test_result[0].startswith('Success')
    failed |= bool(typing_test_result[1])
    print(f'{RED if failed else GREEN}{typing_test_result[0]}{RESET}')
    if typing_test_result[1]:
        print(typing_test_result[1])

    print('Running unit tests')
    tests.unittests()
    # failed |= bool(unit_test_result.errors)
    # failed |= bool(unit_test_result.failures)
    if failed:
        raise Exception('Tests did not pass, read above')
    readme.main()
else:
    sys.argv.pop(sys.argv.index('--notests'))
#%%═════════════════════════════════════════════════════════════════════
# SETUP FUNCTIONS
def header(text: str, linechar = '─', endchar = '┐', headerwidth  =  60):
    titlewidth = headerwidth // 2
    textlen = len(text)
    l_len = ((titlewidth - textlen) // 2 - 1)
    lpad = linechar*l_len
    rpad = f'{(headerwidth - l_len - textlen - 3)*linechar}'
    return f'{lpad} {GREEN}{text}{RESET} {rpad}{endchar}'
#%%═════════════════════════════════════════════════════════════════════
# BUILD INFO

# Loading the pyproject TOML file
pyproject = tomllib.loads(PATH_PYPROJECT.read_text())
build_info = pyproject['project']


if is_verbose:
    print(f'\n{header("Starting packaging setup", "=", "=")}\n')
# Getting package name
build_info['name'] = PATH_INIT.parent.stem
#───────────────────────────────────────────────────────────────────────
# Version
with open(PATH_INIT, 'r', encoding = 'utf8') as f:
    while not (line := f.readline().lstrip()).startswith('__version__'):
        pass
    main_version = line.split('=')[-1].strip().strip("'")
    build_version = int(build_info['version'].split('.')[-1]) + 1
    build_info['version'] = f'{main_version}.{build_version}'
#───────────────────────────────────────────────────────────────────────
# Licence
with open(PATH_LICENCE, 'r', encoding = 'utf8') as f:
    LICENSE_NAME = f'{f.readline().strip()}'
build_info['license'] = {'text': LICENSE_NAME}
#───────────────────────────────────────────────────────────────────────
# URL
URL = f'https://github.com/{build_info["authors"][0]["name"]}/{build_info["name"]}'
GITHUB_MAIN_URL = f'{URL}/blob/main/'
#───────────────────────────────────────────────────────────────────────
# Description
with open(PATH_README, 'r', encoding = 'utf8') as f:
    # The short description is in the README after badges
    while (description := f.readline().lstrip(' ')).startswith(('#', '\n', '[')):
        pass
    while not (line := f.readline().lstrip(' ')).startswith('\n'):
        description += line
build_info['description'] = description[:-1] # Removing trailing linebreak
#───────────────────────────────────────────────────────────────────────
# Long Description
long_description = PATH_README.read_text().replace('./', GITHUB_MAIN_URL)
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
# PRINTING SETUP INFO

if is_verbose:
    for key, value in build_info.items():
        print(f'\n{header(key)}\n')
        pprint.pprint(value)
#%%═════════════════════════════════════════════════════════════════════
# RUNNING THE BUILD
# print(build_info)
pyproject['project'] = build_info
PATH_PYPROJECT.write_text(tomli_w.dumps(pyproject))
for path in (BASE_DIR / 'dist').glob('*'):
    path.unlink()
if is_verbose:
    print(f'\n{header("Replacing README", "=", "=")}\n')
PATH_LOCAL_README = PATH_README.rename(PATH_README.parent /'.README.md')
PATH_README.write_text(long_description)
if is_verbose:
    print(f'\n{header("Calling build", "=", "=")}\n')
build.main([])
if is_verbose:
    print(f'\n{header("Returning README", "=", "=")}\n')
PATH_README.unlink()
PATH_LOCAL_README.rename(PATH_README.parent / 'README.md')
