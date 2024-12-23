import os
import sys
os.system('git config pull.rebase true')
os.system(f'{sys.executable} -m pip install --upgrade pip')
os.system('pip install -e .[dev]')
os.system('pre-commit install --install-hooks')
