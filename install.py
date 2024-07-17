import os

os.system('git config pull.rebase true')
os.system('python -m pip install --upgrade pip')
os.system('pip install -e .[dev]')
os.system('pre-commit install')
os.system('pre-commit run --files ' + __file__)
