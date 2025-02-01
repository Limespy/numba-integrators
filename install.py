import sys
from subprocess import run

run((sys.executable, '-m', 'pip', 'install', 'limeinstall'))

try:
    run((sys.executable, '-m', 'limeinstall', *sys.argv[1:]))
finally:
    run((sys.executable, '-m', 'pip', 'uninstall', 'limeinstall' ,'-y'))
