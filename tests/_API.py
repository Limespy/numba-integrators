'''Unittests for limesqueezer'''
#%%═════════════════════════════════════════════════════════════════════
# IMPORT
import os
import pathlib
import sys
from typing import Callable
from typing import NoReturn
from typing import Optional
from typing import Union

PATH_TESTS = pathlib.Path(__file__).parent
PATH_UNITTESTS = PATH_TESTS / 'unittests'
PATH_LINTCONFIG = PATH_TESTS / '.pylintrc'
PATH_REPO = PATH_TESTS.parent
# First item in src should be the package
PATH_SRC = next((PATH_REPO / 'src').rglob('*'))
#%%═════════════════════════════════════════════════════════════════════
# TEST CASES

#══════════════════════════════════════════════════════════════════════════════
def unittests() -> None:
    import pytest
    CWD = pathlib.Path.cwd()
    os.chdir(str(PATH_UNITTESTS))
    pytest.main(["--cov=numba_integrators", "--cov-report=html"])
    os.chdir(str(CWD))
    return None
#══════════════════════════════════════════════════════════════════════════════
def typing(shell: bool = True) -> Optional[tuple[str, str, int]]:
    args = [str(PATH_SRC), '--config-file', str(PATH_TESTS / "mypy.ini")]
    if shell:
        from mypy.main import main
        main(args = args)
    else:
        from mypy import api
        return api.run(args)
#══════════════════════════════════════════════════════════════════════════════
def lint() -> None:
    from pylint import lint
    lint.Run([str(PATH_SRC),
              f'--rcfile={str(PATH_LINTCONFIG)}',
              '--output-format=colorized',
              '--msg-template="{path}:{line}:{column}:{msg_id}:{symbol}\n'
                              '    {msg}"'])
#══════════════════════════════════════════════════════════════════════════════
def compare() -> None:
    from compare import main as _main
    return _main()
#══════════════════════════════════════════════════════════════════════════════
def performance():
    from performance import main as _main
    return _main()
#═══════════════════════════════════════════════════════════════════════
def profile():
    import numba_integrators as ni
    import cProfile
    import gprof2dot
    import subprocess

    path_profile = PATH_TESTS / 'profile'
    path_pstats = path_profile.with_suffix('.pstats')
    path_dot = path_profile.with_suffix('.dot')
    path_pdf = path_profile.with_suffix('.pdf')

    def profile_run():
        ...

    profile_run()
    with cProfile.Profile() as pr:
        profile_run()
        pr.dump_stats(path_pstats)

    gprof2dot.main(['-f', 'pstats', str(path_pstats), '-o', path_dot])
    path_pstats.unlink()
    try:
        subprocess.run(['dot', '-Tpdf', str(path_dot), '-o', str(path_pdf)])
    except FileNotFoundError:
        raise RuntimeError('Conversion to PDF failed, maybe graphviz dot program is not installed. http://www.graphviz.org/download/')
    path_dot.unlink()
#══════════════════════════════════════════════════════════════════════════════
TESTS: dict[str, Callable] = {function.__name__: function # type: ignore
                              for function in
                              (lint,
                               unittests,
                               compare,
                               typing,
                               performance,
                               profile)}
def main(args: list[str] = sys.argv[1:]) -> Union[list, None, NoReturn]:
    if not args:
        return None
    if args[0] == '--all':
        return [test() for test in TESTS.values()]
    return [TESTS[arg[2:]]() for arg in args if arg.startswith('--')]
#══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    raise SystemExit(main())
