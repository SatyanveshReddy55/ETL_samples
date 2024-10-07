import os
import sys


def pytest_sessionstart(session):
    proj_root = os.path.normpath(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    package_root = os.path.join(proj_root, "function_app")
    sys.path.append(package_root)
    sys.path.append(os.path.join(proj_root, "function_app_tests"))
    os.chdir(package_root)


def pytest_runtest_setup(item):
    import function_app
