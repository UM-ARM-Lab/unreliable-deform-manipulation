## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['link_bot_planning'],
    package_dir={'': 'src'},
)

setup(requires=['control', 'matplotlib', 'numpy', 'tensorflow', 'colorama', 'tabulate', 'gpflow', 'PIL', 'more_itertools',
                'scipy'], **setup_args)
