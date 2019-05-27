from setuptools import setup

requirements = [
    'gpflow>=1.3.0'
]

setup(
    name='link_bot_gaussian_process',
    version='0.0.0',
    package_dir={'': 'src'},
    packages=['link_bot_gaussian_process'],
    install_requires=requirements,
    zip_safe=True,
)

