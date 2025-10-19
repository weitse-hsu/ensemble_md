"""
ensemble_md
A repository for developing ensemble simulation methods
"""
import sys
from setuptools import setup, find_packages
import versioneer

short_description = "A package for setting up, performing, and analyzing molecular dynamics ensembles using GROMACS".split("\n")[0]

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except FileNotFoundError:
    long_description = "\n".join(short_description[2:])


setup(
    # Self-descriptive entries which should always be present
    name='ensemble_md',
    author='Wei-Tse Hsu',
    author_email='wehs7661@colorado.edu',
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',
    project_urls={
        "Documentation": "https://ensemble-md.readthedocs.io/",
        "Source Code": "https://github.com/wehs7661/ensemble_md",
    },
    keywords="molecular mechanics, free energy calculations, advanced sampling",

    # Describes the project using a list of classifiers (https://pypi.org/classifiers/)
    # This makes the project more searchable.
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows ",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line or set include_package_data to False to prevent the files from being packaged with your software
    include_package_data=False,

    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,

    # Add entry points
    entry_points={
        'console_scripts':[
            'run_REXEE = ensemble_md.cli.run_REXEE:main',
            'analyze_REXEE = ensemble_md.cli.analyze_REXEE:main',
            'explore_REXEE = ensemble_md.cli.explore_REXEE:main',
        ],
    },

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='https://github.com/wehs7661/ensemble_md',  # Website

    # Required packages, pulls from pip if needed; do not use for Conda deployment
    install_requires=[
        'numpy',
        'natsort',
        'argparse',
        'pymbar>=4.0.1',
        'alchemlyb>=2.0.0',
        'pyyaml',
        'seaborn',
        'matplotlib<=3.8.4',
        'pyemma',
        'openmpi',
        'mpi4py',
        'ruptures'
    ],

    # (OBSOLETE) The following extra_require directive provides optional dependencies by, in our case, pip install ensemble[gmxapi].
    # ensemble_md requires GROMACS and gmxapi to be installed before use. 
    # If a working version of GROMACS is available, the user can choose to install ensemble_md along with gmxapi 
    # using the following command: `pip install ensemble_md[gmxapi]`. Otherwise, follow installation 
    # instructions of GROMACS and gmxapi to install each package separately. 
    
    # extras_require={
    #    'gmxapi': [
    #        'pybind11>=2.6',
    #        'setuptools>=42.0',
    #        'gmxapi>=0.4.0rc2'
    #    ],
    # },

    platforms=['Linux',
               'Mac OS-X',
               'Unix',
               'Windows'],            # Valid platforms your code works on, adjust to your flavor
    
    python_requires=">=3.8",          # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

)
