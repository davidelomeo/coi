from setuptools import find_packages, setup
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Package meta-data
NAME = "coi"
DESCRIPTION = "Python package to calculate Cyanobacteria Occurrence Index from input dataset"
AUTHOR = "Davide Lomeo"
AUTHOR_EMAIL = "davide.lomeo@kcl.ac.uk"
# URL = "https://github.com/davidelomeo/cyanobacteria_occurrence_index"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "0.1.0"


# Required packages
REQUIRED = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "xarray>=0.20.0",
]


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    python_requires=REQUIRES_PYTHON,
    # url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    include_package_data=True,
    package_data={
        'coi': ['core_files/*'],
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    entry_points={
        'console_scripts': [
            'coi=coi.cli:main',
        ],
    },
)