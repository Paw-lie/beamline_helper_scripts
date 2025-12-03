"""
Setup script for Beamline Helper Scripts with IQM analysis
"""
from setuptools import setup, find_packages

setup(
    name="beamline-helper-scripts",
    version="2.0.0",
    description="Beamline helper scripts with IQM slice analysis integration",
    author="Paul",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "scikit-image>=0.19.0",
        "lmfit>=1.0.3",
    ],
    entry_points={
        'console_scripts': [
            'beamline-helper=beamline_helper_scripts.beamline_console_helper:main',
        ],
    },
)
