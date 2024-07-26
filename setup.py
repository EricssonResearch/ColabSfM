from setuptools import setup, find_packages


setup(
    name = "sfmreg",
    author = "Johan Edstedt",
    packages = find_packages(include = ["sfmreg*","hloc*"]),
    install_requires=open("requirements.txt", "r").read().split("\n"),
)