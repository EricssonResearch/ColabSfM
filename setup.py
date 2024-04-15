from setuptools import setup, find_packages


setup(
    name = "omnireg",
    author = "Johan Edstedt",
    packages = find_packages(include = "omnireg*"),
    install_requires=open("requirements.txt", "r").read().split("\n"),
)