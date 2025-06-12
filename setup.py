from setuptools import setup, find_packages


setup(
    name = "colabsfm",
    version = "0.0.1",
    author = "Johan Edstedt",
    packages = find_packages(include = ["hloc*", "colabsfm*"]),
    install_requires=open("requirements.txt", "r").read().split("\n"),
)