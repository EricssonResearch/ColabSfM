from setuptools import setup, find_packages


setup(
    name = "colabsfm",
    author = "Johan Edstedt",
    packages = find_packages(include = ["src*"]),
    install_requires=open("requirements.txt", "r").read().split("\n"),
)