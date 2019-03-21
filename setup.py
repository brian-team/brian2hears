from setuptools import setup, find_packages

setup(
    name="brian2hears",
    version="2.0b1",
    author="Bertrand Fontaine, Dan Goodman, Marcel Stimberg, Victor Benichoux, Romain Brette",
    author_email="team@briansimulator.org",
    description="Auditory modelling package for brian2 simulator",
    install_requires=['numpy', 'scipy', 'brian2', 'six', 'future'],
    packages=find_packages(),
)
