from setuptools import setup, find_packages
import os

with open(os.path.join(os.path.dirname(__file__),
                       'brian2hears', 'VERSION')) as version_file:
    version = version_file.read().strip()

setup(
    name="brian2hears",
    version=version,
    url='https://github.com/brian-team/brian2hears',
    author="Bertrand Fontaine, Dan Goodman, Marcel Stimberg, Victor Benichoux, Romain Brette",
    author_email="team@briansimulator.org",
    description="Auditory modelling package for brian2 simulator",
    install_requires=['numpy', 'scipy>=0.7', 'brian2>=2.1.3', 'future'],
    packages=find_packages(),
    package_data={'brian2hears': 'VERSION'},
    use_2to3=False,
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: CEA CNRS Inria Logiciel Libre License, version 2.1 (CeCILL-2.1)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ]
)
