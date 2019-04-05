import os
import sys

import pytest

import brian2hears

basedir = os.path.dirname(brian2hears.__file__)
os.chdir(basedir)
sys.exit(pytest.main(['-v']))
