from brian2 import *
del test  ## prevent Brian's test suite from getting picked up
from brian2hears.db import *

def test_db():
    x = (5*dB).gain()
