import brian2hears
import nose
import os

basedir, _ = os.path.split(brian2hears.__file__)
os.chdir(basedir)
nose.run()
