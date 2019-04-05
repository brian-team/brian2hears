import os

from .prefs import *
from .bufferable import *
from .sounds import *
from .erb import *
from .filtering import *
from .hrtf import *
from .db import *
from .plotting import *

with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
