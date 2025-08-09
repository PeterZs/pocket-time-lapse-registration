import os

def _getModulePath():
    return os.path.abspath(os.path.dirname(__file__));

from ptlreg.apy.amedia.media.examplefiles.examplefiles import *
from ptlreg.apy.amedia.media.Image import *
from ptlreg.apy.amedia.media.ImageDraw import *
from ptlreg.apy.amedia.media.Audio import *
from ptlreg.apy.amedia.media.AudioWaveforms import *
from ptlreg.apy.amedia.media.Video import *