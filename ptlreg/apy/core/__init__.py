from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _getModulePath():
    return os.path.abspath(os.path.dirname(__file__));




from ptlreg.apy.core.aobject import *
from ptlreg.apy.core.filepath import *
from ptlreg.apy.core.aobject.AObjectList import *
from ptlreg.apy.core.aobject.AObjectOrderedSet import *
from ptlreg.apy.core.SavesToJSON import *
from ptlreg.apy.core.SavesFeatures import *
from ptlreg.apy.core.SavesDirectories import *
from ptlreg.apy.core.DictDir import *
from ptlreg.apy.core.dicts.ParamDict import *
from ptlreg.apy.core.dicts.FuncDict import *
from ptlreg.apy.core.dicts.IndexDict import *


def IsList(object):
    return isinstance(object, (list, tuple, np.ndarray, AObjectList));