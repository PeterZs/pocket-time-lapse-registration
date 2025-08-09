import unittest
import os
import shutil
import ptlreg.apy.utils

_TESTDIR_NAME = "TEMP_MEDIAGRAPH_TEST_DIR"
ABEPY_TESTDIR = None;

TEST_MEDIA = True;


def TestModule(module):
    loader = unittest.TestLoader()
    start_dir = module._getModulePath()
    tests_subdir = os.path.join(start_dir, 'tests');
    if(os.path.exists(tests_subdir)):
        start_dir = tests_subdir;
    suite = loader.discover(start_dir)
    runner = unittest.TextTestRunner(verbosity=2);
    runner.run(suite, );


def CreateTestDir(path = None):
    global ABEPY_TESTDIR
    if(path is not None):
        ptlreg.apy.utils.make_sure_path_exists(path);
        ABEPY_TESTDIR = path;
        return ABEPY_TESTDIR
    ABEPY_TESTDIR = ptlreg.apy.utils.get_incremented_until_path_does_not_exist(os.path.join('', _TESTDIR_NAME));
    os.mkdir(ABEPY_TESTDIR);
    return ABEPY_TESTDIR;

def SetTestDir(path=None, folder_name=None, create_if_missing=True):
    global ABEPY_TESTDIR

    assert (folder_name is None or path is None), "do not provide both folder name and path for SetTestDir in abepy.testing"
    if(folder_name is None):
        use_folder_name = _TESTDIR_NAME;
    else:
        use_folder_name = folder_name;

    if(path is None):
        path = os.path.join('', use_folder_name);
    if(os.path.exists(path)):
        ABEPY_TESTDIR = path;
        return ABEPY_TESTDIR;
    else:
        return CreateTestDir(path=path);

def GetTestDir():
    global ABEPY_TESTDIR
    if(ABEPY_TESTDIR is None):
        return SetTestDir();
    return ABEPY_TESTDIR;

