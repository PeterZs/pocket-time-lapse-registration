import shutil
import os
from ptlreg.apy.core import *
# from abepy.apy.SavesFeaturesDir import SavesFeaturesDir
from ptlreg.apy.amedia.media import *
import numpy as np
import random
from ptlreg.apy.testing import TEST_MEDIA

VERBOSE_TEST = False;

def tprint(pargs):
    if(VERBOSE_TEST):
        print(pargs);

testvals = dict(testint=1234567,
                testfloat = 3.1415927,
                teststring = 'Hello World!',
                testlist = [1, 'hello', 5.4321, {'innerdictkey':'innerdictvalue'}],
                testdict = dict(a=1,b=2,c=3));




def make_sure_dir_exists(path):
    pparts = os.path.split(path);
    destfolder = pparts[0]+os.sep;
    try:
        os.makedirs(destfolder)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def GetTestDir(test_file_path):
    dir_path = os.path.dirname(os.path.realpath(test_file_path))
    dirname = os.path.splitext(os.path.basename(test_file_path))[0]+'_tempdir';
    counter = 0;
    testdir = os.path.join(dir_path,'tmp',dirname);
    while(os.path.isdir(testdir)):
        counter = counter+1;
        testdir = os.path.join(dir_path, 'tmp', dirname + str(counter));

    if(testdir[-1] !=os.sep):
        testdir = testdir+os.sep;
    return testdir;



def StressFeatureFuncs(a):
    for f in a._feature_funcs():
        a.get_feature(f)
    for f in a._feature_funcs():
        a.get_feature(f)
    for f in a._feature_funcs():
        a.get_feature(f, force_recompute=True)
        tprint("Feature '{}' is OK!".format(f));



# class TestClass(SavesDirectories, SavesFeatures):
#     def __init__(self, path=None):
#         """
#         """
#         super(TestClass, self).__init__(path=path);
#
#
#     def _set_file_path(self, file_path=None, **kwargs):
#         super(TestClass, self)._set_file_path(file_path=file_path, **kwargs);
#         self.addDirIfMissing(name='features', folder_name="Features");
#         self.features_root = self.get_dir('features');
#
#     @FeatureFunction('identitymatrix')
#     def getIdentityMatrix(self, size=None):
#         if(size is None):
#             size = 5;
#         return np.identity(size);
#
#
# class TestClass2(SavesFeaturesDir):
#     def __init__(self, *args, **kwargs):
#         """
#         """
#         super(TestClass2, self).__init__(*args, **kwargs);
#
#     @FeatureFunction('n_letter_of_alphabet')
#     def getNLettersOfAlphabet(self, n=None):
#         if(n is None):
#             n=5;
#         alphabet = 'abcdefghijklmnopqrstuvwxyz';
#         return alphabet[:n];
#
#     @FeatureFunction('onestimesn')
#     def getIdentityMatrix(self, size=None, n=None):
#         if (size is None):
#             size = [2,3];
#         if(n is None):
#             n=3;
#         return np.ones(size)*n;
#
#
# def CreateTestHasFeaturesDirInstance(dest_dir):
#     TC = SavesFeaturesDir(path=dest_dir)
#     featuredict = dict(testint =  random.randint(0, 1000000000),
#          testfloat = random.random() * 1000000000.0,
#          teststring = 'Hello World TESTSTRING!!',
#          testlist=[1, 'hello', 5.839, random.random() * 1000000000.0, random.randint(0, 1000000000)],
#          testdict=dict(integerval=1, stringval='hello', floatval=5.839))
#     # TC.printAsDictionary()
#     for f in featuredict:
#         TC.setFeature(name=f, value=featuredict[f]);
#     TC.saveFeatures(features_to_save=['each', 'all'])
#     TC.set_info(label='testinfostring', value='testvalueforinfostring');
#     TC.set_info(label='testinfofloat', value=3.1415926536);
#     TC.set_info(label='randomlist', value=[random.random() * 1000000000.0, random.randint(0, 1000000000), random.random() * 1000000000.0, random.randint(0, 1000000000), random.random() * 1000000000.0, random.randint(0, 1000000000)]);
#     TC.saveJSON();
#     return TC;