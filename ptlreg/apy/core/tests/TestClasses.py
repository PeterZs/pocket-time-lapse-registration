from ptlreg.apy.core import SavesFeatures, FeatureFunction
from ptlreg.apy.core import SavesDirectories

import numpy as np
import random
import ptlreg.apy.utils

savesfeaturesid = str (ptlreg.apy.utils.GetUUID());
class SavesFeaturesDir(SavesFeatures, SavesDirectories):
    @classmethod
    def aobject_type_name(cls):
        return cls.__name__+savesfeaturesid[:7];

    def __init__(self, path=None):
        """
        """
        super(SavesFeaturesDir, self).__init__(path=path);


    def init_dirs(self, **kwargs):
        super(SavesFeaturesDir, self).init_dirs(**kwargs);
        self.add_dir_if_missing(name='features', folder_name="Features");
        self.features_root = self.get_dir('features', absolute_path=True)


testclassid = str (ptlreg.apy.utils.GetUUID());
class TestClass(SavesFeatures, SavesDirectories):
    @classmethod
    def aobject_type_name(cls):
        return cls.__name__+testclassid[:7];
    def __init__(self, path=None):
        """
        """
        super(TestClass, self).__init__(path=path);


    def init_dirs(self, **kwargs):
        super(TestClass, self).init_dirs(**kwargs);
        self.add_dir_if_missing(name='features', folder_name="Features");
        self.features_root = self.get_dir('features', absolute_path=True);

    @FeatureFunction('identitymatrix')
    def getIdentityMatrix(self, size=None):
        if(size is None):
            size = 5;
        return np.identity(size);

testclass2id = str (ptlreg.apy.utils.GetUUID());
class TestClass2(SavesFeaturesDir):
    @classmethod
    def aobject_type_name(cls):
        return cls.__name__+testclass2id[:7];
    def __init__(self, *args, **kwargs):
        """
        """
        super(TestClass2, self).__init__(*args, **kwargs);

    @FeatureFunction('n_letter_of_alphabet')
    def getNLettersOfAlphabet(self, n=None):
        if(n is None):
            n=5;
        alphabet = 'abcdefghijklmnopqrstuvwxyz';
        return alphabet[:n];

    @FeatureFunction('onestimesn')
    def getIdentityMatrix(self, size=None, n=None):
        if (size is None):
            size = [2,3];
        if(n is None):
            n=3;
        return np.ones(size)*n;


def CreateTestHasFeaturesDirInstance(dest_dir):
    TC = SavesFeaturesDir(path=dest_dir)
    featuredict = dict(testint =  random.randint(0, 1000000000),
         testfloat = random.random() * 1000000000.0,
         teststring = 'Hello World TESTSTRING!!',
         testlist=[1, 'hello', 5.839, random.random() * 1000000000.0, random.randint(0, 1000000000)],
         testdict=dict(integerval=1, stringval='hello', floatval=5.839))
    # TC.printAsDictionary()
    for f in featuredict:
        TC.set_feature(name=f, value=featuredict[f]);
    TC.save_features(features_to_save=['each', 'all'])
    TC.set_info(label='testinfostring', value='testvalueforinfostring');
    TC.set_info(label='testinfofloat', value=3.1415926536);
    TC.set_info(label='randomlist', value=[random.random() * 1000000000.0, random.randint(0, 1000000000), random.random() * 1000000000.0, random.randint(0, 1000000000), random.random() * 1000000000.0, random.randint(0, 1000000000)]);
    TC.save_json();
    return TC;