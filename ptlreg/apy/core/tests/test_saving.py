from ptlreg.apy.core.tests.test_helper import *
import unittest

testdir = GetTestDir(__file__);


class SavingTest(unittest.TestCase):
    """
    saves and loads from same location, same class
    """
    @classmethod
    def setUpClass(cls):
        make_sure_dir_exists(testdir);

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(testdir)
        return;

    def testRegisteredFunctions(self):
        TC = TestClass(path=testdir);
        self.assertTrue(np.array_equal(TC.get_feature('identitymatrix', size=5), np.identity(5)));

    def assertFeaturesEqual(self, reference_instance, test_instance):
        for f in reference_instance.get_features_list():
            self.assertEqual(reference_instance.get_feature(f),
                             test_instance.get_feature(f),
                             "Feature {} mismatch\nreference_instance: {}\ntest_instance: {}"
                             .format(f, reference_instance.get_feature(f), test_instance.get_feature(f)));

    def assertInfoEqual(self, reference_instance, test_instance):
        for infokey in reference_instance._serialize_info():
            self.assertEqual(reference_instance.get_info(infokey),
                             test_instance.get_info(infokey),
                             "Info {} mismatch\nreference_instance: {}\ntest_instance: {}".format(infokey, reference_instance.get_info(infokey), test_instance.get_info(infokey)));

    def TestClassTest(self):
        TC = TestClass(path=testdir)
        for tv in testvals:
            TC.set_feature(name=tv, value=testvals[tv]);
        TC.save_features(features_to_save = ['each', 'all'])
        TC.save_json();
        TCL = TestClass(path=testdir);
        TCL.load_features(features_to_load='all');
        self.assertFeaturesEqual(TC, TCL);
        self.assertInfoEqual(TC,TCL);


        TCLE = TestClass(path=testdir);
        TCLE.load_features(features_to_load='each');

        self.assertFeaturesEqual(TC, TCLE);
        self.assertInfoEqual(TC, TCLE);


    def SavesFeaturesDirClass(self):
        TC = CreateTestHasFeaturesDirInstance(testdir);
        TCL = SavesFeaturesDir(path=testdir);
        TCL.load_features(features_to_load='all');
        # print("TC has {}, TCL has {}".format(TC.getFeaturesList(), TCL.getFeaturesList()));
        self.assertFeaturesEqual(TC, TCL);
        self.assertInfoEqual(TC, TCL);
        TCLE = SavesFeaturesDir(path=testdir);
        TCLE.load_features(features_to_load='each');

        self.assertFeaturesEqual(TC, TCLE);
        self.assertInfoEqual(TC, TCLE);

    def testMoveDir(self):
        return;




def main():
    unittest.main();

if __name__ == '__main__':
    main()
