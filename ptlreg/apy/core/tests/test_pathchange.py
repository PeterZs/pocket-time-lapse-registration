from ptlreg.apy.core.tests.test_helper import *
import unittest

testdir = GetTestDir(__file__);


class PathChangeTest(unittest.TestCase):
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

    def testMoveDir(self):
        initial_dir = os.path.join(testdir, 'initial_dir');
        ptlreg.apy.utils.make_sure_path_exists(initial_dir);
        new_parent_dir = os.path.join(testdir, 'new_parent_dir');
        ptlreg.apy.utils.make_sure_path_exists(new_parent_dir);
        TC = CreateTestHasFeaturesDirInstance(initial_dir);
        new_dir = os.path.join(new_parent_dir, 'new_dir');

        shutil.copytree(initial_dir, new_dir);
        TCL = SavesFeaturesDir(path=new_dir);
        TCL.load_features(features_to_load='all');
        self.assertFeaturesEqual(TC, TCL);
        TCLE = SavesFeaturesDir(path=new_dir);
        TCLE.load_features(features_to_load='each');
        self.assertFeaturesEqual(TC, TCLE);
        self.assertEqual(TCLE.directories, TC.directories);

def main():
    unittest.main();

if __name__ == '__main__':
    main()
