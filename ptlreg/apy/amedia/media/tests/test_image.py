from .test_helper import *
import unittest

testdir = GetTestDir(__file__);

if(not TEST_MEDIA):
    print("abepy.testing.TEST_MEDIA is False; SKIPPING IMAGE TEST!")
else:
    class ImageTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            make_sure_dir_exists(testdir);

        @classmethod
        def tearDownClass(cls):
            shutil.rmtree(testdir)
            return;

        def assertFeaturesEqual(self, reference_instance, test_instance):
            for f in reference_instance.get_features_list():
                self.assertEqual(reference_instance.get_feature(f),
                                 test_instance.get_feature(f),
                                 "Feature {} mismatch\nreference_instance: {}\ntest_instance: {}"
                                 .format(f, reference_instance.get_feature(f), test_instance.get_feature(f)));

        def assertInfoEqual(self, reference_instance, test_instance, info_keys=None):
            for infokey in reference_instance._serialize_info():
                if((info_keys is None) or infokey in info_keys):
                    self.assertEqual(reference_instance.get_info(infokey),
                                     test_instance.get_info(infokey),
                                     "Info {} mismatch\nreference_instance: {}\ntest_instance: {}".format(infokey, reference_instance.get_info(infokey), test_instance.get_info(infokey)));



        def test1(self):
            test_image = GetTestImage();
            test_write_path = os.path.join(testdir, 'testwrite_' + test_image.file_name);
            test_image.write_to_file(test_write_path);
            self.assertEqual(test_image.n_color_channels, 4);
            StressFeatureFuncs(test_image)


if __name__ == '__main__':
    unittest.main()