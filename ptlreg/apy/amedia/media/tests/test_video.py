from .test_helper import *
import unittest

testdir = GetTestDir(__file__);

if(not TEST_MEDIA):
    print("abepy.testing.TEST_MEDIA is False; SKIPPING IMAGE TEST!")
else:
    class VideoTest(unittest.TestCase):
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
            print("getting test video in test1");
            test_vid = GetTestVideo();
            print("got test video in test1");
            test_write_path = os.path.join(testdir, 'testwrite_' + test_vid.file_name);
            test_vid.write_to_file(test_write_path);
            StressFeatureFuncs(test_vid);

        def testFuncs(self):
            print("getting test video in testFuncs");
            vid = GetTestVideo();
            print("got test video in testFuncs");
            frame_index = vid.get_frame(10);

            def getVidFrameFromTime(vidin, t):
                f = t*vidin.sampling_rate;
                return vidin.get_frame(f=f);

            frame_time = getVidFrameFromTime(vid,1.2655);
            # frame_time = vid.getFrameFromTime(1.2655);
            # self.assertEqual(test_vid.n_color_channels, 4);

            StressFeatureFuncs(frame_index);
            StressFeatureFuncs(frame_time);
            self.assertEqual(frame_index.shape[0],frame_time.shape[0]);
            self.assertEqual(frame_index.shape[1],frame_time.shape[1]);

            audio = vid.audio;
            self.assertTrue(np.abs(audio.duration-vid.duration)<1);


if __name__ == '__main__':
    unittest.main()