from ptlreg.apy.core.tests.test_helper import *
import unittest


testdir = GetTestDir(__file__);

class LoadingTest(unittest.TestCase):
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
        TC2 = TestClass2(path=testdir);
        self.assertEqual('abcde', TC2.get_feature('n_letter_of_alphabet'))
        self.assertEqual('abcdefgh', TC2.get_feature('n_letter_of_alphabet', n=8, force_recompute=True));
        self.assertEqual('abcdefgh', TC2.get_feature('n_letter_of_alphabet'));


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

    # def assertAInfoEqual(self, reference_instance, test_instance, info_keys=None):
    #     for infokey in reference_instance.serializeInfo()['_ainfo']:
    #         if((info_keys is None) or infokey in info_keys):
    #             self.assertEqual(reference_instance.get_info(infokey),
    #                              test_instance.get_info(infokey),
    #                              "Info {} mismatch\nreference_instance: {}\ntest_instance: {}".format(infokey, reference_instance.get_info(infokey),                                                                                                 test_instance.get_info(infokey)));



    def test1(self):
        TC = TestClass(path=testdir)

        for tv in testvals:
            TC.set_feature(name=tv, value=testvals[tv]);

        TC.save_features(features_to_save = ['each', 'all'])
        TC.save_json();

        TCL = TestClass(path=testdir);
        TCL.load_features(features_to_load='all', features_dir=TC.features_dir);

        self.assertFeaturesEqual(TC, TCL);
        self.assertInfoEqual(TC,TCL);


        TCLE = TestClass(path=testdir);
        TCLE.load_features(features_to_load='each', features_dir=TC.features_dir);

        self.assertFeaturesEqual(TC, TCLE);
        self.assertInfoEqual(TC, TCLE);




    def test2(self):
        TC = CreateTestHasFeaturesDirInstance(testdir);

        TCL = SavesFeaturesDir(path=testdir);
        TCL.load_features(features_to_load='all', features_dir=TC.features_dir);
        self.assertFeaturesEqual(TC, TCL);
        self.assertInfoEqual(TC, TCL);


        TCLE = SavesFeaturesDir(path=testdir);
        TCLE.load_features(features_to_load='each', features_dir=TC.features_dir);

        self.assertFeaturesEqual(TC, TCLE);
        self.assertInfoEqual(TC, TCLE);

    def testMoveDir(self):
        return;


    def testAObjects(self):
        TestAObjects();

    def testLoadDirectory(self):
        TSFD = CreateTestHasFeaturesDirInstance(testdir);
        TC1 = TestClass(path=testdir);
        TC1.save_json();
        TC2 = TestClass2(path=testdir);
        TC2.save_json();

        c = load_jsons_from_directory(testdir)
        lTSFD = c[SavesFeaturesDir.aobject_type_name()][0];
        lTC1 = c[TestClass.aobject_type_name()][0];
        lTC2 = c[TestClass2.aobject_type_name()][0];
        self.assertInfoEqual(TSFD, lTSFD);
        self.assertInfoEqual(TC1, lTC1);
        self.assertInfoEqual(TC2, lTC2);

        moved_info_keys = ["aobject_type_name", "file_base_name", "file_ext", "file_path","randomlist", "testinfofloat", "testinfostring"]

        newtestdirs = [];
        newtestdirs.append(os.path.join(testdir, 'innerdir'+os.sep));
        newtestdirs.append(GetTestDir(__file__));
        for ntd in newtestdirs:
            # make_sure_dir_exists(ntd);
            shutil.copytree(testdir, ntd);
            c = load_jsons_from_directory(testdir)
            lTSFD = c[SavesFeaturesDir.aobject_type_name()][0];
            lTC1 = c[TestClass.aobject_type_name()][0];
            lTC2 = c[TestClass2.aobject_type_name()][0];
            self.assertInfoEqual(TSFD, lTSFD, info_keys=moved_info_keys);
            self.assertInfoEqual(TC1, lTC1,  info_keys=moved_info_keys);
            self.assertInfoEqual(TC2, lTC2,  info_keys=moved_info_keys);
            shutil.rmtree(ntd)

    def testLoadReferences(self):
        file1_path = os.path.join(testdir, 'ob1.json');
        file2_path = os.path.join(testdir, 'ob2.json');
        ob1 = SavesToJSON(file1_path);
        ob2 = SavesToJSON(file2_path);

        ob1.set_info('ob2', ob2);
        ob2.set_info('ob1', ob1);

        ob1.save_json();
        ob2.save_json();


        path = testdir


    def testAObjectLoadSave(self):
        aoclasses = AObject._registered_aobject_classes();
        for cname in aoclasses:
            c = aoclasses[cname]
            testi1 = c();
            for ti in testvals:
                testi1.set_info(label=ti, value=testvals[ti]);
            d = testi1.to_dictionary();
            testi2 = AObject.create_from_dictionary(d);
            for i in range(5):
                testi2b = AObject.create_from_dictionary(testi2.to_dictionary());
                testi2 = AObject.create_from_dictionary(testi2b.to_dictionary());
            self.assertEqual(testi1.to_dictionary(), testi2.to_dictionary());






def main():
    unittest.main();

if __name__ == '__main__':
    main()
