import warnings
from unicodedata import category

from ptlreg.apy.core.filepath import FilePathList, FilePath
from ptlreg.apydn import FPath
from ptlreg.apydn.imagegraph.ImageUtils import ExposureCorrect
from ptlreg.apydn.imagegraph.capture.CaptureConstants import CaptureConstants
from ptlreg.apydn.imagegraph.capture.CaptureSample import CaptureSampleSet
from ptlreg.apydn.imagegraph.capture.CaptureSession import CaptureSession
from ptlreg.apydn.imagegraph.capture.ops import CaptureSessionSelectionOp
from ptlreg.apydn.imagegraph.colmap import COLMAPper
from ptlreg.apydn.imagegraph.dataset.ImageDataset import ImageDataset
import os
import numpy as np
import pandas as pd

class CaptureTarget(ImageDataset):
    SUBSET_CLASS = CaptureSampleSet;
    SAMPLE_SET_TYPE = CaptureSampleSet;

    def __init__(self,
                 path,
                 name=None,
                 # key_sample_name=None,
                 # key_primary_names=None,
                 # calibration_node=None,
                 target_gps=None,
                 read_only=False,
                 **kwargs
                 ):
        """
        Initializes a CaptureTarget instance.
        :param path: path to the target directory.
        :param name: name of target, if None, will use target_id.
        :param key_sample_name:
        :param key_primary_names:
        :param calibration_node:
        :param target_gps:
        :param read_only:
        :param kwargs:
        """
        fpath = FPath.From(path);
        if (name is None):
            name = fpath.get_directory_name();

        super(CaptureTarget, self).__init__(path=path, **kwargs)
        if (self.name is None):
            self.name = name;
        self._cdb = None;
        if (target_gps is not None):
            self.target_gps = target_gps;
        self.read_only = read_only;


    def get_originals_directory(self):
        return self.get_images_subdir(CaptureConstants.ORIGINALS_SUBDIR_NAME);

    def add_original_sample_category_dir_if_missing(self, category):
        return self.add_images_subdir_if_missing(name=self._get_images_subdir_subpath_for_original_sample_category(category),
                                             add_images=False);

    def init_dirs(self, **kwargs):
        super(CaptureTarget, self).init_dirs(**kwargs);
        self.add_images_subdir_if_missing(name=CaptureConstants.ORIGINALS_SUBDIR_NAME,add_images=False);
        categories = [
            CaptureConstants.PRIMARY_CATEGORY_NAME,
            CaptureConstants.SECONDARY_CATEGORY_NAME,
        ]
        for category in categories:
            self.add_original_sample_category_dir_if_missing(category);

    # <editor-fold desc="Property: 'name'">
    @property
    def name(self):
        return self.get_info("name");

    @name.setter
    def name(self, value):
        self.set_info('name', value);

    # </editor-fold>


    def save(self, *args, **kwargs):
        if (self.read_only):
            return;
        else:
            super(CaptureTarget, self).save(*args, **kwargs);

    @property
    def central_node(self):
        nodepath = self.get_info('central_node_path');
        if (nodepath is None):
            return None;
        return self.get_sample_for_path(nodepath)

    @central_node.setter
    def central_node(self, value):
        print("Setting central node path to {}".format(value));
        self.set_info('central_node_path', value.file_path);

    # <editor-fold desc="Property: 'target_gps'">
    @property
    def target_gps(self):
        return self.get_info("target_gps");

    @target_gps.setter
    def target_gps(self, value):
        self.set_info('target_gps', value);
    # </editor-fold>

    # <editor-fold desc="Property: 'target_timezone'">
    @property
    def target_timezone(self):
        return self.get_info("target_timezone");

    @target_timezone.setter
    def target_timezone(self, value):
        self.set_info('target_timezone', value);
    # </editor-fold>

    @classmethod
    def HomographyTest(cls, matrix, homography_threshold=None, **kwargs):
        if (homography_threshold is None):
            homography_threshold = 0.1

        if (matrix is None):
            return None;
        if (matrix[2, 2] == 0):
            return False;
        matcopy = matrix.copy();
        matcopy = matcopy / matcopy[2, 2];
        detLin = np.linalg.det(matcopy[:2, :2]);
        detHom = np.linalg.det(matcopy);
        if (detLin <= 0 or detHom < 0):
            return False
        score = detHom;
        if (score > 1):
            score = 1 / score;
        if (score < homography_threshold):
            print("REJECTED SCORE {}".format(score))
            return False;
        return score;

    def pull_directory_to_originals(self, source_dir, on_repeat='skip', category=None, force_metadata_timestamps=False, **kwargs):
        """
        Pulls a directory of images into the target's originals subdirectory.
        :param source_dir: The source directory to pull from.
        :param on_repeat: What to do if an image already exists in the target's originals subdirectory.
        :param kwargs: Additional keyword arguments to pass to the pull_new_image method.

        :return: None
        """
        if (not os.path.exists(source_dir)):
            raise ValueError("Source directory does not exist: {}".format(source_dir));
        if(category is None):
            new_samples = self.pull_directory(source_dir=source_dir, images_subdir = CaptureConstants.ORIGINALS_SUBDIR_NAME, on_repeat=on_repeat, **kwargs);
        else:
            new_samples = self.pull_directory(source_dir=source_dir,
                                              images_subdir=self._get_images_subdir_subpath_for_original_sample_category(category), on_repeat=on_repeat,
                                              **kwargs);
        for n in new_samples:
            n._calc_timestamp(force_use_metadata=force_metadata_timestamps);
            self._add_original_sample_labels(n, category=category);
        return new_samples;

    def pull_directory_to_primaries(self, source_dir, on_repeat='skip', **kwargs):
        """
        Pulls a directory of images into the target's primary originals subdirectory.
        :param source_dir:
        :param on_repeat:
        :param kwargs:
        :return:
        """
        return self.pull_directory_to_originals(source_dir, on_repeat=on_repeat, category=CaptureConstants.PRIMARY_CATEGORY_NAME, **kwargs);

    def pull_directory_to_secondaries(self, source_dir, on_repeat='skip', **kwargs):
        """
        Pulls a directory of images into the target's secondary originals subdirectory.
        :param source_dir:
        :param on_repeat:
        :param kwargs:
        :return:
        """
        return self.pull_directory_to_originals(source_dir, on_repeat=on_repeat, category=CaptureConstants.SECONDARY_CATEGORY_NAME, **kwargs);


    ##################//--Categories--\\##################
    # <editor-fold desc="Categories">

    def get_aligned_pano_images_dir(self, tag):
        if (tag is None):
            return self.get_images_subdir(self._get_aligned_directory_name(self._get_panos_derivative_name_part()));
        else:
            return self.get_images_subdir(
                self._get_aligned_directory_name(self._get_panos_derivative_name_part()) + "_{}".format(tag));

    def _get_aligned_directory_name(self, directory_name):
        return directory_name + "_aligned";

    def _add_aligned_panos_dir_if_missing(self, tag=None):
        if (tag is None):
            self.add_images_subdir_if_missing(self._get_aligned_directory_name(self._get_panos_derivative_name_part()));
        else:
            self.add_images_subdir_if_missing(
                self._get_aligned_directory_name(self._get_panos_derivative_name_part()) + "_{}".format(tag));

    def _add_original_sample_labels(self, sample, category=None):
        if (not sample.get_label_value(CaptureConstants.ORIGINAL_SAMPLE_TAG)):
            sample.add_tag_label(CaptureConstants.ORIGINAL_SAMPLE_TAG);
        if (category is not None):
            self._add_sample_category_label(sample, category);

    def _add_sample_category_label(self, sample, category):
        """
        Adds a label to the sample indicating the category of the original sample.
        :param sample: The sample to add the label to.
        :param category: The category of the original sample.
        """
        if (not sample.get_label_value(self._get_original_sample_category_label(category))):
            sample.add_tag_label(self._get_original_sample_category_label(category));

    def _get_original_sample_category_label(self, category):
        """
        Returns the label for the original sample category.
        :param category: The category of the original sample.
        :return: The label for the original sample category.
        """
        # for now just return the category name as a tag label
        return category

    def _add_primary_labels(self, sample):
        self._add_original_sample_labels(sample, category=CaptureConstants.PRIMARY_CATEGORY_NAME);


    def _remove_original_sample_labels(self, sample):
        """
        Removes the original sample labels from the sample.
        :param sample: The sample to remove the labels from.
        """
        if (sample.has_tag(CaptureConstants.ORIGINAL_SAMPLE_TAG)):
            sample.remove_tag_label(CaptureConstants.ORIGINAL_SAMPLE_TAG);
        if (sample.has_tag(CaptureConstants.PRIMARY_CATEGORY_NAME)):
            sample.remove_tag_label(CaptureConstants.PRIMARY_CATEGORY_NAME);
        if (sample.has_tag(CaptureConstants.SECONDARY_CATEGORY_NAME)):
            sample.remove_tag_label(CaptureConstants.SECONDARY_CATEGORY_NAME);

    def _set_sample_to_primary(self):
        """
        Sets the sample to primary by adding the primary category label and removing the secondary category label.
        :return: None
        """
        if (self.has_tag(CaptureConstants.SECONDARY_CATEGORY_NAME)):
            self.remove_tag_label(CaptureConstants.SECONDARY_CATEGORY_NAME);
        self.add_tag_label(CaptureConstants.PRIMARY_CATEGORY_NAME);

    def _set_sample_to_secondary(self):
        """
        Sets the sample to secondary by adding the secondary category label and removing the primary category label.
        :return: None
        """
        if (self.has_tag(CaptureConstants.PRIMARY_CATEGORY_NAME)):
            self.remove_tag_label(CaptureConstants.PRIMARY_CATEGORY_NAME);
        self.add_tag_label(CaptureConstants.SECONDARY_CATEGORY_NAME);

    def _add_secondary_labels(self, sample):
        self._add_original_sample_labels(sample, category=CaptureConstants.SECONDARY_CATEGORY_NAME)

    def _add_pano_labels(self, sample):
        if (not sample.get_label_value(CaptureConstants.PANO_CATEGORY_NAME)):
            sample.add_tag_label(CaptureConstants.PANO_CATEGORY_NAME);

    def get_original_samples(self):
        return self.samples.get_with_tag(CaptureConstants.ORIGINAL_SAMPLE_TAG);

    def get_original_samples_for_category(self, category):
        # return self.samples.get_with_label_value(CaptureConstants.ORIGINAL_SAMPLE_TAG, value=category);
        return self.samples.get_with_tag(CaptureConstants.ORIGINAL_SAMPLE_TAG).get_with_tag(category);

    def get_primary_samples(self):
        return self.get_original_samples_for_category(CaptureConstants.PRIMARY_CATEGORY_NAME);

    def get_secondary_samples(self):
        return self.get_original_samples_for_category(CaptureConstants.SECONDARY_CATEGORY_NAME);

    def _alignment_method_tag(self, method):
        if(method is None):
            return CaptureConstants.ALIGNED_SAMPLE_TAG;
        else:
            return CaptureConstants.ALIGNED_SAMPLE_TAG + "_{}".format(method);

    def _add_aligned_labels(self, sample, method=None):
        if (not sample.get_label_value(CaptureConstants.ALIGNED_SAMPLE_TAG)):
            sample.add_tag_label(self._alignment_method_tag(method));

    def _add_aligned_pano_sample_for_path(self, path, method=None):
        new_pano = self.add_sample_for_image_path(path);
        self._add_aligned_labels(new_pano, method);
        self._add_pano_labels(new_pano);
        return new_pano;
    # </editor-fold>
    ##################\\--Categories--//##################

    ##################//--Pull Data--\\##################
    # <editor-fold desc="Pull Data">

    def _get_derivative_name_part_for_original_sample_category(self, category):
        return category;
        # return os.path.join('originals', category);
    def _get_panos_derivative_name_part(self):
        return self._get_derivative_name_part_for_original_sample_category(CaptureConstants.PANO_CATEGORY_NAME);

    def _get_images_subdir_subpath_for_original_sample_category(self, category):
        return os.path.join(CaptureConstants.ORIGINALS_SUBDIR_NAME, self._get_derivative_name_part_for_original_sample_category(category=category))

    def get_images_dir_for_original_sample_category(self, category):
        return self.get_images_subdir(self._get_images_subdir_subpath_for_original_sample_category(category));

    def get_primary_images_dir(self):
        return self.get_images_dir_for_original_sample_category(CaptureConstants.PRIMARY_CATEGORY_NAME);

    def get_secondary_images_dir(self):
        return self.get_images_dir_for_original_sample_category(CaptureConstants.SECONDARY_CATEGORY_NAME);

    def pull_new_image_for_original_sample_category(self, category, fpath, width=None, filename_fn=None,
                                                    on_repeat='skip', **kwargs):
        newSample = super(CaptureTarget, self).pull_new_image(fpath,
                                                              images_subdir=self._get_images_subdir_subpath_for_original_sample_category(
                                                                  category),
                                                              width=width, filename_fn=filename_fn, on_repeat=on_repeat,
                                                              **kwargs)
        if (newSample is None):
            return;
        self._add_original_sample_labels(newSample, category=category);
        return newSample;

    def pull_new_primary_image(self, fpath, width=None, filename_fn=None, on_repeat='skip'):
        return self.pull_new_image_for_original_sample_category(category=CaptureConstants.PRIMARY_CATEGORY_NAME, fpath=fpath,
                                                                width=width, filename_fn=filename_fn, on_repeat=on_repeat);

    def pull_new_secondary_image(self, fpath, width=None, filename_fn=None, on_repeat='skip', calibration=None, correct_distortion=True):
        return self.pull_new_image_for_original_sample_category(category=CaptureConstants.SECONDARY_CATEGORY_NAME, fpath=fpath,
                                                                width=width, filename_fn=filename_fn, on_repeat=on_repeat, calibration=calibration, correct_distortion=correct_distortion);

    def _update_images_for_original_sample_category(self, category):
        samples = FilePathList.from_directory(self.get_images_dir_for_original_sample_category(category),
                                              extension_list=['.jpeg', '.jpg', '.png']);
        new_samples = []
        for p in samples:
            if (p.file_name[0] != "."):
                existing = self.get_image_sample_for_path(p.absolute_file_path);
                if (existing is None):
                    new_sample = self.add_sample_for_image_path(p.absolute_file_path);
                    self._add_original_sample_labels(new_sample, category=category);
                    new_samples.append(new_sample);
        return self.create_sample_set(new_samples);

    def _update_primary_images(self):
        return self._update_images_for_original_sample_category(CaptureConstants.PRIMARY_CATEGORY_NAME);

    def _update_secondary_images(self):
        return self._update_images_for_original_sample_category(CaptureConstants.SECONDARY_CATEGORY_NAME);

    # </editor-fold>
    ##################\\--Pull Data--//##################


    ##################//--CaptureSessionSelection--\\##################
    # <editor-fold desc="CaptureSessionSelection">

    def cluster_into_sessions(self, seconds_before_new_primary=30):
        samples = self.get_original_samples();
        samples.calc_time_since_prev_sample()
        self._add_primary_labels(samples[0])
        for s in samples[1:]:
            if (s.get_label_value("dt_prev").seconds > 5):
                self._add_primary_labels(s)
            else:
                self._add_secondary_labels(s)
        return self.get_session_set();

    @CaptureSessionSelectionOp("originals")
    def get_session_set(self):
        original = self.get_original_samples().sort_by_timestamp()
        sample_set = self.SAMPLE_SET_TYPE();
        newNode = None;
        for s in original:
            if (s._is_primary_sample):
                newNode = self._create_capture_session_with_primary(s);
                sample_set.append(newNode);
            if (s._is_secondary_sample):
                newNode.append(s);
        return sample_set;

    def _create_capture_session_with_primary(self, primary):
        session = CaptureSession.create_with_primary(primary)
        session._set_capture_target(self);
        return session;

    @CaptureSessionSelectionOp("primary_based_sessions")
    def get_primary_based_session_set(self):
        original = self.get_original_samples().sort_by_timestamp()
        sample_set = self.SAMPLE_SET_TYPE();
        newNode = None;
        for s in original:
            if (s.hasTag("primary")):
                newNode = self._create_capture_session_with_primary(s);
                sample_set.append(newNode);
            if (s.hasTag("secondary")):
                newNode.append(s);
        return sample_set;

    @CaptureSessionSelectionOp("clustered_sessions")
    def get_clustered_session_set(self):
        original = self.get_original_samples().sort_by_timestamp()
        sample_set = self.SAMPLE_SET_TYPE();
        newNode = None;
        for s in original:
            if (s.hasTag("primary")):
                newNode = self._create_capture_session_with_primary(s);
                sample_set.append(newNode);
            if (s.hasTag("secondary")):
                newNode.append(s);
        return sample_set;
    # </editor-fold>
    ##################\\--CaptureSessionSelection--//##################

    def get_aligned_samples(self, method=None):
        return self.samples.get_with_tag(self._alignment_method_tag(method));

    def get_aligned_panos(self, method=None):
        return self.get_aligned_samples(method).get_with_tag(CaptureConstants.PANO_CATEGORY_NAME);

    def _update_aligned_panos(self, method = None):
        aligned_panos_dir_subpath = self.get_images_subdir(self._get_aligned_directory_name(self._get_panos_derivative_name_part()));
        aligned_panos_dir = self.get_dir(aligned_panos_dir_subpath);
        samples = FilePathList.from_directory(aligned_panos_dir,
                                             extension_list=['.jpeg', '.jpg', '.png']);
        new_samples = []
        for p in samples:
            if (p.file_name[0] != "."):
                existing = self.getImageSampleForPath(p.absolute_file_path);
                if (existing is None):
                    new_sample = self._add_aligned_pano_sample_for_path(p.absolute_file_path, method);
                    new_samples.append(new_sample);
        return self.create_sample_set(new_samples);

    ##################//--COLMAP--\\##################
    # <editor-fold desc="COLMAP">

    def run_matching_on_original_samples(self):
        """
        Runs COLMAP feature detection and matching on the original samples.
        :return: COLMAPper instance
        """
        return self._run_colmap(
            n_primary_neighbors=10,
            n_secondary_neighbors=5,
            n_key_primaries=0,
            recompute=True,
            use_one_round_matching=True
        );

    def get_current_colmap_base_directory(self):
        return self.get_originals_directory();

    def _get_colmapper(self, base_dir = None, remake_db=False, attempt_to_load_from_csv=False):
        if(base_dir is None):
            base_dir = self.get_current_colmap_base_directory();

        if(attempt_to_load_from_csv):
            colmapper = COLMAPper.load_from_directory(base_dir);
        else:
            colmapper = None;

        if (colmapper is None or remake_db or self.name == 'TEST'):
            # if (calibration_node is None):
            #     raise ValueError("Must provide calibration node if we are creating a new colmapper.")
            colmapper = COLMAPper(
                root_path=base_dir,
                scene_name=self.name,
                images_path=None,  # will default to root path
            )


            colmapper.save();
        return colmapper;

    def get_colmapdb(self, colmap_base_dir=None):
        # if (recipe_subdir is None):
        #     recipe_subdir = CaptureConstants.UNDISTORTED_SAMPLE_TAG;
        if(colmap_base_dir is None):
            colmap_base_dir = self.get_current_colmap_base_directory()
        if (self._cdb is None):
            colmapper = self._get_colmapper()
            # colmapper = COLMAPper.load_from_directory(colmap_base_dir);
            self._cdb = colmapper.get_colmapdb();
        return self._cdb;

    def _run_colmap(self,
                    base_dir=None,
                    n_primary_neighbors=None,
                    n_secondary_neighbors=None,
                    n_key_primaries='default',
                    recompute=False,
                    use_one_round_matching=True,
                    remake_db=False,
                    ):
        colmapper = self._get_colmapper(base_dir=base_dir);
        colmapper.detect_features(recompute=recompute);
        if(use_one_round_matching):
            colmapper.write_image_match_list(
                self._get_default_image_match_list_pairs(
                    n_primary_neighbors=n_primary_neighbors,
                    n_secondary_neighbors=n_secondary_neighbors,
                    n_key_primaries=n_key_primaries
                ),
                recompute=recompute
            );
            colmapper.match_features(recompute=recompute)
        else:
            raise NotImplementedError
            # colmapper.writeImageMatchList(
            #     self._getDefaultImageMatchListPairs(
            #         n_primary_neighbors=10,
            #         n_key_primaries=0
            #     ),
            #     recompute=recompute
            # );


        return colmapper;

    def clear_colmap(self, subdir=None):
        if (subdir is None):
            subdir = self.get_current_colmap_base_directory();

        def deleteIfExists(path):
            if (os.path.exists(path)):
                os.remove(path);

        colmapfiles = [
            os.path.join(subdir, 'colmap.db'),
            os.path.join(subdir, 'colmap.db-shm'),
            os.path.join(subdir, 'colmap.db-wal'),
            os.path.join(subdir, 'colmapper.csv'),
            os.path.join(subdir, 'colmap.ini'),
            # os.path.join(subdir, ''),
            # os.path.join(subdir, ''),
        ]
        for c in colmapfiles:
            deleteIfExists(c);

    @property
    def key_primary_names(self):
        return self.get_info("key_primary_names");

    @key_primary_names.setter
    def key_primary_names(self, value):
        self.set_info('key_primary_names', value);

    @property
    def key_sample_name(self):
        return self.get_info("key_sample");

    @key_sample_name.setter
    def key_sample_name(self, value):
        self.set_info('key_sample_name', value);


    # def write_image_match_list(self, n_primary_neighbors=None, n_key_primaries=5, output_dir=None, file_name=None):
    #     if (output_dir is None):
    #         output_dir = self.get_current_colmap_base_directory()
    #         # output_dir = self.getUndistortedDir();
    #     output_path = self._get_main_match_list_path(output_dir=output_dir, file_name=file_name);
    #     f = open(output_path, "a");
    #
    #     def relpath(fpin: FilePath):
    #         return fpin.relative(output_dir);
    #
    #     filepath_pair_list = self._get_default_image_match_list_pairs(n_primary_neighbors=n_primary_neighbors,
    #                                                                   n_key_primaries=n_key_primaries);
    #     # print(filepath_pair_list)
    #
    #     for pair in filepath_pair_list:
    #         f.write("{} {}\n".format(relpath(pair[0]), relpath(pair[1])));
    #     f.close()
    #     return output_path;

    def _get_default_image_match_list_pairs(
            self,
            n_primary_neighbors = None,
            match_dict=None,
            n_key_primaries = 5,
            n_secondary_neighbors = 4,
            sessions=None,
            # add_key_secondaries = True
            add_key_secondaries = False
    ):
        """

        :param n_primary_neighbors:
        :param match_dict:
        :param n_key_primaries:
        :param n_secondary_neighbors:
        :param sessions:
        :return:
        """

        def get_n_even_sampled(samples, n=5):
            if (n == 0):
                return []
            if (len(samples) <= n):
                return samples;
            n_samples = len(samples);
            inds = np.rint(np.linspace(0, n_samples - 1, n)).astype(int);
            return [samples[x] for x in inds]

        print("CREATING MATCH LIST\nn_primary_neighbors:{}\nn_secondary_neighbors:{}\nn_key_primaries:{}".format(
            n_primary_neighbors, n_secondary_neighbors, n_key_primaries
        ))

        if(n_primary_neighbors is None):
            n_primary_neighbors = CaptureTarget.DEFAULT_N_PRIMARY_NEIGHBORS;
        if (n_key_primaries is None):
            n_key_primaries = 3;
        if (n_key_primaries == 'default'):
            n_key_primaries = 7;

        if(n_secondary_neighbors is None):
            n_secondary_neighbors = 4
        if (sessions is None):
            sessions = self.get_session_set();
            # sessions = self.GetUndistortedSessions();

        session_primaries = [ses.primary_sample for ses in sessions];
        key_sessions = get_n_even_sampled(sessions, n=n_key_primaries);
        if(n_key_primaries>0):
            if(add_key_secondaries):
                key_samples = []
                for key_session_i in key_sessions:
                    for ssmpl in key_session_i:
                        key_samples.append(ssmpl)
            else:
                key_samples = [x.primary_sample for x in key_sessions];

        else:
            key_samples = []

        key_primary_names = self.key_primary_names;
        if (key_primary_names is None and self.key_sample_name is not None):
            key_primary_names = [self.key_sample_name];

        if (key_primary_names is not None):
            def findInGroup(key_name, group):
                for smp in group:
                    if (smp.file_name == key_name or smp.file_name_base == key_name):
                        return smp;
                return None;

            for k in key_primary_names:
                k_ink = findInGroup(k, key_samples);
                if (k_ink is None):
                    ksamp = findInGroup(k, session_primaries);
                    key_samples.append(ksamp);
            # key_primary_file_name = sessions[0].primary_sample.file_name;
            # key_sample = sessions[0].primary_sample;

        match_list = [];
        if (match_dict is None):
            match_dict = dict();

        def addMatch(match):
            if (match[0] == match[1]):
                return;
            if (match[0] is None or match[1] is None):
                return;
            if (match[0] < match[1]):
                a = match[0];
                b = match[1];
            else:
                a = match[1];
                b = match[0];
            if (a in match_dict):
                match_dict[a][b] = match;
            else:
                match_dict[a] = dict();
                match_dict[a][b] = match;

        def addMatches(matches):
            for m in matches:
                addMatch(m);

        for si in range(len(sessions)):
            s = sessions[si];
            # if (key_primary_file_name == s.primary_sample.file_name and key_sample is None):
            #     key_sample = s;

            addMatches(s._get_default_image_match_list_pairs());
            for ni in range(min(n_primary_neighbors, si)):
                addMatch(
                    [s.primary_sample.file_path, sessions[si - ni - 1].primary_sample.file_path]
                );
        for si in range(len(sessions)):
            for keysample in key_samples:
                addMatch(
                    [keysample.file_path, sessions[si].primary_sample.file_path]
                )
            for sii in range(min(n_secondary_neighbors, si)):
                secondary_neighbors = self._get_image_matches_between_sets(sessions[si].get_secondary_samples(),
                                                                           sessions[si - sii].get_secondary_samples())
                addMatches(secondary_neighbors);

        # extract a list of matches out of the dict
        for key1 in match_dict:
            for key2 in match_dict[key1]:
                match_list.append(match_dict[key1][key2]);
        return match_list;

    def _get_image_matches_between_sets(self, set1, set2):
        """
        gets match list for all images from one set against all of another
        :param set1:
        :param set2:
        :return:
        """
        match_list = [];
        for s1 in set1:
            for s2 in set2:
                if (s1.file_path != s2.file_path):
                    match_list.append([s1.file_path, s2.file_path]);
        return match_list;

    def _get_image_matches_within_set(self, set1):
        """
        Gets match list for all pairs within set
        :param set1:
        :return:
        """
        return self._get_image_matches_between_sets(set1, set1)


    # </editor-fold>
    ##################\\--COLMAP--//##################


    ##################//--Alignment graph--\\##################
    # <editor-fold desc="Alignment graph">
    def calculate_primary_alignment_graph(self):
        cmdb = self.get_colmapdb();
        primaries = self.get_primary_samples();
        primary_alignment_graph = cmdb.get_all_paths_alignment_graph_for_samples(samples=primaries);
        return primary_alignment_graph;

    def calculate_aligned_undistorted_panos(self, target_viewport_btlr='default',
                                            radial_falloff='default', skip_existing=True,
                                            method = None,
                                            alpha_exponent=2.5,
                                            alpha_scale = 5,
                                            network_cutoff=None,
                                            rewrite=False,
                                            alpha_threshold= 0.001,
                                            recompute_central_node = False,
                                            homography_threshold = None,
                                            min_alpha=0.05,
                                            save = True,
                                            use_autoexposure=True,
                                            exposure_blend=0.5,
                                            ):


        def splat(im, toim=None):
            try:
                splatIm = im;
                tpix = toim.fpixels;
                falphc = splatIm.fpixels[:, :, 3];
                talphc = toim.fpixels[:, :, 3];
                from_alpha_im = np.dstack((falphc, falphc, falphc));
                splatpix = np.clip(splatIm.pixels[:, :, :3], 0, 1);

                toim.pixels[:, :, :3] = splatpix * (from_alpha_im) + (1 - from_alpha_im) * tpix[:, :, :3];
                toim.pixels[:, :, 3] = np.clip(talphc + falphc, 0, 1);
                return True;
            except ValueError as error:
                print(error);
                return None;


        def addsplat(im, toim=None, refim=None, auto_expose = False, multiply_alpha=True, exposure_blend=None):
            try:
                if(auto_expose):
                    if(refim is not None):
                        splatout, ratios = ExposureCorrect(im, refim, exposure_blend=exposure_blend)
                        # splatout.pixels[:, :, 0] = splatout.pixels[:, :, 0] * splatout.pixels[:, :, 3];
                        # splatout.pixels[:, :, 1] = splatout.pixels[:, :, 1] * splatout.pixels[:, :, 3];
                        # splatout.pixels[:, :, 2] = splatout.pixels[:, :, 2] * splatout.pixels[:, :, 3];
                    else:
                        splatout = im.get_float_copy();
                else:
                    splatout = im.get_float_copy();
                if(multiply_alpha):
                    splatout._multiply_rgb_by_alpha_channel();
                toim.pixels = toim.pixels+splatout.pixels;
            except ValueError as error:
                print(error);
                # raise error;
                return None;



        cmdb = self.get_colmapdb();
        primaries = self.get_primary_samples();
        primaries.sort_by_timestamp();

        # Primary alignment graph calculated here
        primary_alignment_graph = self.calculate_primary_alignment_graph()

        if(self.central_node is None or recompute_central_node):
            self.central_node = primary_alignment_graph.central_sample;

        key_primary_sample = self.central_node;
        sessions = self.get_session_set();
        if (key_primary_sample is None):
            raise ValueError("Failed to compute central node!");

        # Target alignment graph calculated here
        target_alignment_graph = cmdb.get_central_sample_alignment_graph_for_samples(self.get_original_samples(),
                                                                                     central_sample=key_primary_sample,
                                                                                     cutoff=network_cutoff);
        if (target_viewport_btlr is None):
            pshape = key_primary_sample.image_shape;
            target_viewport_btlr = [0, pshape[0], 0, pshape[1]];
        elif (target_viewport_btlr == 'large'):
            factor = 0.75;
            pshape = key_primary_sample.image_shape;
            target_viewport_btlr = [int(-pshape[0] * factor), int(pshape[0] * (1 + factor)), int(-pshape[1] * factor),
                                    int(pshape[1] * (1 + factor))];
        elif (target_viewport_btlr == 'default'):
            factor = 0.5;
            pshape = key_primary_sample.image_shape;
            target_viewport_btlr = [int(-pshape[0] * factor), int(pshape[0] * (1 + factor)), int(-pshape[1] * factor),
                                    int(pshape[1] * (1 + factor))];
        output_shape = np.array([target_viewport_btlr[1] - target_viewport_btlr[0], target_viewport_btlr[3] - target_viewport_btlr[2], 4]);
        self._add_aligned_panos_dir_if_missing(method);
        output_dir = self.get_aligned_pano_images_dir(tag=method);
        new_samples = [];
        viewport_mat = np.array([[1, 0, -target_viewport_btlr[2]],
                                 [0, 1, -target_viewport_btlr[0]],
                                 [0, 0, 1]]).astype(float);
        ext = '.png';

        for session in sessions:
            primary = session.primary_sample;
            new_sample_path = os.path.join(output_dir, primary.file_name_base+ext);
            if (os.path.exists(new_sample_path) and skip_existing):
                print("Found aligned undistorted sample {}".format(new_sample_path));
            else:
                is_homography=True;
                if (primary.file_name == key_primary_sample.file_name):
                    alignment = np.eye(3);
                else:
                    alignment = primary_alignment_graph.get_alignment_matrix_for_samples(from_sample=primary,
                                                                                         to_sample=key_primary_sample);
                good_homography = is_homography and self.__class__.HomographyTest(alignment, homography_threshold);
                if (good_homography):
                    primary_matrix = viewport_mat @ alignment;
                    warped_primary_reference = primary._get_image_warped_by_matrix(primary_matrix, output_shape=output_shape, radial_alpha = False);
                    output_image = primary._get_image_warped_by_matrix(primary_matrix,
                                                                       output_shape=output_shape,
                                                                       radial_alpha = True,
                                                                       alpha_exponent=alpha_exponent,
                                                                       alpha_scale=alpha_scale,
                                                                       multiply_alpha=True
                                                                       );
                    if(use_autoexposure):
                        warped_primary_splat = primary._get_image_warped_by_matrix(primary_matrix,
                                                                                   output_shape=output_shape,
                                                                                   alpha_exponent=alpha_exponent,
                                                                                   alpha_scale=alpha_scale,
                                                                                   multiply_alpha=False,
                                                                                   radial_alpha=True,
                                                                                   );
                    else:
                        warped_primary_splat = primary._get_image_warped_by_matrix(primary_matrix, output_shape=output_shape,
                                                                                   alpha_exponent=alpha_exponent,
                                                                                   alpha_scale=alpha_scale,
                                                                                   multiply_alpha=False,
                                                                                   );
                    # border_width = 'default'
                    secondaries = session.get_secondary_samples();
                    for secondary in secondaries:
                        secondary_alignment = target_alignment_graph.get_alignment_matrix_for_sample(from_sample=secondary);
                        good_homography = self.__class__.HomographyTest(secondary_alignment, homography_threshold);
                        if(good_homography):
                            smatrix = viewport_mat @ secondary_alignment;
                            s_aligned = secondary._get_image_warped_by_matrix(
                                smatrix,
                                output_shape=output_shape,
                                radial_alpha = True,
                                alpha_exponent=alpha_exponent,
                                alpha_scale=alpha_scale,
                                min_alpha=min_alpha,
                                multiply_alpha=False
                            );
                            addsplat(s_aligned, output_image, refim=warped_primary_reference, auto_expose=use_autoexposure, exposure_blend=exposure_blend, multiply_alpha=True);
                            # coverage = s_aligned.fpixels[:, :, 3].sum() / np.product(s.image_shape[:2]);

                    if(use_autoexposure):
                        output_image = output_image.get_with_alpha_divided(threshold=alpha_threshold);
                        splat(warped_primary_splat, output_image);
                    else:
                        addsplat(warped_primary_splat, output_image);
                        output_image = output_image.get_with_alpha_divided(threshold=alpha_threshold);

                    output_image.pixels = np.clip(output_image.pixels, 0, 1);
                    output_image.write_to_file(new_sample_path);
                    new_pano_sample = self._add_aligned_pano_sample_for_path(new_sample_path, method=None);
                    new_samples.append(new_pano_sample);
                else:
                    # AWARN("Primary {} did not have a good homography".format(p.file_name));
                    warnings.warn(
                        "Primary {} did not have a good homography: {}".format(primary.file_name, good_homography));
        if (save):
            self.save();
        return self.create_sample_set(new_samples);


    def _get_distances_from_central_node_dict(self, network_cutoff = None):
        cmdb = self.get_colmapdb();
        target_alignment_graph = cmdb.get_central_sample_alignment_graph_for_samples(self.get_original_samples(),
                                                                                     central_sample=self.central_node,
                                                                                     cutoff=network_cutoff);
        return target_alignment_graph._paths_from_center_node[0];

    def calculate_alignment_graph_edge_strengths(self, target_viewport_btlr='default',
                                                 network_cutoff=None,
                                                 recompute_central_node=False,
                                                 homography_threshold=None,
                                                 ):

        cmdb = self.get_colmapdb();
        sample_name_to_imageid = pd.Series(cmdb._imageid_to_sample_name.index.values,
                                           index=cmdb._imageid_to_sample_name);

        MAX_IMAGE_ID = 2 ** 31 - 1

        def image_ids_to_pair_id(image_id1, image_id2):
            if image_id1 > image_id2:
                image_id1, image_id2 = image_id2, image_id1
            return image_id1 * MAX_IMAGE_ID + image_id2

        edge_counts = {};

        def addPath(path):
            previous_node = path[0];
            for n in path[1:]:
                fromnode = previous_node;
                tonode = n;
                edge_id = image_ids_to_pair_id(sample_name_to_imageid[fromnode], sample_name_to_imageid[tonode]);
                if (edge_id not in edge_counts):
                    edge_counts[edge_id] = 1;
                else:
                    edge_counts[edge_id] = edge_counts[edge_id] + 1;
                previous_node = tonode;

        # primaries = self.getUndistortedPrimaries();
        primaries = self.get_primary_samples();
        primaries.sort_by_timestamp();
        primary_alignment_graph = self.calculate_primary_alignment_graph()
        if (self.central_node is None or recompute_central_node):
            self.central_node = primary_alignment_graph.central_sample;
        key_primary_sample = self.central_node;
        sessions = self.get_session_set();
        # sessions = self.GetUndistortedSessions();
        if (key_primary_sample is None):
            raise ValueError("Failed to compute central node!");
        target_alignment_graph = cmdb.get_central_sample_alignment_graph_for_samples(self.get_original_samples(),
                                                                                     central_sample=key_primary_sample,
                                                                                     cutoff=network_cutoff);
        for session in sessions:
            primary = session.primary_sample;
            is_homography = True;
            primary_alignment_path = primary_alignment_graph.get_alignment_path_for_samples(from_sample=primary,
                                                                                            to_sample=key_primary_sample);
            alignment = primary_alignment_graph.get_alignment_matrix_for_samples(from_sample=primary,
                                                                                 to_sample=key_primary_sample);
            good_homography = is_homography and self.__class__.HomographyTest(alignment, homography_threshold);
            if (good_homography):
                addPath(primary_alignment_path)
                secondaries = session.get_secondary_samples();
                for secondary in secondaries:
                    secondary_alignment = target_alignment_graph.get_alignment_matrix_for_sample(from_sample=secondary);
                    good_homography = self.__class__.HomographyTest(secondary_alignment, homography_threshold);
                    if (good_homography):
                        addPath(target_alignment_graph.get_alignment_path_for_sample(secondary));
            else:
                # AWARN("Primary {} did not have a good homography".format(p.file_name));
                warnings.warn("Primary {} did not have a good homography: {}".format(primary.file_name,
                                                                           good_homography));

        rawdf = pd.DataFrame.from_dict(
            edge_counts, orient='index', columns=['times_used']
        );
        rawdf.reset_index(names="pair_id", inplace=True);
        image_ids = rawdf.pair_id.map(cmdb.image_ids_from_pair_id);
        r_dataframe = pd.DataFrame(dict(
            source=image_ids.map(lambda x: cmdb._imageid_to_sample_name[x[0]]),
            target=image_ids.map(lambda x: cmdb._imageid_to_sample_name[x[1]]),
            times_used=rawdf.times_used,
        ))
        return r_dataframe;


    # </editor-fold>
    ##################\\--Alignment graph--//##################