import os.path

from ptlreg.apy.aobject import *
from .cameracalibration import *
from .imagenodes.ImageNode import *
from ptlreg.apy.amedia import *
from .CaptureSession import *
from .ops import *
from .ImageDatasetOps import *
from .CaptureSample import *
from timezonefinder import TimezoneFinder
from ptlreg.apydn.imagegraph.colmap import *
import clip
import torch
from ptlreg.apydn.imagegraph.colmap import *

from ptlreg.apydn.imagegraph.ImageSampleConstants import ImageSampleConstants

import pytz;
from pytz.exceptions import UnknownTimeZoneError

from ..dataset.ImageDataset import ImageDataset

_tzf = TimezoneFinder();

try:
    import cv2
    import copyreg


    def _pickle_keypoints(point):
        return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                              point.response, point.octave, point.class_id)


    copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)
except ImportError:
    AWARN("Failed to import opencv!")


def get_n_even_sampled(samples, n=5):
    if(n == 0):
        return []
    if (len(samples) <= n):
        return samples;
    n_samples = len(samples);
    inds = np.rint(np.linspace(0, n_samples - 1, n)).astype(int);
    return [samples[x] for x in inds]


def DateTimeFromReCaptureString(str):
    return datetime.datetime.strptime(str, '%Y-%m-%d %H:%M:%S');


def DateTimeFromMetaData(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path.absolute_file_path))


def DateTimeFromISOString(strng):
    # return datetime.datetime.strptime(strng, "%Y-%m-%dT%H-%M-%S");
    try:
        return datetime.datetime.fromisoformat(strng);
    except ValueError:
        try:
            return datetime.datetime.strptime(strng, "%Y-%m-%dT%H-%M-%S");
        except ValueError:
            print("dash datetime format!")
            return datetime.datetime.strptime(strng, "%Y-%m-%d-%H-%M-%S");


def _compareTimeSampleFilePaths(a, b):
    return DateTimeFromISOString(a.file_name_base).timestamp() - DateTimeFromISOString(b.file_name_base).timestamp()


def _getTimeStampFromImageFilePath(a):
    return DateTimeFromISOString(a.file_name_base);


def AbsoluteTimestampDistance(a, b):
    return abs(a.timestamp - b.timestamp).total_seconds();


class MissingPrimaryImageException(Exception):
    pass;


class CaptureTarget(ImageDataset):
    SUBSET_CLASS = CaptureSampleSet;
    SAMPLE_SET_TYPE = CaptureSampleSet;


    CAPTURE_SESSION_PRODUCTS = {};
    CAPTURE_SAMPLE_PRODUCTS = {};
    DEFAULT_ALIGNED_SUBSET = 200;

    DEFAULT_N_PRIMARY_NEIGHBORS = 20;


    TimeDifferenceMetric = AbsoluteTimestampDistance;

    PANO_THUMBNAIL_TAG = "pano_thumbnail";

    @classmethod
    def HomographyTest(cls, matrix, homography_threshold = None, **kwargs):
        if(homography_threshold is None):
            homography_threshold = 0.1

        if (matrix is None):
            return None;
        if(matrix[2,2]==0):
            return False;
        matcopy = matrix.copy();
        matcopy = matcopy/matcopy[2,2];
        detLin = np.linalg.det(matcopy[:2, :2]);
        detHom = np.linalg.det(matcopy);
        if (detLin <= 0 or detHom <0):
            return False
        score = detHom;
        if (score > 1):
            score = 1/score;
        if(score<homography_threshold):
            print("REJECTED SCORE {}".format(score))
            return False;
        return score;

    def GetAnalysisDataFrame(self):
        df = self.samples.DataNodeSet().nodes;
        df = df.assign(dataset=self.name);
        df['date']=df.timestamp.map(lambda x: x.date());
        return df;


    @classmethod
    def create_sample_set(cls, *args, **kwargs):
        return cls.ImageSetType(*args, **kwargs);

    # @classmethod
    # def SubsetClass(cls):
    #     return CaptureSampleSet;

    @classmethod
    def PrimaryHomographyTest(cls, matrix, **kwargs):
        detLin = np.linalg.det(matrix[:2, :2]);
        detHom = np.linalg.det(matrix);
        if (detLin > 0 and detHom > 0):
            return True;

    # def GetSampleDataNodeSet(self):
    #     for n in self.samples:
    #

    def EstimateSunAngleForTimestamp(self, timestamp):
        gps = self.target_gps;
        target_gps = self.target_gps;
        timezone_name = _tzf.timezone_at(lng=target_gps[0], lat=target_gps[1]);
        try:
            tz = pytz.timezone(timezone_name)
            aware_datetime = timestamp.replace(tzinfo=tz)
            # aware_datetime_in_utc = aware_datetime.astimezone(utc)
            # naive_datetime_as_utc_converted_to_tz = tz.localize(naive_datetime)
        except UnknownTimeZoneError:
            pass  # {handle error}

        sun_args = [gps[0], gps[1], aware_datetime];
        sun_altitude = psol.get_altitude(*sun_args);
        sun_azimuth = psol.get_azimuth(*sun_args);
        return np.array([sun_altitude, sun_azimuth]);

    # <editor-fold desc="Property: 'target_gps'">
    @property
    def target_gps(self):
        return self.get_info("target_gps");

    @target_gps.setter
    def target_gps(self, value):
        self.set_info('target_gps', value);

    # </editor-fold>

    @property
    def key_sample_name(self):
        return self.get_info("key_sample");

    @key_sample_name.setter
    def key_sample_name(self, value):
        self.set_info('key_sample_name', value);

    # </editor-fold>

    # <editor-fold desc="Property: 'target_timezone'">
    @property
    def target_timezone(self):
        return self.get_info("target_timezone");

    @target_timezone.setter
    def target_timezone(self, value):
        self.set_info('target_timezone', value);
    # </editor-fold>

    def CalculateSunAngles(self):
        for s in self.samples:
            sun_angle = s.calculate_sun_angle();
            if (sun_angle is None and s.timestamp is not None):
                s.sun_angle = self.EstimateSunAngleForTimestamp(s.timestamp);

    @classmethod
    def TimeStampFromImageFilePath(cls, a):
        return DateTimeFromISOString(FilePath.From(a).file_name_base);

    @classmethod
    def GetTimestampFromFileAtPath(cls, path):
        filepath = FilePath.From(path);
        try:
            return DateTimeFromISOString(filepath.file_name_base);
        except ValueError:
            print("using modified time")
            return DateTimeFromMetaData(filepath);
        # return DateTimeFromISOString(.file_name_base);

    @classmethod
    def _GetTargetDirectory(cls, parent_dir, name):
        input_path = parent_dir;
        if (isinstance(parent_dir, HasFilePath)):
            input_path = parent_dir.absolute_file_path;
        target_dir = os.path.join(input_path, name);
        return target_dir + os.sep;

    def save(self, *args, **kwargs):
        if (self.read_only):
            return;
        else:
            super(CaptureTarget, self).save(*args, **kwargs);

    def __init__(self,
                 parent_dir=None,
                 target_id=None,
                 name=None,
                 key_sample_name=None,
                 key_primary_names=None,
                 calibration_node=None,
                 target_gps=None,
                 read_only=False,
                 **kwargs
                 ):
        if (name is None):
            name = target_id;
        if (target_id is None and name is not None):
            target_id = name;
        target_dir = self.__class__._GetTargetDirectory(parent_dir, name)
        # make_sure_dir_exists(target_dir);
        super(CaptureTarget, self).__init__(path=target_dir, **kwargs)
        if (self.name is None):
            self.name = name;
        if (self.target_id is None):
            self.target_id = target_id;
        if (key_sample_name is not None):
            self.key_sample_name = key_sample_name;
        if (key_primary_names is not None):
            self.key_primary_names = key_primary_names;
        self._cdb = None;
        self.calibration_node = calibration_node;
        if (target_gps is not None):
            self.target_gps = target_gps;
        self.read_only = read_only;

    @property
    def key_primary_names(self):
        return self.get_info("key_primary_names");

    @key_primary_names.setter
    def key_primary_names(self, value):
        self.set_info('key_primary_names', value);

    # <editor-fold desc="Property: 'target_id'">
    @property
    def target_id(self):
        return self.get_info("target_id");

    @target_id.setter
    def target_id(self, value):
        self.set_info('target_id', value);

    # </editor-fold>

    # <editor-fold desc="Property: 'name'">
    @property
    def name(self):
        return self.get_info("name");

    @name.setter
    def name(self, value):
        self.set_info('name', value);

    # </editor-fold>

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

    def get_original_samples(self):
        return self.samples.get_with_tag(CaptureConstants.ORIGINAL_SAMPLE_TAG);

    def get_aligned_samples(self, method=None):
        return self.samples.get_with_tag(self._alignment_method_tag(method));


    # @property
    # def original_samples(self):
    #     return self.samples.getWithTag(CaptureConstants.ORIGINAL_SAMPLE_TAG);
    #
    # @property
    # def aligned_samples(self):
    #     return self.samples.getWithTag(CaptureConstants.ALIGNED_SAMPLE_TAG);

    # def _ComputeSIFTDatabase(self, sift_database=None, force_recompute=False):
    #     # n_checks = 50;
    #     # knn = 2;
    #     # SIFT feature matcher
    #     sift = cv2.SIFT_create()
    #
    #     if(sift_database is None or force_recompute):
    #         sift_database = dict();
    #     mask = None;
    #     for s in self.samples:
    #         index = self.ImageSetType.main_index_map_func(self.samples, s);
    #         if(force_recompute or sift_database.get(index) is None):
    #             print("Computing SIFT for {}".format(s.file_name));
    #             gray = s.GetImage().GetGrayCopy().ipixels;
    #             if(mask is None or mask.shape !=gray.shape):
    #                 mask = (np.ones_like(gray)*255).astype('uint8')
    #             keypoints, descriptors = sift.detectAndCompute(gray, mask);
    #             sift_database[index]= [keypoints, descriptors];
    #     return sift_database;
    #
    #
    #
    # @FeatureFunction("sift_database")
    # def GetSIFTDatabase(self):
    #     return self._ComputeSIFTDatabase();

    def GetPrimaryTimestamps(self):
        return self.get_primary_samples().get_timestamps();

    def GetSecondaryTimestamps(self):
        return self.get_secondary_samples().get_timestamps();

    # def _WriteTestSubset(self, name, timestamps):
    #     primary_dir = "TestSet_{}_primary".format(name)
    #     primary_aligned_dir = "TestSet_{}_primary_aligned".format(name);
    #     secondary_dir = "TestSet_{}_secondary".format(name);
    #     self.AddImagesSubdir(primary_dir);
    #     self.AddImagesSubdir(primary_aligned_dir);
    #     self.AddImagesSubdir(secondary_dir);
    #     primaries = self.getPrimarySamples();
    #     original_session_set = self.GetSessionSet();
    #
    #     for t in timestamps:
    #         session = original_session_set.get

    def GetPrimarySubsamplingSet(self, n_total=None):
        if (n_total is None):
            n_total = CaptureTarget.DEFAULT_ALIGNED_SUBSET
        primaries = self.get_primary_samples();
        n = primaries.length();
        if (n_total > n):
            return primaries;
        skip = n / n_total;
        subset_inds = np.arange(0, n, skip);
        subset_inds = subset_inds.astype(int);
        if (len(subset_inds) < n_total):
            subset_inds = subset_inds + [n - 1];
        selected = [primaries[a] for a in subset_inds]
        return self.create_sample_set(selected);

    def WritePrimarySubsamplingDirectory(self, n_samples=None, scale=None, overwrite=False):
        if (n_samples is None):
            n_samples = CaptureTarget.DEFAULT_ALIGNED_SUBSET;

        primaries = self.get_primary_samples();
        if (isinstance(n_samples, str) and 'all' == n_samples.lower()):
            n_samples = primaries.length();

        subset_tag = self._primarySubsetTag(n_samples, scale=scale);
        existing = self.samples.get_with_tag(subset_tag);

        if (overwrite):
            self.deleteDir(self.get_images_subdir(subset_tag));
        elif (existing.length() == np.min([n_samples, primaries.length()])):
            return existing;
        elif (existing.length() > 0):
            AWARN("Inconsistent number of images for subset of {} found".format(n_samples));
        subset = self.GetPrimarySubsamplingSet(n_samples);
        sset = self.DuplicateSampleSetToImageSubdirWithTags(sample_set=subset, subdir_name=subset_tag,
                                                            overwrite=overwrite, scale=scale);
        self.add_images_subdir_if_missing(self._get_aligned_directory_name(subset_tag));
        return sset;


    def GetPanoThumbnails(self):
        tset = self.samples.get_with_tag(CaptureConstants.PANO_THUMBNAIL_TAG);
        tset.sort_by_timestamp();
        return tset;

    def WritePanoThumbnails(self, scale=0.2, rotate90x=None, overwrite=False, save=True):
        def _thumbnailTag(_scale):
            return [CaptureConstants.PANO_THUMBNAIL_TAG];

        panos = self.get_aligned_panos();
        thumbnail_tags = _thumbnailTag(scale);
        subdir_name = CaptureConstants.PANO_THUMBNAIL_TAG;
        existing = self.samples.get_with_tags(thumbnail_tags);

        if (overwrite):
            self.deleteDir(self.get_images_subdir(subdir_name));
        ##################//----\\##################
        # <editor-fold desc="">

        new_samples = [];
        self.add_images_subdir_if_missing(subdir_name);
        subdir = self.get_images_subdir(subdir_name);
        for s in panos:
            use_ext = '.jpeg';
            spath = os.path.join(subdir, s.file_name_base + use_ext);
            if ((not os.path.exists(spath)) or overwrite):
                im = s.get_image();
                if(rotate90x is not None):
                    im.rotate90x(rotate90x);
                if (scale is None):
                    im.GetRGBCopy().write_to_file(spath);
                else:
                    im.GetScaledByFactor(scale).GetRGBCopy().write_to_file(spath);

            s.setStringLabel(CaptureSample.PANO_THUMBNAIL_PATH_KEY, spath);

            primary = self.get_primary_samples().get_samples_with_name(s.sample_name)[0];
            primary.setStringLabel(CaptureSample.PANO_THUMBNAIL_PATH_KEY, spath);
            new_sample = self.AddSampleForImagePath(spath);
            for t in thumbnail_tags:
                new_sample.setTagLabel(t);
            s.set_thumbnail_image(new_sample);

            new_samples.append(new_sample);
        if (save):
            self.save();
        return self.create_sample_set(new_samples);

    def WritePrimaryThumbnails(self, scale=None, rotate90x=None, overwrite=False, save=True):
        def _thumbnailSubdirName(_scale):
            return "primary_thumbnail";

        def _thumbnailTag(_scale):
            return [_thumbnailSubdirName(_scale)];

        primaries = self.get_primary_samples();
        thumbnail_tags = _thumbnailTag(scale);
        subdir_name = _thumbnailSubdirName(scale);
        existing = self.samples.get_with_tags(thumbnail_tags);

        if (overwrite):
            self.deleteDir(self.get_images_subdir(subdir_name));
        ##################//----\\##################
        # <editor-fold desc="">

        new_samples = [];
        self.add_images_subdir_if_missing(subdir_name);
        subdir = self.get_images_subdir(subdir_name);
        for s in primaries:
            use_ext = '.jpeg';
            spath = os.path.join(subdir, s.file_name_base + use_ext);
            if ((not os.path.exists(spath)) or overwrite):
                im = s.get_image();
                if (rotate90x is not None):
                    im.rotate90x(rotate90x);
                if (scale is None):
                    im.GetRGBCopy().write_to_file(spath);
                else:
                    im.GetScaledByFactor(scale).GetRGBCopy().write_to_file(spath);

            # s.setStringLabel(ImageSampleConstants.THUMBNAIL_PATH_KEY, spath);

            new_sample = self.AddSampleForImagePath(spath);
            for t in thumbnail_tags:
                new_sample.setTagLabel(t);
            new_samples.append(new_sample);
            s.set_thumbnail_image(new_sample);
        if (save):
            self.save();
        return self.create_sample_set(new_samples);

    # def GetPrimarySubsamplingSet(self, n_samples=None, overwrite=False):
    #     if (n_samples is None):
    #         n_samples = CaptureTarget.DEFAULT_ALIGNED_SUBSET;
    #     subset_tag = self._primarySubsetTag(n_samples);
    #     existing = self.samples.getWithTag(subset_tag);
    #     primaries = self.getPrimarySamples();
    #     if (existing.length() == np.min([n_samples, primaries.length()])):
    #         return existing;
    #     else:
    #         return self.WritePrimarySubsamplingDirectory(n_samples=n_samples, overwrite=overwrite);

    def _primarySubsetTag(self, name, scale=None):
        if (scale is None):
            return "{}_primary_subset".format(name);
        else:
            return "{}_primary_subset_scale{}".format(name, '{0:.2f}'.format(scale).replace('.', 'p'));

    def _alignedPrimarySubsetTag(self, name):
        return self._get_aligned_directory_name(self._primarySubsetTag(name));

    def AddAlignedPrimarySubsetDirectory(self, subset_name=None):
        """
        After aligning a subset with Photoshop and saving it into the aligned subset directory (the extra one created
        when you call WritePrimarySubsamplingDirectory), you can add the aligned subset directory to the capture target
        with this function.
        :param subset_name: the tag (usually n_samples)
        :return:
        """
        if (subset_name is None):
            subset_name = CaptureTarget.DEFAULT_ALIGNED_SUBSET
        # subset_tag = "primary_subset_{}".format(n_samples);
        aligned_subset = self.AddImagesSubdirWithTags(self._alignedPrimarySubsetTag(subset_name),
                                                      tags=[self._alignedPrimarySubsetTag(subset_name), "keyframe"]);
        return aligned_subset;

    def getPrimaryKeyframes(self, subset_name=None):
        if (subset_name is None):
            subset_name = CaptureTarget.DEFAULT_ALIGNED_SUBSET;
        return self.samples.get_with_tag(self._alignedPrimarySubsetTag(subset_name));

    # def getAlignedPrimarySubset(self, n_samples):

    def CalculateSessionProduct(self, sessions, product_func, name, rewrite=False, ext='.png', **kwargs):
        subdir_name = "{}_session_product".format(name);
        self.add_images_subdir_if_missing(subdir_name);
        for s in sessions:
            product_func(s);

    # def CalculateSessionProduct(self, sessions, product_func, name, rewrite=False, ext='.png', **kwargs):
    #     subdir_name = "{}_session_product".format(name);
    #     self.add_images_subdir_if_missing(subdir_name);
    #     subdir_path = self.get_images_subdir(subdir_name);
    #     new_samples = [];
    #     for s in sessions:
    #         product_sample_name = s.primary_sample.file_name_base;
    #         product_sample_path = os.path.join(subdir_path, product_sample_name+ext);
    #         existing_sample = None;
    #         if(rewrite or not os.path.exists(product_sample_path)):
    #             # existing_sample = self.getImageSampleForPath(product_sample_path);
    #             # existing_sample = self.AddSampleForImagePath()
    #             # print(existing_sample)
    #         # if(existing_sample is None or rewrite):
    #             im = product_func(s);
    #             im.write_to_file(product_sample_path);
    #             # im.show();
    #         new_sample = self.AddSampleForImagePath(product_sample_path);
    #         self._addSessionProductLabels(new_sample, name);
    #         new_samples.append(new_sample);
    #     return self.create_sample_set(new_samples);

    def CalculateMedians(self, coverage_threshold=0.3, rewrite=False):
        def calcmed(session):
            return session.GetMedianImage(coverage_threshold=coverage_threshold);

        return self.CalculateSessionProduct(self.GetAlignedSessionSet(), calcmed, "median_image", rewrite=rewrite);

    def CalculateComposteForwards(self, threshold=0.2, rewrite=False, overlap_threshold=0.0, taper=0.0):
        def calcmed(session):
            return session.GetCompositeForwardImage(threshold=threshold, overlap_threshold=overlap_threshold,
                                                    taper=taper);

        return self.CalculateSessionProduct(self.GetAlignedSessionSet(), calcmed, "composite_forward_image",
                                            rewrite=rewrite);


    def stitchPanosOpenCV(self, name="panorama_image_opencv", method='OpenCV', target_viewport_btlr='default', seam_blur='default',
                          align_seq=False, rewrite=False):
        cdb = self.get_colmapdb();
        def calcpano(session):
            return session.get_composite_exposure(session_product_subdir_suffix=method);

        return self.CalculateSessionProduct(self.GetAlignedSessionSet(method=method), calcpano, name=name, rewrite=rewrite);


    # def stitchIndependentPanosOpenCV(self, name="independent_stitched_panos", target_viewport_btlr='default', seam_blur='default', align_seq=False, rewrite=False):
    #     cdb = self.GetCOLMAPDB();
    #
    #     def calcpano(session):
    #         return session.GetCompositeExposure();
    #
    #     return self.CalculateSessionProduct(self.GetAlignedSessionSet(method='OpenCV'), calcpano, name=name,
    #                                         rewrite=rewrite);

    def CalculatePanosOld(self, name="panorama_image", target_viewport_btlr='default', seam_blur='default',
                          align_seq=False, method="OpenCV", rewrite=False):
        """
        Stitch panoramas by registering secondaries with aligned primaries
        :param name:
        :param target_viewport_btlr:
        :param seam_blur:
        :param align_seq:
        :param rewrite:
        :return:
        """
        def calcpano(session):
            return session.get_pano_image(cdb=None, target_viewport_btlr=target_viewport_btlr, seam_blur=seam_blur,
                                          align_seq=align_seq, session_product_subdir_suffix=method);

        return self.CalculateSessionProduct(self._GetAlignedPrimarySessionSet(method=method), calcpano, name=name, rewrite=rewrite);

    # def CalculatePanosIndependent(self, name="panorama_image", target_viewport_btlr='default', seam_blur='default',
    #                               align_seq=False, rewrite=False):
    #     cdb = self.GetCOLMAPDB();
    #
    #     def calcpano(session):
    #         return session.GetPanoImage(cdb=cdb, target_viewport_btlr=target_viewport_btlr, seam_blur=seam_blur,
    #                                     align_seq=align_seq);
    #
    #     return self.CalculateSessionProduct(self.GetUndistortedSessions(), calcpano, name=name, rewrite=rewrite);

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

    # def getUndistortedDir(self):
    #     return self.getRecipeSubDir(CaptureConstants.UNDISTORTED_SAMPLE_TAG);

    def get_colmapdb_path(self, base_directory=None):
        if(dir is None):
            base_directory = self.get_current_colmap_base_directory()
        return os.path.join(base_directory, 'colmap.db');

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
                    AWARN(
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
                AWARN(
                    "Primary {} did not have a good homography: {}".format(primary.file_name,
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

    def plotTimeBetweenCaptures(self, time_range_args=None, bins=None):
        df = self.get_primary_samples().sort_by_timestamp().DataNodeSet().nodes;
        df['timediff'] = df.timestamp.diff()
        fig, ax = plt.subplots()
        series = df["timediff"][1:].astype(np.int64);
        if (bins is None):
            bins = 20;
        if (time_range_args is None):
            series.plot.hist(ax=ax, bins=20)
        else:
            series.plot.hist(ax=ax, bins=20, range=(0, pd.Timedelta(**time_range_args).value))
        labels = ax.get_xticks().tolist()
        labels = pd.to_timedelta(labels)
        ax.set_xticklabels(labels, rotation=90)
        plt.show()

    def _add_session_product_labels(self, sample, name):
        if (not sample.get_label_value(CaptureConstants.SESSION_PRODUCT_TAG)):
            sample.add_tag_label(CaptureConstants.SESSION_PRODUCT_TAG);
        session_product_name_tag = "{}_{}".format(name, CaptureConstants.SESSION_PRODUCT_NAME_KEY);
        if (not sample.get_label_value(session_product_name_tag)):
            sample.add_tag_label(session_product_name_tag);
        if (not sample.has_label(CaptureConstants.SESSION_PRODUCT_NAME_KEY)):
            sample.add_string_label(key=CaptureConstants.SESSION_PRODUCT_NAME_KEY, value=name);
        return sample

    @staticmethod
    def ConvertRecaptureFileName(filename):
        try:
            file_name_parts = os.path.splitext(filename);
            dtime = DateTimeFromReCaptureString(file_name_parts[0]);
            rstring = dtime.isoformat();
            if (len(file_name_parts) > 1):
                rstring = rstring + file_name_parts[1];
            return rstring;
        except:
            return;

    def CreateSampleForImagePath(self, path):
        imsample = super(CaptureTarget, self).create_sample_for_image_path(path);

        imsample.set_timestamp(self.__class__.GetTimestampFromFileAtPath(path));
        # imsample.setTimestamp(self.__class__.TimeStampFromImageFilePath(path));
        return imsample;

    def addOriginalSampleCategoryDirIfMissing(self, category):
        return self.add_images_subdir_if_missing(name=self._get_images_subdir_subpath_for_original_sample_category(category),
                                             add_images=False);

    def init_dirs(self, **kwargs):
        super(CaptureTarget, self).init_dirs(**kwargs);
        categories = [
            CaptureConstants.PRIMARY_CATEGORY_NAME,
            CaptureConstants.SECONDARY_CATEGORY_NAME,
        ]
        self.addDirIfMissing(name=self._getVideosDirectoryName());
        self.addDirIfMissing(name=self._getHEICDirectoryName());

        # self.addDirIfMissing(name=self._getMatchListsDirName());

        self.add_images_subdir_if_missing(name=self.__class__.ORIGINALS_SUBDIR_NAME,add_images=False);
        for category in categories:
            self.addOriginalSampleCategoryDirIfMissing(category);

        # self.add_images_subdir_if_missing(name=self._getExtrasDirectoryName(), add_images=False);

    def _get_derivative_name_part_for_original_sample_category(self, category):
        return category;
        # return os.path.join('originals', category);

    def _get_panos_derivative_name_part(self):
        return self._get_derivative_name_part_for_original_sample_category(CaptureConstants.PANO_CATEGORY_NAME);

    def _getPrimaryDerivativeNamePart(self):
        return self._get_derivative_name_part_for_original_sample_category(CaptureConstants.PRIMARY_CATEGORY_NAME);
        # return "primary_"+self.target_id;

    def _getSecondaryDerivativeNamePart(self):
        return self._get_derivative_name_part_for_original_sample_category(CaptureConstants.SECONDARY_CATEGORY_NAME);
        # return "secondary_"+self.target_id;

    def _getVideosDirectoryName(self):
        return "videos";

    def _getHEICDirectoryName(self):
        return "heic";

    def _getMatchListsDirName(self):
        return "matchlists";

    def _getExtrasDirectoryName(self):
        return "extras_cam0";

    def getVideosDirectory(self):
        return self.get_dir(self._getVideosDirectoryName());

    def _get_aligned_directory_name(self, directory_name):
        return directory_name + "_aligned";

    def _get_images_subdir_subpath_for_original_sample_category(self, category):
        return os.path.join(CaptureConstants.ORIGINALS_SUBDIR_NAME, self._get_derivative_name_part_for_original_sample_category(category=category))

    def get_images_dir_for_original_sample_category(self, category):
        return self.get_images_subdir(self._get_images_subdir_subpath_for_original_sample_category(category));

    def get_primary_images_dir(self):
        return self.get_images_dir_for_original_sample_category(CaptureConstants.PRIMARY_CATEGORY_NAME);

    def get_secondary_images_dir(self):
        return self.get_images_dir_for_original_sample_category(CaptureConstants.SECONDARY_CATEGORY_NAME);

    def getAlignedPrimaryImagesDir(self, tag):
        if (tag is None):
            return self.get_images_subdir(self._get_aligned_directory_name(self._getPrimaryDerivativeNamePart()));
        else:
            return self.get_images_subdir(
                self._get_aligned_directory_name(self._getPrimaryDerivativeNamePart()) + "_{}".format(tag));

    def getAlignedSecondaryImagesDir(self, tag=None):
        if (tag is None):
            return self.get_images_subdir(self._get_aligned_directory_name(self._getSecondaryDerivativeNamePart()));
        else:
            return self.get_images_subdir(
                self._get_aligned_directory_name(self._getSecondaryDerivativeNamePart()) + "_{}".format(tag));

    def get_aligned_pano_images_dir(self, tag):
        if (tag is None):
            return self.get_images_subdir(self._get_aligned_directory_name(self._get_panos_derivative_name_part()));
        else:
            return self.get_images_subdir(
                self._get_aligned_directory_name(self._get_panos_derivative_name_part()) + "_{}".format(tag));

    def reload_capture_data_from_directories(self, clear_first=False):
        if (clear_first):
            originals = self.get_original_samples();
            for e in originals:
                self.samples.remove(e);
        primaries = self._update_primary_images();
        secondaries = self._update_secondary_images();
        self.save();
        return primaries.GetUnion(secondaries);

    def _update_images_for_original_sample_category(self, category):
        samples = FilePathList.from_directory(self.get_images_dir_for_original_sample_category(category),
                                              extension_list=['.jpeg', '.jpg', '.png']);
        new_samples = []
        for p in samples:
            if (p.file_name[0] is not "."):
                existing = self.getImageSampleForPath(p.absolute_file_path);
                if (existing is None):
                    new_sample = self.add_sample_for_image_path(p.absolute_file_path);
                    self._add_original_sample_labels(new_sample, category=category);
                    # self._add_original_label(new_sample);
                    # self._add_original_sample_labels(new_sample, category);
                    new_samples.append(new_sample);
        return self.create_sample_set(new_samples);

    def _update_primary_images(self):
        """
        Update the primaries to include any new images in the primaries image directory
        :return:
        """
        return self._update_images_for_original_sample_category(CaptureConstants.PRIMARY_CATEGORY_NAME);
        # primaries = FilePathList.from_directory(self.getPrimaryImagesDir(), extension_list=['.jpeg','.jpg','.png']);
        # new_primary_samples = []
        # for p in primaries:
        #     if(p.file_name[0] is not "."):
        #         existing = self.getImageSampleForPath(p.absolute_file_path);
        #         if(existing is None):
        #             new_primary = self.AddSampleForImagePath(p.absolute_file_path);
        #             self._addOriginalLabel(new_primary);
        #             self._addPrimaryLabels(new_primary);
        #             new_primary_samples.append(new_primary);
        # return self.create_sample_set(new_primary_samples);

    def _update_secondary_images(self):
        return self._update_images_for_original_sample_category(CaptureConstants.SECONDARY_CATEGORY_NAME);
        # secondaries = FilePathList.from_directory(self.getSecondaryImagesDir(), extension_list=['.jpeg','.jpg','.png']);
        # new_samples = []
        # for p in secondaries:
        #     existing = self.getImageSampleForPath(p.absolute_file_path);
        #     if(existing is None):
        #         new_s = self.AddSampleForImagePath(p.absolute_file_path);
        #         self._addOriginalLabel(new_s);
        #         self._addSecondaryLabels(new_s);
        #         new_samples.append(new_s);
        #     # else:
        #     #     print(existing)
        # return self.create_sample_set(new_samples);

    def _update_aligned_panos(self, method = None):
        aligned_panos_dir_subpath = self.get_images_subdirName(self._get_aligned_directory_name(self._get_panos_derivative_name_part()));
        aligned_panos_dir = self.get_dir(aligned_panos_dir_subpath);
        samples = FilePathList.from_directory(aligned_panos_dir,
                                             extension_list=['.jpeg', '.jpg', '.png']);
        new_samples = []
        for p in samples:
            if (p.file_name[0] is not "."):
                existing = self.getImageSampleForPath(p.absolute_file_path);
                if (existing is None):
                    new_sample = self._add_aligned_pano_sample_for_path(p.absolute_file_path, method);
                    new_samples.append(new_sample);
        return self.create_sample_set(new_samples);


    def _addExtraLabel(self, sample):
        if (not sample.get_label_value(CaptureConstants.EXTRA_SAMPLE_TAG)):
            sample.add_tag_label(CaptureConstants.EXTRA_SAMPLE_TAG);

    # def _add_original_label(self, sample):
    #     if (not sample.get_label_value(CaptureConstants.ORIGINAL_SAMPLE_TAG)):
    #         sample.add_tag_label(CaptureConstants.ORIGINAL_SAMPLE_TAG);

    def _add_original_sample_labels(self, sample, category=None):
        if (not sample.get_label_value(CaptureConstants.ORIGINAL_SAMPLE_TAG)):
            sample.add_tag_label(CaptureConstants.ORIGINAL_SAMPLE_TAG);
        if (category is not None):
            if(not sample.get_label_value(self._get_original_sample_category_label(category))):
                sample.add_tag_label(self._get_original_sample_category_label(category));

    def _add_primary_labels(self, sample):
        self._add_original_sample_labels(sample, category=CaptureConstants.PRIMARY_CATEGORY_NAME);

    def _add_secondary_labels(self, sample):
        self._add_original_sample_labels(sample, category=CaptureConstants.SECONDARY_CATEGORY_NAME)

    def _add_pano_labels(self, sample):
        if (not sample.get_label_value(CaptureConstants.PANO_CATEGORY_NAME)):
            sample.add_tag_label(self._getOriginalSampleCategoryLabel(CaptureConstants.PANO_CATEGORY_NAME));
        # self._add_original_sample_labels(sample, category=CaptureConstants.PANO_CATEGORY_NAME);

    def _addUndistortedLabels(self, sample):
        if (not sample.get_label_value(CaptureConstants.UNDISTORTED_SAMPLE_TAG)):
            sample.add_tag_label(CaptureConstants.UNDISTORTED_SAMPLE_TAG);

    def _alignment_method_tag(self, method):
        if(method is None):
            return CaptureConstants.ALIGNED_SAMPLE_TAG;
        else:
            return CaptureConstants.ALIGNED_SAMPLE_TAG + "_{}".format(method);

    def _add_aligned_labels(self, sample, method=None):
        # alignment_matrix=None, aligned_with=None):
        if (not sample.get_label_value(CaptureConstants.ALIGNED_SAMPLE_TAG)):
            sample.add_tag_label(self._alignment_method_tag(method));
        # if(alignment_matrix is not None):
        #     sample.add_label_instance(LabelInstance(key="alignment_matrix", value=alignment_matrix));
        #     h,w = sample.image_shape[:2];
        #     corners = [
        #         np.array([0, 0, 1]),
        #         np.array([w, 0, 1]),
        #         np.array([w, h, 1]),
        #         np.array([0, h, 1]),
        #     ]
        #     errors = []
        #     corners_after = [];
        #     for c in corners:
        #         tc = alignment_matrix @ c;
        #         corners_after.append(tc);
        #         diff = tc - c;
        #         diff = diff[:2] * np.array([1 / w, 1 / h]);
        #         errors.append(np.linalg.norm(diff, 2))
        #     sample.add_label_instance(LabelInstance(key="corner_mapping", value=dict(corners_before=corners, corners_after=corners_after)));
        #     sample.add_label_instance(LabelInstance(key="max_corner_shift",value=np.max(errors)));
        #
        # if(aligned_with is not None):
        #     sample.add_label_instance(LabelInstance(key="spatial_alignment_target", value=aligned_with));


    # def _addAlignedSecondaryLabels(self, sample):
    #     if(not sample.get_label_value(CaptureConstants.ALIGNED_SECONDARY_SAMPLE_TAG)):
    #         sample.add_tag_label(CaptureConstants.ALIGNED_SECONDARY_SAMPLE_TAG);

    def _pullImage(self, source, dest, width='default', can_use_ffmpeg=False, calibration=None, correct_distortion=True, exif_data=None):
        """
        Pull the image, optionally changing its size, undistorting, and putting calibration information into the exif data of the new file
        :param source: path to source image
        :param dest: path to new file
        :param width: width of new image
        :param can_use_ffmpeg: whether it's ok to use ffmpeg. Probably not anymore...
        :param calibration: calibration object representing the source image's calibration
        :param correct_distortion: whether to undistort image before saving it to the destination
        :return:
        """

        # TODO need to adjust to different input resolutions
        # print(calibration.camera_name)

        if(not correct_distortion and calibration is None):
            AWARN("Not including calibration or EXIF in pulled image metadata!")
            if(width is None):
                shutil.copy2(source, dest);
                return;

            if(can_use_ffmpeg):
                (
                    ffmpeg
                    .input(source, **{'noautorotate': None})
                    .filter('scale', width, -1)
                    .output(dest)
                    .run()
                )
                return;
            else:
                im = Image(path=source, rotate_with_exif=False);
                if(im.shape[1]==width):
                    shutil.copy2(source, dest);
                else:
                    imscaled = im.get_scaled_to_width(width);
                    imscaled.write_to_file(dest, exif=im._get_exif());
            return;

        im = Image(path=source, rotate_with_exif=False);
        if (exif_data is None):
            exif_data = im._get_exif();
        if (width == 'default'):
            small_side = 1080;
            if(im.shape[1] < im.shape[0]):
                width = small_side;
            else:
                width = int(np.round(im.shape[1] * (small_side / im.shape[0])));

        if(calibration and (not correct_distortion)):
            if (im.shape[1] == width or width is None):
                # save calibration without changing image pixels at all
                assert (im.shape[1] == calibration.calibration_shape[0]), "calibration does not match image shape"
                # p0.timestamp.strftime('%Y:%m:%d %H:%M:%S')

                im.write_with_calibration_exif(output_path=dest, calibration=calibration, exif_in=exif_data);
                return;
            else:
                # resize and save modified calibration
                assert (width == calibration.calibration_shape[0]), "calibration does not match image shape"
                imscaled = im.get_scaled_to_width(width);
                imscaled.write_with_calibration_exif(
                    output_path=dest,
                    calibration=calibration.GetScaledToShape(imscaled.shape),
                    original_calibration=calibration,
                    exif_in=exif_data
                );
                return;


        if(correct_distortion):
            assert(calibration is not None), "No calibration provided for distortion correction";
            if(calibration.calibration_shape[0]!=im.shape[0]):
                sratio = im.shape[0]/calibration.calibration_shape[0];
                assert(sratio == im.shape[1]/calibration.calibration_shape[1]), "Calibration shape appears to be wrong! {} with im shape {}".format(calibration.calibration_shape, im.shape)
                calibration = calibration.GetScaledToShape(im.shape);
            # assert(calibration.calibration_shape[0] == im.shape[0] and calibration.calibration_shape[1] == im.shape[1]), "Calibration shape appears to be wrong! {} with im shape {}".format(calibration.calibration_shape, im.shape)
            newImage, newCalibration = calibration.getUndistortedImageAndNewCalibration(im);
            if(width is not None and width != newImage.shape[1]):
                scaledImage = newImage.get_scaled_to_width(width);
                scaledImage.write_with_calibration_exif(
                    output_path=dest,
                    calibration=newCalibration.GetScaledToShape(scaledImage.shape),
                    original_calibration=calibration,
                    exif_in=im._get_exif()
                );
                scaled = width/im.shape[1];
                # print("wrote {} undistorted scaling x{}".format(FilePath.From(dest).file_name, scaled));
                print("wrote {} undist scaling x{}".format(FilePath.From(dest).file_path, scaled));
            else:
                newImage.write_with_calibration_exif(
                    output_path=dest,
                    calibration=newCalibration,
                    original_calibration=calibration,
                    exif_in=im._get_exif()
                );
                # print("wrote {} undistorted no scaling".format(FilePath.From(dest).file_name));
                print("wrote {} undist no scaling".format(FilePath.From(dest).file_path));
            return;



    def pull_new_image_for_original_sample_category(self, category, fpath, width='default', filename_fn=None,
                                                    on_repeat='skip', calibration=None, correct_distortion=True):
        newSample = super(CaptureTarget, self).pull_new_image(fpath,
                                                              images_subdir=self._get_images_subdir_subpath_for_original_sample_category(
                                                                category),
                                                              width=width, filename_fn=filename_fn, on_repeat=on_repeat, calibration=calibration, correct_distortion=correct_distortion)
        if (newSample is None):
            return;
        # self._add_original_label(newSample);
        # self._add_original_sample_labels(newSample, category=category);
        self._add_original_sample_labels(newSample, category=category);
        return newSample;

    def pull_new_primary_image(self, fpath, width='default', filename_fn=None, on_repeat='skip'):
        return self.pull_new_image_for_original_sample_category(category=CaptureConstants.PRIMARY_CATEGORY_NAME, fpath=fpath,
                                                                width=width, filename_fn=filename_fn, on_repeat=on_repeat);
        # newSample = super(CaptureTarget, self).pullNewImage(fpath, images_subdir=self._getPrimaryDirectoryName(), width=width, filename_fn=filename_fn, on_repeat=on_repeat)
        # if(newSample is None):
        #     return;
        # self._addOriginalLabel(newSample);
        # self._addPrimaryLabels(newSample);
        # return newSample;

    def pull_new_secondary_image(self, fpath, width='default', filename_fn=None, on_repeat='skip', calibration=None, correct_distortion=True):
        return self.pull_new_image_for_original_sample_category(category=CaptureConstants.SECONDARY_CATEGORY_NAME, fpath=fpath,
                                                                width=width, filename_fn=filename_fn, on_repeat=on_repeat, calibration=calibration, correct_distortion=correct_distortion);
        # newSample = super(CaptureTarget, self).pullNewImage(fpath, images_subdir=self._getSecondaryDirectoryName(), width=width, filename_fn=filename_fn, on_repeat=on_repeat)
        # if(newSample is None):
        #     return;
        # self._addOriginalLabel(newSample);
        # self._addSecondaryLabels(newSample);
        # return newSample;

    def get_original_samples_for_category(self, category):
        return self.samples.get_with_tag(CaptureConstants.ORIGINAL_SAMPLE_TAG).get_with_tag(category);

    def get_primary_samples(self):
        return self.get_original_samples_for_category(CaptureConstants.PRIMARY_CATEGORY_NAME);
        # return self.samples.getWithTag(CaptureConstants.ORIGINAL_SAMPLE_TAG).getWithTag(CaptureConstants.PRIMARY_CATEGORY_NAME);

    def get_secondary_samples(self):
        return self.get_original_samples_for_category(CaptureConstants.SECONDARY_CATEGORY_NAME);
        # return self.samples.getWithTag(CaptureConstants.ORIGINAL_SAMPLE_TAG).getWithTag(CaptureConstants.SECONDARY_CATEGORY_NAME);


    def _get_camera_and_image_lists_for_colmap(self, use_unique_cameras = True):
        """
        get a dictionary {[calibration_name:string]:camera_id}
        :return:
        """
        samples = self.get_original_samples()

        cam_id_count = 0
        camdict = {}
        cameras = []
        images = []

        originals_dir = self.get_originals_directory()
        for s in samples:
            ims = s.get_image(rotate_with_exif=False)
            camcaldict = ims._read_maker_note_dict_data()
            if ('CurrentCameraCalibration' in camcaldict):
                cdict = camcaldict['CurrentCameraCalibration'];
                cal = CameraCalibrationNode.FromCalibrationDataDict(cdict)
                # cal = cal.GetScaledAndOrRotatedToShape(ims.shape)
                cam_id = -1;


                if ((cal.camera_name not in camdict) or use_unique_cameras):
                    cam_id = cam_id_count;
                    camdict[cal.camera_name] = cam_id;
                    cmdict = cal.get_cmdb_dict()
                    assert(cmdict['model']==1), "WRONG MODEL TYPE FOUND"
                    cmdict['width']=int(ims.shape[0])
                    cmdict['height'] = int(ims.shape[1])
                    params_in = cmdict['params']
                    f = (params_in[0]+params_in[1])*0.5;
                    cmdict['params'][0]=f
                    cmdict['params'][1]=f
                    cmdict['params'][2]= ims.shape[1]*0.5
                    cmdict['params'][3]= ims.shape[0]*0.5
                    cameras.append(cmdict)
                    cam_id_count += 1;
                else:
                    cam_id = camdict[cal.camera_name]
                image_relpath = os.path.relpath(s.file_path, start=originals_dir)
                images.append(dict(name=image_relpath, camera_id=cam_id))

        return cameras, images



    # ####################################

    def PullTimeLapseData(self, primary_directory, staging_directory=None, calibration=None, correct_distortion=True):
        '''
        :param primary_directory: the directory with the first image from each recapture session
        :type primary_directory:
        :param staging_directory: the directory with the extra images from each recapture session
        The staging directory is not necesarilly just for this particular time lapse; it is where you put all of the
        data you regularly pull off the app.
        :type staging_directory:
        :return:
        :rtype:
        '''
        primary_directory = FilePath.From(primary_directory)
        staging_directory = FilePath.From(staging_directory)
        self.pullPrimaryReCaptureData(primary_directory.absolute_file_path, calibration = calibration, correct_distortion=correct_distortion);
        staging_directory_path = staging_directory.absolute_file_path;
        if (staging_directory is not None and os.path.exists(staging_directory_path) and os.path.isdir(
                staging_directory_path)):
            self.pullSecondaryReCaptureDataFromStaging(staging_directory, calibration = calibration, correct_distortion=correct_distortion);
        self.save();

    def UpdateFromReCapturePull(self, path_to_recapture_files, target_id=None, calibration=None, correct_distortion=True):
        '''
        "OverlayTarget" this is where the primary samples are stored
        "RecaptureData" this is where secondary images are stored (a copy of each primary as well...)
        :param path_to_recapture_files: path to staging directory, where files are copied directly from the ReCapture
        iOS App. It should have two directories in it: "OverlayTarget" and "RecaptureData".
        :param target_id: The id used in the app for this capture target. Defaults to self.target_id
        :return:
        '''
        if (target_id is None):
            target_id = self.target_id;
        infp = FilePath.From(path_to_recapture_files);
        primary_directory = os.path.join(infp.absolute_file_path, "OverlayTarget",
                                         target_id) + os.sep;
        secondary_directory = os.path.join(infp.absolute_file_path, "RecaptureData", 'OverlayTarget',
                                           target_id) + os.sep;
        primary_fp = FilePath.From(primary_directory);
        secondary_fp = FilePath.From(secondary_directory);
        self.pullPrimaryReCaptureData(primary_fp.absolute_file_path, calibration = calibration, correct_distortion=correct_distortion);
        self.pullSecondaryReCaptureData(secondary_fp.absolute_file_path, calibration = calibration, correct_distortion=correct_distortion);

    def UpdateFromReCaptureUniversalPull(self, path_to_root, root_to_primaries, use_separate_category=False, calibration=None, correct_distortion=True):
        '''
        Update with recapture universal data
        :param path_to_root: path to app files root
        :param root_to_primaries: path from root to the capture target
        :return:
        '''
        root_path = FilePath.From(path_to_root);
        primaries_subpath = FilePath.From(root_to_primaries);
        print("looking in ReCapUniversal {}".format(primaries_subpath))
        primary_directory = os.path.join(root_path.absolute_file_path, primaries_subpath.file_path) + os.sep;
        secondary_directory = os.path.join(root_path.absolute_file_path, 'RecaptureData',
                                           primaries_subpath.file_path) + os.sep;
        primary_fp = FilePath.From(primary_directory);
        secondary_fp = FilePath.From(secondary_directory);

        primary_category = CaptureConstants.PRIMARY_CATEGORY_NAME;
        secondary_category = CaptureConstants.SECONDARY_CATEGORY_NAME;

        if (use_separate_category):
            primary_category = CaptureConstants.UNIVERSAL_PRIMARY_CATEGORY_NAME;
            secondary_category = CaptureConstants.UNIVERSAL_SECONDARY_CATEGORY_NAME;
            self.addOriginalSampleCategoryDirIfMissing(primary_category);
            self.addOriginalSampleCategoryDirIfMissing(secondary_category);

        self.pullReCaptureDataToOriginalSampleCategory(primary_category,
                                                       primary_fp.absolute_file_path,
                                                       calibration=calibration, correct_distortion=correct_distortion);

        self.pullReCaptureDataToOriginalSampleCategory(secondary_category,
                                                       secondary_fp.absolute_file_path,
                                                       calibration=calibration, correct_distortion=correct_distortion);

    def DeletePrimaryImagesFromSecondarySets(self):
        pfn = self.get_primary_samples().DataFrame().set_index(HasFPath.FILE_NAME_KEY);
        sfn = self.get_secondary_samples().DataFrame().set_index(HasFPath.FILE_NAME_KEY);
        reps = pfn.index.intersection(sfn.index);
        to_remove = sfn.loc[reps].set_index(HasFPath.FILE_PATH_KEY);
        for i in to_remove.index:
            sample = self.get_sample_for_path(i);
            primary_path = pfn.loc[sample.file_name][HasFPath.FILE_PATH_KEY];
            if (not os.path.exists(primary_path)):
                raise MissingPrimaryImageException(
                    "Record of secondary image also being a primary, but there is no copy of the primary on file! Missing: {}".format(
                        primary_path));
            print("Removing primary from secondaries {}".format(sample.file_path));
            self.DeleteSample(sample, for_real=True);
        return;

    # def _GetDuplicatedPrimaries(self):
    #     primary_set = ctarget.getPrimarySamples().DataNodeSet()
    #     secondary_set = ctarget.getSecondarySamples().DataNodeSet()
    #     # ctarget.samples.DataNodeSet().nodes
    #     pdf = primary_set.nodes[['file_path', 'file_name']]
    #     sdf = secondary_set.nodes[['file_path', 'file_name']]
    #     merged = pd.merge(pdf, sdf, how='inner', on='file_name', suffixes=['_primary', '_secondary'])

    def pullReCaptureDataToOriginalSampleCategory(self, category, path, width='default', calibration=None, correct_distortion=True):
        self.pullDataForOriginalSampleCategory(category, path, width=width,
                                               filename_fn=CaptureTarget.ConvertRecaptureFileName, calibration=calibration, correct_distortion=correct_distortion);

    def pullPrimaryReCaptureData(self, path, width='default', calibration=None, correct_distortion=True):
        return self.pullReCaptureDataToOriginalSampleCategory(CaptureConstants.PRIMARY_CATEGORY_NAME, path=path,
                                                              width=width,
                                                              calibration = calibration, correct_distortion=correct_distortion)
        # self.pullPrimaryData(path, width=width, filename_fn = CaptureTarget.ConvertRecaptureFileName);

    def pullSecondaryReCaptureData(self, path, width='default', calibration=None, correct_distortion=True):
        return self.pullReCaptureDataToOriginalSampleCategory(CaptureConstants.SECONDARY_CATEGORY_NAME, path=path,
                                                              width=width,
                                                              calibration = calibration, correct_distortion=correct_distortion);

    def pullPrimaryData(self, path, width='default', filename_fn=None):
        return self.pullDataForOriginalSampleCategory(CaptureConstants.PRIMARY_CATEGORY_NAME, path, width=width,
                                                      filename_fn=filename_fn);
        # remote = FilePathList.from_directorySearch(path, recursive=True, extension_list=['.jpeg','.jpg','.png'], criteriaFunc=FilePathList.NO_FILES_THAT_START_WITH_DOT);
        # for rmim in remote:
        #     self.pullNewPrimaryImage(rmim, width=width, filename_fn=filename_fn);
        #     # shutil.copy2(rmim.absolute_file_path, os.path.join(self.get_dir('primary'), rmim.file_name));
        # self.save()
        # return;

    def pullSecondaryData(self, path, width='default', filename_fn=None):
        return self.pullDataForOriginalSampleCategory(CaptureConstants.SECONDARY_CATEGORY_NAME, path, width=width,
                                                      filename_fn=filename_fn);
        # remote = FilePathList.from_directorySearch(path, recursive=True, extension_list=['.jpeg','.jpg','.png'], criteriaFunc=FilePathList.NO_FILES_THAT_START_WITH_DOT);
        # for rmim in remote:
        #     self.pullNewSecondaryImage(rmim, width=width, filename_fn=filename_fn);
        #     # shutil.copy2(rmim.absolute_file_path, os.path.join(self.get_dir('primary'), rmim.file_name));
        # self.save()
        # return;

    def pullDataForOriginalSampleCategory(self, category, path, width='default', filename_fn=None, calibration=None, correct_distortion=True):
        remote = FilePathList.from_directory_search(path, recursive=True, extension_list=['.jpeg', '.jpg', '.png'],
                                                    criteriaFunc=FilePathList.NO_FILES_THAT_START_WITH_DOT);
        for rmim in remote:
            # print(rmim)
            self.pull_new_image_for_original_sample_category(category, rmim, width=width, filename_fn=filename_fn, calibration=calibration, correct_distortion=correct_distortion);
            # shutil.copy2(rmim.absolute_file_path, os.path.join(self.get_dir('primary'), rmim.file_name));
        self.save()
        return;

    def pullSecondaryReCaptureDataFromStaging(self, rootPath, width='default', target_id=None, calibration=None, correct_distortion=True):
        search_path = FilePath.From(rootPath);
        image_dict = dict();
        if (target_id is None):
            target_id = self.target_id;
        print("Secondary RootPath {}".format(rootPath))

        def criteriaFunc(fpth):
            dpth = FilePath(fpth.get_directory_path());
            parentfolder = os.path.split(dpth.file_path)[-1];
            fpth_converted_name = CaptureTarget.ConvertRecaptureFileName(fpth.file_name);
            if (fpth_converted_name is None):
                # print("Skipping image with non-timestamp file name: {}".format(fpth));
                return False;
            if (parentfolder == target_id and 'RecaptureData' in fpth.parts()):
                if (fpth_converted_name in image_dict):
                    # print("found dubplicate of {}!".format(fpth_converted_name));
                    return False;
                else:
                    image_dict[fpth_converted_name] = fpth
                    return True;
            else:
                return False;

        to_pull = FilePathList.from_directory_search(directory=search_path.absolute_file_path, recursive=True,
                                                     extension_list=['.jpeg', '.jpg', ',png'], criteriaFunc=criteriaFunc);
        for f in to_pull:
            self.pull_new_secondary_image(f, width=width, filename_fn=CaptureTarget.ConvertRecaptureFileName, calibration = calibration, correct_distortion=correct_distortion);
            # self._pullImage(f.absolute_file_path, os.path.join(self.getSecondaryImagesDir(target_id), self._convertImageFileName(f.file_name)), width=width);
            # shutil.copy2(f.absolute_file_path, os.path.join(self.get_dir('secondary'), f.file_name));
        self.save();

    def _create_capture_session_with_primary(self, primary):
        session = CaptureSession.create_with_primary(primary)
        session._set_capture_target(self);
        return session;

    def get_session_dataframe(self):
        def hours(x):
            if (isinstance(x, pd.Timedelta)):
                return x / np.timedelta64(1, 'h')
            else:
                return -1;

        sessions = self.get_session_set().sort_by_timestamp();
        sessions_dict = {}
        for s in sessions:
            sessions_dict[s.timestamp] = dict(
                timestamp=s.primary_sample.timestamp,
                n_samples=s.length()
            )
        df = pd.DataFrame.from_dict(sessions_dict, orient='index')
        df['time_since_previous_session'] = df.timestamp.diff()
        df['hours_since_previous_session'] = df['time_since_previous_session'].map(lambda x: hours(x))
        return df;

    @CaptureSessionSelectionOp("originals")
    def get_session_set(self):
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

    @CaptureSessionSelectionOp("undistorted")
    def GetUndistortedSessions(self):
        assert(False), "should not be getting undistorted sessions!"
        undistorted = self.getUndistortedSamples().sort_by_timestamp()
        nodeset = ImageNodeSet();
        newNode = None;
        if (undistorted is not None):
            for s in undistorted:
                if (s.hasTag("primary")):
                    newNode = self._create_capture_session_with_primary(s);
                    nodeset.append(newNode);
                if (s.hasTag("secondary")):
                    newNode.append(s);
            nodeset.sort_by_timestamp();
            return nodeset;

    def _get_main_match_list_path(self, output_dir=None, file_name=None):
        if (file_name is None):
            file_name = 'match_list.txt';
        if (output_dir is None):
            output_dir = self.get_current_colmap_base_directory();
            # output_dir = self.getUndistortedDir();
        return os.path.join(output_dir, file_name);

    def _get_match_lists_dir(self):
        return self.get_dir(self._getMatchListsDirName());

    def write_image_match_list(self, n_primary_neighbors=None, n_key_primaries=5, output_dir=None, file_name=None):
        if (output_dir is None):
            output_dir = self.get_current_colmap_base_directory()
            # output_dir = self.getUndistortedDir();
        output_path = self._get_main_match_list_path(output_dir=output_dir, file_name=file_name);
        f = open(output_path, "a");

        def relpath(fpin: FilePath):
            return fpin.relative(output_dir);

        filepath_pair_list = self._get_default_image_match_list_pairs(n_primary_neighbors=n_primary_neighbors,
                                                                      n_key_primaries=n_key_primaries);
        # print(filepath_pair_list)

        for pair in filepath_pair_list:
            f.write("{} {}\n".format(relpath(pair[0]), relpath(pair[1])));
        f.close()
        return output_path;

    def _get_primary_image_match_list_path(self, output_dir=None):
        if (output_dir is None):
            output_dir = self._get_match_lists_dir();
        output_path = os.path.join(output_dir, '{}.txt'.format(CaptureConstants.PRIMARY_CATEGORY_NAME))
        return output_path;

    def write_primary_image_match_list(self, output_dir=None):
        output_path = self._get_primary_image_match_list_path(output_dir);
        f = open(output_path, "a");
        primaries = self.get_primary_samples()
        for primary in primaries:
            f.write("{}\n".format(os.path.join(CaptureConstants.PRIMARY_CATEGORY_NAME, primary.file_name)));
        f.close()
        return output_path;

    def write_session_image_match_lists(self, output_dir=None):
        if (output_dir is None):
            output_dir = self._get_match_lists_dir();

        sessions = self.get_session_set();
        for s in sessions:
            output_path = os.path.join(output_dir, '{}.txt'.format(s.primary_sample.sample_name))
            f = open(output_path, "a");
            f.write("{}\n".format(os.path.join(CaptureConstants.PRIMARY_CATEGORY_NAME, s.primary_sample.file_name)));
            secondaries = s.get_secondary_samples();
            for sec in secondaries:
                f.write("{}\n".format(os.path.join(CaptureConstants.SECONDARY_CATEGORY_NAME, sec.file_name)));
            f.close()



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

        print("CREATING MATCH LIST\nn_primary_neighbors:{}\nn_secondary_neighbors:{}\nn_key_primaries:{}".format(
            n_primary_neighbors, n_secondary_neighbors, n_key_primaries
        ))

        if(n_primary_neighbors is None):
            n_primary_neighbors = CaptureTarget.DEFAULT_N_PRIMARY_NEIGHBORS;
        if (n_key_primaries is None):
            n_key_primaries = 3;
        if (n_key_primaries is 'default'):
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
            if (match[0].file_path < match[1].file_path):
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

    @CaptureSessionSelectionOp("aligned_primaries")
    def _GetAlignedPrimarySessionSet(self, method=None):
        # original = self.getOriginalSamples().sortByTimestamp()
        samples = self.GetAlignedPrimaries(method) | self.get_secondary_samples();
        samples.sort_by_timestamp();
        nodeset = ImageNodeSet();
        newNode = None;
        for s in samples:
            if (s.hasTag("primary")):
                newNode = self._create_capture_session_with_primary(s);
                nodeset.append(newNode);
            if (s.hasTag("secondary")):
                newNode.append(s);
        return nodeset;

    @CaptureSessionSelectionOp("all_aligned")
    def GetAlignedSessionSet(self, method):
        aligned = self.get_aligned_samples(method).sort_by_timestamp()
        nodeset = ImageNodeSet();
        newNode = None;
        for s in aligned:
            if (s.hasTag("primary")):
                newNode = self._create_capture_session_with_primary(s);
                nodeset.append(newNode);
            if (s.hasTag("secondary")):
                newNode.append(s);
        return nodeset;

    def create_colmap_project(self, base_dir = None):
        return self._get_colmapper(base_dir = base_dir);

    def create_undistorted_colmap_project(self, calibration_node=None):
        assert(False), "Do away with undistorted"
        self._GetCOLMAPPERForOriginals(calibration_node=calibration_node);
        # raise NotImplementedError;
        # self._GetCOLMAPPERForRecipeSubdir(recipe_subdir=CaptureConstants.UNDISTORTED_SAMPLE_TAG,
        #                                   calibration_node=calibration_node);

    def _get_colmapper_for_recipe_subdir(self, recipe_subdir, calibration_node=None):
        raise NotImplementedError;
        colmapper = COLMAPper.load_from_directory(self.get_recipe_subdir(recipe_subdir));
        if (colmapper is None):
            if (calibration_node is None):
                raise ValueError("Must provide calibration node if we are creating a new colmapper.")
            colmapper = COLMAPper(
                root_path=self.get_recipe_subdir(recipe_subdir),
                scene_name=self.name,
                images_path=None,  # will default to root path
                calibration_node=calibration_node
            )
            colmapper.save();
        return colmapper;


    def _get_colmapper(self, base_dir = None, create_custom_db = False, remake_db=False):
        if(base_dir is None):
            base_dir = self.get_current_colmap_base_directory();
        colmapper = COLMAPper.load_from_directory(base_dir);
        if (colmapper is None or remake_db or self.name == 'TEST'):
            # if (calibration_node is None):
            #     raise ValueError("Must provide calibration node if we are creating a new colmapper.")
            colmapper = COLMAPper(
                root_path=base_dir,
                scene_name=self.name,
                images_path=None,  # will default to root path
            )

            if(create_custom_db):
                if(not os.path.exists(colmapper.db_path) or remake_db or self.name == 'TEST'):
                    print("CREATING DB")
                    cameras, images = self._get_camera_and_image_lists_for_colmap()
                    if(self.name == 'TEST' and os.path.exists(colmapper.db_path)):
                        colmapper.create_project(cameras=cameras, images=images, remake_db=True);
                    else:
                        colmapper.create_project(cameras=cameras, images=images, remake_db = remake_db);


            colmapper.save();
        return colmapper;

    def _run_colmap(self,
                    base_dir=None,
                    n_primary_neighbors=None,
                    n_secondary_neighbors=None,
                    n_key_primaries='default',
                    recompute=False,
                    use_one_round_matching=True,
                    remake_db=False,
                    ):
        # TODO: Left off here April 18 2024
        colmapper = self._get_colmapper(base_dir=base_dir);
        # cameras, images = self._getCameraAndImageListsForCOLMAP()
        # colmapper.createProject(remake_db=remake_db, cameras=cameras, images=images);
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

    # def _RunCOLMAPOnRecipeSubdir(self, recipe_subdir, calibration_node=None, n_primary_neighbors=None,
    #                              n_key_primaries='default', recompute=False, remake_db=False):
    #     colmapper = self._GetCOLMAPPERForRecipeSubdir(recipe_subdir=recipe_subdir, calibration_node=calibration_node);
    #     colmapper.createProject(calibration_node=calibration_node, remake_db=remake_db);
    #     colmapper.detectFeatures(calibration_node=calibration_node, recompute=recompute);
    #     colmapper.writeImageMatchList(self._getDefaultImageMatchListPairs(n_primary_neighbors=n_primary_neighbors,
    #                                                                       n_key_primaries=n_key_primaries),
    #                                   recompute=recompute);
    #     colmapper.matchFeatures(recompute=recompute)
    #     return colmapper;

    def _UpdateMatches(self, recipe_subdir, update_features=True, n_primary_neighbors=None, n_key_primaries=None, n_secondary_neighbors=None,
                       recompute=False):
        colmapper = self._get_colmapper_for_recipe_subdir(recipe_subdir=recipe_subdir);
        if (update_features):
            colmapper.detect_features(recompute=True);
        colmapper.write_image_match_list(self._get_default_image_match_list_pairs(n_primary_neighbors=n_primary_neighbors,
                                                                                  n_key_primaries=n_key_primaries, n_secondary_neighbors=n_secondary_neighbors),
                                         recompute=recompute);
        colmapper.match_features(recompute=recompute)
        return colmapper;

    def _UpdateUndistortedMatches(self, update_features=True, n_primary_neighbors=None, recompute=False,
                                  n_key_primaries=None):
        self._UpdateMatches(recipe_subdir=CaptureConstants.UNDISTORTED_SAMPLE_TAG, update_features=update_features,
                            n_key_primaries=n_key_primaries, n_primary_neighbors=n_primary_neighbors,
                            recompute=recompute);


    def runCOLMAPDict(self, recipe_subdir = None):
        # if(recipe_subdir is None):
        #     recipe_subdir = CaptureConstants.UNDISTORTED_SAMPLE_TAG;
        # colmapper = self._GetCOLMAPPERForRecipeSubdir(recipe_subdir=recipe_subdir);
        colmapper = self._GetCOLMAPPERForOriginals();
        colmapper.detect_features(recompute=True);
        # colmapper.writeImageMatchList(self._getDefaultImageMatchListPairs(n_primary_neighbors=n_primary_neighbors,
        #                                                                   n_key_primaries=n_key_primaries),
        #                               recompute=recompute);
        colmapper.match_features(recompute=True, default_dictionary_mode=True);
        return colmapper;

    def _write_match_list_for_subdir(self, recipe_subdir, match_list=None):
        colmapper = self._get_colmapper_for_recipe_subdir(recipe_subdir=recipe_subdir);
        if (match_list is None):
            match_list = self._get_default_image_match_list_pairs()
        colmapper.write_image_match_list(match_list);

    def _CalculateFeaturesForRecipeSubdir(self, recipe_subdir):
        colmapper = self._get_colmapper_for_recipe_subdir(recipe_subdir=recipe_subdir);
        colmapper.detect_features(recompute=True);

    # def _CalculateFeaturesForUndistorted(self):
    #     return self._CalculateFeaturesForRecipeSubdir(recipe_subdir=CaptureConstants.UNDISTORTED_SAMPLE_TAG);

    def calculate_image_features(self, base_dir = None):
        colmapper = self._get_colmapper(base_dir=base_dir);
        colmapper.detect_features(recompute=True)
        # return self._CalculateImageFeaturesForDirectory(base_dir = base_dir);


    def _write_match_list_for_undistorted(self):
        self._write_match_list_for_subdir(recipe_subdir=CaptureConstants.UNDISTORTED_SAMPLE_TAG);

    def _run_colmapper(self,
                       n_primary_neighbors=None,
                       n_secondary_neighbors=None,
                       n_key_primaries='default',
                       recompute=False,
                       remake_db=False
                       ):
        return self._run_colmap(
            base_dir=self.get_current_colmap_base_directory(),
            n_primary_neighbors=n_primary_neighbors,
            n_secondary_neighbors=n_secondary_neighbors,
            n_key_primaries=n_key_primaries,
            recompute=recompute,
            remake_db=remake_db
        )
        # return self._RunCOLMAPOnRecipeSubdir(CaptureConstants.UNDISTORTED_SAMPLE_TAG,
        #                                      n_primary_neighbors=n_primary_neighbors, n_key_primaries=n_key_primaries,
        #                                      calibration_node=calibration_node, recompute=recompute);

    def calc_colmapdb(self, n_primary_neighbors=None, n_key_primaries='default',
                      recompute=False, **kwargs):
        colmapper = self._run_colmapper(
            n_primary_neighbors=n_primary_neighbors,
            n_key_primaries=n_key_primaries,
            recompute=recompute,
            **kwargs
        );
        return colmapper.get_colmapdb()

    def get_colmapdb(self, colmap_base_dir=None):
        # if (recipe_subdir is None):
        #     recipe_subdir = CaptureConstants.UNDISTORTED_SAMPLE_TAG;
        if(colmap_base_dir is None):
            colmap_base_dir = self.get_current_colmap_base_directory()
        if (self._cdb is None):
            colmapper = COLMAPper.load_from_directory(colmap_base_dir);
            self._cdb = colmapper.get_colmapdb();
        return self._cdb;

    def _add_aligned_primary_dir_if_missing(self, tag=None):
        if (tag is None):
            self.add_images_subdir_if_missing(self._get_aligned_directory_name(self._getPrimaryDerivativeNamePart()));
        else:
            self.add_images_subdir_if_missing(
                self._get_aligned_directory_name(self._getPrimaryDerivativeNamePart()) + "_{}".format(tag));


    def _get_aligned_panos_dir(self, tag=None):
        if (tag is None):
            return self.get_images_subdir(self._get_aligned_directory_name(self._get_panos_derivative_name_part()));
        else:
            return self.get_images_subdir(
                self._get_aligned_directory_name(self._get_panos_derivative_name_part()) + "_{}".format(tag)
            );

    def _add_aligned_panos_dir_if_missing(self, tag=None):
        if (tag is None):
            self.add_images_subdir_if_missing(self._get_aligned_directory_name(self._get_panos_derivative_name_part()));
        else:
            self.add_images_subdir_if_missing(
                self._get_aligned_directory_name(self._get_panos_derivative_name_part()) + "_{}".format(tag));

    def _add_aligned_secondary_dir_if_missing(self, tag=None):
        if (tag is None):
            self.add_images_subdir_if_missing(self._get_aligned_directory_name(self._getSecondaryDerivativeNamePart()));
        else:
            self.add_images_subdir_if_missing(
                self._get_aligned_directory_name(self._getSecondaryDerivativeNamePart()) + "_{}".format(tag));

    def CalculateKeyframeAlignedPrimaryImages(self, keyframe_sample_set, metric=None, skip_existing=True, ext='.png',
                                              alpha=True, method="keyframe", save=True):
        return self._CalculateKeyframeAlignedPrimaryImages(keyframe_sample_set, tag=None, metric=metric,
                                                           skip_existing=skip_existing, ext=ext, alpha=alpha, method=method, save=save)

    def CalculateAlignedPrimaryImagesByKeyframes(self, keyframe_sample_set, metric=None, skip_existing=True, ext='.png',
                                                 alpha=True, method = "keyframe", save=True):
        self._add_aligned_primary_dir_if_missing(method);
        output_dir = self.getAlignedPrimaryImagesDir(method);
        if metric is None:
            metric = CaptureTarget.TimeDifferenceMetric;

        primaries = self.get_primary_samples();
        primaries.sort_by_timestamp();
        new_samples = [];
        for p in primaries:
            # Check if the primary is part of the keyframe set
            existing_set = keyframe_sample_set.get_with_timestamp(p.timestamp);
            new_sample_path = os.path.join(output_dir, p.file_name_base + ext);
            if (os.path.exists(new_sample_path) and skip_existing):
                print("Found Aligned Primary {}".format(new_sample_path));
            else:
                # if it is, just copy its image
                if (existing_set.length() > 0):
                    keysample = existing_set[0];
                    keysample.get_image().write_to_file(new_sample_path);
                    new_primary = self._addAlignedPrimarySampleForPath(new_sample_path, method=method);
                    new_samples.append(new_primary);
                else:
                    # We will sort keyframes by their metric distances to the current primary
                    # We will then go down the list trying alignment with each
                    def cmp(a, b):
                        return metric(p, a) - metric(p, b);

                    keyframe_sample_set.sort(cmp=cmp);
                    target = keyframe_sample_set[0];
                    aligned_sample, matrix, mask = p._get_aligned_with(target, border_width=None, return_mask=False);
                    good_homography = self.__class__.HomographyTest(matrix);
                    if (good_homography):
                        aligned_sample.write_to_file(new_sample_path);
                        new_primary = self._addAlignedPrimarySampleForPath(new_sample_path, method=method);
                        new_primary._set_alignment_details(alignment_matrix=matrix, input_shape=p.image_shape);
                        new_samples.append(new_primary);
                    else:
                        AWARN("Primary {} did not have a good homography".format(p.file_name));
        return self.create_sample_set(new_samples);

    def calculate_aligned_undistorted_primary_images(self, key_primary_sample=None, target_viewport_btlr='default',
                                                     skip_existing=True, ext='.png', seam_blur=None, n_key_primaries=None,
                                                     method=None, alpha=True, save=True):
        '''
        Calculate undistorted images using COLMAP db
        '''

        # self._UpdateUndistortedMatches(n_key_primaries=n_key_primaries);

        primaries = self.get_primary_samples();
        # primaries = self.getUndistortedPrimaries();
        primaries.sort_by_timestamp();
        cmdb = self.get_colmapdb();
        alignment_graph = cmdb.get_all_paths_alignment_graph_for_samples(samples=primaries);
        if (key_primary_sample is None):
            key_primary_sample = alignment_graph.central_sample;

        # if(isinstance(key_primary_sample, str)):
        #     print("KEY PRIMARY {}".format(key_primary_sample));
        #     for prm in primaries:
        #         if(prm.file_name_base == key_primary_sample):
        #             key_primary_sample = prm;
        #             break;

        if (key_primary_sample is None):
            raise ValueError("Failed to compute central node!");
            # key_primary_sample = primaries[0];

        if (target_viewport_btlr is None):
            pshape = key_primary_sample.image_shape;
            target_viewport_btlr = [0, pshape[0], 0, pshape[1]];
        elif (target_viewport_btlr == 'default'):
            factor = 0.5;
            pshape = key_primary_sample.image_shape;
            target_viewport_btlr = [int(-pshape[0] * factor), int(pshape[0] * (1 + factor)), int(-pshape[1] * factor),
                                    int(pshape[1] * (1 + factor))];
        output_shape = np.array(
            [target_viewport_btlr[1] - target_viewport_btlr[0], target_viewport_btlr[3] - target_viewport_btlr[2], 4]);

        self._add_aligned_primary_dir_if_missing(tag=method);
        output_dir = self.getAlignedPrimaryImagesDir(tag=method);

        new_samples = [];
        viewport_mat = np.array([[1, 0, -target_viewport_btlr[2]],
                                 [0, 1, -target_viewport_btlr[0]],
                                 [0, 0, 1]]).astype(float);
        for p in primaries:
            new_sample_path = os.path.join(output_dir, p.file_name_base + ext);
            if (os.path.exists(new_sample_path) and skip_existing):
                print("Found aligned undistorted primary {}".format(new_sample_path));
            else:
                is_homography = True;
                if (p.file_name == key_primary_sample.file_name):
                    alignment = np.eye(3);
                else:
                    # source_keypoints, target_keypoints = cmdb.GetKeypointMatchesForSamplePair(p, key_primary_sample);
                    # source_keypoints, target_keypoints = cmdb.GetKeypointMatchesForSamplePair(p, key_primary_sample);
                    # if(source_keypoints is None or target_keypoints is None):
                    #     AWARN("NO MATCHES FOR {} and {}".format(p.file_name, key_primary_sample.file_name))
                    # alignment, mask = p.GetMatrixForCorrespondences(source_keypoints, target_keypoints, target_viewport_btlr=target_viewport_btlr);
                    # tvg = cmdb.GetGeometryForSamplePair(p, key_primary_sample);
                    # alignment = tvg.H(from_sample=p, to_sample=key_primary_sample);
                    alignment = alignment_graph.get_alignment_matrix_for_samples(from_sample=p,
                                                                                 to_sample=key_primary_sample);
                    # is_homography = tvg.is_homography;
                good_homography = is_homography and self.__class__.HomographyTest(alignment);
                if (good_homography):
                    primary_matrix = viewport_mat @ alignment;
                    new_im = p._get_image_warped_by_matrix(primary_matrix, output_shape=output_shape,
                                                           border_width=seam_blur);
                    new_im.write_to_file(new_sample_path);
                    new_primary = self._addAlignedPrimarySampleForPath(new_sample_path, method=None);
                    new_primary._set_alignment_details(alignment_matrix=primary_matrix, input_shape=p.image_shape);
                    new_samples.append(new_primary);
                else:
                    # AWARN("Primary {} did not have a good homography".format(p.file_name));
                    AWARN(
                        "Primary {} did not have a good homography: {}".format(p.file_name, good_homography));

        if (save):
            self.key_sample_name = key_primary_sample.sample_name
            self.save();
        return self.create_sample_set(new_samples);

    def CalculateAlignedPrimaryImagesOpenCV(self, key_primary_sample=None, target_viewport_btlr='default',
                                                       skip_existing=True, ext='.png', seam_blur=None,
                                                       n_key_primaries=None, alpha=True, save=True, method='OpenCV'):
        # primaries = self.getUndistortedPrimaries();
        primaries = self.get_primary_samples()
        primaries.sort_by_timestamp();
        if (key_primary_sample is None):
            key_primary_sample = self.key_sample_name;
        if (isinstance(key_primary_sample, str)):
            # print("KEY PRIMARY {}".format(key_primary_sample));
            for prm in primaries:
                if (prm.file_name_base == key_primary_sample):
                    key_primary_sample = prm;
                    break;

        if (key_primary_sample is None):
            key_primary_sample = primaries[0];

        if (target_viewport_btlr is None):
            pshape = key_primary_sample.image_shape;
            target_viewport_btlr = [0, pshape[0], 0, pshape[1]];
        elif (target_viewport_btlr == 'default'):
            factor = 0.5;
            pshape = key_primary_sample.image_shape;
            target_viewport_btlr = [int(-pshape[0] * factor), int(pshape[0] * (1 + factor)), int(-pshape[1] * factor),
                                    int(pshape[1] * (1 + factor))];
        output_shape = np.array(
            [target_viewport_btlr[1] - target_viewport_btlr[0], target_viewport_btlr[3] - target_viewport_btlr[2], 4]);

        # self._UpdateUndistortedMatches(n_key_primaries=n_key_primaries);
        self._add_aligned_primary_dir_if_missing(method);
        output_dir = self.getAlignedPrimaryImagesDir(method);
        new_samples = [];
        cdb = self.get_colmapdb();

        def getPrimaryVersionFromSet(prm, set):
            timestamp = prm.timestamp
            smatches = set.get_with_timestamp(timestamp);
            if (smatches.length() == 0):
                return None;
            elif (smatches.length() == 1):
                return smatches[0]
            else:
                raise ValueError("Too many primary matches!")
        existing_aligned_primaries = self.getAlignedPrimaries(method=method);


        last_aligned_primary = primaries[0]
        for p in primaries:
            # print("try {}".format(p.file_name))
            if(method == "sequential"):
                target_sample = last_aligned_primary
            else:
                target_sample = key_primary_sample


            new_sample_path = os.path.join(output_dir, p.file_name_base + ext);
            old_sample = getPrimaryVersionFromSet(p, existing_aligned_primaries); # check if we already have
            # if (os.path.exists(new_sample_path) and skip_existing):
            #     print("Found {}".format(new_sample_path));
            # if (os.path.exists(new_sample_path) and skip_existing):
                # print("Found {}".format(new_sample_path));
            if(old_sample is None or (not skip_existing)):
                if(old_sample is not None and (not skip_existing)):
                    self.samples.remove(old_sample)
                if (p.file_name == target_sample.file_name):
                    viewport_mat = np.array([[1, 0, -target_viewport_btlr[2]],
                                             [0, 1, -target_viewport_btlr[0]],
                                             [0, 0, 1]]).astype(float);
                    alignment = viewport_mat;
                else:
                    source_keypoints, target_keypoints = cdb.get_keypoint_matches_for_sample_pair(p, target_sample);
                    if (source_keypoints is None or target_keypoints is None):
                        AWARN("NO MATCHES FOR {} and {}".format(p.file_name,
                                                                target_sample.file_name))
                    alignment, mask = p.get_matrix_for_correspondences(source_keypoints, target_keypoints,
                                                                       target_viewport_btlr=target_viewport_btlr);
                good_homography = self.__class__.HomographyTest(alignment);
                if (good_homography):
                    new_im = p._get_image_warped_by_matrix(alignment, output_shape=output_shape,
                                                           border_width=seam_blur);
                    new_im.write_to_file(new_sample_path);
                    new_primary = self._addAlignedPrimarySampleForPath(new_sample_path, method=method);
                    new_primary._set_alignment_details(alignment_matrix=alignment, input_shape=p.image_shape);
                    # print("ADDING NEW PRIMARY")
                    # print(new_primary)
                    new_samples.append(new_primary);
                    last_aligned_primary = p
                else:
                    # AWARN("Primary {} did not have a good homography".format(p.file_name));
                    AWARN(
                        "Primary {} did not have a good homography: {}".format(p.file_name, good_homography));

        if (save):
            self.save();
        return self.create_sample_set(new_samples);

    def CalculateAlignedSecondaryImagesOpenCV(self, border_width=None, aligned_targets=True,
                                                         target_viewport_btlr=None, recompute_homographies=False,
                                                         force_recompute=False, seam_blur='default', method='OpenCV', save=True):
        if (force_recompute):
            old = self.getAlignedSecondaries(method);
            for e in old:
                self.samples.remove(e);
        self._add_aligned_secondary_dir_if_missing(method)
        # self.add_images_subdir_if_missing(self._getAlignedDirectoryName(self._getSecondaryDerivativeNamePart()));
        colmap_db = self.get_colmapdb();
        original_sessions = self.get_session_set();
        # undistorted_sessions = self.GetUndistortedSessions();

        def getPrimaryVersionFromSet(prm, set):
            timestamp = prm.timestamp
            smatches = set.get_with_timestamp(timestamp);
            if (smatches.length() == 0):
                return None;
            elif (smatches.length() == 1):
                return smatches[0]
            else:
                raise ValueError("Too many primary matches!")

        def match_key_func(sample):
            return sample.file_path.relative(sample.file_path.get_directory_path() + os.sep + ".." + os.sep);

        # undistorted_primaries = self.getUndistortedPrimaries();
        # if (aligned_targets):
        #     aligned_primaries = self.getAlignedPrimaries();
        # else:
        #     aligned_primaries = undistorted_primaries;
        aligned_primaries = self.getAlignedPrimaries(method);

        aligned = self.get_aligned_samples(method);
        new_samples = [];
        for session in original_sessions:
            # undistorted_primary = getPrimaryVersionFromSet(session.primary_sample, aligned_primaries);
            aligned_primary = getPrimaryVersionFromSet(session.primary_sample, aligned_primaries);
            primary = aligned_primary;
            if (primary is None):
                AWARN("Missing primary for {}".format(session))
            else:
                secondaries = session.get_secondary_samples();
                # fprimary = primary.GetImage().get_float_copy();
                if (aligned_targets):
                    primary_alignment_details = primary.getAlignmentDetails();
                    primary_matrix = primary_alignment_details['alignment_matrix'];
                if ((not aligned_targets) or primary_matrix is not None):
                    for s in secondaries:
                        # existing = aligned.getWithTimestamp(s.timestamp);
                        # print("self.getAlignedSecondaryImagesDir(method): {}".format(self.getAlignedSecondaryImagesDir(method)))
                        # print("s.file_name_base: {}".format(s.file_name_base))
                        new_secondary_path = os.path.join(self.getAlignedSecondaryImagesDir(method),
                                                          s.file_name_base + '.png');
                        existing_sample = self.getImageSampleForPath(new_secondary_path);
                        if (existing_sample is None or force_recompute):
                            if ((not force_recompute) and os.path.exists(new_secondary_path)):
                                new_s = self._addAlignedSecondarySampleForPath(new_secondary_path, method)
                                # new_s = self.AddSampleForImagePath(new_secondary_path);
                                # new_s._setAlignmentDetails(alignment_matrix=None, input_shape=s.image_shape);
                                # self._addAlignedLabels(new_s, method=method);
                                # self._addSecondaryLabels(new_s);
                                new_samples.append(new_s);
                            else:
                                if (not recompute_homographies):
                                    existing_homography_samples = aligned.get_with_timestamp(s.timestamp);
                                    if (existing_homography_samples.length() > 0):
                                        # Old version had alignment matrix as separate label
                                        if (existing_homography_samples[0].has_label("alignment_matrix")):
                                            matrix = existing_homography_samples[0].get_label_value("alignment_matrix");
                                        # New version uses alignment details
                                        elif (
                                        existing_homography_samples[0].has_label(ImageSampleConstants.ALIGNMENT_DETAILS_KEY)):
                                            alignment_details = existing_homography_samples[0].get_label_value(
                                                ImageSampleConstants.ALIGNMENT_DETAILS_KEY)
                                            matrix = alignment_details["alignment_matrix"];
                                        ####
                                source_keypoints, target_keypoints = colmap_db.get_keypoint_matches_for_sample_pair(s, session.primary_sample, key_function=match_key_func);
                                to_primary_matrix, mask = s.get_matrix_for_correspondences(source_keypoints,
                                                                                           target_keypoints,
                                                                                           target_viewport_btlr=target_viewport_btlr);
                                # if (matrix is None):
                                #     s_aligned, matrix, mask = s._GetAlignedWith(primary, border_width=border_width,
                                #                                                 return_mask=False);
                                # else:
                                #     s_aligned = cv2.warpPerspective(s.GetImage().get_rgba_copy().ipixels, matrix, (w, h),
                                #                                     flags=cv2.INTER_LINEAR)

                                good_homography = self.__class__.HomographyTest(to_primary_matrix);
                                if (good_homography):
                                    if (aligned_targets):
                                        matrix = primary_matrix @ to_primary_matrix;
                                    else:
                                        matrix = to_primary_matrix;
                                    s_aligned = s._get_image_warped_by_matrix(matrix, output_shape=primary.image_shape,
                                                                              border_width=seam_blur);
                                    s_aligned.write_to_file(new_secondary_path);

                                    new_s = self._addAlignedSecondarySampleForPath(new_secondary_path, method)
                                    new_s._set_alignment_details(alignment_matrix=matrix, input_shape=s.image_shape,
                                                                 alignment_target_path=primary.file_path);
                                    new_samples.append(new_s);
                                else:
                                    if (to_primary_matrix is not None):
                                        detLin = np.linalg.det(to_primary_matrix[:2, :2]);
                                        detHom = np.linalg.det(to_primary_matrix);
                                        AWARN("Poor Homography, skipping: {} | {}".format(detLin, detHom));
                                    else:
                                        AWARN("Poor Homography, skipping: (No to_primary_matrix!)");
        if (save):
            self.save();
        return self.create_sample_set(new_samples);

    def CalculateAlignedUndistortedSecondaryImages(self, border_width=None,
                                                   target_viewport_btlr='default',
                                                   force_recompute=False,
                                                   cutoff=None,
                                                   # seam_blur='default',
                                                   seam_blur=None,
                                                   save=True,
                                                   coverage_threshold=0.25,
                                                   method=None,
                                                   tag=None):
        if (force_recompute):
            old = self.getAlignedSecondaries();
            for e in old:
                self.samples.remove(e);

        self._add_aligned_secondary_dir_if_missing(tag=tag);
        output_dir = self.getAlignedSecondaryImagesDir(tag=tag);

        self.add_images_subdir_if_missing(self._get_aligned_directory_name(self._getSecondaryDerivativeNamePart()));
        # colmap_db = self.GetCOLMAPDB();
        # original_sessions = self.GetSessionSet();
        # sessions = self.GetUndistortedSessions();
        sessions = self.get_session_set();
        cmdb = self.get_colmapdb();
        # alignment_graph = cmdb.GetAlignmentGraphForSamples(self.getUndistortedSamples());

        primaries = self.get_primary_samples();
            # self.getUndistortedPrimaries());
        primaries.sort_by_timestamp();

        if (self.central_node is None):
            primary_alignment_graph = cmdb.get_all_paths_alignment_graph_for_samples(samples=primaries);
            self.central_node = primary_alignment_graph.central_sample;
            self.save();

        central_node = self.central_node;
        central_node_shape = central_node.image_shape;

        # self.getUndistortedSamples()
        alignment_graph = cmdb.get_central_sample_alignment_graph_for_samples(self.get_original_samples(),
                                                                              central_sample=central_node, cutoff=cutoff);

        if (target_viewport_btlr is None):
            target_viewport_btlr = [0, central_node_shape[0], 0, central_node_shape[1]];
            viewport_mat = np.eye;
        elif (target_viewport_btlr == 'default'):
            factor = 0.5;
            target_viewport_btlr = [int(-central_node_shape[0] * factor), int(central_node_shape[0] * (1 + factor)),
                                    int(-central_node_shape[1] * factor),
                                    int(central_node_shape[1] * (1 + factor))];
            viewport_mat = np.array([[1, 0, -target_viewport_btlr[2]],
                                     [0, 1, -target_viewport_btlr[0]],
                                     [0, 0, 1]]).astype(float);
        else:
            viewport_mat = np.array([[1, 0, -target_viewport_btlr[2]],
                                     [0, 1, -target_viewport_btlr[0]],
                                     [0, 0, 1]]).astype(float);
        output_shape = np.array(
            [target_viewport_btlr[1] - target_viewport_btlr[0], target_viewport_btlr[3] - target_viewport_btlr[2], 4]);

        def getPrimaryVersionFromSet(prm, set):
            timestamp = prm.timestamp
            smatches = set.get_with_timestamp(timestamp);
            if (smatches.length() == 0):
                return None;
            elif (smatches.length() == 1):
                return smatches[0]
            else:
                raise ValueError("Too many primary matches!")

        def match_key_func(sample):
            return sample.file_path.relative(sample.file_path.get_directory_path() + os.sep + ".." + os.sep);

        new_samples = [];
        for session in sessions:
            undistorted_primary = getPrimaryVersionFromSet(session.primary_sample, primaries);
            primary = undistorted_primary;
            if (primary is None or undistorted_primary is None):
                AWARN("Missing primary for {}".format(session))
            else:
                secondaries = session.get_secondary_samples();
                for s in secondaries:
                    new_secondary_path = os.path.join(self.getAlignedSecondaryImagesDir(tag),
                                                      s.file_name_base + '.png');
                    existing_sample = self.getImageSampleForPath(new_secondary_path);
                    if (existing_sample is None or force_recompute):
                        is_homography = True;  # can probably remove this
                        alignment = alignment_graph.get_alignment_matrix_for_sample(from_sample=s);
                        if ((not force_recompute) and os.path.exists(new_secondary_path)):
                            new_s = self._addAlignedSecondarySampleForPath(new_secondary_path, method)
                            # new_s = self.AddSampleForImagePath(new_secondary_path);
                            # self._addAlignedLabels(new_s);
                            # self._addSecondaryLabels(new_s);
                            new_s._set_alignment_details(alignment_matrix=alignment, input_shape=s.image_shape);
                            new_samples.append(new_s);
                        else:
                            good_homography = is_homography and self.__class__.HomographyTest(alignment);
                            if (good_homography):
                                matrix = viewport_mat @ alignment;
                                s_aligned = s._get_image_warped_by_matrix(matrix, output_shape=output_shape,
                                                                          border_width=seam_blur);
                                coverage = s_aligned.fpixels[:, :, 3].sum() / np.product(s.image_shape[:2]);
                                if (coverage > coverage_threshold):
                                    s_aligned.write_to_file(new_secondary_path);
                                    new_s = self._addAlignedSecondarySampleForPath(new_secondary_path, method);
                                    # new_s = self.AddSampleForImagePath(new_secondary_path);
                                    # self._addAlignedLabels(new_s);
                                    # self._addSecondaryLabels(new_s);
                                    new_s._set_alignment_details(alignment_matrix=matrix, input_shape=s.image_shape,
                                                                 alignment_target_path=central_node.file_path);
                                    new_samples.append(new_s);
                            else:
                                if (alignment is not None):
                                    detLin = np.linalg.det(alignment[:2, :2]);
                                    detHom = np.linalg.det(alignment);
                                    AWARN("Poor Homography, skipping: {} | {}".format(detLin, detHom));
                                else:
                                    AWARN("Poor Homography, skipping: (No to_primary_matrix!)");
        if (save):
            self.save();
        return self.create_sample_set(new_samples);

    # def CalculateAlignedPrimarySet()

    def _CalculateKeyframeAlignedPrimaryImages(self, keyframe_sample_set, tag=None, metric=None, skip_existing=True,
                                               ext='.png', alpha=True, method="keyframe", save=True):
        """

        :param tag:
        :param keyframe_sample_set: The set of keyframes to use for anchoring the alignment
        :param metric: The metric for finding the nearest neighbor keyframe to align each image with
        :param skip_existing: Whether to skip existing images or re-calculate them
        :param ext: image file extension to use (png by default)
        :param alpha: include alpha (not implemented)
        :param save: save
        :return:
        """
        # self.add_images_subdir_if_missing(self._getAlignedDirectoryName(self._getPrimaryDirectoryName()));

        self._add_aligned_primary_dir_if_missing(method);
        output_dir = self.getAlignedPrimaryImagesDir(method);
        if metric is None:
            metric = CaptureTarget.TimeDifferenceMetric;

        primaries = self.get_primary_samples();
        primaries.sort_by_timestamp();
        new_samples = [];
        for p in primaries:
            # Check if the primary is part of the keyframe set
            existing_set = keyframe_sample_set.get_with_timestamp(p.timestamp);
            new_sample_path = os.path.join(output_dir, p.file_name_base + ext);
            if (os.path.exists(new_sample_path) and skip_existing):
                print("Found primary in calc keyframe aligned, {}".format(new_sample_path));
            else:
                # if it is, just copy its image
                if (existing_set.length() > 0):
                    keysample = existing_set[0];
                    keysample.get_image().write_to_file(new_sample_path);
                    new_primary = self._addAlignedPrimarySampleForPath(new_sample_path, method=method);
                    new_samples.append(new_primary);
                else:
                    # We will sort keyframes by their metric distances to the current primary
                    # We will then go down the list trying alignment with each
                    def cmp(a, b):
                        return metric(p, a) - metric(p, b);

                    keyframe_sample_set.sort(cmp=cmp);
                    target = keyframe_sample_set[0];
                    aligned_sample, matrix, mask = p._get_aligned_with(target, border_width=None, return_mask=False);
                    good_homography = self.__class__.HomographyTest(matrix);
                    if (good_homography):
                        aligned_sample.write_to_file(new_sample_path);
                        new_primary = self._addAlignedPrimarySampleForPath(new_sample_path, method=method);
                        new_primary._set_alignment_details(alignment_matrix=matrix, input_shape=p.image_shape);
                        new_samples.append(new_primary);
                    else:
                        AWARN("Primary {} did not have a good homography".format(p.file_name));
        return self.create_sample_set(new_samples);

    def CreatePomotedSecondaryImageSet(self, threshold=0.1, calculate_new_alignments=False):
        if (calculate_new_alignments):
            self.CalculateAlignedSecondaryImages()
        secondaries = self.GetAlignedSecondaries()
        closeToAligned = [];
        for s in secondaries:
            if (s.has_label("max_corner_shift")):
                corner_error = s.get_label_value("max_corner_shift");
                if (corner_error < 0.1):
                    closeToAligned.append(s);
            else:
                if (s.has_label("alignment_matrix")):
                    h, w = s.image_shape[:2];
                    # s.GetImage().show

    def CreatePrimarySetAlignedWithTargetSample(self, tag, target_sample=None, recompute_homographies=False,
                                                force_recompute=False, save=True):
        primaries = self.get_primary_samples();
        primaries.sort_by_timestamp();
        if (target_sample is None):
            target_sample = primaries[0]

        if (force_recompute):
            old = self.getWithTag(tag);
            for e in old:
                self.samples.remove(e);
        self.addRecipeSubdir(tag);
        output_dir = self.getRecipeSubDir(tag);
        new_samples = [];

        def addlabels(sample):
            sample.add_tag_label(tag);

        for p in primaries:
            timestamp = p.timestamp;
            new_image_path = os.path.join(output_dir, p.file_name_base + '.png');
            existing_sample = self.getImageSampleForPath(new_image_path);
            matrix = None;
            target_image = target_sample.GetImage();

            if (existing_sample is None or force_recompute):

                # If the file exists but we just haven't loaded it into the database
                if ((not force_recompute) and os.path.exists(new_image_path)):
                    new_i = self.AddSampleForImagePath(new_image_path);
                    addlabels(new_i);
                    new_samples.append(new_s);
                else:
                    matrix = None;
                    if (not recompute_homographies):
                        existing_homography_samples = primaries.get_with_timestamp(p.timestamp);
                        if (existing_homography_samples.length() > 0):
                            # Older caclulations only had alignment matrix
                            if (existing_homography_samples[0].has_label("alignment_matrix")):
                                matrix = existing_homography_samples[0].get_label_value("alignment_matrix");
                            elif (existing_homography_samples[0].has_label(ImageSampleConstants.ALIGNMENT_DETAILS_KEY)):
                                matrix = existing_homography_samples[0].get_alignment_matrix();
                                # matrix = alignment_details["alignment_matrix"];
                    if (matrix is None):
                        i_aligned, matrix, mask = p._get_aligned_with(target_sample,
                                                                      return_mask=False);
                    else:
                        i_aligned = cv2.warpPerspective(p.get_image().get_rgba_copy().ipixels, matrix,
                                                        (target_image.shape[1], target_image.shape[0]),
                                                        flags=cv2.INTER_LINEAR)
                    good_homography = self.__class__.HomographyTest(matrix);
                    if (good_homography):
                        i_aligned.write_to_file(new_image_path);
                        new_s = self.AddSampleForImagePath(new_image_path);
                        new_s._set_alignment_details(alignment_matrix=matrix, input_shape=p.image_shape);
                        addlabels(new_s);
                        new_samples.append(new_s);
                    else:
                        detLin = np.linalg.det(matrix[:2, :2]);
                        detHom = np.linalg.det(matrix);
                        AWARN("Poor Homography, skipping: {} | {}".format(detLin, detHom));
        if (save):
            self.save();
        return self.create_sample_set(new_samples);

    # @ImageDatasetRecipe("CroppedPrimaries")
    # def SaveCroppedAlignedPrimaries(self, minpoint, maxpoint, tag=None, **kwargs):
    #     if(tag is None):
    #         tag = "crop_x{}y{}X{}y{}".format(minpoint[1],minpoint[0],maxpoint[1],maxpoint[0]);
    #     recipe_path_args = dict(name="CroppedPrimaries", tag=tag);
    #     recipe_subdir_name = self.getRecipeSubdirName(**recipe_path_args);
    #     self.addRecipeSubdir(**recipe_path_args);
    #     output_dir = self.getRecipeSubDir(**recipe_path_args);
    #
    #     primaries = self.getAlignedPrimaries();
    #     for p in primaries:
    #         newim = p.GetImage().GetCropped(x_range=[minpoint[1],maxpoint[1]], y_range=[minpoint[0],maxpoint[0]]);
    #         newim.write_to_file(os.path.join(output_dir, p.file_name));

    def CalculateUndistortedSubset(self, subset, subset_subpath, tags, calibration_node: CameraCalibrationNode,
                                   alpha=None, force_recompute=False, **kwargs):
        if (alpha is None):
            alpha = 0
        if (tags is None):
            tags = [];
        new_tags = tags;
        recipe_path_args = dict(parent_recipe=CaptureConstants.UNDISTORTED_SAMPLE_TAG, name=subset_subpath);
        self.addRecipeSubdir(CaptureConstants.UNDISTORTED_SAMPLE_TAG);
        # recipe_subdir_name = self.getRecipeSubdirName(**recipe_path_args);
        self.addRecipeSubdir(**recipe_path_args);
        output_dir = self.getRecipeSubDir(**recipe_path_args);

        def _addLabelsForNewUndistortedSample(sample):
            for t in tags:
                sample.add_tag_label(t);
            self._addUndistortedLabels(sample);

        new_samples = [];
        for p in subset:
            new_image_path = os.path.join(output_dir, p.file_name);

            if (os.path.exists(new_image_path)):
                newim = Image(path=new_image_path);
                new_matrix = calibration_node.getUnDistortedMatrix(alpha=alpha);
            else:
                oldim = p.get_image();
                [undistorted_pix, new_matrix] = calibration_node.getUndistorted(oldim.pixels, alpha);
                newim = Image(pixels=undistorted_pix);
                newim.write_to_file(new_image_path);

            old_sample = self.get_sample_for_path(new_image_path);
            if (old_sample is None):
                new_sample = self.AddSampleForImagePath(new_image_path);
                _addLabelsForNewUndistortedSample(new_sample)
                new_sample.set_info("intrinsic_matrix", new_matrix);
                new_samples.append(new_sample);
        return new_samples;

    @ImageDatasetRecipe("UndistortedPrimaries")
    def CalculateUndistortedPrimaries(self, calibration_node: CameraCalibrationNode, alpha=None, **kwargs):
        subset_subpath = self._getPrimaryDerivativeNamePart()
        tags = [self._getOriginalSampleCategoryLabel(CaptureConstants.PRIMARY_CATEGORY_NAME)];
        primaries = self.get_primary_samples();
        return self.CalculateUndistortedSubset(subset=primaries, subset_subpath=subset_subpath, tags=tags,
                                               calibration_node=calibration_node, alpha=alpha, **kwargs)

    @ImageDatasetRecipe("UndistortedSecondaries")
    def CalculateUndistortedSecondaries(self, calibration_node: CameraCalibrationNode, alpha=None, **kwargs):
        subset_subpath = self._getSecondaryDerivativeNamePart()
        tags = [self._getOriginalSampleCategoryLabel(CaptureConstants.SECONDARY_CATEGORY_NAME)];
        secondaries = self.get_secondary_samples();
        return self.CalculateUndistortedSubset(subset=secondaries, subset_subpath=subset_subpath, tags=tags,
                                               calibration_node=calibration_node, alpha=alpha, **kwargs)

    # def CalculateAlignedSecondaryImages(self, border_width = 'default', force_recompute=False, save=True):
    def CalculateAlignedSecondaryImages(self, border_width=None, target_viewport_btlr=None,
                                        recompute_homographies=False, force_recompute=False, method='OpenCVStale', save=True):
        raise ValueError("Probably looking for CalculateAlignedUndistortedSecondaryImages")
        if (force_recompute):
            old = self.getAlignedSecondaries();
            for e in old:
                self.samples.remove(e);
        self._add_aligned_secondary_dir_if_missing(method)
        # self.add_images_subdir_if_missing(self._getAlignedDirectoryName(self._getSecondaryDerivativeNamePart()));
        original_sessions = self.get_session_set();
        aligned_primaries = self.getAlignedPrimaries();
        aligned = self.get_aligned_samples(method);
        new_samples = [];
        if (target_viewport_btlr is None):
            pshape = aligned_primaries[0].get_image().shape;
            target_viewport_btlr = [0, pshape[0], 0, pshape[1]];
        w = target_viewport_btlr[3] - target_viewport_btlr[2];
        h = target_viewport_btlr[1] - target_viewport_btlr[0];
        viewport_mat = np.array([
            [1, 0, target_viewport_btlr[0]],
            [0, 1, target_viewport_btlr[2]],
            [0, 0, 1]
        ])
        for session in original_sessions:
            timestamp = session.primary_sample.timestamp
            primary_matches = aligned_primaries.get_with_timestamp(timestamp);
            primary = None;
            if (primary_matches.length() == 1):
                primary = primary_matches[0];
            else:
                assert (primary_matches.length() < 2), "too many aligned primaries for {}".format(timestamp);
            if (primary is None):
                AWARN("Missing aligned primary for {}".format(timestamp))
            else:
                secondaries = session.get_secondary_samples();
                # fprimary = primary.GetImage().get_float_copy();
                for s in secondaries:
                    # existing = aligned.getWithTimestamp(s.timestamp);
                    new_secondary_path = os.path.join(self.getAlignedSecondaryImagesDir(),
                                                      s.file_name_base + '.png');
                    existing_sample = self.get_image_sample_for_path(new_secondary_path);
                    matrix = None;
                    if (existing_sample is None or force_recompute):
                        if ((not force_recompute) and os.path.exists(new_secondary_path)):
                            new_s = self.add_sample_for_image_path(new_secondary_path);
                            new_s._set_alignment_details(alignment_matrix=matrix, input_shape=s.image_shape);
                            self._add_aligned_labels(new_s);
                            self._add_secondary_labels(new_s);
                            new_samples.append(new_s);
                        else:
                            if (not recompute_homographies):
                                existing_homography_samples = aligned.get_with_timestamp(s.timestamp);
                                if (existing_homography_samples.length() > 0):
                                    # Old version had alignment matrix as separate label
                                    if (existing_homography_samples[0].has_label("alignment_matrix")):
                                        matrix = existing_homography_samples[0].get_label_value("alignment_matrix");
                                    # New version uses alignment details
                                    elif (existing_homography_samples[0].has_label(ImageSampleConstants.ALIGNMENT_DETAILS_KEY)):
                                        matrix = existing_homography_samples[0].get_alignment_matrix();
                                        # alignment_details = existing_homography_samples[0].getAlignmentDetails()
                                        # matrix = alignment_details["alignment_matrix"];
                                    ####
                            if (matrix is None):
                                s_aligned, matrix, mask = s._get_aligned_with(primary, border_width=border_width,
                                                                              return_mask=False);
                            else:
                                s_aligned = cv2.warpPerspective(s.get_image().get_rgba_copy().ipixels, matrix, (w, h),
                                                                flags=cv2.INTER_LINEAR)

                            good_homography = self.__class__.HomographyTest(matrix);
                            if (good_homography):
                                target_primary = primary.GetImage().get_float_copy();
                                corrected, ratios = ExposureCorrect(s_aligned, target_primary);
                                corrected.pixels = np.clip(corrected.pixels, 0, 1);
                                # s_aligned.write_to_file(new_secondary_path);
                                corrected.write_to_file(new_secondary_path);
                                new_s = self._addAlignedSecondarySampleForPath(new_secondary_path, method)
                                # new_s = self.AddSampleForImagePath(new_secondary_path);
                                # self._addAlignedLabels(new_s);
                                # self._addSecondaryLabels(new_s);
                                new_s._set_alignment_details(alignment_matrix=matrix, input_shape=s.image_shape,
                                                             alignment_target_path=primary.file_name);
                                new_samples.append(new_s);
                            else:
                                detLin = np.linalg.det(matrix[:2, :2]);
                                detHom = np.linalg.det(matrix);
                                AWARN("Poor Homography, skipping: {} | {}".format(detLin, detHom));
        if (save):
            self.save();
        return self.create_sample_set(new_samples);

    def GetAlignedPrimaries(self, method=None):
        return self.get_aligned_samples(method).get_with_tag(CaptureConstants.PRIMARY_CATEGORY_NAME);

    def GetAlignedSecondaries(self, method=None):
        return self.get_aligned_samples(method).get_with_tag(CaptureConstants.SECONDARY_CATEGORY_NAME);

    def get_originals_directory(self):
        return self.get_images_subdir(self.__class__.ORIGINALS_SUBDIR_NAME);

    def get_current_colmap_base_directory(self):
        dbdir = self.get_info("COLMAPDBDIRECTORY");
        if(dbdir is None):
            return self.get_originals_directory();

    # def GetAlignedSessions(self):
    #     aligned = self.getAlignedSamples().sortByTimestamp()
    #     nodeset = ImageNodeSet();
    #     newNode = None;
    #     for s in aligned:
    #         if(s.hasTag("primary")):
    #             newNode = self._CreateCaptureSessionWithPrimary(s);
    #             nodeset.append(newNode);
    #         if(s.hasTag("secondary")):
    #             newNode.append(s);
    #     return nodeset;

    def getUndistortedPrimaries(self):
        assert(False), "should not be using undistorted primaries anymore"
        # return self.getPrimarySamples();
        return self.getUndistortedSamples().get_with_tag(CaptureConstants.PRIMARY_CATEGORY_NAME);

    def getUndistortedSecondaries(self):
        # return self.getSecondarySamples();
        assert (False), "should not be using undistorted primaries anymore"
        return self.getUndistortedSamples().get_with_tag(CaptureConstants.SECONDARY_CATEGORY_NAME);

    def getAlignedPrimaries(self, method=None):
        return self.get_aligned_samples(method).get_with_tag(CaptureConstants.PRIMARY_CATEGORY_NAME);

    def getAlignedSecondaries(self, method):
        return self.get_aligned_samples(method).get_with_tag(CaptureConstants.SECONDARY_CATEGORY_NAME);

    def get_aligned_panos(self, method=None):
        return self.get_aligned_samples(method).get_with_tag(CaptureConstants.PANO_CATEGORY_NAME);

    # @property
    # def aligned_primaries(self):
    #     return self.aligned_samples.getWithTag(CaptureConstants.PRIMARY_SAMPLE_TAG);
    #
    # @property
    # def aligned_secondaries(self):
    #     return self.aligned_samples.getWithTag(CaptureConstants.SECONDARY_SAMPLE_TAG);

    def _addAlignedPrimarySampleForPath(self, path, method=None):
        new_primary = self.AddSampleForImagePath(path);
        self._add_aligned_labels(new_primary, method);
        self._add_primary_labels(new_primary);
        return new_primary;

    def _add_aligned_pano_sample_for_path(self, path, method=None):
        new_pano = self.add_sample_for_image_path(path);
        self._add_aligned_labels(new_pano, method);
        self._add_pano_labels(new_pano);
        return new_pano;

    def _addAlignedSecondarySampleForPath(self, path, method=None):
        new_s = self.AddSampleForImagePath(path);
        self._add_aligned_labels(new_s, method);
        self._add_secondary_labels(new_s);
        return new_s;

    def AddAlignedPrimaries(self, method=None, force_reload=False):
        self.add_images_subdir_if_missing(self._get_aligned_directory_name(self._getPrimaryDerivativeNamePart()));
        if (force_reload):
            aligned_primaries = self.getAlignedPrimaries();
            for e in aligned_primaries:
                self.samples.remove(e);
        aligned_primaries = FilePathList.from_directory_search(self.getAlignedPrimaryImagesDir(method), recursive=False,
                                                               extension_list=['.jpeg', '.jpg', '.png'],
                                                               criteriaFunc=FilePathList.NO_FILES_THAT_START_WITH_DOT);
        new_primary_samples = []
        for p in aligned_primaries:
            existing = self.getImageSampleForPath(p.absolute_file_path);
            if (existing is None):
                new_primary = self._addAlignedPrimarySampleForPath(p.absolute_file_path);
                new_primary_samples.append(new_primary);
        return self.create_sample_set(new_primary_samples);


    def AddAlignedSecondariesDirectory(self, load_existing_files=True):
        self.add_images_subdir_if_missing(self._get_aligned_directory_name(self.getAlignedSecondaryImagesDir()));
        new_samples = []
        if (load_existing_files):
            secondaries = FilePathList.from_directory_search(self.getAlignedSecondaryImagesDir(), recursive=False,
                                                             extension_list=['.jpeg', '.jpg', '.png'],
                                                             criteriaFunc=FilePathList.NO_FILES_THAT_START_WITH_DOT);
            for p in secondaries:
                existing = self.getImageSampleForPath(p.absolute_file_path);
                if (existing is None):
                    new_s = self.AddSampleForImagePath(p.absolute_file_path);
                    self._add_aligned_labels(new_s);
                    self._add_secondary_labels(new_s);
                    new_samples.append(new_s);
        return self.create_sample_set(new_samples);

# def CaptureSessionProduct(func):
#     setattr(CaptureTarget, func.__name__, func)
#     CaptureTarget.CAPTURE_SESSION_PRODUCTS[func.__name__]=func;
#     return getattr(Image, func.__name__);
#
# def CaptureSessionProduct(func):
#     setattr(CaptureTarget, func.__name__, func)
#     CaptureTarget.CAPTURE_SESSION_PRODUCTS[func.__name__]=func;
#     return getattr(Image, func.__name__);


# def CaptureSampleProduct()
