from ptlreg.apy.amedia import Image
from .CaptureSample import CaptureSampleSet
from ptlreg.apydn.imagegraph.imagefilesample.ImageFileSampleSet import *
from .ops import *
from ..ImageUtils import ExposureCorrect
from ... import DataNodeConstants
import numpy as np

class CaptureSession(CaptureSampleSet):
    SUBSET_CLASS = CaptureSampleSet

    # INDEX_MAP_LABEL_KEY = DataNodeConstants.CREATED_TIMESTAMP_KEY;

    def __init__(self, *args, **kwargs):
        self._capture_target = None;
        super(CaptureSession, self).__init__(*args, **kwargs);

    def _set_capture_target(self, capture_target):
        self._capture_target = capture_target;

    @property
    def capture_target(self):
        return self._capture_target;

    # def main_index_map_func(self, o):
    #     return o.get_label_value(DataNodeConstants.CREATED_TIMESTAMP_KEY);

    # @classmethod
    # def SubsetClass(cls):
    #     return ImageDatasetFileSampleSet;

    # <editor-fold desc="Property: '_primary_id'">
    @property
    def _primary_id(self):
        return self.get_info("_primary_id");

    @_primary_id.setter
    def _primary_id(self, value):
        self.set_info('_primary_id', value);
    # </editor-fold>

    @classmethod
    def pano_homography_test(cls, matrix, target_shape=None, **kwargs):
        detLin = np.linalg.det(matrix[:2,:2]);
        detHom = np.linalg.det(matrix);
        if(detLin>0 and detHom>0):
            return True;


    @property
    def primary_sample(self):
        return self.get_sample_for_id(self._primary_id);



    def get_secondary_samples(self):
        primary = self.primary_sample;
        slist = [];
        for sample in self:
            if(sample is not primary):
                slist.append(sample);
        return self.__class__.SubsetConstructor(slist);

    def _get_default_image_match_list_pairs(self):
        pairs = [];
        secondaries = self.get_secondary_samples();
        primary = self.primary_sample;
        for si in range(len(secondaries)):
            s = secondaries[si];
            pairs.append([s.file_path, primary.file_path]);
            for sii in range(si+1,len(secondaries)):
                ss = secondaries[sii];
                pairs.append([s.file_path, ss.file_path]);
        return pairs;


    @classmethod
    def create_with_primary(cls, primary_sample):
        rval = cls();
        rval.set_primary(primary_sample);
        return rval;

    def set_primary(self, primary_sample):
        self.append(primary_sample);
        self._primary_id=self.main_index_map_func(primary_sample);
        # self._set_data_labels(primary_sample.data_labels);
        # self._primary_id=primary_sample.node_id;
        # self.add_label_instance(self.primary_sample.get_label(DataNodeConstants.CREATED_TIMESTAMP_KEY).clone());
        self.set_timestamp(primary_sample.get_label_value(DataNodeConstants.CREATED_TIMESTAMP_KEY));

    # def getImageNodeID(self):
    #     return self._primary_id;
    #
    # def setImageNodeID(self, value):
    #     self._primary_id = value;

    @property
    def timestamp(self):
        return self.get_label_value(DataNodeConstants.CREATED_TIMESTAMP_KEY);


    @CaptureSessionProductOp("panorama_image")
    def get_pano_image(self, cdb=None, target_viewport_btlr='default', seam_blur='default', align_seq=False):
        '''
        viewport is relative to primary image
        :param target_viewport_btlr:
        :[int, int, int, int] target_viewport_btlr: [miny, maxy, minx, maxx]
        :return:
        :rtype:
        '''
        if (target_viewport_btlr is None):
            pshape = self.primary_sample.get_image().shape;
            target_viewport_btlr = [0, pshape[0], 0, pshape[1]];
        elif (target_viewport_btlr == 'default'):
            factor = 0.5;
            pshape = self.primary_sample.get_image().shape;
            target_viewport_btlr = [int(-pshape[0] * factor), int(pshape[0] * (1 + factor)), int(-pshape[1] * factor),
                                    int(pshape[1] * (1 + factor))];
        w = target_viewport_btlr[3] - target_viewport_btlr[2];
        h = target_viewport_btlr[1] - target_viewport_btlr[0];
        viewport_mat = np.array([[1, 0, -target_viewport_btlr[2]],
                                 [0, 1, -target_viewport_btlr[0]],
                                 [0, 0, 1]]).astype(float);

        primary = self.primary_sample;
        primaryIm = primary._get_image_warped_by_matrix(viewport_mat, output_shape=[h, w, 4], border_width=seam_blur);
        outputIm = primaryIm.get_rgba_copy().get_float_copy();

        def splat(im, toim=None):
            if (toim is None):
                toim = outputIm;
            splatIm, ratios = ExposureCorrect(im, toim)
            tpix = toim.fpixels;
            falphc = splatIm.fpixels[:, :, 3];
            talphc = toim.fpixels[:, :, 3];
            from_alpha_im = np.dstack((falphc, falphc, falphc));
            splatpix = np.clip(splatIm.pixels[:, :, :3], 0, 1);

            toim.pixels[:, :, :3] = splatpix * (from_alpha_im) + (1 - from_alpha_im) * tpix[:, :, :3];
            toim.pixels[:, :, 3] = np.clip(talphc + falphc, 0, 1);

        secondaries = self.get_secondary_samples();
        for si in range(secondaries.length()):
            s = secondaries[si];
            if (cdb is None and align_seq):
                alignto = outputIm.GetUIntCopy();

            if (s.file_name_base != primary.file_name_base):
                if (cdb is None):
                    raise NotImplementedError("should be using colmap for alignment!");
                    alignment_matrix, mask = s._get_alignment_matrix(alignto);
                else:
                    pkp, skp = cdb.get_keypoint_matches_for_sample_pair(primary, s)
                    if (pkp is not None and skp is not None):
                        pkp_use = pkp - np.array([target_viewport_btlr[2], target_viewport_btlr[0]])
                        alignment_matrix, mask = cv2.findHomography(skp, pkp_use, cv2.RANSAC, 5.0)
                    else:
                        alignment_matrix = None;

                labelkey = "alignment_pano_{}_{}_{}_{}".format(*target_viewport_btlr);
                if (not s.has_label(labelkey)):
                    s.add_label(key=labelkey, value=alignment_matrix);
                    # s.add_label_instance(LabelInstance(key=labelkey, value=alignment_matrix));
                else:
                    s.set_label_value(key=labelkey, value=alignment_matrix);
                if (alignment_matrix is not None):
                    good_homography = self.__class__.pano_homography_test(alignment_matrix, outputIm.shape);
                    if (good_homography):
                        warped_s = s._get_image_warped_by_matrix(alignment_matrix, output_shape=outputIm.shape,
                                                                 border_width=seam_blur);
                        splat(warped_s.get_float_copy());
                    # else:
                    #     warped_s = s._GetImageWarpedByMatrix(alignment_matrix, output_shape=outputIm.shape, border_width=seam_blur);
                    #     warped_s.show()
        splat(primaryIm);
        return outputIm;

    @CaptureSessionProductOp("composite_exposure")
    def get_composite_exposure(self, **kwargs):
        '''
        viewport is relative to primary image
        :param target_viewport_btlr:
        :[int, int, int, int] target_viewport_btlr: [miny, maxy, minx, maxx]
        :return:
        :rtype:
        '''
        return self._get_composite_exposure(primary = self.primary_sample, secondaries = self.get_secondary_samples())


    def _get_composite_exposure(self, primary, secondaries, **kwargs):
        '''
                viewport is relative to primary image
                :param target_viewport_btlr:
                :[int, int, int, int] target_viewport_btlr: [miny, maxy, minx, maxx]
                :return:
                :rtype:
                '''
        # primary = self.primary_sample;
        primaryIm = primary.get_image();
        outputIm = primaryIm.get_rgba_copy().get_float_copy();
        def splat(im, toim=None):
            if (toim is None):
                toim = outputIm;
            try:
                splatIm, ratios = ExposureCorrect(im, toim)
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

        # secondaries = self.getSecondarySamples();
        for si in range(secondaries.length()):
            s = secondaries[si];
            if (s.file_name_base != primary.file_name_base):
                splat(s.get_image());
        splat(primaryIm);
        return outputIm.get_with_alpha_divided(threshold=0.001);
        # return outputIm;

