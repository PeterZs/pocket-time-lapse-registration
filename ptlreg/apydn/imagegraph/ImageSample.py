import warnings

from ptlreg.apy.core import AObject, datetime_from_formatted_timestamp_string
from ptlreg.apydn import DataNodeConstants
from ptlreg.apydn.datanode.datasample.MapsToDataNodeMixin import MapsToDataNodeMixin
import numpy as np

from .HasImageSampleLabelsMixin import HasImageSampleLabelsMixin
from .ImageSampleConstants import ImageSampleConstants
from ptlreg.apydn.datanode import DataNode
from ptlreg.apydn.datanode.datasample.DataSample import DataSampleMixin, DataSampleBase
from ptlreg.apydn.imagegraph.ImageUtils import *


class ImageSampleMixin(HasImageSampleLabelsMixin, DataSampleMixin):
    """
    ImageSample is a base class for image samples in the image graph. Whereas ImageNode can reference one or more images, ImageSample is assumed to reference just one.
    """
    def __init__(self, *args, **kwargs):
        super(ImageSampleMixin, self).__init__(*args, **kwargs);

    def get_aligned_with(self, target_sample, n_keypoints=50, n_checks=50, knn=2, **kwargs):
        secondary_warped_im, matrix, matrix_mask = self._get_aligned_with(target_sample=target_sample, n_keypoints=n_keypoints, n_checks=n_checks, knn=knn, **kwargs);
        return secondary_warped_im;

    def get_matrix_for_correspondences(self, source_keypoints, target_keypoints, target_viewport_btlr=None, seam_blur='default'):
        if (target_viewport_btlr is None):
            pshape = self.image_shape;
            target_viewport_btlr = [0, pshape[0], 0, pshape[1]];
        elif (target_viewport_btlr == 'default'):
            factor = 0.5;
            pshape = self.primary_sample.get_image().shape;
            target_viewport_btlr = [int(-pshape[0] * factor), int(pshape[0] * (1 + factor)), int(-pshape[1] * factor),
                                    int(pshape[1] * (1 + factor))];
        if (target_keypoints is not None and source_keypoints is not None):
            pkp_use = target_keypoints - np.array([target_viewport_btlr[2], target_viewport_btlr[0]])
            return cv2.findHomography(source_keypoints, pkp_use, cv2.RANSAC, 5.0)
        else:
            return None, None;

    def _get_alignment_matrix(self, target_sample, n_keypoints=50, n_checks=50, knn=2, **kwargs):
        if(isinstance(target_sample, Image)):
            targetIm = target_sample;
            sift = cv2.SIFT_create()
            gray = targetIm.GetGrayCopy().ipixels;
            mask = (np.ones_like(gray)*255).astype('uint8');
            keypoints, descriptors = sift.detectAndCompute(gray, mask);
            target_sift = dict(keypoints=keypoints, descriptors=descriptors);
        else:
            targetIm = target_sample.get_image();
            target_sift = target_sample.get_sift();
        target_keypoints = target_sift['keypoints'];
        target_descriptors = target_sift['descriptors'];
        source_sift = self.get_sift();
        source_keypoints = source_sift['keypoints'];
        source_descriptors = source_sift['descriptors'];
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=n_checks)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        # find correspondence points between 2 frames by SIFT features.
        matches = flann.knnMatch(target_descriptors, source_descriptors, k=knn);
        # getting good matches
        good_points=[]
        # looking at two nearest neighbors, m and n
        for m, n in matches:
            good_points.append((m, m.distance/n.distance)) # what is with m.distance/n.distance?

        # Sort the correspondence points by confidence, by default we only use the best 50.
        good_points.sort(key=lambda y: y[1])
        output_points = np.float32([target_keypoints[m.queryIdx]
                                   .pt for m,d in good_points[:n_keypoints]]).reshape(-1, 1, 2)
        input_points = np.float32([source_keypoints[m.trainIdx]
                                  .pt for m,d in good_points[:n_keypoints]]).reshape(-1, 1, 2)

        # targetIm.show();

        # Compute homography by the correspondence pairs
        matrix, mask = cv2.findHomography(input_points, output_points, cv2.RANSAC, 5.0)
        return matrix, mask;

    def _get_aligned_with(self, target_sample, border_width=None, n_keypoints=50, n_checks=50, knn=2, return_mask = False, **kwargs):
        alignment_matrix, mask = self._get_alignment_matrix(target_sample, n_keypoints=n_keypoints, n_checks=n_checks, knn=knn, **kwargs);
        sourceIm = self.get_image();
        if(border_width == 'default'):
            border_width = int(min(sourceIm.width, sourceIm.height)*0.05);
        if(border_width is not None):
            sourceIm = sourceIm.get_with_tapered_alpha_boundary(border_width);
        rgbaIm = self.get_image().get_rgba_copy();
        targetIm = target_sample.get_image();
        target_shape = targetIm.shape;
        secondary_to_primary = cv2.warpPerspective(rgbaIm.ipixels, alignment_matrix, (target_shape[1], target_shape[0]), flags=cv2.INTER_LINEAR)
        if(return_mask):
            mask = cv2.warpPerspective((np.ones(rgbaIm.shape[:2])*255).astype(np.uint8), alignment_matrix, (target_shape[1], target_shape[0]), flags=cv2.INTER_LINEAR)
        secondary_warped_im = Image(pixels=secondary_to_primary);
        return secondary_warped_im, alignment_matrix, mask

    def _set_alignment_details(self, alignment_matrix, input_shape, alignment_target_path=None):
        h, w = input_shape[:2];
        # corners = np.array([
        #     np.array([0, 0, 1]),
        #     np.array([w, 0, 1]),
        #     np.array([w, h, 1]),
        #     np.array([0, h, 1]),
        # ]);
        errors = []
        # corners_after = [];
        # for c in corners:
        #     tc = alignment_matrix @ c;
        #     corners_after.append(tc);
        #     diff = tc - c;
        #     diff = diff[:2] * np.array([1 / w, 1 / h]);
        #     errors.append(np.linalg.norm(diff, 2))

        # corners_after = np.ndarray()
        output_shape=self.image_shape;

        alignment_details = dict(
            alignment_matrix=alignment_matrix.tolist(),
            input_shape=input_shape.tolist(),
            output_shape=output_shape.tolist(),
            # corners_in=corners.tolist(),
            # corners_out=corners_after.tolist(),
            alignment_target=alignment_target_path
        )
        if (not self.has_alignment_details):
            self.add_label(key=ImageSampleConstants.ALIGNMENT_DETAILS_KEY, value=alignment_details);
            # self.add_label_instance(LabelInstance(key=ImageSampleConstants.ALIGNMENT_DETAILS_KEY, value=alignment_details));

    def get_alignment_details(self):
        return self.get_label_value(ImageSampleConstants.ALIGNMENT_DETAILS_KEY);

    def get_alignment_matrix(self):
        adets = self.get_alignment_details()
        matlist = adets['alignment_matrix'];
        return np.ndarray(matlist);


    @property
    def has_alignment_details(self):
        return self.has_label(ImageSampleConstants.ALIGNMENT_DETAILS_KEY);


    def _get_image_warped_by_matrix(self, matrix, output_shape=None, border_width=None, radial_alpha = False, alpha_exponent=None, alpha_scale=None, multiply_alpha=None, min_alpha=0):

        im = self.get_image().get_rgba_copy().get_float_copy()
        if(radial_alpha):
              im = im.get_with_radial_alpha(exponent=alpha_exponent, scale=alpha_scale, multiply_alpha=multiply_alpha, min_alpha=min_alpha);
        # if(border_width is 'default'):
        #     border_width = int(min(im.width, im.height)*Image);
        border_alpha = None;
        if(border_width is not None):
            border_alpha = im.GetTaperedBorderAlpha(border_width);
        if(output_shape is None):
            output_shape = im.shape;
        # print(rgbaIm.ipixels)
        # wpix = rgbaIm.GetUIntCopy().pixels;
        # print(matrix);
        # print(wpix.shape)
        # print(wpix.shape.dtype)
        # warped_pix = cv2.warpPerspective(wpix, matrix, (output_shape[1], output_shape[0]), flags=cv2.INTER_LINEAR)
        warped_pix = cv2.warpPerspective(im.ipixels, matrix, (output_shape[1], output_shape[0]), flags=cv2.INTER_LINEAR)
        warped_pix = warped_pix/255.0;
        if(border_alpha is not None):
            border_alpha = cv2.warpPerspective((border_alpha*255).astype(np.uint8), matrix, (output_shape[1], output_shape[0]), flags=cv2.INTER_LINEAR)/255.0
            warped_pix[:,:,3]=warped_pix[:,:,3]*(border_alpha);
            # print(warped_pix)
        # if(return_mask):
        #     mask = cv2.warpPerspective((np.ones(rgbaIm.shape[:2])*255).astype(np.uint8), matrix, (w, h), flags=cv2.INTER_LINEAR)
        warped_im = Image(pixels=warped_pix);
        return warped_im;


    # <editor-fold desc="BInfo Property: 'sift_features'">
    def get_sift(self):
        rval = self._getBInfo("sift_features");
        if(rval is None):
            sift = cv2.SIFT_create()
            gray = self.get_image().GetGrayCopy().ipixels;
            mask = (np.ones_like(gray)*255).astype('uint8');
            keypoints, descriptors = sift.detectAndCompute(gray, mask);
            rval = dict(keypoints=keypoints, descriptors=descriptors);
            self._setBInfo("sift_features", rval);
        return rval;
    # </editor-fold>



    @classmethod
    def for_path(cls, path):
        raise NotImplementedError;

    def get_image(self):
        raise NotImplementedError;

    @property
    def image_shape(self):
        if(self.has_label(ImageSampleConstants.IMAGE_SHAPE_KEY)):
            return np.array(self.get_label_value(ImageSampleConstants.IMAGE_SHAPE_KEY));
        else:
            image = self.get_image();
            return image.shape;


    @image_shape.setter
    def image_shape(self, value):
        oldValue = self.get_label_value(ImageSampleConstants.IMAGE_SHAPE_KEY);
        if(oldValue is not None):
            if(not np.array_equal(oldValue,value)):
                warnings.warn("Old shape value {} is different from new one {}".format(oldValue, value), UserWarning);
            self.set_label_value(ImageSampleConstants.IMAGE_SHAPE_KEY, list(value));
        else:
            self.add_label(key=ImageSampleConstants.IMAGE_SHAPE_KEY, value=list(value));
            # self.add_label_instance(LabelInstance(key=ImageSampleConstants.IMAGE_SHAPE_KEY, value=list(value)));



    @property
    def sun_angle(self):
        return np.array([self.get_label_value(ImageSampleConstants.SUN_ALTITUDE_KEY), self.get_label_value(ImageSampleConstants.SUN_AZIMUTH_KEY)]);
        # return self.get_label_value(ImageSampleConstants.SUN_ANGLE_KEY);

    @sun_angle.setter
    def sun_angle(self, value):
        self.sun_altitude=value[0];
        self.sun_azimuth=value[1];
        # if (self.labels.get(ImageSampleConstants.SUN_ANGLE_KEY) is None):
        #     self.addVec2Label(key=ImageSampleConstants.SUN_ANGLE_KEY, value=value);
        # else:
        #     self.setLabelValue(ImageSampleConstants.SUN_ANGLE_KEY, value);

    @property
    def sun_altitude(self):
        return self.get_label_value(ImageSampleConstants.SUN_ALTITUDE_KEY);

    @sun_altitude.setter
    def sun_altitude(self, value):
        if (self.labels.get(ImageSampleConstants.SUN_ALTITUDE_KEY) is None):
            self.addScalarLabel(key=ImageSampleConstants.SUN_ALTITUDE_KEY, value=value);
        else:
            self.set_label_value(ImageSampleConstants.SUN_ALTITUDE_KEY, value);
        # if (self.labels.get(ImageSampleConstants.SUN_ANGLE_KEY) is None):
        #     self.addVec2Label(key=ImageSampleConstants.SUN_ANGLE_KEY, value=value);
        # else:
        #     self.setLabelValue(ImageSampleConstants.SUN_ANGLE_KEY, value);
        # return self.get_label(ImageSampleConstants.SUN_ANGLE_KEY)[0];

    @property
    def sun_azimuth(self):
        return self.get_label_value(ImageSampleConstants.SUN_AZIMUTH_KEY);

    @sun_azimuth.setter
    def sun_azimuth(self, value):
        if (self.labels.get(ImageSampleConstants.SUN_AZIMUTH_KEY) is None):
            self.addScalarLabel(key=ImageSampleConstants.SUN_AZIMUTH_KEY, value=value);
        else:
            self.set_label_value(ImageSampleConstants.SUN_AZIMUTH_KEY, value);

    def calculate_sun_angle(self):
        if((self.gps is not None) and (self.timestamp is not None)):
            sun_args = [self.gps[0], self.gps[1], self.timestamp];
            self.sun_altitude = psol.get_altitude(*sun_args);
            self.sun_azimuth = psol.get_azimuth(*sun_args);
            return self.sun_angle;
        else:
            return None;


class ImageSample(ImageSampleMixin, DataSampleBase):
    """
    ImageSample is a base class for image samples in the image graph. Whereas ImageNode can reference one or more images, ImageSample is assumed to reference just one.
    """

    def init_node_id(self, *args, **kwargs):
        """
        Initializes the node_id if it is not set.
        """
        if ((self.node_id is None)):
            self.set_node_id(DataNode.generate_node_id());

