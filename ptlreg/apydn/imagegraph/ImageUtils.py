import warnings
import pysolar.solar as psol
from ptlreg.apy.amedia.media.Image import Image
import numpy as np

try:
    import cv2
    import copyreg
    def _pickle_keypoints(point):
        return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                              point.response, point.octave, point.class_id)
    copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)
except ImportError:
    warnings.warn("Failed to import opencv!")



def ImageWarpedByMatrix(image, matrix, output_size):
    # inliers = matrix_mask.sum()
    # rgbaIm = sourceIm.get_rgba_copy();
    warped = cv2.warpPerspective(image.ipixels, matrix, (output_size[1], output_size[0]), flags=cv2.INTER_LINEAR)
    # if(return_mask):
    #     mask = cv2.warpPerspective((np.ones(rgbaIm.shape[:2])*255).astype(np.uint8), matrix, (w, h), flags=cv2.INTER_LINEAR)
    rval = Image(pixels=warped);
    return rval;


def ExposureCorrect(image, reference, exposure_blend=None):
    encoding_gamma = 2.2;
    decoding_gamma = 1/2.2;

    imlin = np.power(image.fpixels[:, :, :3], decoding_gamma);
    reflin = np.power(reference.fpixels[:, :, :3], decoding_gamma)
    mask = reference.pixels[:, :, 3] * image.pixels[:, :, 3];
    ratios = [];
    for i in range(3):
        denom = np.sum(imlin[:, :, i] * mask);
        if (denom < 0.1):
            print("Sum of pixels in masked region is suspiciously low: {}".format(denom));
            raise ValueError("No pixels in mask!")
        ratios.append((np.sum(reflin[:, :, i] * mask)) / denom);
    ratios = np.array(ratios);
    corrected = image.get_float_copy();
    corrected.pixels[:, :, :3] = np.power(imlin * ratios, encoding_gamma);
    if (exposure_blend is not None):
        corrected.pixels[:, :, :3] = (corrected.fpixels[:, :, :3] * exposure_blend) + (image.fpixels[:, :, :3] * (1.0 - exposure_blend));
    return corrected, ratios;