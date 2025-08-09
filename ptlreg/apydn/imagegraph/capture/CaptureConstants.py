

PRIMARY_TYPE = "primary"  # primary image in a session
SECONDARY_TYPE = "secondary"  # secondary image in a session
EXTRA_TYPE = "extra"  # extra image
VIDEO_FRAME_TYPE = "video_frame"  # frame from a video session
SESSION_PRODUCT_TYPE = "session_product"  # product computed as s

class CaptureConstants(object):
    """
    Constants for the Capture module.
    """
    SESSION_KEY = "capture_session";
    TARGET_KEY = "capture_target";
    SAMPLE_TYPE_KEY = "sample_type";
    SAMPLE_TYPE_CATEGORIES = [
        PRIMARY_TYPE,
        SECONDARY_TYPE,
        EXTRA_TYPE,
        VIDEO_FRAME_TYPE,
        SESSION_PRODUCT_TYPE
    ]

    PRIMARY_CATEGORY_NAME = "primary";
    SECONDARY_CATEGORY_NAME = "secondary";
    ORIGINAL_DEFAULT_CATEGORY_NAME = "original";
    PANO_CATEGORY_NAME = "pano"
    UNIVERSAL_PRIMARY_CATEGORY_NAME = "universal_primary";
    UNIVERSAL_SECONDARY_CATEGORY_NAME = "universal_secondary";

    UNDISTORTED_SAMPLE_TAG = "undistorted"
    ALIGNED_SAMPLE_TAG = "aligned_sample"
    ORIGINAL_SAMPLE_TAG = "original_sample"
    SESSION_PRODUCT_NAME_KEY = "session_product"
    SESSION_PRODUCT_TAG = "is_session_product"
    ORIGINALS_SUBDIR_NAME = 'originals'


