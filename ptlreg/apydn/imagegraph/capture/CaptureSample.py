from ptlreg.apydn import FileDataNode, DataNodeConstants
from ptlreg.apydn.imagegraph.capture.CaptureConstants import CaptureConstants
from ptlreg.apydn.imagegraph.imagefilesample.ImageFileSample import ImageFileSample
import pandas as pd
import pandera.pandas as pa
from .CaptureConstants import CaptureConstants
from ..imagefilesample.ImageFileSampleSet import ImageFileSampleSet
import warnings

class CaptureSampleDataNode(FileDataNode):
    DATANODE_SCHEMA = pa.DataFrameSchema({
        "file_path": pa.Column(str),
        DataNodeConstants.CREATED_TIMESTAMP_KEY: pa.Column(pa.dtypes.Timestamp),
        CaptureConstants.SESSION_KEY:pa.Column(str, required=False),
        CaptureConstants.TARGET_KEY: pa.Column(str, required=False),
        CaptureConstants.SAMPLE_TYPE_KEY: pa.Column(pa.Category, checks=pa.Check.isin(CaptureConstants.SAMPLE_TYPE_CATEGORIES))
    },
    index=pa.Index(str)
    );


class CaptureSampleNode(object):
    """
    CaptureSampleNode is a mixin class for Capture
    """
    pass;



class CaptureSample(CaptureSampleNode, ImageFileSample):
    DATANODE_MAP_TYPE = CaptureSampleDataNode;
    PANO_THUMBNAIL_PATH_KEY = 'pano_thumbnail_path';

    @property
    def _is_primary_sample(self):
        return self.has_tag(CaptureConstants.PRIMARY_CATEGORY_NAME);

    @property
    def _is_secondary_sample(self):
        return self.has_tag(CaptureConstants.SECONDARY_CATEGORY_NAME);

    @property
    def _is_original_sample(self):
        return self.has_tag(CaptureConstants.ORIGINAL_SAMPLE_TAG);

    # @property
    # def original_sample_type(self):
    #     return self.get_label_value(CaptureConstants.ORIGINAL_SAMPLE_TAG);

    #
    # # <editor-fold desc="Property: 'gps'">
    # @property
    # def gps(self):
    #     return self.get_info("gps");
    # @gps.setter
    # def gps(self, value):
    #     self.set_info('gps', value);
    # # </editor-fold>
    #
    # # <editor-fold desc="Property: 'sun_angle'">
    # @property
    # def sun_angle(self):
    #     np.array([self.sun_altitude, self.sun_azimuth]);
    # @sun_angle.setter
    # def sun_angle(self, value):
    #     assert(isinstance(value, np.ndarray)), "can only set sun_angle to an np.ndarray of [altitude, azimuth]";
    #     self.sun_altitude = value[0];
    #     self.sun_azimuth = value[1];
    # # </editor-fold>
    #
    # # <editor-fold desc="Property: 'sun_altitude'">
    # @property
    # def sun_altitude(self):
    #     return self.get_info("sun_altitude");
    # @sun_altitude.setter
    # def sun_altitude(self, value):
    #     self.set_info('sun_altitude', value);
    # # </editor-fold>
    #
    # # <editor-fold desc="Property: 'sun_azimuth'">
    # @property
    # def sun_azimuth(self):
    #     return self.get_info("sun_azimuth");
    # @sun_azimuth.setter
    # def sun_azimuth(self, value):
    #     self.set_info('sun_azimuth', value);
    # # </editor-fold>






class CaptureSampleSet(CaptureSampleNode, ImageFileSampleSet):
    ElementClass = CaptureSampleNode;
    # DATANODE_SET_MAP_TYPE = CaptureSampleDataNodeSet;

    @classmethod
    def new_element(cls, *args, **kwargs):
        """
        This is so that a more generic class ElementClass can be validated, while constructor will focus on a specific class.
        :param args:
        :param kwargs:
        :return:
        """
        return CaptureSample(*args, **kwargs);

    # def _validate_element(self, elem):
    #     nelem = elem;
    #     if(not isinstance(elem, CaptureSampleNode)):
    #         nelem = CaptureSample.for_path(elem.file_path);
    #         # nelem._ainfo = not elem._ainfo;
    #         warnings.warn("Changing element type from {} to {}".format(type(elem), self.__class__.ElementClass));
    #         nelem._set_data_labels(elem.data_labels)
    #         # nelem.data_labels = elem.data_labels;
    #     return super(CaptureSampleSet, self)._validate_element(nelem);

CaptureSampleSet.SUBSET_CLASS = CaptureSampleSet