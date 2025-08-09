from ptlreg.apy.core import AObjectOrderedSet
from .ImageFileSample import ImageFileSample
from ptlreg.apydn.imagegraph.ImageSampleConstants import ImageSampleConstants
from ptlreg.apydn.imagegraph.ImageSampleSet import ImageSampleSetMixin
from ptlreg.apydn.DataNodeConstants import DataNodeConstants
from ptlreg.apydn.datanode import DataNode
from ptlreg.apydn.datanode.datasample.DataSampleSet import DataSampleSetBase
from ptlreg.apydn.datanode.filedatanode import FileDataNodeSet


class ImageFileSampleSetMixin(ImageSampleSetMixin):
    DATANODE_MAP_TYPE = ImageFileSample;
    # DATANODE_SET_MAP_TYPE = None;
    ElementClass = ImageFileSample

    # INDEX_MAP_LABEL_KEY = ImageSampleConstants.DATASET_RELATIVE_PATH_KEY;
    INDEX_MAP_LABEL_KEY = DataNodeConstants.NODE_ID_KEY;


    def __init__(self, *args, **kwargs):
        super(ImageFileSampleSetMixin, self).__init__(*args, **kwargs);
        # if (self.node_id is None):
        #     self.init_node_id();

    def init_node_id(self, **kwargs):
        """
        Initializes the node_id if it is not set.
        This method should be called in the constructor of the subclass.
        """
        if (self.node_id is not None):
            return;

        if(self.has_label(DataNodeConstants.FILE_PATH_KEY)):
            self.set_node_id(self.file_path)
            return;
        else:
            raise ValueError("Cannot initialize node_id: no file path label found in ImageFileSampleSetMixin.");
            # self.set_node_id(DataNode.generate_node_id());

            # if(self.node_id != self.file_path):
            #     raise ValueError("Node ID is not the same as file path: {} != {}".format(self.node_id, self.file_path));

    def get_samples_with_name(self, sample_name):
        rval = [];
        for s in self:
            if(s.sample_name == sample_name):
                rval.append(s);
        return self.__class__(rval);

    def get_sample_name_list(self):
        rval = [];
        for s in self:
            rval.append(s.sample_name)
        return rval;

    # def main_index_map_func(self, o):
    #     return o.get_label_value(ImageSampleConstants.DATASET_RELATIVE_PATH_KEY);

    def calc_timestamps(self, force_use_metadata=False):
        """
        Calculate the timestamps for all file nodes in the set.
        This method should be called after the file paths are set.
        :param force_use_metadata: If True, use metadata to calculate timestamps.
        """
        for sample in self:
            sample._calc_timestamp(force_use_metadata=force_use_metadata);


    def calc_time_since_prev_sample(self, force_use_metadata=False):
        """
        Calculate the time since the previous sample for each sample in the set.
        :param force_use_metadata: force to use file metadata instead of file name
        :return:
        """

        self.calc_timestamps(force_use_metadata=force_use_metadata);
        self.sort_by_timestamp()
        last_s = self[0];
        for s in self:
            s.set_label_value(
                ImageSampleConstants.TIME_SINCE_PREV_SAMPLE_KEY,
                s.timestamp_datetime - last_s.timestamp_datetime
            );
            last_s = s;



class ImageFileSampleSet(ImageFileSampleSetMixin, DataSampleSetBase):
    """
    A set of ImageFileSample objects.
    This class is a concrete implementation of the ImageFileSampleSetMixin.
    It inherits from DataSampleSetBase to provide additional functionality for data samples.
    """
    DATANODE_MAP_TYPE = FileDataNodeSet;
    ElementClass = ImageFileSample;
    SUBSET_CLASS = None;

    def init_node_id(self, *args, **kwargs):
        """
        Initializes the node_id if it is not set.
        """
        if ((self.node_id is None)):
            self.set_node_id(DataNode.generate_node_id());



ImageFileSampleSet.SUBSET_CLASS = ImageFileSampleSet;