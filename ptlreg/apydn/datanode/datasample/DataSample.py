from ptlreg.apydn.datanode import DataNode
from ptlreg.apydn.datanode.datasample.MapsToDataNodeMixin import MapsToDataNodeMixin
from ptlreg.apy.core.aobject.AObject import AObject


class DataSampleMixin(MapsToDataNodeMixin):
    """
    A mixin for AObjects that map to DataNodes.
    """

    def __init__(self, *args, **kwargs):
        super(DataSampleMixin, self).__init__(*args, **kwargs)



class DataSampleBase(AObject):
    """
    A base class for DataSample that inherits from AObject.
    This should be mixed with a DataSampleMixin to create a full DataSample class.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the ImageFileSample with a file path.
        :param path: The file path to the image file.
        :param root_path: The root path for the file.
        """
        super(DataSampleBase, self).__init__(*args, **kwargs);
        # self.init_node_id(*args, **kwargs);

    def init_node_id(self, *args, **kwargs):
        """
        Initializes the node_id if it is not set.
        This method should be called in the constructor of the subclass.
        """
        raise NotImplementedError("init_node_id must be overridden.");
        # if ((self.node_id is None) and (self.has_label(DataNodeConstants.FILE_PATH_KEY))):
        #     self.set_node_id(self.file_path)
        # else:
        #     if (self.node_id != self.file_path):
        #         raise ValueError("Node ID is not the same as file path: {} != {}".format(self.node_id, self.file_path));


class DataSample(DataSampleMixin, DataSampleBase):
    """
    A DataSample is a sample of data that maps to a DataNode.
    It is a mixin for AObjects that map to DataNodes.
    """

    def init_node_id(self, *args, **kwargs):
        """
        Initializes the node_id if it is not set.
        """
        if ((self.node_id is None)):
            self.set_node_id(DataNode.generate_node_id());