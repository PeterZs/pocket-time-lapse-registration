import six

from ptlreg.apy.core.aobject.AObject import register_serializable_class, AObject
from ptlreg.apydn import HasDataLabels, DataNodeConstants
from ptlreg.apydn.datanode.DataNode import register_node_class, DataNode
import pandas as pd
import numpy as np
import os

class MapsToDataNodeMixin(HasDataLabels):
    """
    MapsToDataNode is a mixin for objects that map to a DataNode.
    When added to a class, the class is assumed to inherit from AObject.
    """
    DATANODE_MAP_TYPE = DataNode # This should be set to the DataNode type that this class maps to.

    def __init__(self, *args, **kwargs):
        super(MapsToDataNodeMixin, self).__init__(*args, **kwargs);
        # if (self.node_id is None):
        #     self.init_node_id(**kwargs);


    def DataNode(self, label_keys=None):
        # raise NotImplementedError("Subclass needs to implement DataNode method that returns a DataNode instance.");
        return self.__class__.DATANODE_MAP_TYPE.from_series(series= self.data_labels_for_keys(label_keys))

    @property
    def node_id(self):
        return self.get_label("node_id");

    # @node_id.setter
    # def node_id(self, value):
    #     self.set_label("node_id", value);

    def set_node_id(self, node_id):
        self.set_label(DataNodeConstants.NODE_ID_KEY, node_id);

    def init_node_id(self, **kwargs):
        """
        Initializes the node_id if it is not set.
        This method should be called in the constructor of the subclass.
        """
        if self.node_id is None:
            self.node_id = self.__class__.DATANODE_MAP_TYPE.generate_node_id();

    @property
    def _filename_or_none(self):
        if (self.has_label(DataNodeConstants.FILE_PATH_KEY)):
            return os.path.basename(self.get_label_value(DataNodeConstants.FILE_PATH_KEY));
        else:
            return None;
            # return fnl.value;
        # else:
        #     return None;






