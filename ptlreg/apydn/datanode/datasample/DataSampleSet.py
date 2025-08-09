from ptlreg.apy.core import AObjectOrderedSet
from ptlreg.apydn.datanode import DataNode
from ptlreg.apydn.datanode.datasample.MapsToDataNodeSetMixin import MapsToDataNodeSetMixin


class DataSampleSetMixin(MapsToDataNodeSetMixin):
    """
    A mixin for AObjects that map to DataNodeSets.
    Should be mixed with AObjectOrderedSet.
    """

    def __str__(self):
        rstr = "{} with {} samples:\n".format(self.__class__.__name__, len(self));
        for a in self:
            rstr = rstr + "{},\n".format(a._filename_or_none);
        return rstr;

    def get_sample_for_id(self, id):
        return self._main_index_map.get(id);

    def get_sample_list(self):
        return self.asList();


class DataSampleSetBase(AObjectOrderedSet):
    """
    A base class for DataSampleSet that inherits from AObjectOrderedSet.
    This should be mixed with a DataSampleSetMixin to create a full DataSampleSet class.
    """

    def __init__(self, *args, **kwargs):
        super(DataSampleSetBase, self).__init__(*args, **kwargs);
        self.init_node_id(*args, **kwargs);

    def init_node_id(self, *args, **kwargs):
        """
        Initializes the node_id if it is not set.
        This method should be called in the constructor of the subclass.
        """
        raise NotImplementedError("init_node_id must be overridden.");


class DataSampleSet(DataSampleSetMixin, DataSampleSetBase):
    """
    A DataSampleSet is a set of DataSamples that maps to a DataNodeSet.
    It is a mixin for AObjects that map to DataNodeSets.
    """

    def init_node_id(self, *args, **kwargs):
        """
        Initializes the node_id if it is not set.
        """
        if ((self.node_id is None)):
            self.set_node_id(DataNode.generate_node_id());